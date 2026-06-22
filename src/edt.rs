use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::volume::BinaryVolume;

const INF: f64 = 1.0e30;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Phase {
    Foreground,
    Background,
}

impl Phase {
    #[inline(always)]
    pub fn is_selected(self, value: u8) -> bool {
        match self {
            Self::Foreground => value != 0,
            Self::Background => value == 0,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Foreground => "foreground",
            Self::Background => "background",
        }
    }
}

#[derive(Default)]
struct Scratch {
    input: Vec<f64>,
    output: Vec<f64>,
    vertices: Vec<usize>,
    boundaries: Vec<f64>,
}

impl Scratch {
    fn resize(&mut self, n: usize) {
        self.input.resize(n, 0.0);
        self.output.resize(n, 0.0);
        self.vertices.resize(n, 0);
        self.boundaries.resize(n + 1, 0.0);
    }
}

/// Exact Euclidean distance transform of the selected binary phase.
/// Selected voxels receive their distance to the nearest voxel in the opposite phase.
pub fn euclidean_distance_transform(volume: &BinaryVolume, phase: Phase) -> Result<Vec<f32>> {
    if volume.data.iter().all(|&value| phase.is_selected(value)) {
        bail!(
            "distance transform is undefined for an all-{} volume unless an exterior convention is specified",
            phase.name()
        );
    }

    let width = volume.width;
    let height = volume.height;
    let depth = volume.depth;
    let slice_len = volume.slice_len();

    let mut data: Vec<f64> = volume
        .data
        .par_iter()
        .map(|&value| if phase.is_selected(value) { INF } else { 0.0 })
        .collect();

    // X pass: each row is contiguous.
    data.par_chunks_mut(width)
        .for_each_init(Scratch::default, |scratch, row| {
            scratch.resize(width);
            scratch.input.copy_from_slice(row);
            transform_1d(
                &scratch.input,
                row,
                &mut scratch.vertices,
                &mut scratch.boundaries,
            );
        });

    // Y pass: each z-slice is independent.
    data.par_chunks_mut(slice_len)
        .for_each_init(Scratch::default, |scratch, slice| {
            scratch.resize(height);

            for x in 0..width {
                for y in 0..height {
                    scratch.input[y] = slice[y * width + x];
                }

                transform_1d(
                    &scratch.input,
                    &mut scratch.output,
                    &mut scratch.vertices,
                    &mut scratch.boundaries,
                );

                for y in 0..height {
                    slice[y * width + x] = scratch.output[y];
                }
            }
        });

    // Z pass: each output chunk stores one complete z-line contiguously.
    let mut z_lines = vec![0.0f64; data.len()];
    z_lines.par_chunks_mut(depth).enumerate().for_each_init(
        Scratch::default,
        |scratch, (line_id, line_out)| {
            scratch.resize(depth);

            for z in 0..depth {
                scratch.input[z] = data[z * slice_len + line_id];
            }

            transform_1d(
                &scratch.input,
                line_out,
                &mut scratch.vertices,
                &mut scratch.boundaries,
            );
        },
    );

    drop(data);

    // Convert the transposed z-line layout directly to ordinary z/y/x layout.
    let mut distances = vec![0.0f32; volume.len()];
    distances
        .par_chunks_mut(slice_len)
        .enumerate()
        .for_each(|(z, slice)| {
            for line_id in 0..slice_len {
                let index = z * slice_len + line_id;

                if phase.is_selected(volume.data[index]) {
                    slice[line_id] = z_lines[line_id * depth + z].sqrt() as f32;
                }
            }
        });

    Ok(distances)
}

/// Felzenszwalb-Huttenlocher lower-envelope transform in O(n).
fn transform_1d(f: &[f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    debug_assert_eq!(d.len(), n);
    debug_assert!(v.len() >= n);
    debug_assert!(z.len() > n);

    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;

    for q in 1..n {
        let qf = q as f64;
        let mut p = v[k];
        let mut s = intersection(f, p, q, qf);

        while s <= z[k] {
            if k == 0 {
                break;
            }
            k -= 1;
            p = v[k];
            s = intersection(f, p, q, qf);
        }

        if k == 0 && s <= z[0] {
            v[0] = q;
            z[0] = f64::NEG_INFINITY;
            z[1] = f64::INFINITY;
        } else {
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f64::INFINITY;
        }
    }

    let mut envelope = 0usize;
    for (q, output) in d.iter_mut().enumerate() {
        let qf = q as f64;

        while z[envelope + 1] < qf {
            envelope += 1;
        }

        let p = v[envelope];
        let delta = qf - p as f64;
        *output = delta * delta + f[p];
    }
}

#[inline(always)]
fn intersection(f: &[f64], p: usize, q: usize, qf: f64) -> f64 {
    let pf = p as f64;
    ((f[q] + qf * qf) - (f[p] + pf * pf)) / (2.0 * (qf - pf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_selection_direction_is_correct() {
        assert!(Phase::Foreground.is_selected(1));
        assert!(Phase::Foreground.is_selected(255));
        assert!(!Phase::Foreground.is_selected(0));

        assert!(Phase::Background.is_selected(0));
        assert!(!Phase::Background.is_selected(1));
        assert!(!Phase::Background.is_selected(255));
    }

    #[test]
    fn edt_center_background() {
        let mut data = vec![1u8; 27];
        data[13] = 0;

        let volume = BinaryVolume::new(data, 3, 3, 3).unwrap();
        let edt = euclidean_distance_transform(&volume, Phase::Foreground).unwrap();

        assert_eq!(edt[13], 0.0);
        assert!(
            (edt[12] - 1.0).abs() < 1e-6,
            "expected edt[12] = 1, got {}; edt = {:?}",
            edt[12],
            edt
        );
        assert!(
            (edt[0] - 3.0f32.sqrt()).abs() < 1e-6,
            "expected edt[0] = sqrt(3), got {}; edt = {:?}",
            edt[0],
            edt
        );
    }

    #[test]
    fn background_phase_uses_foreground_as_sites() {
        let mut data = vec![0u8; 27];
        data[13] = 1;

        let volume = BinaryVolume::new(data, 3, 3, 3).unwrap();
        let edt = euclidean_distance_transform(&volume, Phase::Background).unwrap();

        assert_eq!(edt[13], 0.0);
        assert!((edt[12] - 1.0).abs() < 1e-6);
        assert!((edt[0] - 3.0f32.sqrt()).abs() < 1e-6);
    }
}
