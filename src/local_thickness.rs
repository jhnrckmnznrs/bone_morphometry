use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::edt::{euclidean_distance_transform, Phase};
use crate::volume::BinaryVolume;

/// Reproduces localthickness.local_thickness(..., scale=1) for a selected 3-D phase.
pub fn local_thickness(volume: &BinaryVolume, phase: Phase) -> Result<Vec<f32>> {
    let radii = euclidean_distance_transform(volume, phase)?;
    let max_radius = radii.par_iter().copied().reduce(|| 0.0, f32::max).floor() as usize;

    if max_radius == 0 {
        return Ok(radii);
    }

    let denominator = 6.0f32.sqrt() + 3.0f32.sqrt() + 2.0f32.sqrt();
    let face_weight = 6.0f32.sqrt() / denominator;
    let edge_weight = 3.0f32.sqrt() / denominator;
    let corner_weight = 2.0f32.sqrt() / denominator;

    let width = volume.width;
    let height = volume.height;
    let depth = volume.depth;
    let slice_len = volume.slice_len();

    // Add a one-voxel zero halo around the volume. This makes every neighbor
    // access valid and removes coordinate bounds checks from the hot loop.
    let padded_width = width.checked_add(2).context("padded width overflow")?;
    let padded_height = height.checked_add(2).context("padded height overflow")?;
    let padded_depth = depth.checked_add(2).context("padded depth overflow")?;

    let padded_slice_len = padded_width
        .checked_mul(padded_height)
        .context("padded slice size overflow")?;

    let padded_len = padded_slice_len
        .checked_mul(padded_depth)
        .context("padded volume size overflow")?;

    // Copy the ordinary EDT volume into the center of the padded volume.
    let mut current = vec![0.0f32; padded_len];

    current
        .par_chunks_mut(padded_slice_len)
        .enumerate()
        .skip(1)
        .take(depth)
        .for_each(|(padded_z, padded_slice)| {
            let z = padded_z - 1;

            for y in 0..height {
                let source_start = z * slice_len + y * width;
                let destination_start = (y + 1) * padded_width + 1;

                padded_slice[destination_start..destination_start + width]
                    .copy_from_slice(&radii[source_start..source_start + width]);
            }
        });

    // Fixed linear offsets in the padded volume.
    let sx = 1isize;
    let sy = padded_width as isize;
    let sz = padded_slice_len as isize;

    let face_offsets = [-sx, sx, -sy, sy, -sz, sz];

    let edge_offsets = [
        -sx - sy,
        -sx + sy,
        sx - sy,
        sx + sy,
        -sx - sz,
        -sx + sz,
        sx - sz,
        sx + sz,
        -sy - sz,
        -sy + sz,
        sy - sz,
        sy + sz,
    ];

    let corner_offsets = [
        -sx - sy - sz,
        -sx - sy + sz,
        -sx + sy - sz,
        -sx + sy + sz,
        sx - sy - sz,
        sx - sy + sz,
        sx + sy - sz,
        sx + sy + sz,
    ];

    let mut next = vec![0.0f32; padded_len];

    for radius in 0..max_radius {
        let threshold = radius as f32;

        next.par_chunks_mut(padded_slice_len)
            .enumerate()
            .skip(1)
            .take(depth)
            .for_each(|(padded_z, output_slice)| {
                let slice_base = padded_z * padded_slice_len;

                for y in 1..=height {
                    let row_base = y * padded_width;

                    for x in 1..=width {
                        let local_index = row_base + x;
                        let index = slice_base + local_index;
                        let center = current[index];

                        if center <= threshold {
                            output_slice[local_index] = center;
                            continue;
                        }

                        let face_max = max_at_offsets(&current, index, &face_offsets);

                        let edge_max = max_at_offsets(&current, index, &edge_offsets);

                        let corner_max = max_at_offsets(&current, index, &corner_offsets);

                        output_slice[local_index] = face_weight * face_max
                            + edge_weight * edge_max
                            + corner_weight * corner_max;
                    }
                }
            });

        std::mem::swap(&mut current, &mut next);
    }

    // Remove the halo and return the data in the normal volume layout.
    let mut output = vec![0.0f32; volume.len()];

    output
        .par_chunks_mut(slice_len)
        .enumerate()
        .for_each(|(z, output_slice)| {
            let padded_slice = &current[(z + 1) * padded_slice_len..(z + 2) * padded_slice_len];

            for y in 0..height {
                let source_start = (y + 1) * padded_width + 1;
                let destination_start = y * width;

                output_slice[destination_start..destination_start + width]
                    .copy_from_slice(&padded_slice[source_start..source_start + width]);
            }
        });

    Ok(output)
}

#[inline(always)]
fn max_at_offsets(values: &[f32], index: usize, offsets: &[isize]) -> f32 {
    let base = index as isize;
    let mut maximum = values[index];

    for &offset in offsets {
        maximum = maximum.max(values[(base + offset) as usize]);
    }

    maximum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolated_voxel_has_radius_one() {
        let mut data = vec![0u8; 27];
        data[13] = 1;

        let volume = BinaryVolume::new(data, 3, 3, 3).unwrap();

        let thickness = local_thickness(&volume, Phase::Foreground).unwrap();

        assert!((thickness[13] - 1.0).abs() < 1e-6);
        assert_eq!(thickness.iter().filter(|&&value| value > 0.0).count(), 1);
    }

    #[test]
    fn background_phase_matches_complemented_foreground_phase() {
        let mut data = vec![0u8; 125];

        for z in 1..4 {
            for y in 1..4 {
                for x in 1..4 {
                    data[(z * 5 + y) * 5 + x] = 1;
                }
            }
        }

        let volume = BinaryVolume::new(data, 5, 5, 5).unwrap();

        let complement = volume.complement();

        let direct = local_thickness(&volume, Phase::Background).unwrap();

        let complemented = local_thickness(&complement, Phase::Foreground).unwrap();

        assert_eq!(direct, complemented);
    }
}
