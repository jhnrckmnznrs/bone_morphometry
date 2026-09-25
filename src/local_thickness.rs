use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Result;
use rayon::prelude::*;

use crate::edt::{euclidean_distance_transform_squared, Phase};
use crate::volume::BinaryVolume;

/// Compute a BoneJ/Fiji-compatible local-thickness map in voxel units.
///
/// The implementation follows the discrete construction used by Fiji's
/// `LocalThicknessWrapper`: exact EDT, distance-ridge selection, propagation
/// of the largest discrete sphere, cleanup of jagged surface values, and
/// masking with the selected input phase. Returned values are diameters, not
/// radii. No implicit exterior padding is introduced.
pub fn local_thickness(volume: &BinaryVolume, phase: Phase) -> Result<Vec<f32>> {
    let squared_distances = euclidean_distance_transform_squared(volume, phase)?;
    let (distance_values, distance_indices) = distance_index(&squared_distances);
    let templates = create_templates(&distance_values);
    let ridges = distance_ridges(
        volume,
        phase,
        &squared_distances,
        &distance_indices,
        &templates,
    );

    let propagated = propagate_ridges(volume, &ridges);
    let cleaned = clean_up_surface_jaggies(&propagated, volume.width, volume.height, volume.depth);

    Ok(cleaned
        .into_par_iter()
        .zip(volume.data.par_iter())
        .map(
            |(value, &voxel)| {
                if phase.is_selected(voxel) {
                    value
                } else {
                    0.0
                }
            },
        )
        .collect())
}

fn distance_index(squared_distances: &[u32]) -> (Vec<u32>, Vec<usize>) {
    let maximum = squared_distances.iter().copied().max().unwrap_or(0) as usize;
    let mut occurs = vec![false; maximum + 1];
    for &distance in squared_distances {
        occurs[distance as usize] = true;
    }

    let distance_values: Vec<u32> = occurs
        .iter()
        .enumerate()
        .filter_map(|(value, &present)| present.then_some(value as u32))
        .collect();

    let mut distance_indices = vec![usize::MAX; maximum + 1];
    for (index, &value) in distance_values.iter().enumerate() {
        distance_indices[value as usize] = index;
    }

    (distance_values, distance_indices)
}

fn create_templates(distance_values: &[u32]) -> [Vec<u32>; 3] {
    [
        scan_cube(1, 0, 0, distance_values),
        scan_cube(1, 1, 0, distance_values),
        scan_cube(1, 1, 1, distance_values),
    ]
}

/// Port of the search-template construction in Fiji's `Distance_Ridge`.
fn scan_cube(dx: u32, dy: u32, dz: u32, distance_values: &[u32]) -> Vec<u32> {
    distance_values
        .iter()
        .map(|&radius_squared| {
            let radius = integer_sqrt(radius_squared) + 1;
            let mut maximum = 0u32;

            for k in 0..=radius {
                let scan_k = k * k;
                let dk = (k + dz) * (k + dz);

                for j in 0..=radius {
                    let scan_kj = scan_k + j * j;
                    if scan_kj > radius_squared {
                        continue;
                    }

                    let i_plus = integer_sqrt(radius_squared - scan_kj) + dx;
                    let candidate = dk + (j + dy) * (j + dy) + i_plus * i_plus;
                    maximum = maximum.max(candidate);
                }
            }

            maximum
        })
        .collect()
}

fn distance_ridges(
    volume: &BinaryVolume,
    phase: Phase,
    squared_distances: &[u32],
    distance_indices: &[usize],
    templates: &[Vec<u32>; 3],
) -> Vec<(usize, usize, usize, u32)> {
    let slice_len = volume.slice_len();

    (0..volume.len())
        .into_par_iter()
        .filter_map(|index| {
            if !phase.is_selected(volume.data[index]) {
                return None;
            }

            let radius_squared = squared_distances[index];
            if radius_squared == 0 {
                return None;
            }

            let z = index / slice_len;
            let remainder = index - z * slice_len;
            let y = remainder / volume.width;
            let x = remainder - y * volume.width;
            let radius_index = distance_indices[radius_squared as usize];

            for dz in -1isize..=1 {
                let nz = z as isize + dz;
                if nz < 0 || nz >= volume.depth as isize {
                    continue;
                }

                for dy in -1isize..=1 {
                    let ny = y as isize + dy;
                    if ny < 0 || ny >= volume.height as isize {
                        continue;
                    }

                    for dx in -1isize..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let nx = x as isize + dx;
                        if nx < 0 || nx >= volume.width as isize {
                            continue;
                        }

                        let components =
                            (dx != 0) as usize + (dy != 0) as usize + (dz != 0) as usize;
                        let neighbor = volume.index(nx as usize, ny as usize, nz as usize);
                        let required = templates[components - 1][radius_index];

                        if squared_distances[neighbor] >= required {
                            return None;
                        }
                    }
                }
            }

            Some((x, y, z, radius_squared))
        })
        .collect()
}

fn propagate_ridges(volume: &BinaryVolume, ridges: &[(usize, usize, usize, u32)]) -> Vec<f32> {
    let output: Vec<AtomicU32> = (0..volume.len()).map(|_| AtomicU32::new(0)).collect();

    ridges.par_iter().for_each(|&(x, y, z, radius_squared)| {
        let radius = ceiling_sqrt(radius_squared) as usize;
        let x_start = x.saturating_sub(radius);
        let y_start = y.saturating_sub(radius);
        let z_start = z.saturating_sub(radius);
        let x_stop = x.saturating_add(radius).min(volume.width - 1);
        let y_stop = y.saturating_add(radius).min(volume.height - 1);
        let z_stop = z.saturating_add(radius).min(volume.depth - 1);
        let radius_squared_u64 = u64::from(radius_squared);

        for nz in z_start..=z_stop {
            let dz = nz.abs_diff(z) as u64;
            let dz_squared = dz * dz;

            for ny in y_start..=y_stop {
                let dy = ny.abs_diff(y) as u64;
                let dyz_squared = dz_squared + dy * dy;
                if dyz_squared > radius_squared_u64 {
                    continue;
                }

                for nx in x_start..=x_stop {
                    let dx = nx.abs_diff(x) as u64;
                    if dyz_squared + dx * dx <= radius_squared_u64 {
                        output[volume.index(nx, ny, nz)]
                            .fetch_max(radius_squared, Ordering::Relaxed);
                    }
                }
            }
        }
    });

    output
        .into_par_iter()
        .map(|value| (2.0_f64 * (value.into_inner() as f64).sqrt()) as f32)
        .collect()
}

/// Port of Fiji's `Clean_Up_Local_Thickness` surface correction.
fn clean_up_surface_jaggies(input: &[f32], width: usize, height: usize, depth: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let index = index_3d(x, y, z, width, height);
                let value = input[index];
                if value == 0.0 {
                    continue;
                }

                let borders_background = NEIGHBORS_26.iter().any(|&(dx, dy, dz)| {
                    look(
                        input,
                        x as isize + dx,
                        y as isize + dy,
                        z as isize + dz,
                        width,
                        height,
                        depth,
                    ) == 0.0
                });

                output[index] = if borders_background { -1.0 } else { value };
            }
        }
    }

    // BoneJ's cleanup is intentionally order-dependent: processed surface
    // values are stored as negative numbers and are excluded from later
    // neighborhood averages until the final absolute-value pass.
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let index = index_3d(x, y, z, width, height);
                if output[index] != -1.0 {
                    continue;
                }

                let mut count = 0usize;
                let mut sum = 0.0f32;
                for &(dx, dy, dz) in &NEIGHBORS_26 {
                    let value = look(
                        &output,
                        x as isize + dx,
                        y as isize + dy,
                        z as isize + dz,
                        width,
                        height,
                        depth,
                    );
                    if value > 0.0 {
                        count += 1;
                        sum += value;
                    }
                }

                let corrected = if count > 0 {
                    sum / count as f32
                } else {
                    input[index]
                };
                output[index] = -corrected;
            }
        }
    }

    output.par_iter_mut().for_each(|value| *value = value.abs());
    output
}

// Exact visitation order used by Fiji's Clean_Up_Local_Thickness. Keeping the
// order also keeps floating-point summation behavior aligned with ImageJ.
const NEIGHBORS_26: [(isize, isize, isize); 26] = [
    (0, 0, -1),
    (0, 0, 1),
    (0, -1, 0),
    (0, 1, 0),
    (-1, 0, 0),
    (1, 0, 0),
    (0, 1, -1),
    (0, 1, 1),
    (1, -1, 0),
    (1, 1, 0),
    (-1, 0, 1),
    (1, 0, 1),
    (0, -1, -1),
    (0, -1, 1),
    (-1, -1, 0),
    (-1, 1, 0),
    (-1, 0, -1),
    (1, 0, -1),
    (1, 1, 1),
    (1, -1, 1),
    (-1, 1, 1),
    (-1, -1, 1),
    (1, 1, -1),
    (1, -1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
];

#[allow(clippy::too_many_arguments)]
#[inline]
fn look(
    values: &[f32],
    x: isize,
    y: isize,
    z: isize,
    width: usize,
    height: usize,
    depth: usize,
) -> f32 {
    if x < 0 || y < 0 || z < 0 || x >= width as isize || y >= height as isize || z >= depth as isize
    {
        -1.0
    } else {
        values[index_3d(x as usize, y as usize, z as usize, width, height)]
    }
}

#[inline]
fn index_3d(x: usize, y: usize, z: usize, width: usize, height: usize) -> usize {
    (z * height + y) * width + x
}

#[inline]
fn integer_sqrt(value: u32) -> u32 {
    let mut root = (value as f64).sqrt() as u32;
    while u64::from(root + 1) * u64::from(root + 1) <= u64::from(value) {
        root += 1;
    }
    while u64::from(root) * u64::from(root) > u64::from(value) {
        root -= 1;
    }
    root
}

#[inline]
fn ceiling_sqrt(value: u32) -> u32 {
    let floor = integer_sqrt(value);
    if floor * floor == value {
        floor
    } else {
        floor + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolated_voxel_has_diameter_two() {
        let mut data = vec![0u8; 27];
        data[13] = 1;
        let volume = BinaryVolume::new(data, 3, 3, 3).unwrap();

        let thickness = local_thickness(&volume, Phase::Foreground).unwrap();

        assert!((thickness[13] - 2.0).abs() < 1e-6);
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

    #[test]
    fn search_template_matches_known_single_voxel_radius() {
        let templates = create_templates(&[1]);
        assert_eq!(templates[0][0], 4);
        assert_eq!(templates[1][0], 5);
        assert_eq!(templates[2][0], 6);
    }
}
