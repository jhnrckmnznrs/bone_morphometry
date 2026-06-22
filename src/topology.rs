use anyhow::{ensure, Context, Result};

use crate::volume::BinaryVolume;

#[derive(Clone, Copy, Debug)]
pub struct Topology {
    pub corrected_euler: f64,
    pub beta0: usize,
    pub beta1: usize,
    pub beta2: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct CubicalMeasures {
    pub raw_euler: i64,
    pub surface_face_count: i64,
    pub mean_breadth_lattice_units: f64,
}

pub fn analyze_topology(
    volume: &BinaryVolume,
    raw_euler: i64,
    include_boundary_correction: bool,
) -> Result<Topology> {
    let corrected_euler = if include_boundary_correction {
        raw_euler as f64 - bonej_edge_correction(volume)
    } else {
        f64::NAN
    };
    let beta0 = count_components(volume, true, Connectivity::TwentySix);
    let beta2 = count_enclosed_background_components_6(volume);

    let beta0_i64 = i64::try_from(beta0).context("beta0 does not fit in i64")?;
    let beta2_i64 = i64::try_from(beta2).context("beta2 does not fit in i64")?;
    let beta1_i64 = beta0_i64
        .checked_add(beta2_i64)
        .and_then(|sum| sum.checked_sub(raw_euler))
        .context("beta1 calculation overflowed i64")?;
    ensure!(
        beta1_i64 >= 0,
        "computed a negative beta1 ({beta1_i64}); check the 26/6 topology convention"
    );
    let beta1 = usize::try_from(beta1_i64).context("beta1 does not fit in usize")?;

    Ok(Topology {
        corrected_euler,
        beta0,
        beta1,
        beta2,
    })
}

/// Counts the cells of the closed cubical complex formed by foreground voxels.
/// The implied digital topology is 26-connected foreground and 6-connected background.
pub fn cubical_measures(volume: &BinaryVolume) -> CubicalMeasures {
    let w = volume.width;
    let h = volume.height;
    let d = volume.depth;

    let cubes = volume.data.iter().filter(|&&value| value != 0).count() as i64;

    let mut shared_faces = 0i64;
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                if volume.get(x, y, z) == 0 {
                    continue;
                }
                if x + 1 < w && volume.get(x + 1, y, z) != 0 {
                    shared_faces += 1;
                }
                if y + 1 < h && volume.get(x, y + 1, z) != 0 {
                    shared_faces += 1;
                }
                if z + 1 < d && volume.get(x, y, z + 1) != 0 {
                    shared_faces += 1;
                }
            }
        }
    }
    let faces = 6 * cubes - shared_faces;

    let mut edges = 0i64;
    // x-directed lattice edges
    for zv in 0..=d {
        for yv in 0..=h {
            for x in 0..w {
                if any_foreground(
                    volume,
                    &[
                        (x as isize, yv as isize - 1, zv as isize - 1),
                        (x as isize, yv as isize, zv as isize - 1),
                        (x as isize, yv as isize - 1, zv as isize),
                        (x as isize, yv as isize, zv as isize),
                    ],
                ) {
                    edges += 1;
                }
            }
        }
    }
    // y-directed lattice edges
    for zv in 0..=d {
        for y in 0..h {
            for xv in 0..=w {
                if any_foreground(
                    volume,
                    &[
                        (xv as isize - 1, y as isize, zv as isize - 1),
                        (xv as isize, y as isize, zv as isize - 1),
                        (xv as isize - 1, y as isize, zv as isize),
                        (xv as isize, y as isize, zv as isize),
                    ],
                ) {
                    edges += 1;
                }
            }
        }
    }
    // z-directed lattice edges
    for z in 0..d {
        for yv in 0..=h {
            for xv in 0..=w {
                if any_foreground(
                    volume,
                    &[
                        (xv as isize - 1, yv as isize - 1, z as isize),
                        (xv as isize, yv as isize - 1, z as isize),
                        (xv as isize - 1, yv as isize, z as isize),
                        (xv as isize, yv as isize, z as isize),
                    ],
                ) {
                    edges += 1;
                }
            }
        }
    }

    let mut vertices = 0i64;
    for zv in 0..=d {
        for yv in 0..=h {
            for xv in 0..=w {
                let mut occupied = false;
                'search: for dz in [-1isize, 0] {
                    for dy in [-1isize, 0] {
                        for dx in [-1isize, 0] {
                            if volume.get_signed(
                                xv as isize + dx,
                                yv as isize + dy,
                                zv as isize + dz,
                            ) {
                                occupied = true;
                                break 'search;
                            }
                        }
                    }
                }
                if occupied {
                    vertices += 1;
                }
            }
        }
    }

    let raw_euler = vertices - edges + faces - cubes;
    let surface_face_count = 2 * (faces - 3 * cubes);
    let mean_breadth_lattice_units = 0.5 * (edges - 2 * faces + 3 * cubes) as f64;

    CubicalMeasures {
        raw_euler,
        surface_face_count,
        mean_breadth_lattice_units,
    }
}

#[inline(always)]
fn any_foreground(volume: &BinaryVolume, coordinates: &[(isize, isize, isize)]) -> bool {
    coordinates
        .iter()
        .any(|&(x, y, z)| volume.get_signed(x, y, z))
}

/// Exact port of BoneJ's stack-boundary correction.
pub fn bonej_edge_correction(volume: &BinaryVolume) -> f64 {
    let f = stack_vertices(volume);
    let e = stack_edges(volume) + 3 * f;
    let c = stack_faces(volume) + 2 * e - 3 * f;
    let d = edge_vertices(volume) + f;
    let a = face_vertices(volume);
    let b = face_edges(volume);

    let chi_zero = f as f64;
    let chi_one = (d - e) as f64;
    let chi_two = (a - b + c) as f64;
    chi_two / 2.0 + chi_one / 4.0 + chi_zero / 8.0
}

#[inline]
fn end_indices(length: usize) -> [usize; 2] {
    [0, length - 1]
}

fn stack_vertices(v: &BinaryVolume) -> i64 {
    let mut count = 0;
    for z in end_indices(v.depth) {
        for y in end_indices(v.height) {
            for x in end_indices(v.width) {
                count += i64::from(v.get(x, y, z) != 0);
            }
        }
    }
    count
}

fn stack_edges(v: &BinaryVolume) -> i64 {
    let mut count = 0;
    for z in end_indices(v.depth) {
        for y in end_indices(v.height) {
            for x in 1..v.width - 1 {
                count += i64::from(v.get(x, y, z) != 0);
            }
        }
    }
    for z in end_indices(v.depth) {
        for x in end_indices(v.width) {
            for y in 1..v.height - 1 {
                count += i64::from(v.get(x, y, z) != 0);
            }
        }
    }
    for y in end_indices(v.height) {
        for x in end_indices(v.width) {
            for z in 1..v.depth - 1 {
                count += i64::from(v.get(x, y, z) != 0);
            }
        }
    }
    count
}

fn stack_faces(v: &BinaryVolume) -> i64 {
    let mut count = 0;
    for z in end_indices(v.depth) {
        for y in 1..v.height - 1 {
            for x in 1..v.width - 1 {
                count += i64::from(v.get(x, y, z) != 0);
            }
        }
    }
    for y in end_indices(v.height) {
        for z in 1..v.depth - 1 {
            for x in 1..v.width - 1 {
                count += i64::from(v.get(x, y, z) != 0);
            }
        }
    }
    for x in end_indices(v.width) {
        for y in 1..v.height - 1 {
            for z in 1..v.depth - 1 {
                count += i64::from(v.get(x, y, z) != 0);
            }
        }
    }
    count
}

fn face_vertices(v: &BinaryVolume) -> i64 {
    let mut count = 0;
    for z in end_indices(v.depth) {
        for y in 0..=v.height {
            for x in 0..=v.width {
                if v.get_signed(x as isize, y as isize, z as isize)
                    || v.get_signed(x as isize, y as isize - 1, z as isize)
                    || v.get_signed(x as isize - 1, y as isize - 1, z as isize)
                    || v.get_signed(x as isize - 1, y as isize, z as isize)
                {
                    count += 1;
                }
            }
        }
    }
    for x in end_indices(v.width) {
        for y in 0..=v.height {
            for z in 1..v.depth {
                if v.get_signed(x as isize, y as isize, z as isize)
                    || v.get_signed(x as isize, y as isize - 1, z as isize)
                    || v.get_signed(x as isize, y as isize - 1, z as isize - 1)
                    || v.get_signed(x as isize, y as isize, z as isize - 1)
                {
                    count += 1;
                }
            }
        }
    }
    for y in end_indices(v.height) {
        for x in 1..v.width {
            for z in 1..v.depth {
                if v.get_signed(x as isize, y as isize, z as isize)
                    || v.get_signed(x as isize, y as isize, z as isize - 1)
                    || v.get_signed(x as isize - 1, y as isize, z as isize - 1)
                    || v.get_signed(x as isize - 1, y as isize, z as isize)
                {
                    count += 1;
                }
            }
        }
    }
    count
}

fn face_edges(v: &BinaryVolume) -> i64 {
    let mut count = 0;
    for z in end_indices(v.depth) {
        for y in 0..=v.height {
            for x in 0..=v.width {
                if v.get_signed(x as isize, y as isize, z as isize) {
                    count += 2;
                } else {
                    count += i64::from(v.get_signed(x as isize, y as isize - 1, z as isize));
                    count += i64::from(v.get_signed(x as isize - 1, y as isize, z as isize));
                }
            }
        }
    }
    for y in end_indices(v.height) {
        for z in 1..v.depth {
            for x in 0..v.width {
                if v.get_signed(x as isize, y as isize, z as isize)
                    || v.get_signed(x as isize, y as isize, z as isize - 1)
                {
                    count += 1;
                }
            }
        }
    }
    for y in end_indices(v.height) {
        for z in 0..v.depth {
            for x in 0..=v.width {
                if v.get_signed(x as isize, y as isize, z as isize)
                    || v.get_signed(x as isize - 1, y as isize, z as isize)
                {
                    count += 1;
                }
            }
        }
    }
    for x in end_indices(v.width) {
        for z in 1..v.depth {
            for y in 0..v.height {
                if v.get_signed(x as isize, y as isize, z as isize)
                    || v.get_signed(x as isize, y as isize, z as isize - 1)
                {
                    count += 1;
                }
            }
        }
    }
    for x in end_indices(v.width) {
        for z in 0..v.depth {
            for y in 1..v.height {
                if v.get_signed(x as isize, y as isize, z as isize)
                    || v.get_signed(x as isize, y as isize - 1, z as isize)
                {
                    count += 1;
                }
            }
        }
    }
    count
}

fn edge_vertices(v: &BinaryVolume) -> i64 {
    let mut count = 0;
    for z in end_indices(v.depth) {
        for y in end_indices(v.height) {
            for x in 1..v.width {
                if v.get(x, y, z) != 0 || v.get(x - 1, y, z) != 0 {
                    count += 1;
                }
            }
        }
    }
    for z in end_indices(v.depth) {
        for x in end_indices(v.width) {
            for y in 1..v.height {
                if v.get(x, y, z) != 0 || v.get(x, y - 1, z) != 0 {
                    count += 1;
                }
            }
        }
    }
    for x in end_indices(v.width) {
        for y in end_indices(v.height) {
            for z in 1..v.depth {
                if v.get(x, y, z) != 0 || v.get(x, y, z - 1) != 0 {
                    count += 1;
                }
            }
        }
    }
    count
}

#[derive(Clone, Copy)]
enum Connectivity {
    Six,
    TwentySix,
}

fn count_components(volume: &BinaryVolume, foreground: bool, connectivity: Connectivity) -> usize {
    let mut visited = vec![false; volume.len()];
    let mut components = 0usize;
    let mut stack = Vec::new();

    for seed in 0..volume.len() {
        let matches_phase = (volume.data[seed] != 0) == foreground;
        if !matches_phase || visited[seed] {
            continue;
        }
        components += 1;
        flood(
            volume,
            seed,
            foreground,
            connectivity,
            &mut visited,
            &mut stack,
        );
    }
    components
}

fn count_enclosed_background_components_6(volume: &BinaryVolume) -> usize {
    let mut visited = vec![false; volume.len()];
    let mut stack = Vec::new();

    // First inspect the two z-facing boundary planes:
    // z = 0 and z = depth - 1.
    for y in 0..volume.height {
        for x in 0..volume.width {
            flood_boundary_background(volume, volume.index(x, y, 0), &mut visited, &mut stack);

            flood_boundary_background(
                volume,
                volume.index(x, y, volume.depth - 1),
                &mut visited,
                &mut stack,
            );
        }
    }

    // Inspect the two y-facing boundary planes.
    //
    // Exclude z = 0 and z = depth - 1 because those voxels were
    // already inspected by the preceding loops.
    for z in 1..volume.depth - 1 {
        for x in 0..volume.width {
            flood_boundary_background(volume, volume.index(x, 0, z), &mut visited, &mut stack);

            flood_boundary_background(
                volume,
                volume.index(x, volume.height - 1, z),
                &mut visited,
                &mut stack,
            );
        }
    }

    // Inspect the two x-facing boundary planes.
    //
    // Exclude the y and z edges because those voxels were already
    // inspected by the preceding loops.
    for z in 1..volume.depth - 1 {
        for y in 1..volume.height - 1 {
            flood_boundary_background(volume, volume.index(0, y, z), &mut visited, &mut stack);

            flood_boundary_background(
                volume,
                volume.index(volume.width - 1, y, z),
                &mut visited,
                &mut stack,
            );
        }
    }

    // Any background voxel that remains unvisited cannot reach the
    // exterior through 6-connectivity, so it belongs to a cavity.
    let mut cavities = 0usize;

    for seed in 0..volume.len() {
        if volume.data[seed] == 0 && !visited[seed] {
            cavities += 1;

            flood(
                volume,
                seed,
                false,
                Connectivity::Six,
                &mut visited,
                &mut stack,
            );
        }
    }

    cavities
}

#[inline]
fn flood_boundary_background(
    volume: &BinaryVolume,
    seed: usize,
    visited: &mut [bool],
    stack: &mut Vec<usize>,
) {
    if volume.data[seed] == 0 && !visited[seed] {
        flood(volume, seed, false, Connectivity::Six, visited, stack);
    }
}

fn flood(
    volume: &BinaryVolume,
    seed: usize,
    foreground: bool,
    connectivity: Connectivity,
    visited: &mut [bool],
    stack: &mut Vec<usize>,
) {
    stack.clear();
    stack.push(seed);
    visited[seed] = true;
    let slice_len = volume.slice_len();

    while let Some(index) = stack.pop() {
        let z = index / slice_len;
        let remainder = index - z * slice_len;
        let y = remainder / volume.width;
        let x = remainder - y * volume.width;

        match connectivity {
            Connectivity::Six => {
                const OFFSETS: [(isize, isize, isize); 6] = [
                    (-1, 0, 0),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                ];
                for &(dx, dy, dz) in &OFFSETS {
                    visit_neighbor(volume, x, y, z, dx, dy, dz, foreground, visited, stack);
                }
            }
            Connectivity::TwentySix => {
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            visit_neighbor(volume, x, y, z, dx, dy, dz, foreground, visited, stack);
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn visit_neighbor(
    volume: &BinaryVolume,
    x: usize,
    y: usize,
    z: usize,
    dx: isize,
    dy: isize,
    dz: isize,
    foreground: bool,
    visited: &mut [bool],
    stack: &mut Vec<usize>,
) {
    let nx = x as isize + dx;
    let ny = y as isize + dy;
    let nz = z as isize + dz;
    if nx < 0
        || ny < 0
        || nz < 0
        || nx >= volume.width as isize
        || ny >= volume.height as isize
        || nz >= volume.depth as isize
    {
        return;
    }
    let neighbor = volume.index(nx as usize, ny as usize, nz as usize);
    if !visited[neighbor] && ((volume.data[neighbor] != 0) == foreground) {
        visited[neighbor] = true;
        stack.push(neighbor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn volume_with_points(points: &[(usize, usize, usize)]) -> BinaryVolume {
        let mut data = vec![0u8; 125];
        for &(x, y, z) in points {
            data[(z * 5 + y) * 5 + x] = 1;
        }
        BinaryVolume::new(data, 5, 5, 5).unwrap()
    }

    #[test]
    fn single_voxel_cubical_measures() {
        let v = volume_with_points(&[(2, 2, 2)]);
        let measures = cubical_measures(&v);
        assert_eq!(measures.raw_euler, 1);
        assert_eq!(measures.surface_face_count, 6);
        assert!((measures.mean_breadth_lattice_units - 1.5).abs() < 1e-12);
    }

    #[test]
    fn two_face_connected_voxels_form_a_two_by_one_by_one_box() {
        let v = volume_with_points(&[(1, 1, 1), (2, 1, 1)]);
        let measures = cubical_measures(&v);
        assert_eq!(measures.raw_euler, 1);
        assert_eq!(measures.surface_face_count, 10);
        assert!((measures.mean_breadth_lattice_units - 2.0).abs() < 1e-12);
    }

    #[test]
    fn corner_touching_voxels_are_one_26_component() {
        let v = volume_with_points(&[(1, 1, 1), (2, 2, 2)]);
        let measures = cubical_measures(&v);
        let topology = analyze_topology(&v, measures.raw_euler, false).unwrap();
        assert_eq!(measures.raw_euler, 1);
        assert_eq!(topology.beta0, 1);
        assert_eq!(topology.beta1, 0);
        assert_eq!(topology.beta2, 0);
    }

    #[test]
    fn separated_voxels_have_two_components() {
        let v = volume_with_points(&[(1, 1, 1), (3, 3, 3)]);
        let measures = cubical_measures(&v);
        let topology = analyze_topology(&v, measures.raw_euler, false).unwrap();
        assert_eq!(measures.raw_euler, 2);
        assert_eq!(topology.beta0, 2);
        assert_eq!(topology.beta1, 0);
        assert_eq!(topology.beta2, 0);
    }

    #[test]
    fn hollow_cube_has_one_cavity_and_euler_two() {
        let mut data = vec![0u8; 125];
        for z in 1..=3 {
            for y in 1..=3 {
                for x in 1..=3 {
                    if x == 1 || x == 3 || y == 1 || y == 3 || z == 1 || z == 3 {
                        data[(z * 5 + y) * 5 + x] = 1;
                    }
                }
            }
        }
        let v = BinaryVolume::new(data, 5, 5, 5).unwrap();
        let measures = cubical_measures(&v);
        let topology = analyze_topology(&v, measures.raw_euler, false).unwrap();
        assert_eq!(measures.raw_euler, 2);
        assert_eq!(topology.beta0, 1);
        assert_eq!(topology.beta1, 0);
        assert_eq!(topology.beta2, 1);
    }
}
