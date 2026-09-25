use anyhow::{bail, Result};

use crate::volume::BinaryVolume;

#[derive(Clone, Copy, Debug)]
pub struct DaParameters {
    pub directions: usize,
    pub lines: usize,
    pub sampling_increment_voxels: f64,
    pub repetitions: usize,
    pub seed: u64,
}

impl Default for DaParameters {
    fn default() -> Self {
        Self {
            directions: 1024,
            lines: 64,
            sampling_increment_voxels: 3.0_f64.sqrt(),
            repetitions: 5,
            seed: 0x424f4e454a5f4441,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DaStats {
    pub mean: f64,
    pub sd: f64,
    pub cv: f64,
    pub min: f64,
    pub max: f64,
    pub principal_angle_to_z_mean_deg: f64,
    pub radius_a_mean: f64,
    pub radius_b_mean: f64,
    pub radius_c_mean: f64,
    pub domain_side_voxels: usize,
    pub domain_depth_voxels: usize,
}

/// Native MIL-based degree of anisotropy.
///
/// The measurement definition follows BoneJ2: sample mean-intercept-length
/// vectors, fit the full translated quadric, and report
/// `DA = 1 - lambda_min / lambda_max`, where the eigenvalues are inverse
/// squared ellipsoid radii. Sampling is stochastic but explicitly seeded here,
/// unlike the normal BoneJ2 wrapper UI.
pub fn degree_of_anisotropy(
    bone: &BinaryVolume,
    roi: &BinaryVolume,
    params: DaParameters,
) -> Result<DaStats> {
    if params.directions < 9 {
        bail!("DA requires at least 9 directions");
    }
    if params.lines == 0 || params.repetitions == 0 {
        bail!("DA lines and repetitions must be positive");
    }
    if !params.sampling_increment_voxels.is_finite()
        || params.sampling_increment_voxels < 3.0_f64.sqrt() - 1e-12
    {
        bail!("DA sampling increment must be finite and at least sqrt(3) voxels");
    }
    if !bone.same_shape(roi) {
        bail!("DA binary and ROI shapes differ");
    }

    let domain = largest_central_square_inside_all_slices(roi)?;
    let mut da_values = Vec::with_capacity(params.repetitions);
    let mut angles = Vec::with_capacity(params.repetitions);
    let mut radii_a = Vec::with_capacity(params.repetitions);
    let mut radii_b = Vec::with_capacity(params.repetitions);
    let mut radii_c = Vec::with_capacity(params.repetitions);

    for repetition in 0..params.repetitions {
        let mut rng =
            SplitMix64::new(params.seed ^ (repetition as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15));
        let mut points = Vec::<[f64; 3]>::with_capacity(params.directions);
        for _ in 0..params.directions {
            // BoneJ samples each direction from an independent isotropic rotation.
            // The same rotation also orients the stratified plane of parallel lines.
            let rotation = random_rotation(&mut rng);
            let direction = rotate(rotation, [0.0, 0.0, 1.0]);
            if let Some(mil) = mil_for_direction(bone, domain, rotation, params, &mut rng) {
                points.push([direction[0] * mil, direction[1] * mil, direction[2] * mil]);
            }
        }
        if points.len() < 9 {
            bail!("DA produced only {} valid MIL directions", points.len());
        }
        let (eigenvalues, eigenvectors) = fit_general_quadric(&points)?;
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));
        let lmin = eigenvalues[order[0]];
        let lmid = eigenvalues[order[1]];
        let lmax = eigenvalues[order[2]];
        if !(lmin > 0.0 && lmid > 0.0 && lmax > 0.0) {
            bail!("DA quadric is not positive definite: {:?}", eigenvalues);
        }
        let da = 1.0 - lmin / lmax;
        let principal = [
            eigenvectors[0][order[0]],
            eigenvectors[1][order[0]],
            eigenvectors[2][order[0]],
        ];
        let angle = principal[2].abs().clamp(0.0, 1.0).acos().to_degrees();
        da_values.push(da);
        angles.push(angle);
        // BoneJ radii are commonly printed shortest -> longest as a,b,c.
        radii_a.push(1.0 / lmax.sqrt());
        radii_b.push(1.0 / lmid.sqrt());
        radii_c.push(1.0 / lmin.sqrt());
    }

    let da_mean = mean(&da_values);
    let da_sd = sample_sd(&da_values);
    Ok(DaStats {
        mean: da_mean,
        sd: da_sd,
        cv: if da_mean != 0.0 {
            da_sd.abs() / da_mean.abs()
        } else {
            f64::NAN
        },
        min: da_values.iter().copied().fold(f64::INFINITY, f64::min),
        max: da_values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        principal_angle_to_z_mean_deg: mean(&angles),
        radius_a_mean: mean(&radii_a),
        radius_b_mean: mean(&radii_b),
        radius_c_mean: mean(&radii_c),
        domain_side_voxels: domain.side,
        domain_depth_voxels: bone.depth,
    })
}

#[derive(Clone, Copy, Debug)]
struct SquareDomain {
    x0: usize,
    y0: usize,
    side: usize,
    depth: usize,
}

fn largest_central_square_inside_all_slices(roi: &BinaryVolume) -> Result<SquareDomain> {
    let w = roi.width;
    let h = roi.height;
    let mut common = vec![true; w * h];
    for z in 0..roi.depth {
        for y in 0..h {
            for x in 0..w {
                if roi.get(x, y, z) == 0 {
                    common[y * w + x] = false;
                }
            }
        }
    }
    let stride = w + 1;
    let mut prefix = vec![0usize; (w + 1) * (h + 1)];
    for y in 0..h {
        let mut row_sum = 0usize;
        for x in 0..w {
            row_sum += usize::from(!common[y * w + x]);
            prefix[(y + 1) * stride + (x + 1)] = prefix[y * stride + (x + 1)] + row_sum;
        }
    }
    for side in (1..=w.min(h)).rev() {
        let x0 = (w - side) / 2;
        let y0 = (h - side) / 2;
        let x1 = x0 + side;
        let y1 = y0 + side;
        let outside =
            prefix[y1 * stride + x1] - prefix[y0 * stride + x1] - prefix[y1 * stride + x0]
                + prefix[y0 * stride + x0];
        if outside == 0 {
            return Ok(SquareDomain {
                x0,
                y0,
                side,
                depth: roi.depth,
            });
        }
    }
    bail!("could not find a non-empty central square inside the ROI on all slices")
}

fn mil_for_direction(
    bone: &BinaryVolume,
    domain: SquareDomain,
    rotation: [[f64; 3]; 3],
    params: DaParameters,
    rng: &mut SplitMix64,
) -> Option<f64> {
    let diagonal = ((domain.side * domain.side * 2 + domain.depth * domain.depth) as f64).sqrt();
    let target_length = params.lines as f64 * diagonal;
    let center = [
        domain.x0 as f64 + domain.side as f64 / 2.0,
        domain.y0 as f64 + domain.side as f64 / 2.0,
        domain.depth as f64 / 2.0,
    ];
    let direction = rotate(rotation, [0.0, 0.0, 1.0]);
    let plane_x = rotate(rotation, [1.0, 0.0, 0.0]);
    let plane_y = rotate(rotation, [0.0, 1.0, 0.0]);

    // BoneJ's PlaneParallelLineGenerator divides a square plane of side equal
    // to the domain diagonal into floor(sqrt(lines))^2 strata. Each cycle uses
    // one common random offset within every stratum and a shuffled visit order.
    let sections = (params.lines as f64).sqrt().floor() as usize;
    if sections == 0 {
        return None;
    }
    let section_size = 1.0 / sections as f64;
    let mut order = (0..sections * sections).collect::<Vec<_>>();
    let mut cycle = 0usize;
    let mut u_offset = 0.0;
    let mut t_offset = 0.0;

    let mut total_length = 0.0;
    let mut total_intercepts = 0u64;
    let min = [domain.x0 as f64, domain.y0 as f64, 0.0];
    let max = [
        (domain.x0 + domain.side) as f64,
        (domain.y0 + domain.side) as f64,
        domain.depth as f64,
    ];

    while target_length - total_length > 1e-12 {
        if cycle == 0 {
            shuffle_in_place(&mut order, rng);
            u_offset = rng.next_f64() * section_size;
            t_offset = rng.next_f64() * section_size;
        }
        let idx = order[cycle];
        let u_section = idx / sections;
        let t_section = idx - u_section * sections;
        let u = u_section as f64 * section_size + u_offset;
        let t = t_section as f64 * section_size + t_offset;
        let x = (t - 0.5) * diagonal;
        let y = (u - 0.5) * diagonal;
        let point = [
            center[0] + x * plane_x[0] + y * plane_y[0],
            center[1] + x * plane_x[1] + y * plane_y[1],
            center[2] + x * plane_x[2] + y * plane_y[2],
        ];
        cycle += 1;
        if cycle >= order.len() {
            cycle = 0;
        }

        let Some((mut t0, mut t1)) = line_box_intersection(point, direction, min, max) else {
            continue;
        };
        if t1 < t0 {
            std::mem::swap(&mut t0, &mut t1);
        }
        let available = t1 - t0;
        if available <= 0.0 {
            continue;
        }
        if total_length + available > target_length {
            t1 = t0 + (target_length - total_length);
        }
        let length = t1 - t0;
        if length <= 0.0 {
            continue;
        }

        let start_t = t0 + rng.next_f64() * params.sampling_increment_voxels;
        let samples = ((t1 - start_t) / params.sampling_increment_voxels).ceil() as isize;
        if samples < 1 {
            continue;
        }
        let mut previous = false;
        let mut phase_changes = 0u64;
        for sample in 0..samples as usize {
            let tt = start_t + sample as f64 * params.sampling_increment_voxels;
            let p = [
                point[0] + tt * direction[0],
                point[1] + tt * direction[1],
                point[2] + tt * direction[2],
            ];
            let x = p[0].floor() as isize;
            let y = p[1].floor() as isize;
            let z = p[2].floor() as isize;
            let current = if x >= domain.x0 as isize
                && y >= domain.y0 as isize
                && z >= 0
                && x < (domain.x0 + domain.side) as isize
                && y < (domain.y0 + domain.side) as isize
                && z < domain.depth as isize
            {
                bone.get(x as usize, y as usize, z as usize) != 0
            } else {
                false
            };
            // BoneJ ParallelLineMIL counts every phase transition, not only
            // background->foreground entries.
            if current != previous {
                phase_changes += 1;
            }
            previous = current;
        }
        total_length += length;
        total_intercepts += phase_changes;
    }

    if total_length <= 0.0 {
        None
    } else {
        Some(total_length / total_intercepts.max(1) as f64)
    }
}

fn shuffle_in_place(values: &mut [usize], rng: &mut SplitMix64) {
    for i in (1..values.len()).rev() {
        let j = (rng.next_f64() * (i + 1) as f64).floor() as usize;
        values.swap(i, j.min(i));
    }
}

fn line_box_intersection(
    point: [f64; 3],
    direction: [f64; 3],
    min: [f64; 3],
    max: [f64; 3],
) -> Option<(f64, f64)> {
    let mut tmin = f64::NEG_INFINITY;
    let mut tmax = f64::INFINITY;
    for axis in 0..3 {
        if direction[axis].abs() < 1e-14 {
            if point[axis] < min[axis] || point[axis] > max[axis] {
                return None;
            }
            continue;
        }
        let mut a = (min[axis] - point[axis]) / direction[axis];
        let mut b = (max[axis] - point[axis]) / direction[axis];
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        tmin = tmin.max(a);
        tmax = tmax.min(b);
        if tmax <= tmin {
            return None;
        }
    }
    // Match JOML Intersectiond.intersectRayAab, which BoneJ uses: the
    // intersection must lie at least partly in the forward ray direction.
    // A chord entirely behind the plane origin (tmax < 0) is rejected.
    if tmax < 0.0 {
        return None;
    }
    Some((tmin, tmax))
}

fn fit_general_quadric(points: &[[f64; 3]]) -> Result<([f64; 3], [[f64; 3]; 3])> {
    // Match ImageJ Ops Quadric: solve
    // Ax² + By² + Cz² + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1
    // by ordinary least squares.
    let mut ata = [[0.0f64; 9]; 9];
    let mut atb = [0.0f64; 9];
    for &[x, y, z] in points {
        let row = [
            x * x,
            y * y,
            z * z,
            2.0 * x * y,
            2.0 * x * z,
            2.0 * y * z,
            2.0 * x,
            2.0 * y,
            2.0 * z,
        ];
        for i in 0..9 {
            atb[i] += row[i];
            for j in 0..9 {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    let q = solve_linear(ata, atb).ok_or_else(|| anyhow::anyhow!("singular DA quadric fit"))?;
    let quadratic = [[q[0], q[3], q[4]], [q[3], q[1], q[5]], [q[4], q[5], q[2]]];
    let linear = [q[6], q[7], q[8]];

    // BoneJ QuadricToEllipsoid first translates the quadric to its centre.
    let center_rhs = [-linear[0], -linear[1], -linear[2]];
    let center = solve_linear(quadratic, center_rhs)
        .ok_or_else(|| anyhow::anyhow!("singular DA quadric centre"))?;
    let qc = mat_vec_3(quadratic, center);
    let centered_constant = dot3(center, qc) + 2.0 * dot3(linear, center) - 1.0;
    if !centered_constant.is_finite() || centered_constant.abs() < 1e-18 {
        bail!("invalid centered DA quadric constant: {centered_constant}");
    }
    let scale = -1.0 / centered_constant;
    let normalized = [
        [quadratic[0][0] * scale, quadratic[0][1] * scale, quadratic[0][2] * scale],
        [quadratic[1][0] * scale, quadratic[1][1] * scale, quadratic[1][2] * scale],
        [quadratic[2][0] * scale, quadratic[2][1] * scale, quadratic[2][2] * scale],
    ];
    Ok(jacobi_eigen_3x3(normalized))
}

fn mat_vec_3(a: [[f64; 3]; 3], x: [f64; 3]) -> [f64; 3] {
    [
        a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
        a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
        a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2],
    ]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// Pivoted Gaussian elimination is kept explicit so the small normal-equation
// solve remains auditable for both the 9-parameter quadric and 3-D centre.
#[allow(clippy::needless_range_loop)]
fn solve_linear<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        let mut pivot = col;
        for row in (col + 1)..N {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() < 1e-18 {
            return None;
        }
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }
        let p = a[col][col];
        for j in col..N {
            a[col][j] /= p;
        }
        b[col] /= p;
        for row in 0..N {
            if row == col {
                continue;
            }
            let f = a[row][col];
            if f == 0.0 {
                continue;
            }
            for j in col..N {
                a[row][j] -= f * a[col][j];
            }
            b[row] -= f * b[col];
        }
    }
    Some(b)
}

// The Jacobi rotations intentionally update matrix rows/columns by index.
#[allow(clippy::needless_range_loop)]
fn jacobi_eigen_3x3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..64 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max = a[0][1].abs();
        for &(i, j) in &[(0usize, 2usize), (1, 2)] {
            if a[i][j].abs() > max {
                max = a[i][j].abs();
                p = i;
                q = j;
            }
        }
        if max < 1e-14 {
            break;
        }
        let phi = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let c = phi.cos();
        let s = phi.sin();
        for k in 0..3 {
            let aik = a[p][k];
            let aqk = a[q][k];
            a[p][k] = c * aik - s * aqk;
            a[q][k] = s * aik + c * aqk;
        }
        for k in 0..3 {
            let akp = a[k][p];
            let akq = a[k][q];
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
        }
        for k in 0..3 {
            let vip = v[k][p];
            let viq = v[k][q];
            v[k][p] = c * vip - s * viq;
            v[k][q] = s * vip + c * viq;
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

#[cfg(test)]
fn fibonacci_direction(i: usize, n: usize) -> [f64; 3] {
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let theta = golden * i as f64;
    [r * theta.cos(), r * theta.sin(), z]
}

fn random_rotation(rng: &mut SplitMix64) -> [[f64; 3]; 3] {
    // Shoemake's uniform random unit quaternion.
    let u1 = rng.next_f64();
    let u2 = rng.next_f64();
    let u3 = rng.next_f64();
    let q = [
        (1.0 - u1).sqrt() * (std::f64::consts::TAU * u2).sin(),
        (1.0 - u1).sqrt() * (std::f64::consts::TAU * u2).cos(),
        u1.sqrt() * (std::f64::consts::TAU * u3).sin(),
        u1.sqrt() * (std::f64::consts::TAU * u3).cos(),
    ];
    let [x, y, z, w] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn rotate(m: [[f64; 3]; 3], x: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * x[0] + m[0][1] * x[1] + m[0][2] * x[2],
        m[1][0] * x[0] + m[1][1] * x[1] + m[1][2] * x[2],
        m[2][0] * x[0] + m[2][1] * x[1] + m[2][2] * x[2],
    ]
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_sd(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    (values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt()
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotropic_sphere_point_cloud_has_near_zero_da() {
        let points = (0..512)
            .map(|i| fibonacci_direction(i, 512))
            .collect::<Vec<_>>();
        let (e, _) = fit_general_quadric(&points).unwrap();
        let min = e.iter().copied().fold(f64::INFINITY, f64::min);
        let max = e.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!((1.0 - min / max).abs() < 1e-10);
    }

    #[test]
    fn translated_ellipsoid_quadric_recovers_axis_ratios() {
        let radii = [2.0, 3.0, 5.0];
        let center = [0.7, -1.2, 2.1];
        let mut points = Vec::new();
        for i in 0..1024 {
            let d = fibonacci_direction(i, 1024);
            points.push([
                center[0] + radii[0] * d[0],
                center[1] + radii[1] * d[1],
                center[2] + radii[2] * d[2],
            ]);
        }
        let (e, _) = fit_general_quadric(&points).unwrap();
        let mut rs = e.map(|x| 1.0 / x.sqrt());
        rs.sort_by(f64::total_cmp);
        for (got, expected) in rs.into_iter().zip(radii) {
            assert!((got - expected).abs() < 1e-8, "got={got}, expected={expected}");
        }
    }

    #[test]
    fn phase_change_semantics_count_both_transition_directions() {
        let samples = [false, true, true, false, true, false];
        let mut previous = false;
        let mut changes = 0usize;
        for current in samples {
            if current != previous {
                changes += 1;
            }
            previous = current;
        }
        assert_eq!(changes, 4);
    }

    #[test]
    fn ray_box_intersection_rejects_chord_entirely_behind_origin() {
        let hit = line_box_intersection(
            [2.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        );
        assert!(hit.is_none());
    }

    #[test]
    fn ray_box_intersection_keeps_forward_and_origin_inside_hits() {
        let forward = line_box_intersection(
            [-1.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        )
        .unwrap();
        assert!((forward.0 - 1.0).abs() < 1e-12);
        assert!((forward.1 - 2.0).abs() < 1e-12);

        let inside = line_box_intersection(
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        )
        .unwrap();
        assert!((inside.0 + 0.5).abs() < 1e-12);
        assert!((inside.1 - 0.5).abs() < 1e-12);
    }
}
