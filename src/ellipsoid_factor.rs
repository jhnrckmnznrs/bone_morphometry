use crate::volume::BinaryVolume;

#[derive(Clone, Copy, Debug)]
pub struct EllipsoidFactorStats {
    pub mean: f64,
    pub sd: f64,
    pub median: f64,
    pub plate_fraction: f64,
    pub rod_fraction: f64,
    pub intermediate_fraction: f64,
    pub sampled_seeds: usize,
}

/// Native ellipsoid-factor candidate.
///
/// This deliberately does *not* claim exact BoneJ Ellipsoid Factor equivalence.
/// It uses topology-preserving skeleton seeds and a local covariance ellipsoid
/// of foreground voxels, then applies BoneJ's EF scalar definition
/// `a/b - b/c` for `a <= b <= c`. The included Fiji reference script is the
/// acceptance oracle; until that comparison passes, CSV columns remain marked
/// `native candidate`.
pub fn ellipsoid_factor_candidate_from_skeleton(
    bone: &BinaryVolume,
    skeleton: &BinaryVolume,
    window_radius: usize,
) -> EllipsoidFactorStats {
    let mut values = Vec::<f64>::new();

    for idx in 0..skeleton.len() {
        if skeleton.data[idx] == 0 {
            continue;
        }
        let (x, y, z) = xyz(skeleton, idx);
        if let Some(ef) = local_ef(bone, x, y, z, window_radius) {
            values.push(ef.clamp(-1.0, 1.0));
        }
    }

    if values.is_empty() {
        return EllipsoidFactorStats {
            mean: f64::NAN,
            sd: f64::NAN,
            median: f64::NAN,
            plate_fraction: f64::NAN,
            rod_fraction: f64::NAN,
            intermediate_fraction: f64::NAN,
            sampled_seeds: 0,
        };
    }

    values.sort_by(f64::total_cmp);
    let n = values.len() as f64;
    let plate = values.iter().filter(|&&x| x < -0.25).count();
    let rod = values.iter().filter(|&&x| x > 0.25).count();
    let intermediate = values.len() - plate - rod;
    EllipsoidFactorStats {
        mean: mean(&values),
        sd: sample_sd(&values),
        median: linear_quantile(&values, 0.5),
        plate_fraction: plate as f64 / n,
        rod_fraction: rod as f64 / n,
        intermediate_fraction: intermediate as f64 / n,
        sampled_seeds: values.len(),
    }
}

// Small fixed-size covariance loops are kept index-based to preserve the
// symmetry updates explicitly.
#[allow(clippy::needless_range_loop)]
fn local_ef(bone: &BinaryVolume, x: usize, y: usize, z: usize, radius: usize) -> Option<f64> {
    let r = radius as isize;
    let mut points = Vec::<[f64; 3]>::new();
    for dz in -r..=r {
        for dy in -r..=r {
            for dx in -r..=r {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                let nz = z as isize + dz;
                if nx < 0
                    || ny < 0
                    || nz < 0
                    || nx >= bone.width as isize
                    || ny >= bone.height as isize
                    || nz >= bone.depth as isize
                {
                    continue;
                }
                if bone.get(nx as usize, ny as usize, nz as usize) != 0 {
                    points.push([dx as f64, dy as f64, dz as f64]);
                }
            }
        }
    }
    if points.len() < 6 {
        return None;
    }
    let n_points = points.len();
    let centroid = [
        points.iter().map(|p| p[0]).sum::<f64>() / points.len() as f64,
        points.iter().map(|p| p[1]).sum::<f64>() / points.len() as f64,
        points.iter().map(|p| p[2]).sum::<f64>() / points.len() as f64,
    ];
    let mut covariance = [[0.0f64; 3]; 3];
    for p in points {
        let d = [p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2]];
        for i in 0..3 {
            for j in i..3 {
                covariance[i][j] += d[i] * d[j];
            }
        }
    }
    for i in 0..3 {
        for j in i..3 {
            covariance[i][j] /= (n_points.saturating_sub(1).max(1)) as f64;
            covariance[j][i] = covariance[i][j];
        }
    }
    // Scale cancels in EF; only axis ratios matter.
    let (e, _) = jacobi_eigen_3x3(covariance);
    let mut axes = [
        e[0].max(0.0).sqrt(),
        e[1].max(0.0).sqrt(),
        e[2].max(0.0).sqrt(),
    ];
    axes.sort_by(f64::total_cmp);
    let [a, b, c] = axes;
    if a <= 0.0 || b <= 0.0 || c <= 0.0 {
        return None;
    }
    Some(a / b - b / c)
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
        if max < 1e-12 {
            break;
        }
        let phi = 0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p]);
        let c = phi.cos();
        let s = phi.sin();
        for k in 0..3 {
            let aip = a[k][p];
            let aiq = a[k][q];
            a[k][p] = c * aip - s * aiq;
            a[k][q] = s * aip + c * aiq;
        }
        for k in 0..3 {
            let apk = a[p][k];
            let aqk = a[q][k];
            a[p][k] = c * apk - s * aqk;
            a[q][k] = s * apk + c * aqk;
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

fn xyz(volume: &BinaryVolume, idx: usize) -> (usize, usize, usize) {
    let slice = volume.width * volume.height;
    let z = idx / slice;
    let rem = idx % slice;
    (rem % volume.width, rem / volume.width, z)
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

fn linear_quantile(sorted: &[f64], q: f64) -> f64 {
    let p = q * (sorted.len() - 1) as f64;
    let lo = p.floor() as usize;
    let hi = p.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (p - lo as f64) * (sorted[hi] - sorted[lo])
    }
}
