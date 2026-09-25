use crate::volume::BinaryVolume;

#[derive(Clone, Copy, Debug)]
pub struct PlateRodProxyStats {
    pub plate_seed_fraction: f64,
    pub rod_seed_fraction: f64,
    pub intermediate_seed_fraction: f64,
    pub rod_axial_fraction: f64,
    pub plate_normal_axial_fraction: f64,
    pub mean_abs_ef: f64,
    pub classified_seeds: usize,
}

/// ITS-inspired, open and reproducible plate/rod proxy.
///
/// This is intentionally *not* named ITS in CSV output. Individual Trabecula
/// Segmentation is a specific separately distributed method. Here we classify
/// topology-preserving skeleton seeds from a local covariance ellipsoid and
/// report rod/plate orientation summaries that can be compared to licensed ITS
/// output when it is available.
pub fn plate_rod_proxy_from_skeleton(
    bone: &BinaryVolume,
    skeleton: &BinaryVolume,
    window_radius: usize,
) -> PlateRodProxyStats {
    let mut plate = 0usize;
    let mut rod = 0usize;
    let mut intermediate = 0usize;
    let mut rod_axial = 0usize;
    let mut plate_normal_axial = 0usize;
    let mut abs_ef_sum = 0.0;
    let mut classified = 0usize;
    let cos_30 = (30.0_f64.to_radians()).cos();

    for idx in 0..skeleton.len() {
        if skeleton.data[idx] == 0 {
            continue;
        }
        let (x, y, z) = xyz(skeleton, idx);
        let Some(shape) = local_shape(bone, x, y, z, window_radius) else {
            continue;
        };
        classified += 1;
        abs_ef_sum += shape.ef.abs();
        if shape.ef < -0.25 {
            plate += 1;
            if shape.short_axis[2].abs() >= cos_30 {
                plate_normal_axial += 1;
            }
        } else if shape.ef > 0.25 {
            rod += 1;
            if shape.long_axis[2].abs() >= cos_30 {
                rod_axial += 1;
            }
        } else {
            intermediate += 1;
        }
    }

    if classified == 0 {
        return PlateRodProxyStats {
            plate_seed_fraction: f64::NAN,
            rod_seed_fraction: f64::NAN,
            intermediate_seed_fraction: f64::NAN,
            rod_axial_fraction: f64::NAN,
            plate_normal_axial_fraction: f64::NAN,
            mean_abs_ef: f64::NAN,
            classified_seeds: 0,
        };
    }
    let n = classified as f64;
    PlateRodProxyStats {
        plate_seed_fraction: plate as f64 / n,
        rod_seed_fraction: rod as f64 / n,
        intermediate_seed_fraction: intermediate as f64 / n,
        rod_axial_fraction: if rod > 0 {
            rod_axial as f64 / rod as f64
        } else {
            f64::NAN
        },
        plate_normal_axial_fraction: if plate > 0 {
            plate_normal_axial as f64 / plate as f64
        } else {
            f64::NAN
        },
        mean_abs_ef: abs_ef_sum / n,
        classified_seeds: classified,
    }
}

#[derive(Clone, Copy)]
struct LocalShape {
    ef: f64,
    short_axis: [f64; 3],
    long_axis: [f64; 3],
}

// Small fixed-size covariance loops are kept index-based to preserve the
// symmetry updates explicitly.
#[allow(clippy::needless_range_loop)]
fn local_shape(
    bone: &BinaryVolume,
    x: usize,
    y: usize,
    z: usize,
    radius: usize,
) -> Option<LocalShape> {
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
    let n = points.len();
    let centroid = [
        points.iter().map(|p| p[0]).sum::<f64>() / n as f64,
        points.iter().map(|p| p[1]).sum::<f64>() / n as f64,
        points.iter().map(|p| p[2]).sum::<f64>() / n as f64,
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
            covariance[i][j] /= (n.saturating_sub(1).max(1)) as f64;
            covariance[j][i] = covariance[i][j];
        }
    }
    let (e, v) = jacobi_eigen_3x3(covariance);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| e[a].total_cmp(&e[b]));
    let axes = [
        e[order[0]].max(0.0).sqrt(),
        e[order[1]].max(0.0).sqrt(),
        e[order[2]].max(0.0).sqrt(),
    ];
    if axes.iter().any(|&a| a <= 0.0) {
        return None;
    }
    let ef = axes[0] / axes[1] - axes[1] / axes[2];
    Some(LocalShape {
        ef: ef.clamp(-1.0, 1.0),
        short_axis: normalize([v[0][order[0]], v[1][order[0]], v[2][order[0]]]),
        long_axis: normalize([v[0][order[2]], v[1][order[2]], v[2][order[2]]]),
    })
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

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if n > 0.0 {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        [0.0, 0.0, 0.0]
    }
}

fn xyz(volume: &BinaryVolume, idx: usize) -> (usize, usize, usize) {
    let slice = volume.width * volume.height;
    let z = idx / slice;
    let rem = idx % slice;
    (rem % volume.width, rem / volume.width, z)
}
