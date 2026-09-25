//! Small dependency-free linear-algebra helpers used by descriptor families.
//!
//! The crate already used independent 3x3 Jacobi eigensolvers in several legacy
//! modules. New consolidated descriptors share this implementation so that the
//! mathematical conventions are reviewable in one place.

pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

pub fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = norm(a);
    if n > 0.0 && n.is_finite() {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        [f64::NAN; 3]
    }
}

/// Deterministic sign convention for an unoriented axis.
///
/// Convention: inspect z, then y, then x; the first
/// component with magnitude above 1e-15 is made positive.
pub fn canonical_axis(v: [f64; 3]) -> [f64; 3] {
    let mut a = normalize(v);
    for component in [a[2], a[1], a[0]] {
        if component.abs() > 1e-15 {
            if component < 0.0 {
                a = [-a[0], -a[1], -a[2]];
            }
            break;
        }
    }
    a
}

/// Acute angle in degrees between unoriented axes.
pub fn axis_angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    dot(normalize(a), normalize(b))
        .abs()
        .clamp(0.0, 1.0)
        .acos()
        .to_degrees()
}

/// Symmetric 3x3 eigendecomposition by Jacobi rotations.
///
/// Eigenvectors are returned as columns, matching `numpy.linalg.eigh`'s layout.
#[allow(clippy::needless_range_loop)]
pub fn symmetric_eigen_3x3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..96 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max = a[0][1].abs();
        for &(i, j) in &[(0usize, 2usize), (1usize, 2usize)] {
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

pub fn sorted_eigenpairs(a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let (evals, evecs) = symmetric_eigen_3x3(a);
    let mut order = [0usize, 1usize, 2usize];
    order.sort_by(|&i, &j| evals[i].total_cmp(&evals[j]));
    let values = [evals[order[0]], evals[order[1]], evals[order[2]]];
    let vectors = [
        [evecs[0][order[0]], evecs[0][order[1]], evecs[0][order[2]]],
        [evecs[1][order[0]], evecs[1][order[1]], evecs[1][order[2]]],
        [evecs[2][order[0]], evecs[2][order[1]], evecs[2][order[2]]],
    ];
    (values, vectors)
}

pub fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_diagonal_is_exact() {
        let (e, v) = sorted_eigenpairs([[3.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,2.0]]);
        assert!((e[0]-1.0).abs()<1e-12 && (e[1]-2.0).abs()<1e-12 && (e[2]-3.0).abs()<1e-12);
        assert!((v[1][0].abs()-1.0).abs()<1e-12);
    }

    #[test]
    fn canonical_axis_is_sign_invariant() {
        assert_eq!(canonical_axis([1.0,2.0,-3.0]), canonical_axis([-1.0,-2.0,3.0]));
    }
}
