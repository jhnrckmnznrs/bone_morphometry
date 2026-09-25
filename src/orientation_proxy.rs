//! Plate--rod orientation descriptors used by the SOI and orientation-field families.
//!
//! This is an open proxy, not Individual Trabecula Segmentation (ITS).  The
//! topology-preserving skeleton is used only as sampling support.  Local shape
//! is estimated from the binary bone phase by a thickness-adaptive covariance
//! neighbourhood, following the independently implemented Python reference used for validation.

use std::collections::HashMap;

use anyhow::{bail, Result};

use crate::edt::{euclidean_distance_transform_squared, Phase};
use crate::linalg::{axis_angle_deg, canonical_axis, median, sorted_eigenpairs};
use crate::volume::BinaryVolume;

const KAPPA: f64 = 2.5;
const EF_THRESHOLD: f64 = 0.20;
const MIN_RADIUS_VOX: usize = 3;
const MAX_RADIUS_VOX: usize = 24;
const MIN_POINTS: usize = 8;
const MIN_CLASS_SAMPLES: usize = 25;
const ORIENTATION_FIELD_RADIUS_T50: f64 = 2.0;

const THETA_MIN: f64 = -std::f64::consts::FRAC_PI_2;
const THETA_MAX: f64 = std::f64::consts::FRAC_PI_2;
const PHI_MIN: f64 = 0.0;
const PHI_MAX: f64 = std::f64::consts::FRAC_PI_2;
const BANDWIDTH: f64 = 4.5 * std::f64::consts::PI / 180.0;
const GRID_STEP: f64 = 0.5 * std::f64::consts::PI / 180.0;
const CU_2D: f64 = 1.0 / (std::f64::consts::PI * std::f64::consts::FRAC_PI_2);
// NumPy `isclose(x, -pi/2, atol=1e-14)` also applies its default
// rtol=1e-5. Preserve that exact convention for cross-language equivalence.
const NUMPY_ISCLOSE_RTOL: f64 = 1.0e-5;
const NUMPY_ISCLOSE_ATOL: f64 = 1.0e-14;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrientationKind {
    Plate,
    Rod,
    Intermediate,
}

/// Per-sample diagnostic record. Several fields are retained for QC inspection.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct OrientationSample {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub kind: OrientationKind,
    pub vector: [f64; 3],
    pub ef: f64,
    pub local_thickness_vox: f64,
    pub window_radius_vox: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct SoiStats {
    pub soi_proxy: f64,
    pub plate_organization: f64,
    pub rod_organization: f64,
    pub plate_rod_overlap: f64,
    pub plate_samples: usize,
    pub rod_samples: usize,
    pub classified_samples: usize,
    pub grid_representatives: usize,
    pub median_local_thickness_um: f64,
}

/// Production field summaries plus center-count QC diagnostics.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct OrientationFieldStats {
    pub plate_local_misorientation_deg: f64,
    pub rod_local_misorientation_deg: f64,
    pub plate_rod_local_orthogonality_deviation_deg: f64,
    pub plate_local_centers: usize,
    pub rod_local_centers: usize,
    pub plate_rod_plate_centers: usize,
    pub plate_rod_rod_centers: usize,
}

/// Consolidated orientation output; raw samples are retained for auditability.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct OrientationProxyStats {
    pub soi: SoiStats,
    pub field: OrientationFieldStats,
    /// Primary 1.0*t50 representatives with valid local orientation estimates.
    pub samples: Vec<OrientationSample>,
}

/// Compute the primary SOI proxy and orientation-field descriptors.
///
/// Scientific constants are intentionally not exposed as tuning parameters:
/// they were fixed before outcome analysis. The primary spatial sampling cell
/// is one specimen-specific median local thickness (1.0*t50).
pub fn orientation_proxy_stats(
    masked_bone: &BinaryVolume,
    skeleton: &BinaryVolume,
    spacing: [f64; 3],
) -> Result<OrientationProxyStats> {
    if !masked_bone.same_shape(skeleton) {
        bail!("bone and skeleton dimensions differ in orientation proxy");
    }
    let h = require_isotropic_spacing(spacing)?;
    if masked_bone.data.iter().all(|&v| v == 0) {
        bail!("orientation proxy requires nonempty bone");
    }

    let skeleton_xyz: Vec<[usize; 3]> = skeleton
        .data
        .iter()
        .enumerate()
        .filter_map(|(idx, &v)| {
            if v == 0 {
                return None;
            }
            let z = idx / skeleton.slice_len();
            let rem = idx % skeleton.slice_len();
            let y = rem / skeleton.width;
            let x = rem % skeleton.width;
            Some([x, y, z])
        })
        .collect();
    if skeleton_xyz.is_empty() {
        bail!("orientation proxy requires a nonempty skeleton");
    }
    for &[x, y, z] in &skeleton_xyz {
        if masked_bone.get(x, y, z) == 0 {
            bail!("skeleton support contains a voxel outside masked bone");
        }
    }

    let edt_sq = euclidean_distance_transform_squared(masked_bone, Phase::Foreground)?;
    let mut skeleton_thickness_um = Vec::with_capacity(skeleton_xyz.len());
    for &[x, y, z] in &skeleton_xyz {
        let idx = masked_bone.index(x, y, z);
        skeleton_thickness_um.push(2.0 * (edt_sq[idx] as f64).sqrt() * h);
    }
    let t50 = median(skeleton_thickness_um);
    if !t50.is_finite() || t50 <= 0.0 {
        bail!("invalid median local thickness on skeleton support");
    }

    let representative_indices = grid_representatives(&skeleton_xyz, h, t50)?;
    let mut samples = Vec::new();
    for &i in &representative_indices {
        let [x, y, z] = skeleton_xyz[i];
        if let Some(sample) = local_shape_at_seed(masked_bone, &edt_sq, x, y, z) {
            samples.push(sample);
        }
    }

    let plates: Vec<[f64; 3]> = samples
        .iter()
        .filter(|s| s.kind == OrientationKind::Plate)
        .map(|s| s.vector)
        .collect();
    let rods: Vec<[f64; 3]> = samples
        .iter()
        .filter(|s| s.kind == OrientationKind::Rod)
        .map(|s| s.vector)
        .collect();
    let classified_samples = plates.len() + rods.len();

    let (soi_proxy, p_o, r_o, pr_o) = if plates.len() >= MIN_CLASS_SAMPLES
        && rods.len() >= MIN_CLASS_SAMPLES
    {
        compute_soi_2d(&plates, &rods)
    } else {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    };

    let field = orientation_field(&samples, h, t50);
    Ok(OrientationProxyStats {
        soi: SoiStats {
            soi_proxy,
            plate_organization: p_o,
            rod_organization: r_o,
            plate_rod_overlap: pr_o,
            plate_samples: plates.len(),
            rod_samples: rods.len(),
            classified_samples,
            grid_representatives: representative_indices.len(),
            median_local_thickness_um: t50,
        },
        field,
        samples,
    })
}

fn require_isotropic_spacing(spacing: [f64; 3]) -> Result<f64> {
    if spacing.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        bail!("voxel spacing must be finite and positive");
    }
    let h = spacing[0];
    let tol = 1e-9 * h.abs().max(1.0);
    if (spacing[1] - h).abs() > tol || (spacing[2] - h).abs() > tol {
        bail!(
            "SOI/orientation-field descriptors require isotropic spacing; got [{}, {}, {}] µm",
            spacing[0], spacing[1], spacing[2]
        );
    }
    Ok(h)
}

fn local_shape_at_seed(
    bone: &BinaryVolume,
    edt_sq: &[u32],
    x: usize,
    y: usize,
    z: usize,
) -> Option<OrientationSample> {
    if bone.get(x, y, z) == 0 {
        return None;
    }
    let local_thickness_vox = 2.0 * (edt_sq[bone.index(x, y, z)] as f64).sqrt();
    if !local_thickness_vox.is_finite() || local_thickness_vox <= 0.0 {
        return None;
    }
    let rv = ((KAPPA * local_thickness_vox).ceil() as usize)
        .clamp(MIN_RADIUS_VOX, MAX_RADIUS_VOX);
    let r2 = rv * rv;

    let x0 = x.saturating_sub(rv);
    let y0 = y.saturating_sub(rv);
    let z0 = z.saturating_sub(rv);
    let x1 = (x + rv + 1).min(bone.width);
    let y1 = (y + rv + 1).min(bone.height);
    let z1 = (z + rv + 1).min(bone.depth);

    let mut points = Vec::<[f64; 3]>::new();
    for zz in z0..z1 {
        for yy in y0..y1 {
            for xx in x0..x1 {
                if bone.get(xx, yy, zz) == 0 {
                    continue;
                }
                let dx = xx.abs_diff(x);
                let dy = yy.abs_diff(y);
                let dz = zz.abs_diff(z);
                if dx * dx + dy * dy + dz * dz <= r2 {
                    points.push([xx as f64, yy as f64, zz as f64]);
                }
            }
        }
    }
    if points.len() < MIN_POINTS {
        return None;
    }

    let n = points.len() as f64;
    let mut mean = [0.0; 3];
    for q in &points {
        for k in 0..3 {
            mean[k] += q[k];
        }
    }
    for value in &mut mean {
        *value /= n;
    }
    let mut cov = [[0.0; 3]; 3];
    for q in &points {
        let d = [q[0] - mean[0], q[1] - mean[1], q[2] - mean[2]];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }
    let denom = ((points.len() - 1).max(1)) as f64;
    for row in &mut cov {
        for value in row {
            *value /= denom;
        }
    }

    let (evals, evecs) = sorted_eigenpairs(cov);
    let axes = covariance_axis_lengths_numpy_compatible(evals);
    if axes.iter().any(|&a| a <= 1e-9 || !a.is_finite()) {
        return None;
    }
    let ef = axes[0] / axes[1] - axes[1] / axes[2];
    let (kind, col) = if ef < -EF_THRESHOLD {
        (OrientationKind::Plate, 0usize)
    } else if ef > EF_THRESHOLD {
        (OrientationKind::Rod, 2usize)
    } else {
        (OrientationKind::Intermediate, 2usize)
    };
    let vector = canonical_axis([evecs[0][col], evecs[1][col], evecs[2][col]]);
    Some(OrientationSample {
        x,
        y,
        z,
        kind,
        vector,
        ef,
        local_thickness_vox,
        window_radius_vox: rv,
    })
}

/// Convert covariance eigenvalues to the axis lengths used by the validated
/// NumPy reference.  A covariance matrix is positive semidefinite, but
/// independent eigensolvers can return a tiny negative value for an exact
/// zero eigenvalue.  NumPy/LAPACK returned a tiny *positive* value for one
/// validation neighbourhood; clipping the Jacobi value directly to zero
/// would therefore reject a sample that the reference implementation retained.
///
/// Only roundoff-scale negative values are lifted, using one machine epsilon
/// at the spectral scale.  Substantive negative values still clip to zero and
/// fail the existing minimum-axis gate.
fn covariance_axis_lengths_numpy_compatible(evals: [f64; 3]) -> [f64; 3] {
    let scale = evals
        .iter()
        .map(|x| x.abs())
        .fold(1.0_f64, f64::max);
    let roundoff = 1.0e-12 * scale;
    let floor = f64::EPSILON * scale;
    evals.map(|value| {
        let stabilized = if value < 0.0 && value >= -roundoff {
            floor
        } else {
            value.max(0.0)
        };
        stabilized.sqrt()
    })
}

type GridCell = (i64, i64, i64);
type GridCandidate = (f64, usize, usize, usize, usize);

/// Deterministic physical-grid sampling used by the SOI descriptor.
fn grid_representatives(xyz: &[[usize; 3]], spacing_um: f64, cell_side_um: f64) -> Result<Vec<usize>> {
    if xyz.is_empty() || !spacing_um.is_finite() || spacing_um <= 0.0 || !cell_side_um.is_finite() || cell_side_um <= 0.0 {
        bail!("invalid input to deterministic orientation grid");
    }
    // cell -> (distance squared, z, y, x, input index)
    let mut best: HashMap<GridCell, GridCandidate> = HashMap::new();
    for (i, &[x, y, z]) in xyz.iter().enumerate() {
        let pos = [
            (x as f64 + 0.5) * spacing_um,
            (y as f64 + 0.5) * spacing_um,
            (z as f64 + 0.5) * spacing_um,
        ];
        let cell = [
            (pos[0] / cell_side_um).floor() as i64,
            (pos[1] / cell_side_um).floor() as i64,
            (pos[2] / cell_side_um).floor() as i64,
        ];
        let center = [
            (cell[0] as f64 + 0.5) * cell_side_um,
            (cell[1] as f64 + 0.5) * cell_side_um,
            (cell[2] as f64 + 0.5) * cell_side_um,
        ];
        let d2 = (pos[0] - center[0]).powi(2)
            + (pos[1] - center[1]).powi(2)
            + (pos[2] - center[2]).powi(2);
        let candidate = (d2, z, y, x, i);
        let key = (cell[0], cell[1], cell[2]);
        let replace = match best.get(&key) {
            None => true,
            Some(old) => candidate.0.total_cmp(&old.0).is_lt()
                || (candidate.0.total_cmp(&old.0).is_eq()
                    && (candidate.1, candidate.2, candidate.3) < (old.1, old.2, old.3)),
        };
        if replace {
            best.insert(key, candidate);
        }
    }
    let mut cells: Vec<_> = best.into_iter().collect();
    // Match NumPy lexsort's primary cell x,y,z keys.
    cells.sort_by_key(|(cell, _)| (cell.0, cell.1, cell.2));
    Ok(cells.into_iter().map(|(_, item)| item.4).collect())
}

fn vector_to_angles(mut v: [f64; 3]) -> (f64, f64) {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v = [v[0] / n, v[1] / n, v[2] / n];
    let eqz = v[2].abs() <= 1e-14;
    let flip = v[2] < 0.0 || (eqz && (v[0] < 0.0 || (v[0].abs() <= 1e-14 && v[1] < 0.0)));
    if flip {
        v = [-v[0], -v[1], -v[2]];
    }
    let mut theta = v[1].atan2(v[0]);
    theta = (theta + std::f64::consts::FRAC_PI_2).rem_euclid(std::f64::consts::PI)
        - std::f64::consts::FRAC_PI_2;
    let theta_endpoint_tol = NUMPY_ISCLOSE_ATOL
        + NUMPY_ISCLOSE_RTOL * std::f64::consts::FRAC_PI_2;
    if (theta + std::f64::consts::FRAC_PI_2).abs() <= theta_endpoint_tol {
        theta = std::f64::consts::FRAC_PI_2;
    }
    let phi = v[2].clamp(0.0, 1.0).acos();
    (theta, phi)
}

fn kde2d(vectors: &[[f64; 3]]) -> Vec<f64> {
    let nt = ((THETA_MAX - THETA_MIN) / GRID_STEP).round() as usize + 1;
    let np = ((PHI_MAX - PHI_MIN) / GRID_STEP).round() as usize + 1;
    let mut out = vec![0.0; nt * np];
    let norm = 1.0 / (2.0 * std::f64::consts::PI * BANDWIDTH * BANDWIDTH * vectors.len() as f64);

    for &vector in vectors {
        let (theta, phi) = vector_to_angles(vector);
        let tr = [theta, 2.0 * THETA_MIN - theta, 2.0 * THETA_MAX - theta];
        let pr = [phi, 2.0 * PHI_MIN - phi, 2.0 * PHI_MAX - phi];
        let mut tv = vec![0.0; nt];
        let mut pv = vec![0.0; np];
        for (i, value) in tv.iter_mut().enumerate() {
            let g = THETA_MIN + i as f64 * GRID_STEP;
            *value = tr.iter().map(|&r| (-0.5 * ((g - r) / BANDWIDTH).powi(2)).exp()).sum();
        }
        for (j, value) in pv.iter_mut().enumerate() {
            let g = PHI_MIN + j as f64 * GRID_STEP;
            *value = pr.iter().map(|&r| (-0.5 * ((g - r) / BANDWIDTH).powi(2)).exp()).sum();
        }
        for i in 0..nt {
            for j in 0..np {
                out[i * np + j] += tv[i] * pv[j];
            }
        }
    }
    for value in &mut out {
        *value *= norm;
    }
    out
}

fn integrate2d(values: &[f64]) -> f64 {
    let nt = ((THETA_MAX - THETA_MIN) / GRID_STEP).round() as usize + 1;
    let np = ((PHI_MAX - PHI_MIN) / GRID_STEP).round() as usize + 1;
    debug_assert_eq!(values.len(), nt * np);
    let mut sum = 0.0;
    for i in 0..nt {
        let wi = if i == 0 || i + 1 == nt { 0.5 } else { 1.0 };
        for j in 0..np {
            let wj = if j == 0 || j + 1 == np { 0.5 } else { 1.0 };
            sum += wi * wj * values[i * np + j];
        }
    }
    sum * GRID_STEP * GRID_STEP
}

fn vmax_2d() -> f64 {
    let center = [[std::f64::consts::FRAC_1_SQRT_2, 0.0, std::f64::consts::FRAC_1_SQRT_2]];
    let pdf = kde2d(&center);
    let excess: Vec<f64> = pdf.iter().map(|&x| (x - CU_2D).max(0.0)).collect();
    integrate2d(&excess)
}

fn organization(pdf: &[f64], vmax: f64) -> f64 {
    let excess: Vec<f64> = pdf.iter().map(|&x| (x - CU_2D).max(0.0)).collect();
    integrate2d(&excess) / vmax
}

fn compute_soi_2d(plates: &[[f64; 3]], rods: &[[f64; 3]]) -> (f64, f64, f64, f64) {
    let pp = kde2d(plates);
    let rp = kde2d(rods);
    let vmax = vmax_2d();
    let mut po = organization(&pp, vmax);
    let mut ro = organization(&rp, vmax);
    let absdiff: Vec<f64> = pp.iter().zip(&rp).map(|(&a, &b)| (a - b).abs()).collect();
    let mut pro = 1.0 - 0.5 * integrate2d(&absdiff);
    for value in [&mut po, &mut ro, &mut pro] {
        if *value >= -1e-10 && *value <= 1.0 + 1e-10 {
            *value = (*value).clamp(0.0, 1.0);
        }
    }
    (po * ro * pro, po, ro, pro)
}

fn orientation_field(samples: &[OrientationSample], spacing_um: f64, t50_um: f64) -> OrientationFieldStats {
    let plate: Vec<&OrientationSample> = samples.iter().filter(|s| s.kind == OrientationKind::Plate).collect();
    let rod: Vec<&OrientationSample> = samples.iter().filter(|s| s.kind == OrientationKind::Rod).collect();
    let (pm, pc) = same_type_field(&plate, spacing_um, t50_um);
    let (rm, rc) = same_type_field(&rod, spacing_um, t50_um);
    let (xm, xpc, xrc) = cross_field(&plate, &rod, spacing_um, t50_um);
    OrientationFieldStats {
        plate_local_misorientation_deg: pm,
        rod_local_misorientation_deg: rm,
        plate_rod_local_orthogonality_deviation_deg: xm,
        plate_local_centers: pc,
        rod_local_centers: rc,
        plate_rod_plate_centers: xpc,
        plate_rod_rod_centers: xrc,
    }
}

fn normalized_position(s: &OrientationSample, spacing_um: f64, t50_um: f64) -> [f64; 3] {
    [
        (s.x as f64 + 0.5) * spacing_um / t50_um,
        (s.y as f64 + 0.5) * spacing_um / t50_um,
        (s.z as f64 + 0.5) * spacing_um / t50_um,
    ]
}

fn distance_squared(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn same_type_field(samples: &[&OrientationSample], spacing_um: f64, t50_um: f64) -> (f64, usize) {
    if samples.len() < 2 {
        return (f64::NAN, 0);
    }
    let positions: Vec<_> = samples.iter().map(|s| normalized_position(s, spacing_um, t50_um)).collect();
    let mut local = Vec::new();
    for i in 0..samples.len() {
        let mut angles = Vec::new();
        for j in 0..samples.len() {
            if i != j && distance_squared(positions[i], positions[j]) <= ORIENTATION_FIELD_RADIUS_T50.powi(2) {
                angles.push(axis_angle_deg(samples[i].vector, samples[j].vector));
            }
        }
        if !angles.is_empty() {
            local.push(angles.iter().sum::<f64>() / angles.len() as f64);
        }
    }
    if local.is_empty() {
        (f64::NAN, 0)
    } else {
        (local.iter().sum::<f64>() / local.len() as f64, local.len())
    }
}

fn cross_field(
    plates: &[&OrientationSample],
    rods: &[&OrientationSample],
    spacing_um: f64,
    t50_um: f64,
) -> (f64, usize, usize) {
    if plates.is_empty() || rods.is_empty() {
        return (f64::NAN, 0, 0);
    }
    let pp: Vec<_> = plates.iter().map(|s| normalized_position(s, spacing_um, t50_um)).collect();
    let rp: Vec<_> = rods.iter().map(|s| normalized_position(s, spacing_um, t50_um)).collect();

    let mut plate_local = Vec::new();
    for (i, plate) in plates.iter().enumerate() {
        let mut dev = Vec::new();
        for (j, rod) in rods.iter().enumerate() {
            if distance_squared(pp[i], rp[j]) <= ORIENTATION_FIELD_RADIUS_T50.powi(2) {
                let d = crate::linalg::dot(plate.vector, rod.vector).abs().clamp(0.0, 1.0);
                dev.push(d.asin().to_degrees());
            }
        }
        if !dev.is_empty() {
            plate_local.push(dev.iter().sum::<f64>() / dev.len() as f64);
        }
    }
    let mut rod_local = Vec::new();
    for (j, rod) in rods.iter().enumerate() {
        let mut dev = Vec::new();
        for (i, plate) in plates.iter().enumerate() {
            if distance_squared(rp[j], pp[i]) <= ORIENTATION_FIELD_RADIUS_T50.powi(2) {
                let d = crate::linalg::dot(plate.vector, rod.vector).abs().clamp(0.0, 1.0);
                dev.push(d.asin().to_degrees());
            }
        }
        if !dev.is_empty() {
            rod_local.push(dev.iter().sum::<f64>() / dev.len() as f64);
        }
    }
    if plate_local.is_empty() || rod_local.is_empty() {
        return (f64::NAN, plate_local.len(), rod_local.len());
    }
    let p = plate_local.iter().sum::<f64>() / plate_local.len() as f64;
    let r = rod_local.iter().sum::<f64>() / rod_local.len() as f64;
    (0.5 * (p + r), plate_local.len(), rod_local.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_angles_are_sign_invariant() {
        let a = vector_to_angles([0.4, -0.2, 0.9]);
        let b = vector_to_angles([-0.4, 0.2, -0.9]);
        assert!((a.0 - b.0).abs() < 1e-12);
        assert!((a.1 - b.1).abs() < 1e-12);
    }

    #[test]
    fn deterministic_grid_picks_one_per_cell() {
        let mut xyz = Vec::new();
        for z in 0..3 {
            for y in 0..5 {
                for x in 0..7 {
                    xyz.push([x, y, z]);
                }
            }
        }
        let a = grid_representatives(&xyz, 20.0, 60.0).unwrap();
        let b = grid_representatives(&xyz, 20.0, 60.0).unwrap();
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn theta_endpoint_matches_numpy_isclose_convention() {
        let eps = 0.5 * NUMPY_ISCLOSE_RTOL * std::f64::consts::FRAC_PI_2;
        let theta = -std::f64::consts::FRAC_PI_2 + eps;
        let v = [theta.cos(), theta.sin(), 0.25];
        let (mapped, _) = vector_to_angles(v);
        assert!((mapped - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn psd_roundoff_does_not_drop_rank_deficient_sample() {
        let axes = covariance_axis_lengths_numpy_compatible([-1.7e-16, 1.0, 11.0]);
        assert!(axes[0] > 1e-9);
        assert!(axes[1] > 0.0 && axes[2] > axes[1]);
    }

    #[test]
    fn single_orientation_vmax_matches_reference_scale() {
        // Frozen Python reference reports ~0.954077 for the 2-D functional.
        assert!((vmax_2d() - 0.954077).abs() < 5e-5);
    }
}
