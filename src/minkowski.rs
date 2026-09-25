//! Surface-normal Minkowski tensor descriptor block.
//!
//! Production predictors are the two stable invariants of W_1^{0,2} at the
//! prespecified 30 µm smoothing scale. W_2^{0,2} was rejected by the
//! outcome-blind stability gate and is deliberately not exposed here.

use anyhow::{bail, Result};

use crate::edt::{euclidean_distance_transform_squared, Phase};
use crate::linalg::sorted_eigenpairs;
use crate::mesh::{marching_cubes_scalar, triangle_area};
use crate::sdf_curvature::{
    gaussian_smooth_nearest_in_place, require_isotropic_spacing, signed_distance_field_um,
};
use crate::volume::BinaryVolume;

pub const W102_SIGMA_UM: f64 = 30.0;
pub const W102_SUPPORT_MARGIN_UM: f64 = 160.0; // 4 * max sensitivity sigma (40 µm)
const MIN_SELECTED_TRIANGLES: usize = 1000;

/// Production invariants plus audit/QC quantities retained for validation.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct MinkowskiW102Stats {
    pub degree_of_anisotropy: f64,
    pub mid_over_max: f64,
    pub min_over_max: f64,
    pub eig_min_abs: f64,
    pub eig_mid_abs: f64,
    pub eig_max_abs: f64,
    pub selected_triangles: usize,
    pub selected_surface_area_um2: f64,
    pub mesh_triangles: usize,
    pub sigma_um: f64,
    pub support_margin_um: f64,
}

/// Compute W_1^{0,2} = (1/3) ∫ n⊗n dA on the boundary-safe selected surface.
///
/// The SDF, Gaussian bandwidth, nearest/clamped Gaussian boundary extension,
/// common 160 µm support margin and tensor invariant formulas define the
/// descriptor. The Rust crate uses its own scalar marching-cubes implementation.
/// The independent Python reference used Lewiner marching cubes, so validation
/// uses an explicit cross-backend
/// numerical-equivalence gate rather than claiming mesh-level identity.
pub fn minkowski_w102_stats(
    bone: &BinaryVolume,
    roi: &BinaryVolume,
    spacing: [f64; 3],
) -> Result<MinkowskiW102Stats> {
    if !bone.same_shape(roi) {
        bail!("bone and ROI dimensions differ in W102 computation");
    }
    let h = require_isotropic_spacing(spacing)?;
    if bone.data.iter().all(|&v| v == 0) || bone.data.iter().all(|&v| v != 0) {
        bail!("W102 requires both bone and background in the unmasked binary image");
    }
    if roi.data.iter().all(|&v| v == 0) || roi.data.iter().all(|&v| v != 0) {
        bail!("W102 boundary-safe support requires a nontrivial ROI mask");
    }

    // Build the SDF on uncut bone and apply the ROI
    // only as an interior support criterion.
    let mut phi = signed_distance_field_um(bone, h)?;
    gaussian_smooth_nearest_in_place(
        &mut phi,
        bone.width,
        bone.height,
        bone.depth,
        W102_SIGMA_UM / h,
    );
    let (lo, hi) = phi.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
        (lo.min(v), hi.max(v))
    });
    if !(lo < 0.0 && hi > 0.0) {
        bail!("W102 smoothing removed the zero level set: [{lo}, {hi}]");
    }

    let roi_d2 = euclidean_distance_transform_squared(roi, Phase::Foreground)?;
    let roi_distance: Vec<f64> = roi_d2.into_iter().map(|x| (x as f64).sqrt() * h).collect();
    let mesh = marching_cubes_scalar(
        &phi,
        bone.width,
        bone.height,
        bone.depth,
        spacing,
        0.0,
    );
    if mesh.triangles.is_empty() {
        bail!("W102 zero-level marching cubes returned no triangles");
    }

    let mut w = [[0.0f64; 3]; 3];
    let mut selected_triangles = 0usize;
    let mut selected_area = 0.0;
    for &[ia, ib, ic] in &mesh.triangles {
        let a = mesh.vertices[ia];
        let b = mesh.vertices[ib];
        let c = mesh.vertices[ic];
        let bary = [
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ];
        let roi_d = trilinear_sample(&roi_distance, bone.width, bone.height, bone.depth, h, bary);
        let image_d = image_margin_um(bary, bone.width, bone.height, bone.depth, h);
        if roi_d < W102_SUPPORT_MARGIN_UM || image_d < W102_SUPPORT_MARGIN_UM {
            continue;
        }

        let cr = cross(sub(b, a), sub(c, a));
        let dbl = norm(cr);
        if dbl <= 0.0 || !dbl.is_finite() {
            continue;
        }
        let area = triangle_area(a, b, c);
        let n = [cr[0] / dbl, cr[1] / dbl, cr[2] / dbl];
        for i in 0..3 {
            for j in 0..3 {
                w[i][j] += (area / 3.0) * n[i] * n[j];
            }
        }
        selected_triangles += 1;
        selected_area += area;
    }
    if selected_triangles < MIN_SELECTED_TRIANGLES {
        bail!("only {selected_triangles} W102 surface triangles survive the support margin; at least {MIN_SELECTED_TRIANGLES} are required");
    }

    let (evals, _) = sorted_eigenpairs(w);
    let mut ae = [evals[0].abs(), evals[1].abs(), evals[2].abs()];
    ae.sort_by(f64::total_cmp);
    if !ae[2].is_finite() || ae[2] <= 0.0 {
        bail!("W102 tensor has invalid eigenvalues");
    }
    let min_over_max = ae[0] / ae[2];
    Ok(MinkowskiW102Stats {
        degree_of_anisotropy: 1.0 - min_over_max,
        mid_over_max: ae[1] / ae[2],
        min_over_max,
        eig_min_abs: ae[0],
        eig_mid_abs: ae[1],
        eig_max_abs: ae[2],
        selected_triangles,
        selected_surface_area_um2: selected_area,
        mesh_triangles: mesh.triangles.len(),
        sigma_um: W102_SIGMA_UM,
        support_margin_um: W102_SUPPORT_MARGIN_UM,
    })
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn image_margin_um(p: [f64; 3], width: usize, height: usize, depth: usize, h: f64) -> f64 {
    let upper = [
        (width - 1) as f64 * h,
        (height - 1) as f64 * h,
        (depth - 1) as f64 * h,
    ];
    p[0].min(upper[0] - p[0])
        .min(p[1].min(upper[1] - p[1]))
        .min(p[2].min(upper[2] - p[2]))
}

/// scipy.ndimage.map_coordinates(order=1, mode="constant", cval=0) equivalent
/// for in-bounds face barycentres on a z/y/x scalar grid.
fn trilinear_sample(
    field: &[f64],
    width: usize,
    height: usize,
    depth: usize,
    h: f64,
    p: [f64; 3],
) -> f64 {
    let gx = p[0] / h;
    let gy = p[1] / h;
    let gz = p[2] / h;
    if gx < 0.0 || gy < 0.0 || gz < 0.0
        || gx > (width - 1) as f64 || gy > (height - 1) as f64 || gz > (depth - 1) as f64
    {
        return 0.0;
    }
    let x0 = gx.floor() as usize;
    let y0 = gy.floor() as usize;
    let z0 = gz.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let z1 = (z0 + 1).min(depth - 1);
    let tx = gx - x0 as f64;
    let ty = gy - y0 as f64;
    let tz = gz - z0 as f64;
    let idx = |x: usize, y: usize, z: usize| (z * height + y) * width + x;
    let lerp = |a: f64, b: f64, t: f64| a * (1.0 - t) + b * t;
    let c00 = lerp(field[idx(x0,y0,z0)], field[idx(x1,y0,z0)], tx);
    let c10 = lerp(field[idx(x0,y1,z0)], field[idx(x1,y1,z0)], tx);
    let c01 = lerp(field[idx(x0,y0,z1)], field[idx(x1,y0,z1)], tx);
    let c11 = lerp(field[idx(x0,y1,z1)], field[idx(x1,y1,z1)], tx);
    lerp(lerp(c00,c10,ty), lerp(c01,c11,ty), tz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotropic_tensor_has_zero_da() {
        let w = [[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,2.0]];
        let (e,_) = sorted_eigenpairs(w);
        assert!((1.0 - e[0]/e[2]).abs() < 1e-12);
    }

    #[test]
    fn trilinear_constant_field_is_constant() {
        let field=vec![3.0; 27];
        assert!((trilinear_sample(&field,3,3,3,1.0,[0.7,1.2,1.4])-3.0).abs()<1e-12);
    }
}
