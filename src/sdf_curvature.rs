use anyhow::{bail, Result};

use crate::edt::{euclidean_distance_transform_squared, Phase};
use crate::mesh::{marching_cubes_scalar, triangle_area, Mesh};
use crate::volume::BinaryVolume;

#[derive(Clone, Copy, Debug)]
pub struct SdfCurvatureParameters {
    pub sigma_um: f64,
    /// Number of Gaussian standard deviations excluded from the image boundary.
    /// This avoids using curvature whose SDF/derivative support is affected by
    /// the cropped image boundary. The Gaussian kernel itself is truncated at
    /// 4 sigma, matching SciPy's default calibration prototype.
    pub image_support_sigma: f64,
}

impl Default for SdfCurvatureParameters {
    fn default() -> Self {
        Self {
            sigma_um: 30.0,
            image_support_sigma: 4.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SdfCurvatureStats {
    pub sigma_um: f64,
    pub support_margin_um: f64,
    pub selected_surface_area_um2: f64,
    pub mean_curvature_mean: f64,
    pub mean_curvature_sd: f64,
    pub mean_curvature_abs_mean: f64,
    /// Area-weighted signed Gaussian-curvature median.
    pub gaussian_curvature_median: f64,
    /// Area-weighted signed Gaussian-curvature interquartile range.
    pub gaussian_curvature_iqr: f64,
    /// Area-weighted 90th percentile of |K|.
    pub gaussian_curvature_abs_q90: f64,
    /// Area-weighted 99th percentile of |K|.
    pub gaussian_curvature_abs_q99: f64,
    pub saddle_fraction: f64,
    pub convex_fraction: f64,
    pub concave_fraction: f64,
    /// Surface-area fraction where independently estimated H and K violate
    /// H²-K >= 0. This is a QC diagnostic, not a curvature descriptor.
    pub discriminant_negative_fraction: f64,
    pub selected_vertices: usize,
    pub roi_excluded_vertices: usize,
    pub support_excluded_vertices: usize,
    pub invalid_vertices: usize,
}

#[derive(Clone, Copy, Debug)]
struct PointCurvature {
    h: f64,
    k: f64,
}

/// Scale-aware curvature from a Gaussian-smoothed signed Euclidean distance field.
///
/// Scientific definition:
/// 1. Build the signed EDT on the *uncut* binary bone image, positive inside.
/// 2. Smooth the SDF with a physical Gaussian bandwidth `sigma_um`.
/// 3. Extract the zero level set of that same smoothed field.
/// 4. Evaluate implicit-surface mean and Gaussian curvature from the gradient
///    and Hessian of the smoothed field.
/// 5. Restrict statistical support to the supplied ROI without cutting the
///    surface itself, and exclude an image-boundary support margin.
/// 6. Aggregate by barycentric surface area rather than by vertex count.
///
/// The default 30 µm bandwidth is a prespecified compromise from outcome-blind
/// synthetic calibration at 20--25 µm voxel spacing. Other bandwidths are
/// allowed for sensitivity analyses and should be reported explicitly.
pub fn sdf_curvature_stats(
    bone: &BinaryVolume,
    roi: &BinaryVolume,
    spacing: [f64; 3],
    parameters: SdfCurvatureParameters,
) -> Result<SdfCurvatureStats> {
    if !bone.same_shape(roi) {
        bail!("bone and ROI dimensions differ in SDF curvature");
    }
    let h = require_isotropic_spacing(spacing)?;
    if !parameters.sigma_um.is_finite() || parameters.sigma_um <= 0.0 {
        bail!("curvature sigma must be finite and positive");
    }
    if !parameters.image_support_sigma.is_finite() || parameters.image_support_sigma < 0.0 {
        bail!("curvature image-support sigma multiplier must be finite and non-negative");
    }
    if bone.data.iter().all(|&v| v == 0) {
        bail!("SDF curvature is undefined for an all-background bone volume");
    }
    if bone.data.iter().all(|&v| v != 0) {
        bail!("SDF curvature requires background around the bone phase");
    }

    let mut phi = signed_distance_field_um(bone, h)?;
    let sigma_vox = parameters.sigma_um / h;
    gaussian_smooth_nearest_in_place(
        &mut phi,
        bone.width,
        bone.height,
        bone.depth,
        sigma_vox,
    );

    let (phi_min, phi_max) = phi
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    if !(phi_min <= 0.0 && phi_max >= 0.0) {
        bail!(
            "curvature smoothing removed the zero level set: phi range [{phi_min}, {phi_max}]"
        );
    }

    let mesh = marching_cubes_scalar(
        &phi,
        bone.width,
        bone.height,
        bone.depth,
        spacing,
        0.0,
    );
    if mesh.vertices.is_empty() || mesh.triangles.is_empty() {
        bail!("SDF zero-level marching cubes returned an empty mesh");
    }

    let weights = barycentric_vertex_areas(&mesh);
    let support_margin_um = parameters.image_support_sigma * parameters.sigma_um;
    aggregate_selected_curvature(
        &mesh,
        &weights,
        &phi,
        bone.width,
        bone.height,
        bone.depth,
        h,
        roi,
        parameters.sigma_um,
        support_margin_um,
    )
}

pub(crate) fn require_isotropic_spacing(spacing: [f64; 3]) -> Result<f64> {
    if spacing.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        bail!("voxel spacing must be finite and positive");
    }
    let h = spacing[0];
    let tol = 1e-9 * h.abs().max(1.0);
    if (spacing[1] - h).abs() > tol || (spacing[2] - h).abs() > tol {
        bail!(
            "advanced SDF curvature currently requires isotropic spacing; got [{}, {}, {}] µm",
            spacing[0], spacing[1], spacing[2]
        );
    }
    Ok(h)
}

pub(crate) fn signed_distance_field_um(bone: &BinaryVolume, h: f64) -> Result<Vec<f64>> {
    let inside_sq = euclidean_distance_transform_squared(bone, Phase::Foreground)?;
    let mut phi = vec![0.0f64; bone.len()];
    for (i, (&b, &d2)) in bone.data.iter().zip(&inside_sq).enumerate() {
        if b != 0 {
            phi[i] = (d2 as f64).sqrt() * h;
        }
    }
    drop(inside_sq);

    let outside_sq = euclidean_distance_transform_squared(bone, Phase::Background)?;
    for (i, (&b, &d2)) in bone.data.iter().zip(&outside_sq).enumerate() {
        if b == 0 {
            phi[i] = -(d2 as f64).sqrt() * h;
        }
    }
    Ok(phi)
}

fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return vec![1.0];
    }
    let radius = (4.0 * sigma + 0.5).floor() as isize;
    let mut kernel = Vec::with_capacity((2 * radius + 1) as usize);
    let mut sum = 0.0;
    for i in -radius..=radius {
        let x = i as f64;
        let w = (-0.5 * x * x / (sigma * sigma)).exp();
        kernel.push(w);
        sum += w;
    }
    for w in &mut kernel {
        *w /= sum;
    }
    kernel
}

pub(crate) fn gaussian_smooth_nearest_in_place(
    data: &mut Vec<f64>,
    width: usize,
    height: usize,
    depth: usize,
    sigma_vox: f64,
) {
    let kernel = gaussian_kernel(sigma_vox);
    if kernel.len() == 1 {
        return;
    }
    let radius = (kernel.len() / 2) as isize;
    let mut tmp = vec![0.0f64; data.len()];

    let idx = |x: usize, y: usize, z: usize| (z * height + y) * width + x;
    let clamp = |v: isize, n: usize| -> usize {
        if v < 0 {
            0
        } else if v >= n as isize {
            n - 1
        } else {
            v as usize
        }
    };

    // X pass.
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let mut acc = 0.0;
                for (ki, &w) in kernel.iter().enumerate() {
                    let dx = ki as isize - radius;
                    acc += w * data[idx(clamp(x as isize + dx, width), y, z)];
                }
                tmp[idx(x, y, z)] = acc;
            }
        }
    }
    std::mem::swap(data, &mut tmp);

    // Y pass.
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let mut acc = 0.0;
                for (ki, &w) in kernel.iter().enumerate() {
                    let dy = ki as isize - radius;
                    acc += w * data[idx(x, clamp(y as isize + dy, height), z)];
                }
                tmp[idx(x, y, z)] = acc;
            }
        }
    }
    std::mem::swap(data, &mut tmp);

    // Z pass.
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let mut acc = 0.0;
                for (ki, &w) in kernel.iter().enumerate() {
                    let dz = ki as isize - radius;
                    acc += w * data[idx(x, y, clamp(z as isize + dz, depth))];
                }
                tmp[idx(x, y, z)] = acc;
            }
        }
    }
    std::mem::swap(data, &mut tmp);
}

fn barycentric_vertex_areas(mesh: &Mesh) -> Vec<f64> {
    let mut area = vec![0.0f64; mesh.vertices.len()];
    for &[a, b, c] in &mesh.triangles {
        let tri_area = triangle_area(mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]);
        let share = tri_area / 3.0;
        area[a] += share;
        area[b] += share;
        area[c] += share;
    }
    area
}

#[allow(clippy::too_many_arguments)]
fn aggregate_selected_curvature(
    mesh: &Mesh,
    weights: &[f64],
    phi: &[f64],
    width: usize,
    height: usize,
    depth: usize,
    h: f64,
    roi: &BinaryVolume,
    sigma_um: f64,
    support_margin_um: f64,
) -> Result<SdfCurvatureStats> {
    debug_assert_eq!(mesh.vertices.len(), weights.len());

    let max_x = (width - 1) as f64 * h;
    let max_y = (height - 1) as f64 * h;
    let max_z = (depth - 1) as f64 * h;

    let mut selected_vertices = 0usize;
    let mut roi_excluded_vertices = 0usize;
    let mut support_excluded_vertices = 0usize;
    let mut invalid_vertices = 0usize;

    let mut w_sum = 0.0;
    let mut h_sum = 0.0;
    let mut h_abs_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut saddle_w = 0.0;
    let mut convex_w = 0.0;
    let mut concave_w = 0.0;
    let mut discriminant_negative_w = 0.0;

    // Raw Gaussian-curvature mean/SD are deliberately not production
    // descriptors. Outcome-blind robustness analysis showed that vanishingly small surface tails
    // can dominate K² while leaving the bulk distribution stable. Preserve the
    // full selected distribution only long enough to compute outcome-blind,
    // area-weighted robust summaries.
    let mut k_samples: Vec<(f64, f64)> = Vec::new();

    for (i, &p) in mesh.vertices.iter().enumerate() {
        let w = weights[i];
        if !w.is_finite() || w <= 0.0 {
            invalid_vertices += 1;
            continue;
        }

        if p[0] < support_margin_um
            || p[1] < support_margin_um
            || p[2] < support_margin_um
            || max_x - p[0] < support_margin_um
            || max_y - p[1] < support_margin_um
            || max_z - p[2] < support_margin_um
        {
            support_excluded_vertices += 1;
            continue;
        }

        let q = [p[0] / h, p[1] / h, p[2] / h];
        if !roi_contains_trilinear(roi, q) {
            roi_excluded_vertices += 1;
            continue;
        }

        let Some(curv) = sample_curvature_trilinear(phi, width, height, depth, h, q) else {
            invalid_vertices += 1;
            continue;
        };
        if !(curv.h.is_finite() && curv.k.is_finite()) {
            invalid_vertices += 1;
            continue;
        }

        let discriminant = curv.h * curv.h - curv.k;

        selected_vertices += 1;
        w_sum += w;
        h_sum += w * curv.h;
        h_abs_sum += w * curv.h.abs();
        h2_sum += w * curv.h * curv.h;
        k_samples.push((curv.k, w));

        if discriminant < 0.0 {
            discriminant_negative_w += w;
        }
        if curv.k < 0.0 {
            saddle_w += w;
        } else if curv.h >= 0.0 {
            convex_w += w;
        } else {
            concave_w += w;
        }
    }

    if selected_vertices == 0 || w_sum <= 0.0 {
        bail!(
            "no SDF-curvature surface vertices remained after ROI/image-support selection; sigma={} µm, support margin={} µm",
            sigma_um,
            support_margin_um
        );
    }

    let h_mean = h_sum / w_sum;
    let h_var = (h2_sum / w_sum - h_mean * h_mean).max(0.0);

    // Signed K median and IQR.
    k_samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let k_q25 = weighted_quantile_sorted(&k_samples, 0.25);
    let k_median = weighted_quantile_sorted(&k_samples, 0.50);
    let k_q75 = weighted_quantile_sorted(&k_samples, 0.75);

    // Reuse the same allocation for |K| magnitude quantiles.
    for sample in &mut k_samples {
        sample.0 = sample.0.abs();
    }
    k_samples.sort_by(|a, b| a.0.total_cmp(&b.0));
    let abs_k_q90 = weighted_quantile_sorted(&k_samples, 0.90);
    let abs_k_q99 = weighted_quantile_sorted(&k_samples, 0.99);

    Ok(SdfCurvatureStats {
        sigma_um,
        support_margin_um,
        selected_surface_area_um2: w_sum,
        mean_curvature_mean: h_mean,
        mean_curvature_sd: h_var.sqrt(),
        mean_curvature_abs_mean: h_abs_sum / w_sum,
        gaussian_curvature_median: k_median,
        gaussian_curvature_iqr: k_q75 - k_q25,
        gaussian_curvature_abs_q90: abs_k_q90,
        gaussian_curvature_abs_q99: abs_k_q99,
        saddle_fraction: saddle_w / w_sum,
        convex_fraction: convex_w / w_sum,
        concave_fraction: concave_w / w_sum,
        discriminant_negative_fraction: discriminant_negative_w / w_sum,
        selected_vertices,
        roi_excluded_vertices,
        support_excluded_vertices,
        invalid_vertices,
    })
}

fn weighted_quantile_sorted(samples: &[(f64, f64)], probability: f64) -> f64 {
    debug_assert!(!samples.is_empty());
    debug_assert!((0.0..=1.0).contains(&probability));

    let total_weight = samples.iter().map(|x| x.1).sum::<f64>();
    debug_assert!(total_weight > 0.0);

    let first_fraction = samples[0].1 / total_weight;
    if probability <= first_fraction {
        return samples[0].0;
    }

    let mut previous_fraction = first_fraction;
    let mut previous_value = samples[0].0;
    let mut cumulative = samples[0].1;

    for &(value, weight) in &samples[1..] {
        cumulative += weight;
        let fraction = cumulative / total_weight;
        if probability <= fraction {
            let span = fraction - previous_fraction;
            if span <= 0.0 {
                return value;
            }
            let t = (probability - previous_fraction) / span;
            return previous_value + t * (value - previous_value);
        }
        previous_fraction = fraction;
        previous_value = value;
    }

    samples.last().expect("non-empty samples").0
}

fn roi_contains_trilinear(roi: &BinaryVolume, q: [f64; 3]) -> bool {
    if q.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let x0 = q[0].floor() as isize;
    let y0 = q[1].floor() as isize;
    let z0 = q[2].floor() as isize;
    let tx = q[0] - x0 as f64;
    let ty = q[1] - y0 as f64;
    let tz = q[2] - z0 as f64;

    let mut value = 0.0;
    for dz in 0..=1 {
        for dy in 0..=1 {
            for dx in 0..=1 {
                let wx = if dx == 0 { 1.0 - tx } else { tx };
                let wy = if dy == 0 { 1.0 - ty } else { ty };
                let wz = if dz == 0 { 1.0 - tz } else { tz };
                if roi.get_signed(x0 + dx, y0 + dy, z0 + dz) {
                    value += wx * wy * wz;
                }
            }
        }
    }
    value >= 0.5
}

fn sample_curvature_trilinear(
    phi: &[f64],
    width: usize,
    height: usize,
    depth: usize,
    h: f64,
    q: [f64; 3],
) -> Option<PointCurvature> {
    let x0 = q[0].floor() as isize;
    let y0 = q[1].floor() as isize;
    let z0 = q[2].floor() as isize;
    let tx = q[0] - x0 as f64;
    let ty = q[1] - y0 as f64;
    let tz = q[2] - z0 as f64;

    if x0 < 1
        || y0 < 1
        || z0 < 1
        || x0 + 2 >= width as isize
        || y0 + 2 >= height as isize
        || z0 + 2 >= depth as isize
    {
        return None;
    }

    let mut h_acc = 0.0;
    let mut k_acc = 0.0;
    for dz in 0..=1 {
        for dy in 0..=1 {
            for dx in 0..=1 {
                let wx = if dx == 0 { 1.0 - tx } else { tx };
                let wy = if dy == 0 { 1.0 - ty } else { ty };
                let wz = if dz == 0 { 1.0 - tz } else { tz };
                let c = curvature_at_grid(
                    phi,
                    width,
                    height,
                    depth,
                    h,
                    (x0 + dx) as usize,
                    (y0 + dy) as usize,
                    (z0 + dz) as usize,
                )?;
                let w = wx * wy * wz;
                h_acc += w * c.h;
                k_acc += w * c.k;
            }
        }
    }
    Some(PointCurvature { h: h_acc, k: k_acc })
}

#[inline(always)]
fn phi_at(
    phi: &[f64],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    z: usize,
) -> f64 {
    phi[(z * height + y) * width + x]
}

#[inline(always)]
fn gradient_at_grid(
    phi: &[f64],
    width: usize,
    height: usize,
    h: f64,
    x: usize,
    y: usize,
    z: usize,
) -> [f64; 3] {
    let inv_2h = 0.5 / h;
    [
        (phi_at(phi, width, height, x + 1, y, z)
            - phi_at(phi, width, height, x - 1, y, z))
            * inv_2h,
        (phi_at(phi, width, height, x, y + 1, z)
            - phi_at(phi, width, height, x, y - 1, z))
            * inv_2h,
        (phi_at(phi, width, height, x, y, z + 1)
            - phi_at(phi, width, height, x, y, z - 1))
            * inv_2h,
    ]
}

#[inline(always)]
fn outward_normal_at_grid(
    phi: &[f64],
    width: usize,
    height: usize,
    h: f64,
    x: usize,
    y: usize,
    z: usize,
) -> [f64; 3] {
    let g = gradient_at_grid(phi, width, height, h, x, y, z);
    let norm = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    let safe = norm.max(f64::EPSILON);
    [-g[0] / safe, -g[1] / safe, -g[2] / safe]
}

#[allow(clippy::too_many_arguments)]
fn curvature_at_grid(
    phi: &[f64],
    width: usize,
    height: usize,
    depth: usize,
    h: f64,
    x: usize,
    y: usize,
    z: usize,
) -> Option<PointCurvature> {
    // The independent Python reference uses np.gradient twice. At interior points this
    // requires a two-voxel stencil for pure second derivatives and a one-voxel
    // stencil for mixed derivatives.
    if x < 2
        || y < 2
        || z < 2
        || x + 2 >= width
        || y + 2 >= height
        || z + 2 >= depth
    {
        return None;
    }

    let inv_2h = 0.5 / h;
    let inv_4h2 = 0.25 / (h * h);

    let g = gradient_at_grid(phi, width, height, h, x, y, z);
    let gx = g[0];
    let gy = g[1];
    let gz = g[2];

    // Mean curvature follows the independent reference convention exactly:
    // H = 0.5 div(n), n = -grad(phi)/max(|grad(phi)|, eps).
    let nx_p = outward_normal_at_grid(phi, width, height, h, x + 1, y, z)[0];
    let nx_m = outward_normal_at_grid(phi, width, height, h, x - 1, y, z)[0];
    let ny_p = outward_normal_at_grid(phi, width, height, h, x, y + 1, z)[1];
    let ny_m = outward_normal_at_grid(phi, width, height, h, x, y - 1, z)[1];
    let nz_p = outward_normal_at_grid(phi, width, height, h, x, y, z + 1)[2];
    let nz_m = outward_normal_at_grid(phi, width, height, h, x, y, z - 1)[2];
    let h_mean = 0.5 * ((nx_p - nx_m) + (ny_p - ny_m) + (nz_p - nz_m)) * inv_2h;

    // np.gradient(np.gradient(phi)) pure second derivatives use +/-2.
    let f0 = phi_at(phi, width, height, x, y, z);
    let fxx = (phi_at(phi, width, height, x + 2, y, z)
        - 2.0 * f0
        + phi_at(phi, width, height, x - 2, y, z))
        * inv_4h2;
    let fyy = (phi_at(phi, width, height, x, y + 2, z)
        - 2.0 * f0
        + phi_at(phi, width, height, x, y - 2, z))
        * inv_4h2;
    let fzz = (phi_at(phi, width, height, x, y, z + 2)
        - 2.0 * f0
        + phi_at(phi, width, height, x, y, z - 2))
        * inv_4h2;

    let fxy = (phi_at(phi, width, height, x + 1, y + 1, z)
        - phi_at(phi, width, height, x + 1, y - 1, z)
        - phi_at(phi, width, height, x - 1, y + 1, z)
        + phi_at(phi, width, height, x - 1, y - 1, z))
        * inv_4h2;
    let fxz = (phi_at(phi, width, height, x + 1, y, z + 1)
        - phi_at(phi, width, height, x + 1, y, z - 1)
        - phi_at(phi, width, height, x - 1, y, z + 1)
        + phi_at(phi, width, height, x - 1, y, z - 1))
        * inv_4h2;
    let fyz = (phi_at(phi, width, height, x, y + 1, z + 1)
        - phi_at(phi, width, height, x, y + 1, z - 1)
        - phi_at(phi, width, height, x, y - 1, z + 1)
        + phi_at(phi, width, height, x, y - 1, z - 1))
        * inv_4h2;

    let g2 = gx * gx + gy * gy + gz * gz;
    if !g2.is_finite() {
        return None;
    }
    let safe2 = g2.max(f64::EPSILON * f64::EPSILON);

    let cxx = fyy * fzz - fyz * fyz;
    let cyy = fxx * fzz - fxz * fxz;
    let czz = fxx * fyy - fxy * fxy;
    let cxy = fxz * fyz - fxy * fzz;
    let cxz = fxy * fyz - fxz * fyy;
    let cyz = fxy * fxz - fyz * fxx;
    let numer = gx * gx * cxx
        + gy * gy * cyy
        + gz * gz * czz
        + 2.0 * gx * gy * cxy
        + 2.0 * gx * gz * cxz
        + 2.0 * gy * gz * cyz;
    let k_gauss = numer / (safe2 * safe2);

    if !(h_mean.is_finite() && k_gauss.is_finite()) {
        return None;
    }

    Some(PointCurvature {
        h: h_mean,
        k: k_gauss,
    })
}

/// Crate-internal cross-language validation hook. Compiled only with
/// `--features validation-hooks`; it does not change production CLI behavior.
#[cfg(feature = "validation-hooks")]
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SdfCurvatureGridSample {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub phi_um: f64,
    pub h_um_inv: f64,
    pub k_um_inv2: f64,
}

#[cfg(feature = "validation-hooks")]
#[allow(dead_code)]
pub(crate) fn sdf_curvature_grid_samples(
    bone: &BinaryVolume,
    spacing: [f64; 3],
    parameters: SdfCurvatureParameters,
    points: &[[usize; 3]],
) -> Result<Vec<SdfCurvatureGridSample>> {
    let h = require_isotropic_spacing(spacing)?;
    if !parameters.sigma_um.is_finite() || parameters.sigma_um <= 0.0 {
        bail!("curvature sigma must be finite and positive");
    }

    let mut phi = signed_distance_field_um(bone, h)?;
    gaussian_smooth_nearest_in_place(
        &mut phi,
        bone.width,
        bone.height,
        bone.depth,
        parameters.sigma_um / h,
    );

    let mut out = Vec::with_capacity(points.len());
    for &[x, y, z] in points {
        if x >= bone.width || y >= bone.height || z >= bone.depth {
            bail!("validation grid sample [{x}, {y}, {z}] is outside the volume");
        }
        let curv = curvature_at_grid(
            &phi,
            bone.width,
            bone.height,
            bone.depth,
            h,
            x,
            y,
            z,
        )
        .ok_or_else(|| {
            anyhow::anyhow!(
                "validation grid sample [{x}, {y}, {z}] lacks a valid curvature stencil"
            )
        })?;
        out.push(SdfCurvatureGridSample {
            x,
            y,
            z,
            phi_um: phi[(z * bone.height + y) * bone.width + x],
            h_um_inv: curv.h,
            k_um_inv2: curv.k,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(radius_vox: f64, margin: usize) -> BinaryVolume {
        let half = radius_vox.ceil() as usize + margin;
        let n = 2 * half + 1;
        let c = half as f64;
        let mut data = vec![0u8; n * n * n];
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let d2 = (x as f64 - c).powi(2)
                        + (y as f64 - c).powi(2)
                        + (z as f64 - c).powi(2);
                    if d2 <= radius_vox * radius_vox {
                        data[(z * n + y) * n + x] = 1;
                    }
                }
            }
        }
        BinaryVolume::new(data, n, n, n).unwrap()
    }

    #[test]
    fn gaussian_kernel_is_normalized() {
        for sigma in [0.5, 1.0, 1.5, 2.0] {
            let k = gaussian_kernel(sigma);
            assert!((k.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn gaussian_filter_preserves_constant_field() {
        let mut data = vec![3.25f64; 9 * 8 * 7];
        gaussian_smooth_nearest_in_place(&mut data, 9, 8, 7, 1.5);
        assert!(data.iter().all(|&x| (x - 3.25).abs() < 1e-12));
    }


    #[test]
    fn weighted_quantile_matches_reference_interpolation_convention() {
        // numpy.interp(p, cumulative_weight_fraction, sorted_values)
        let samples = vec![(0.0, 1.0), (10.0, 1.0)];
        assert_eq!(weighted_quantile_sorted(&samples, 0.25), 0.0);
        assert_eq!(weighted_quantile_sorted(&samples, 0.50), 0.0);
        assert!((weighted_quantile_sorted(&samples, 0.75) - 5.0).abs() < 1e-15);
        assert_eq!(weighted_quantile_sorted(&samples, 1.00), 10.0);
    }

    #[test]
    fn smoothed_sdf_sphere_has_positive_curvature_and_reasonable_scale() {
        let bone = sphere(12.0, 10);
        let roi = BinaryVolume::new(vec![1u8; bone.len()], bone.width, bone.height, bone.depth)
            .unwrap();
        let stats = sdf_curvature_stats(
            &bone,
            &roi,
            [1.0; 3],
            SdfCurvatureParameters {
                sigma_um: 1.5,
                image_support_sigma: 4.0,
            },
        )
        .unwrap();

        let expected_h = 1.0 / 12.0;
        let expected_k = 1.0 / 144.0;
        assert!(stats.mean_curvature_mean > 0.0);
        assert!(stats.gaussian_curvature_median > 0.0);
        assert!((stats.mean_curvature_mean - expected_h).abs() / expected_h < 0.20);
        assert!((stats.gaussian_curvature_median - expected_k).abs() / expected_k < 0.35);
        assert!(stats.gaussian_curvature_iqr >= 0.0);
        assert!(stats.gaussian_curvature_abs_q99 >= stats.gaussian_curvature_abs_q90);
        assert!(stats.convex_fraction > 0.90);
    }
}
