use anyhow::Result;

use crate::anisotropy::{degree_of_anisotropy, DaParameters, DaStats};
use crate::ellipsoid_factor::{ellipsoid_factor_candidate_from_skeleton, EllipsoidFactorStats};
use crate::mesh::marching_cubes_bonej;
use crate::plate_rod::{plate_rod_proxy_from_skeleton, PlateRodProxyStats};
use crate::skeleton::{graph_stats, skeletonize, GraphStats};
use crate::sdf_curvature::{sdf_curvature_stats, SdfCurvatureParameters, SdfCurvatureStats};
use crate::volume::BinaryVolume;

#[derive(Clone, Copy, Debug)]
pub struct SurfaceMetrics {
    pub mesh_surface_area: f64,
    pub mesh_surface_to_bone_volume: f64,
    pub mesh_surface_to_total_volume: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct AdvancedMetrics {
    pub curvature: SdfCurvatureStats,
    pub graph: GraphStats,
    pub plate_rod_proxy: PlateRodProxyStats,
}

#[derive(Clone, Copy, Debug)]
pub struct ExtendedMetrics {
    pub surface: SurfaceMetrics,
    pub da: DaStats,
    pub ef_candidate: EllipsoidFactorStats,
    pub advanced: Option<AdvancedMetrics>,
}

// The analysis inputs are kept explicit because they are independent scientific
// controls rather than an incidental parameter bundle.
#[allow(clippy::too_many_arguments)]
pub fn analyze_extended(
    bone: &BinaryVolume,
    roi: &BinaryVolume,
    spacing: [f64; 3],
    bone_volume: f64,
    total_volume: f64,
    da_parameters: DaParameters,
    ef_window_radius: usize,
    advanced: bool,
    curvature_parameters: SdfCurvatureParameters,
) -> Result<ExtendedMetrics> {
    let analysis_bone = bone.phase_inside(roi, true)?;
    let surface_mesh = marching_cubes_bonej(&analysis_bone, spacing);
    let mesh_surface_area = surface_mesh.surface_area();
    let surface = SurfaceMetrics {
        mesh_surface_area,
        mesh_surface_to_bone_volume: mesh_surface_area / bone_volume,
        mesh_surface_to_total_volume: mesh_surface_area / total_volume,
    };

    let skeleton = skeletonize(&analysis_bone);
    let ef_candidate =
        ellipsoid_factor_candidate_from_skeleton(&analysis_bone, &skeleton, ef_window_radius);
    let da = degree_of_anisotropy(&analysis_bone, roi, da_parameters)?;

    let advanced_metrics = if advanced {
        // Curvature is deliberately computed from the *uncut* bone image and
        // only then restricted to the ROI. This avoids manufacturing a curved
        // bone/ROI interface. The SDF bandwidth is specified in physical units.
        Some(AdvancedMetrics {
            curvature: sdf_curvature_stats(bone, roi, spacing, curvature_parameters)?,
            graph: graph_stats(&skeleton, spacing),
            plate_rod_proxy: plate_rod_proxy_from_skeleton(
                &analysis_bone,
                &skeleton,
                ef_window_radius,
            ),
        })
    } else {
        None
    };

    Ok(ExtendedMetrics {
        surface,
        da,
        ef_candidate,
        advanced: advanced_metrics,
    })
}
