use anyhow::{bail, Result};
use clap::ValueEnum;

use crate::edt::Phase;
use crate::local_thickness::local_thickness;
use crate::topology::{
    analyze_topology, bone_marrow_interface_faces, bonej_connectivity, cubical_measures,
    BonejConnectivity, Topology,
};
use crate::volume::BinaryVolume;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum FeatureMode {
    /// The same 13 measurements and definitions as the ROI-aware BoneJ batch script.
    Bonej,
    /// Extended BoneJ-comparable suite. `bonej` remains the 13-column compatibility schema.
    Standard,
    /// Mean/std/max thickness and spacing plus ROI-clipped topology outputs.
    Classic,
    /// Classic outputs plus cubical bone surface area and mean breadth.
    Refined,
    /// Study feature set with robust quantiles, Betti densities, and normalized surface measures.
    Experimental,
}

#[derive(Clone, Copy, Debug)]
pub struct DistributionStats {
    pub mean: f64,
    pub standard_deviation: f64,
    pub maximum: f64,
    pub median: f64,
    pub percentile_10: f64,
    pub percentile_25: f64,
    pub percentile_75: f64,
    pub percentile_90: f64,
    pub interquartile_range: f64,
    pub coefficient_of_variation: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct GeometryMetrics {
    pub bone_surface: f64,
    pub bone_surface_to_bone_volume: f64,
    pub bone_surface_to_total_volume: f64,
    pub mean_breadth: f64,
}

#[derive(Clone, Debug)]
pub struct Metrics {
    pub thickness: DistributionStats,
    pub spacing: DistributionStats,
    pub bone_volume: f64,
    pub total_volume: f64,
    pub bone_volume_fraction: f64,
    pub topology: Topology,
    pub bonej_connectivity: BonejConnectivity,
    pub bonej_connectivity_density: f64,
    pub connectivity: f64,
    pub connectivity_density: f64,
    pub roi_boundary_bone_fraction: f64,
    pub geometry: GeometryMetrics,
}

pub fn analyze(
    bone: &BinaryVolume,
    roi: &BinaryVolume,
    voxel_spacing: [f64; 3],
    feature_mode: FeatureMode,
) -> Result<Metrics> {
    if voxel_spacing
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        bail!(
            "voxel spacings must be finite and positive; got {:?}",
            voxel_spacing
        );
    }
    if !bone.same_shape(roi) {
        bail!(
            "binary image dimensions {}x{}x{} do not match ROI dimensions {}x{}x{}",
            bone.width,
            bone.height,
            bone.depth,
            roi.width,
            roi.height,
            roi.depth
        );
    }
    if !roi.data.iter().any(|&value| value != 0) {
        bail!("ROI mask contains no included voxels");
    }
    if feature_mode == FeatureMode::Bonej && roi.data.iter().all(|&value| value != 0) {
        bail!("BoneJ compatibility mode requires an ROI mask with both inside and outside voxels");
    }

    let analysis_bone = bone.phase_inside(roi, true)?;
    let analysis_marrow = bone.phase_inside(roi, false)?;
    if !analysis_bone.data.iter().any(|&value| value != 0) {
        bail!("ROI contains no bone voxels");
    }
    if !analysis_marrow.data.iter().any(|&value| value != 0) {
        bail!("ROI contains no marrow voxels");
    }

    let include_quantiles = matches!(
        feature_mode,
        FeatureMode::Standard | FeatureMode::Experimental
    );

    let mut thickness_diameters = local_thickness(&analysis_bone, Phase::Foreground)?;
    let thickness = nonzero_stats(
        &mut thickness_diameters,
        voxel_spacing[0],
        include_quantiles,
    )?;
    drop(thickness_diameters);

    let mut spacing_diameters = local_thickness(&analysis_marrow, Phase::Foreground)?;
    let spacing = nonzero_stats(&mut spacing_diameters, voxel_spacing[0], include_quantiles)?;
    drop(spacing_diameters);

    let voxel_volume = voxel_spacing.iter().product::<f64>();
    let bone_voxels = analysis_bone
        .data
        .iter()
        .filter(|&&value| value != 0)
        .count();
    let roi_voxels = roi.data.iter().filter(|&&value| value != 0).count();
    let bone_volume = bone_voxels as f64 * voxel_volume;
    let total_volume = roi_voxels as f64 * voxel_volume;
    let bone_volume_fraction = bone_volume / total_volume;

    let cubical = cubical_measures(&analysis_bone);
    let topology = analyze_topology(&analysis_bone, cubical.raw_euler)?;
    let bonej_connectivity = bonej_connectivity(&analysis_bone, cubical.raw_euler);
    let bonej_connectivity_density = bonej_connectivity.connectivity / total_volume;
    let connectivity = topology.beta1 as f64;
    let connectivity_density = connectivity / total_volume;

    let interface_faces = bone_marrow_interface_faces(&analysis_bone, roi)?;
    let bone_surface = interface_faces as f64 * voxel_spacing[0] * voxel_spacing[1];
    let geometry = GeometryMetrics {
        bone_surface,
        bone_surface_to_bone_volume: bone_surface / bone_volume,
        bone_surface_to_total_volume: bone_surface / total_volume,
        mean_breadth: cubical.mean_breadth_lattice_units * voxel_spacing[0],
    };

    Ok(Metrics {
        thickness,
        spacing,
        bone_volume,
        total_volume,
        bone_volume_fraction,
        topology,
        bonej_connectivity,
        bonej_connectivity_density,
        connectivity,
        connectivity_density,
        roi_boundary_bone_fraction: roi_boundary_bone_fraction(&analysis_bone, roi),
        geometry,
    })
}

fn roi_boundary_bone_fraction(analysis_bone: &BinaryVolume, roi: &BinaryVolume) -> f64 {
    const OFFSETS: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    let mut bone_voxels = 0usize;
    let mut boundary_bone_voxels = 0usize;
    for z in 0..analysis_bone.depth {
        for y in 0..analysis_bone.height {
            for x in 0..analysis_bone.width {
                if analysis_bone.get(x, y, z) == 0 {
                    continue;
                }
                bone_voxels += 1;
                let touches_boundary = OFFSETS.iter().any(|&(dx, dy, dz)| {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    let nz = z as isize + dz;
                    nx < 0
                        || ny < 0
                        || nz < 0
                        || nx >= roi.width as isize
                        || ny >= roi.height as isize
                        || nz >= roi.depth as isize
                        || roi.get(nx as usize, ny as usize, nz as usize) == 0
                });
                boundary_bone_voxels += touches_boundary as usize;
            }
        }
    }
    boundary_bone_voxels as f64 / bone_voxels as f64
}

fn nonzero_stats(
    values: &mut Vec<f32>,
    scale: f64,
    include_quantiles: bool,
) -> Result<DistributionStats> {
    let mut count = 0u64;
    let mut sum = 0.0f64;
    let mut sum_of_squares = 0.0f64;
    let mut maximum = 0.0f64;
    let scale_f32 = scale as f32;

    for &raw in values.iter() {
        if !raw.is_finite() || raw <= 0.0 {
            continue;
        }
        // Fiji calibrates the 32-bit thickness map with FloatProcessor.multiply
        // before StackStatistics is evaluated, so retain the same f32 rounding.
        let value = (raw * scale_f32) as f64;
        count += 1;
        sum += value;
        sum_of_squares += value * value;
        maximum = maximum.max(value);
    }

    if count == 0 {
        bail!("cannot compute statistics because the selected phase is empty");
    }

    let count_f64 = count as f64;
    let mean = sum / count_f64;
    // ImageJ's ImageStatistics.calculateStdDev() reports the sample standard
    // deviation. It returns zero when there is only one finite map value.
    let centered_sum_of_squares = (count_f64 * sum_of_squares - sum * sum) / count_f64;
    let standard_deviation = if count > 1 && centered_sum_of_squares > 0.0 {
        (centered_sum_of_squares / (count_f64 - 1.0)).sqrt()
    } else {
        0.0
    };
    let (median, percentile_10, percentile_25, percentile_75, percentile_90) = if include_quantiles
    {
        values.retain(|value| value.is_finite() && *value > 0.0);

        (
            linear_quantile_unstable(values, 0.5, scale_f32),
            linear_quantile_unstable(values, 0.1, scale_f32),
            linear_quantile_unstable(values, 0.25, scale_f32),
            linear_quantile_unstable(values, 0.75, scale_f32),
            linear_quantile_unstable(values, 0.9, scale_f32),
        )
    } else {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    };

    Ok(DistributionStats {
        mean,
        standard_deviation,
        maximum,
        median,
        percentile_10,
        percentile_25,
        percentile_75,
        percentile_90,
        interquartile_range: percentile_75 - percentile_25,
        coefficient_of_variation: standard_deviation / mean,
    })
}

fn linear_quantile_unstable(values: &mut [f32], quantile: f64, scale: f32) -> f64 {
    debug_assert!(!values.is_empty());
    debug_assert!((0.0..=1.0).contains(&quantile));

    let position = quantile * (values.len() - 1) as f64;

    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;

    let lower_value = select_value(values, lower);

    let upper_value = if upper == lower {
        lower_value
    } else {
        select_value(values, upper)
    };

    let quantile_value = lower_value + fraction * (upper_value - lower_value);
    (quantile_value as f32 * scale) as f64
}

#[inline]
fn select_value(values: &mut [f32], index: usize) -> f64 {
    let (_, value, _) = values.select_nth_unstable_by(index, |left, right| left.total_cmp(right));

    *value as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantiles_use_numpy_linear_interpolation() {
        let mut values = vec![1.0f32, 2.0, 3.0, 4.0];

        assert!((linear_quantile_unstable(&mut values, 0.1, 1.0) - 1.3).abs() < 1e-6);

        assert!((linear_quantile_unstable(&mut values, 0.5, 1.0) - 2.5).abs() < 1e-6);

        assert!((linear_quantile_unstable(&mut values, 0.9, 1.0) - 3.7).abs() < 1e-6);
    }

    #[test]
    fn distribution_statistics_include_cv() {
        let mut values = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let stats = nonzero_stats(&mut values, 2.0, true).unwrap();
        assert!((stats.mean - 5.0).abs() < 1e-12);
        assert!((stats.median - 5.0).abs() < 1e-12);
        assert!((stats.percentile_10 - 2.6).abs() < 1e-6);
        assert!((stats.percentile_90 - 7.4).abs() < 1e-6);
        assert!((stats.maximum - 8.0).abs() < 1e-12);
        assert!((stats.standard_deviation - 2.581_988_897_471_611).abs() < 1e-12);
        assert!((stats.coefficient_of_variation - stats.standard_deviation / 5.0).abs() < 1e-12);
    }

    #[test]
    fn roi_controls_volume_and_excludes_cut_faces_from_surface() {
        let mut bone_data = vec![0u8; 125];
        bone_data[(2 * 5 + 2) * 5 + 1] = 1;
        let bone = BinaryVolume::new(bone_data, 5, 5, 5).unwrap();

        let mut roi_data = vec![0u8; 125];
        for z in 1..=3 {
            for y in 1..=3 {
                for x in 1..=3 {
                    roi_data[(z * 5 + y) * 5 + x] = 1;
                }
            }
        }
        let roi = BinaryVolume::new(roi_data, 5, 5, 5).unwrap();
        let metrics = analyze(&bone, &roi, [1.0; 3], FeatureMode::Refined).unwrap();

        assert_eq!(metrics.bone_volume, 1.0);
        assert_eq!(metrics.total_volume, 27.0);
        assert!((metrics.bone_volume_fraction - 1.0 / 27.0).abs() < 1e-12);
        assert_eq!(metrics.geometry.bone_surface, 5.0);
        assert_eq!(metrics.topology.raw_euler, 1);
        assert_eq!(metrics.topology.beta0, 1);
        assert_eq!(metrics.topology.beta1, 0);
        assert_eq!(metrics.topology.beta2, 0);
        assert_eq!(metrics.roi_boundary_bone_fraction, 1.0);
    }
}
