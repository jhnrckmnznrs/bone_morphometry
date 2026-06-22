use anyhow::{bail, Result};
use clap::ValueEnum;

use crate::edt::Phase;
use crate::local_thickness::local_thickness;
use crate::topology::{analyze_topology, cubical_measures, Topology};
use crate::volume::BinaryVolume;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum FeatureMode {
    /// Original mean/std/max thickness and spacing plus corrected connectivity outputs.
    Classic,
    /// Robust quantiles, Betti densities, cubical surface area, and mean breadth.
    Refined,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ConnectivityMode {
    /// Generalized value: beta0 + beta2 - corrected Euler.
    Generalized,
    /// BoneJ convention for a purified stack: 1 - corrected Euler.
    Bonej,
}

#[derive(Clone, Copy, Debug)]
pub struct DistributionStats {
    pub mean: f64,
    pub standard_deviation: f64,
    pub maximum: f64,
    pub median: f64,
    pub percentile_10: f64,
    pub percentile_90: f64,
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
    pub connectivity: f64,
    pub connectivity_density: f64,
    pub geometry: GeometryMetrics,
}

pub fn analyze(
    bone: &BinaryVolume,
    voxel_size: f64,
    feature_mode: FeatureMode,
    connectivity_mode: ConnectivityMode,
) -> Result<Metrics> {
    if !voxel_size.is_finite() || voxel_size <= 0.0 {
        bail!("voxel size must be finite and positive; got {voxel_size}");
    }
    if !bone.data.iter().any(|&value| value != 0) {
        bail!("volume contains no bone voxels");
    }
    if !bone.data.contains(&0) {
        bail!("volume contains no background voxels");
    }

    let include_quantiles = feature_mode == FeatureMode::Refined;

    let mut thickness_radii = local_thickness(bone, Phase::Foreground)?;
    let thickness = nonzero_stats(&mut thickness_radii, 2.0 * voxel_size, include_quantiles)?;
    drop(thickness_radii);

    let mut spacing_radii = local_thickness(bone, Phase::Background)?;
    let spacing = nonzero_stats(&mut spacing_radii, 2.0 * voxel_size, include_quantiles)?;
    drop(spacing_radii);

    let voxel_volume = voxel_size.powi(3);
    let bone_voxels = bone.data.iter().filter(|&&value| value != 0).count();
    let bone_volume = bone_voxels as f64 * voxel_volume;
    let total_volume = bone.len() as f64 * voxel_volume;
    let bone_volume_fraction = bone_volume / total_volume;

    let cubical = cubical_measures(bone);
    let topology = analyze_topology(
        bone,
        cubical.raw_euler,
        feature_mode == FeatureMode::Classic,
    )?;
    let connectivity = if feature_mode == FeatureMode::Classic {
        match connectivity_mode {
            ConnectivityMode::Generalized => {
                topology.beta0 as f64 + topology.beta2 as f64 - topology.corrected_euler
            }
            ConnectivityMode::Bonej => 1.0 - topology.corrected_euler,
        }
    } else {
        f64::NAN
    };
    let connectivity_density = connectivity / total_volume;

    let bone_surface = cubical.surface_face_count as f64 * voxel_size.powi(2);
    let geometry = GeometryMetrics {
        bone_surface,
        bone_surface_to_bone_volume: bone_surface / bone_volume,
        bone_surface_to_total_volume: bone_surface / total_volume,
        mean_breadth: cubical.mean_breadth_lattice_units * voxel_size,
    };

    Ok(Metrics {
        thickness,
        spacing,
        bone_volume,
        total_volume,
        bone_volume_fraction,
        topology,
        connectivity,
        connectivity_density,
        geometry,
    })
}

fn nonzero_stats(
    values: &mut Vec<f32>,
    scale: f64,
    include_quantiles: bool,
) -> Result<DistributionStats> {
    let mut count = 0u64;
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    let mut maximum = 0.0f64;

    for &raw in values.iter() {
        if raw <= 0.0 {
            continue;
        }
        let value = raw as f64 * scale;
        count += 1;
        let delta = value - mean;
        mean += delta / count as f64;
        let delta2 = value - mean;
        m2 += delta * delta2;
        maximum = maximum.max(value);
    }

    if count == 0 {
        bail!("cannot compute statistics because the selected phase is empty");
    }

    let standard_deviation = (m2 / count as f64).sqrt();
    let (median, percentile_10, percentile_90) = if include_quantiles {
        values.retain(|value| *value > 0.0);

        (
            linear_quantile_unstable(values, 0.5) * scale,
            linear_quantile_unstable(values, 0.1) * scale,
            linear_quantile_unstable(values, 0.9) * scale,
        )
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    };

    Ok(DistributionStats {
        mean,
        standard_deviation,
        maximum,
        median,
        percentile_10,
        percentile_90,
        coefficient_of_variation: standard_deviation / mean,
    })
}

fn linear_quantile_unstable(values: &mut [f32], quantile: f64) -> f64 {
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

    lower_value + fraction * (upper_value - lower_value)
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

        assert!((linear_quantile_unstable(&mut values, 0.1) - 1.3).abs() < 1e-12);

        assert!((linear_quantile_unstable(&mut values, 0.5) - 2.5).abs() < 1e-12);

        assert!((linear_quantile_unstable(&mut values, 0.9) - 3.7).abs() < 1e-12);
    }

    #[test]
    fn distribution_statistics_include_cv() {
        let mut values = vec![0.0f32, 1.0, 2.0, 3.0, 4.0];
        let stats = nonzero_stats(&mut values, 2.0, true).unwrap();
        assert!((stats.mean - 5.0).abs() < 1e-12);
        assert!((stats.median - 5.0).abs() < 1e-12);
        assert!((stats.percentile_10 - 2.6).abs() < 1e-12);
        assert!((stats.percentile_90 - 7.4).abs() < 1e-12);
        assert!((stats.maximum - 8.0).abs() < 1e-12);
        assert!((stats.coefficient_of_variation - stats.standard_deviation / 5.0).abs() < 1e-12);
    }
}
