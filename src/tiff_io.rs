use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{bail, Context, Result};
use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;
use tiff::ColorType;

use crate::volume::BinaryVolume;

#[derive(Clone, Copy)]
enum AcceptedSamples {
    Any,
    ZeroOr255,
    RoiMask,
}

pub fn read_binary_tiff(path: &Path, threshold: u8, strict_binary: bool) -> Result<BinaryVolume> {
    let accepted = if strict_binary {
        AcceptedSamples::ZeroOr255
    } else {
        AcceptedSamples::Any
    };
    read_binary_tiff_impl(path, threshold, accepted)
}

/// Read the ROI encodings accepted by the Fiji batch script: 0/1 or 0/255.
pub fn read_roi_tiff(path: &Path) -> Result<BinaryVolume> {
    read_binary_tiff_impl(path, 0, AcceptedSamples::RoiMask)
}

fn read_binary_tiff_impl(
    path: &Path,
    threshold: u8,
    accepted_samples: AcceptedSamples,
) -> Result<BinaryVolume> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut decoder = Decoder::new(BufReader::new(file))
        .with_context(|| format!("initializing TIFF decoder for {}", path.display()))?;

    let mut volume = Vec::new();
    let mut expected_width = None;
    let mut expected_height = None;
    let mut depth = 0usize;

    loop {
        let (width_u32, height_u32) = decoder
            .dimensions()
            .with_context(|| format!("reading dimensions from {}", path.display()))?;
        let width = usize::try_from(width_u32)?;
        let height = usize::try_from(height_u32)?;

        match decoder
            .colortype()
            .with_context(|| format!("reading color type from {}", path.display()))?
        {
            ColorType::Gray(8) => {}
            other => bail!(
                "{} must contain 8-bit grayscale pages; found {:?}",
                path.display(),
                other
            ),
        }

        if let (Some(w), Some(h)) = (expected_width, expected_height) {
            if width != w || height != h {
                bail!(
                    "TIFF pages in {} do not have consistent dimensions: expected {}x{}, got {}x{}",
                    path.display(),
                    w,
                    h,
                    width,
                    height
                );
            }
        } else {
            expected_width = Some(width);
            expected_height = Some(height);
        }

        let page = match decoder
            .read_image()
            .with_context(|| format!("decoding page {} of {}", depth + 1, path.display()))?
        {
            DecodingResult::U8(values) => values,
            _other => bail!(
                "{} decoded to an unsupported pixel type; expected unsigned 8-bit samples",
                path.display()
            ),
        };

        let expected_page_len = width
            .checked_mul(height)
            .ok_or_else(|| anyhow::anyhow!("TIFF page dimensions overflow usize"))?;
        if page.len() != expected_page_len {
            bail!(
                "decoded page {} of {} has {} samples; expected {}",
                depth + 1,
                path.display(),
                page.len(),
                expected_page_len
            );
        }

        let invalid = page
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| match accepted_samples {
                AcceptedSamples::Any => false,
                AcceptedSamples::ZeroOr255 => *value != 0 && *value != 255,
                AcceptedSamples::RoiMask => *value != 0 && *value != 1 && *value != 255,
            });
        if let Some((offset, value)) = invalid {
            let expected = match accepted_samples {
                AcceptedSamples::Any => unreachable!(),
                AcceptedSamples::ZeroOr255 => "0 or 255",
                AcceptedSamples::RoiMask => "0/1 or 0/255",
            };
            bail!(
                "{} is not a valid binary image: page {}, sample {} has value {} (expected {})",
                path.display(),
                depth + 1,
                offset,
                value,
                expected
            );
        }

        volume.extend(page.into_iter().map(|value| u8::from(value > threshold)));
        depth += 1;

        if decoder.more_images() {
            decoder
                .next_image()
                .with_context(|| format!("advancing to the next page of {}", path.display()))?;
        } else {
            break;
        }
    }

    BinaryVolume::new(
        volume,
        expected_width.unwrap_or(0),
        expected_height.unwrap_or(0),
        depth,
    )
}

/// Read an isotropic voxel side length in micrometres.
///
/// TIFF X/Y resolution values are pixels per ResolutionUnit, not physical
/// pixel sizes. ResolutionUnit=2 means inch, so spacing is 25_400 divided by
/// pixels per inch. ResolutionUnit=3 means centimetre. ImageJ
/// `spacing=`/`unit=` metadata supplies the z calibration. All three spatial
/// spacings are required and checked for isotropy specimen by specimen.
pub fn read_isotropic_spacing_um(path: &Path) -> Result<[f64; 3]> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut decoder = Decoder::new(BufReader::new(file))
        .with_context(|| format!("initializing TIFF decoder for {}", path.display()))?;

    let description = decoder.get_tag_ascii_string(Tag::ImageDescription).ok();
    let named_unit_um = description
        .as_deref()
        .and_then(|text| description_value(text, "unit"))
        .and_then(unit_scale_um);

    let resolution_unit: Option<u16> = decoder
        .find_tag_unsigned(Tag::ResolutionUnit)
        .with_context(|| format!("reading ResolutionUnit from {}", path.display()))?;
    let resolution_unit_um = match resolution_unit {
        Some(2) => Some(25_400.0),
        Some(3) => Some(10_000.0),
        Some(1) | None => named_unit_um,
        Some(code) => {
            bail!(
                "{} has unsupported TIFF ResolutionUnit code {}",
                path.display(),
                code
            )
        }
    };

    let x_pixels_per_unit = decoder.get_tag_f64(Tag::XResolution).ok();
    let y_pixels_per_unit = decoder.get_tag_f64(Tag::YResolution).ok();
    let xy_spacing = if let (Some(x_resolution), Some(y_resolution), Some(unit_um)) =
        (x_pixels_per_unit, y_pixels_per_unit, resolution_unit_um)
    {
        Some(isotropic_spacing_from_resolution(
            x_resolution,
            y_resolution,
            unit_um,
            path,
        )?)
    } else {
        None
    };

    let z_spacing = if let Some(description) = description {
        if let Some(value) = description_value(&description, "spacing") {
            let spacing_in_named_units: f64 = value
                .parse()
                .with_context(|| format!("parsing spacing metadata in {}", path.display()))?;
            let scale_um = named_unit_um.unwrap_or(1.0);
            let spacing_um = spacing_in_named_units * scale_um;
            if !spacing_um.is_finite() || spacing_um <= 0.0 {
                bail!(
                    "spacing in {} must be finite and positive; got {} µm",
                    path.display(),
                    spacing_um
                );
            }
            Some(spacing_um)
        } else {
            None
        }
    } else {
        None
    };

    combine_isotropic_spacing(xy_spacing, z_spacing, path)
}

fn combine_isotropic_spacing(
    xy_spacing_um: Option<[f64; 2]>,
    z_spacing_um: Option<f64>,
    path: &Path,
) -> Result<[f64; 3]> {
    let xy = xy_spacing_um.ok_or_else(|| {
        anyhow::anyhow!(
            "x/y spacing was not found in the TIFF resolution metadata of {}",
            path.display()
        )
    })?;
    let z = z_spacing_um.ok_or_else(|| {
        anyhow::anyhow!(
            "z spacing was not found in the 'spacing=' ImageDescription metadata of {}; use --spacing-dir if the binarized TIFF lost calibration",
            path.display()
        )
    })?;

    let values = [xy[0], xy[1], z];
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
    if (maximum - minimum) / maximum > 1.0e-6 {
        bail!(
            "{} is not isotropic in 3D: x spacing={} µm, y spacing={} µm, z spacing={} µm",
            path.display(),
            xy[0],
            xy[1],
            z
        );
    }
    Ok(values)
}

fn isotropic_spacing_from_resolution(
    x_pixels_per_unit: f64,
    y_pixels_per_unit: f64,
    unit_um: f64,
    path: &Path,
) -> Result<[f64; 2]> {
    if !x_pixels_per_unit.is_finite()
        || !y_pixels_per_unit.is_finite()
        || x_pixels_per_unit <= 0.0
        || y_pixels_per_unit <= 0.0
    {
        bail!(
            "{} has invalid X/Y resolution values ({}, {})",
            path.display(),
            x_pixels_per_unit,
            y_pixels_per_unit
        );
    }

    let spacing_x_um = unit_um / x_pixels_per_unit;
    let spacing_y_um = unit_um / y_pixels_per_unit;
    let mean = 0.5 * (spacing_x_um + spacing_y_um);
    let relative_difference = (spacing_x_um - spacing_y_um).abs() / mean;
    if relative_difference > 0.01 {
        bail!(
            "{} is not isotropic in XY: x spacing={} µm, y spacing={} µm",
            path.display(),
            spacing_x_um,
            spacing_y_um
        );
    }
    Ok([spacing_x_um, spacing_y_um])
}

fn description_value<'a>(description: &'a str, key: &str) -> Option<&'a str> {
    description.lines().find_map(|raw_line| {
        let line = raw_line.trim_matches('\0').trim();
        line.strip_prefix(key)
            .and_then(|rest| rest.strip_prefix('='))
            .map(str::trim)
    })
}

fn unit_scale_um(unit: &str) -> Option<f64> {
    match unit
        .trim()
        .to_ascii_lowercase()
        .replace(['µ', 'μ'], "u")
        .as_str()
    {
        "nm" | "nanometer" | "nanometers" => Some(0.001),
        "um" | "micron" | "microns" | "micrometer" | "micrometers" => Some(1.0),
        "mm" | "millimeter" | "millimeters" => Some(1_000.0),
        "cm" | "centimeter" | "centimeters" => Some(10_000.0),
        "in" | "inch" | "inches" => Some(25_400.0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inch_resolution_is_converted_to_micrometres_per_voxel() {
        let spacing = isotropic_spacing_from_resolution(
            1_000_000.0,
            1_000_000.0,
            25_400.0,
            Path::new("test.tif"),
        )
        .unwrap();
        assert!((spacing[0] - 0.0254).abs() < 1e-12);
        assert!((spacing[1] - 0.0254).abs() < 1e-12);
    }

    #[test]
    fn imagej_description_values_and_units_are_parsed() {
        let description = "ImageJ=1.54\nimages=10\nspacing=0.025\nunit=mm\n";
        assert_eq!(description_value(description, "spacing"), Some("0.025"));
        assert_eq!(description_value(description, "unit"), Some("mm"));
        assert_eq!(unit_scale_um("mm"), Some(1_000.0));
        assert_eq!(unit_scale_um("micron"), Some(1.0));
    }

    #[test]
    fn three_dimensional_isotropy_is_checked() {
        let spacing =
            combine_isotropic_spacing(Some([20.0, 20.0]), Some(20.0), Path::new("test.tif"))
                .unwrap();
        assert_eq!(spacing, [20.0, 20.0, 20.0]);

        assert!(
            combine_isotropic_spacing(Some([20.0, 20.0]), Some(21.0), Path::new("test.tif"),)
                .is_err()
        );
    }
}
