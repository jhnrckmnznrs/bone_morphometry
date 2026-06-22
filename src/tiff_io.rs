use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{bail, Context, Result};
use tiff::decoder::{Decoder, DecodingResult};
use tiff::tags::Tag;
use tiff::ColorType;

use crate::volume::BinaryVolume;

pub fn read_binary_tiff(path: &Path, threshold: u8, strict_binary: bool) -> Result<BinaryVolume> {
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

        if strict_binary {
            if let Some((offset, value)) = page
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| *value != 0 && *value != 255)
            {
                bail!(
                    "{} is not strictly binary: page {}, sample {} has value {} (expected 0 or 255)",
                    path.display(),
                    depth + 1,
                    offset,
                    value
                );
            }
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

pub fn read_spacing(path: &Path) -> Result<f64> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut decoder = Decoder::new(BufReader::new(file))
        .with_context(|| format!("initializing TIFF decoder for {}", path.display()))?;
    let description = decoder
        .get_tag_ascii_string(Tag::ImageDescription)
        .with_context(|| format!("reading ImageDescription from {}", path.display()))?;

    for raw_line in description.lines() {
        let line = raw_line.trim_matches('\0').trim();
        if let Some(value) = line.strip_prefix("spacing=") {
            let spacing: f64 = value
                .trim()
                .parse()
                .with_context(|| format!("parsing spacing metadata in {}", path.display()))?;
            if !spacing.is_finite() || spacing <= 0.0 {
                bail!(
                    "spacing in {} must be finite and positive; got {}",
                    path.display(),
                    spacing
                );
            }
            return Ok(spacing);
        }
    }

    bail!(
        "'spacing=' was not found in the first-page ImageDescription of {}",
        path.display()
    )
}
