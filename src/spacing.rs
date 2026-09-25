use std::collections::HashMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{bail, Context, Result};

const FILENAME_COLUMN: &str = "filename";
const SPACING_UM_COLUMN: &str = "spacing_micrometers";

/// Specimen-specific isotropic voxel spacings indexed by TIFF basename.
#[derive(Debug)]
pub struct SpacingTable {
    spacing_um_by_filename: HashMap<String, f64>,
}

impl SpacingTable {
    /// Load the authoritative isotropic calibration table.
    ///
    /// The required columns are `filename` and `spacing_micrometers`. Other
    /// columns are ignored deliberately: in particular, the dataset's
    /// dimensionless/legacy `spacing` column is not used for physical units.
    pub fn from_csv(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        Self::from_reader(file, &path.display().to_string())
    }

    pub fn isotropic_spacing_for(&self, filename: &OsStr) -> Result<[f64; 3]> {
        let filename = filename
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("input TIFF filename is not valid UTF-8"))?;
        let key = normalize_filename(filename)?;
        let spacing_um = self.spacing_um_by_filename.get(&key).copied().ok_or_else(|| {
            anyhow::anyhow!(
                "no row for '{}' was found in the spacing CSV (matching uses the TIFF basename, case-insensitively)",
                filename
            )
        })?;
        Ok([spacing_um; 3])
    }

    pub fn row_count(&self) -> usize {
        self.spacing_um_by_filename.len()
    }

    fn from_reader<R: Read>(reader: R, source: &str) -> Result<Self> {
        let mut reader = csv::ReaderBuilder::new()
            .trim(csv::Trim::All)
            .from_reader(reader);
        let headers = reader
            .headers()
            .with_context(|| format!("reading the header of {source}"))?
            .clone();

        let filename_index = required_column_index(&headers, FILENAME_COLUMN, source)?;
        let spacing_index = required_column_index(&headers, SPACING_UM_COLUMN, source)?;
        let mut spacing_um_by_filename = HashMap::new();

        for (record_index, record) in reader.records().enumerate() {
            let line_number = record_index + 2;
            let record =
                record.with_context(|| format!("reading line {line_number} of {source}"))?;
            let raw_filename = record.get(filename_index).unwrap_or_default();
            let key = normalize_filename(raw_filename)
                .with_context(|| format!("invalid filename on line {line_number} of {source}"))?;
            let raw_spacing = record.get(spacing_index).unwrap_or_default();
            let spacing_um: f64 = raw_spacing.parse().with_context(|| {
                format!(
                    "parsing '{}' as {} on line {} of {}",
                    raw_spacing, SPACING_UM_COLUMN, line_number, source
                )
            })?;
            if !spacing_um.is_finite() || spacing_um <= 0.0 {
                bail!(
                    "{} on line {} of {} must be finite and positive; got {}",
                    SPACING_UM_COLUMN,
                    line_number,
                    source,
                    spacing_um
                );
            }
            if spacing_um_by_filename
                .insert(key.clone(), spacing_um)
                .is_some()
            {
                bail!(
                    "duplicate filename '{}' on line {} of {}",
                    raw_filename,
                    line_number,
                    source
                );
            }
        }

        if spacing_um_by_filename.is_empty() {
            bail!("{source} contains no specimen spacing rows");
        }

        Ok(Self {
            spacing_um_by_filename,
        })
    }
}

fn required_column_index(
    headers: &csv::StringRecord,
    required: &str,
    source: &str,
) -> Result<usize> {
    headers
        .iter()
        .position(|header| header.trim_start_matches('\u{feff}') == required)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{} is missing the required '{}' column; found: {}",
                source,
                required,
                headers.iter().collect::<Vec<_>>().join(", ")
            )
        })
}

fn normalize_filename(raw: &str) -> Result<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("filename is empty");
    }
    let basename = Path::new(trimmed)
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| anyhow::anyhow!("'{}' has no valid UTF-8 basename", trimmed))?;
    Ok(basename.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_the_dataset_schema_and_uses_micrometres() {
        let csv = concat!(
            "filename,spacing,spacing_micrometers\n",
            "CX09T1.tif,0.000984251968503937,25\n",
            "HA01T3.tif,0.0009685038802223634,24.59999855764803\n",
        );
        let table = SpacingTable::from_reader(Cursor::new(csv), "test.csv").unwrap();
        assert_eq!(
            table
                .isotropic_spacing_for(OsStr::new("CX09T1.tif"))
                .unwrap(),
            [25.0; 3]
        );
        assert_eq!(
            table
                .isotropic_spacing_for(OsStr::new("ha01t3.TIF"))
                .unwrap(),
            [24.59999855764803; 3]
        );
    }

    #[test]
    fn bundled_dataset_table_contains_all_24_specimens() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("spacing.csv");
        let table = SpacingTable::from_csv(&path).unwrap();
        assert_eq!(table.row_count(), 24);
        assert_eq!(
            table
                .isotropic_spacing_for(OsStr::new("CX09T1.tif"))
                .unwrap(),
            [25.0; 3]
        );
    }

    #[test]
    fn missing_sample_is_an_error() {
        let csv = "filename,spacing_micrometers\nCX09T1.tif,25\n";
        let table = SpacingTable::from_reader(Cursor::new(csv), "test.csv").unwrap();
        assert!(table
            .isotropic_spacing_for(OsStr::new("missing.tif"))
            .is_err());
    }

    #[test]
    fn duplicate_filenames_are_rejected_case_insensitively() {
        let csv = concat!(
            "filename,spacing_micrometers\n",
            "CX09T1.tif,25\n",
            "cx09t1.TIF,25\n",
        );
        assert!(SpacingTable::from_reader(Cursor::new(csv), "test.csv").is_err());
    }

    #[test]
    fn ambiguous_legacy_spacing_column_is_not_accepted_alone() {
        let csv = "filename,spacing\nCX09T1.tif,0.000984251968503937\n";
        assert!(SpacingTable::from_reader(Cursor::new(csv), "test.csv").is_err());
    }
}
