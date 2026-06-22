mod edt;
mod local_thickness;
mod metrics;
mod tiff_io;
mod topology;
mod volume;

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use metrics::{analyze, ConnectivityMode, FeatureMode, Metrics};
use tiff_io::{read_binary_tiff, read_spacing};

#[derive(Debug, Parser)]
#[command(
    name = "bone-morphometry",
    version,
    about = "Fast morphometry for 8-bit 3-D binary TIFF stacks"
)]
struct Cli {
    /// Root containing binary TIFFs, optionally grouped into subdirectories.
    #[arg(long, default_value = "images/binary/otsu2D_5_5")]
    binary_dir: PathBuf,

    /// Matching root containing original TIFFs with first-page `spacing=` metadata.
    #[arg(long, default_value = "images/original")]
    original_dir: PathBuf,

    /// Comma-delimited subdirectories to process. Empty means files directly in binary-dir.
    #[arg(long, value_delimiter = ',', default_value = "ALN,OA,ELD")]
    subdirs: Vec<String>,

    /// Output CSV file.
    #[arg(long, default_value = "bonej_otsu2D_5_5.csv")]
    output: PathBuf,

    /// Select the CSV feature schema.
    #[arg(long, value_enum, default_value_t = FeatureMode::Classic)]
    feature_mode: FeatureMode,

    /// Override isotropic voxel size for all files instead of reading `spacing=` metadata.
    #[arg(long)]
    voxel_size: Option<f64>,

    /// A sample is bone when its value is greater than this threshold.
    #[arg(long, default_value_t = 0)]
    threshold: u8,

    /// Reject any sample that is not exactly 0 or 255.
    #[arg(long)]
    strict_binary: bool,

    /// Connectivity definition used only by `--feature-mode classic`.
    #[arg(long, value_enum, default_value_t = ConnectivityMode::Generalized)]
    connectivity_mode: ConnectivityMode,

    /// Number of Rayon worker threads. Omit to use Rayon's default.
    #[arg(long)]
    threads: Option<usize>,

    /// Log an error and continue with remaining files instead of failing immediately.
    #[arg(long)]
    continue_on_error: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Some(threads) = cli.threads {
        if threads == 0 {
            bail!("--threads must be at least 1");
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("initializing Rayon thread pool")?;
    }

    let files = collect_files(&cli)?;
    if files.is_empty() {
        bail!(
            "no .tif or .tiff files were found under {}",
            cli.binary_dir.display()
        );
    }

    let mut writer = csv::Writer::from_path(&cli.output)
        .with_context(|| format!("creating {}", cli.output.display()))?;
    write_header(&mut writer, cli.feature_mode)?;

    let mut failed = 0usize;
    for (relative_dir, binary_path) in files {
        eprintln!("Processing: {}", binary_path.display());
        match process_one(&cli, &relative_dir, &binary_path, &mut writer) {
            Ok(()) => {}
            Err(error) if cli.continue_on_error => {
                failed += 1;
                eprintln!("ERROR: {error:#}");
            }
            Err(error) => return Err(error),
        }
    }
    writer.flush()?;

    if failed > 0 {
        eprintln!("Completed with {failed} failed file(s).");
    }
    Ok(())
}

fn write_header(writer: &mut csv::Writer<std::fs::File>, mode: FeatureMode) -> Result<()> {
    match mode {
        FeatureMode::Classic => writer.write_record([
            "filename",
            "Tb.Th Mean (µm)",
            "Tb.Th Std Dev (µm)",
            "Tb.Th Max (µm)",
            "Tb.Sp Mean (µm)",
            "Tb.Sp Std Dev (µm)",
            "Tb.Sp Max (µm)",
            "BV (micron^3)",
            "TV (micron^3)",
            "BV/TV",
            "Corr. Euler (χ + Δχ)",
            "Connectivity",
            "Connectivity Density (µm⁻³)",
        ])?,
        FeatureMode::Refined => writer.write_record([
            "filename",
            "BV (micron^3)",
            "TV (micron^3)",
            "BV/TV",
            "Tb.Th Median (µm)",
            "Tb.Th P10 (µm)",
            "Tb.Th P90 (µm)",
            "Tb.Th CV",
            "Tb.Sp Median (µm)",
            "Tb.Sp P10 (µm)",
            "Tb.Sp P90 (µm)",
            "Tb.Sp CV",
            "Beta0 Density (µm⁻³)",
            "Beta1 Density (µm⁻³)",
            "Beta2 Density (µm⁻³)",
            "BS (µm²)",
            "BS/BV (µm⁻¹)",
            "BS/TV (µm⁻¹)",
            "Mean Breadth (µm)",
        ])?,
    }
    Ok(())
}

fn collect_files(cli: &Cli) -> Result<Vec<(PathBuf, PathBuf)>> {
    let relative_dirs: Vec<PathBuf> =
        if cli.subdirs.is_empty() || (cli.subdirs.len() == 1 && cli.subdirs[0].is_empty()) {
            vec![PathBuf::new()]
        } else {
            cli.subdirs
                .iter()
                .map(|subdir| PathBuf::from(subdir.as_str()))
                .collect()
        };

    let mut files = Vec::new();
    for relative_dir in relative_dirs {
        let directory = cli.binary_dir.join(&relative_dir);
        let entries = fs::read_dir(&directory)
            .with_context(|| format!("reading directory {}", directory.display()))?;
        for entry in entries {
            let path = entry?.path();
            if path.is_file() && is_tiff(&path) {
                files.push((relative_dir.clone(), path));
            }
        }
    }
    files.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(files)
}

fn is_tiff(path: &Path) -> bool {
    path.extension()
        .and_then(OsStr::to_str)
        .map(|extension| {
            extension.eq_ignore_ascii_case("tif") || extension.eq_ignore_ascii_case("tiff")
        })
        .unwrap_or(false)
}

fn process_one(
    cli: &Cli,
    relative_dir: &Path,
    binary_path: &Path,
    writer: &mut csv::Writer<std::fs::File>,
) -> Result<()> {
    let file_name = binary_path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("{} has no filename", binary_path.display()))?;
    let original_path = cli.original_dir.join(relative_dir).join(file_name);

    let voxel_size = match cli.voxel_size {
        Some(value) => value,
        None => read_spacing(&original_path)?,
    };
    let volume = read_binary_tiff(binary_path, cli.threshold, cli.strict_binary)?;
    let metrics = analyze(&volume, voxel_size, cli.feature_mode, cli.connectivity_mode)
        .with_context(|| format!("analyzing {}", binary_path.display()))?;

    if matches!(cli.feature_mode, FeatureMode::Classic)
        && matches!(cli.connectivity_mode, ConnectivityMode::Bonej)
        && (metrics.topology.beta0 != 1 || metrics.topology.beta2 != 0)
    {
        eprintln!(
            "WARNING: BoneJ mode assumes one 26-connected bone component and no enclosed 6-connected marrow cavities; found beta0={} and beta2={} in {}",
            metrics.topology.beta0,
            metrics.topology.beta2,
            binary_path.display()
        );
    }

    let filename = file_name.to_string_lossy().into_owned();
    let row = match cli.feature_mode {
        FeatureMode::Classic => classic_row(filename, &metrics),
        FeatureMode::Refined => refined_row(filename, &metrics),
    };
    writer.write_record(&row)?;
    Ok(())
}

fn classic_row(filename: String, metrics: &Metrics) -> Vec<String> {
    vec![
        filename,
        format_float(metrics.thickness.mean),
        format_float(metrics.thickness.standard_deviation),
        format_float(metrics.thickness.maximum),
        format_float(metrics.spacing.mean),
        format_float(metrics.spacing.standard_deviation),
        format_float(metrics.spacing.maximum),
        format_float(metrics.bone_volume),
        format_float(metrics.total_volume),
        format_float(metrics.bone_volume_fraction),
        format_float(metrics.topology.corrected_euler),
        format_float(metrics.connectivity),
        format_float(metrics.connectivity_density),
    ]
}

fn refined_row(filename: String, metrics: &Metrics) -> Vec<String> {
    let beta0_density = metrics.topology.beta0 as f64 / metrics.total_volume;
    let beta1_density = metrics.topology.beta1 as f64 / metrics.total_volume;
    let beta2_density = metrics.topology.beta2 as f64 / metrics.total_volume;

    vec![
        filename,
        format_float(metrics.bone_volume),
        format_float(metrics.total_volume),
        format_float(metrics.bone_volume_fraction),
        format_float(metrics.thickness.median),
        format_float(metrics.thickness.percentile_10),
        format_float(metrics.thickness.percentile_90),
        format_float(metrics.thickness.coefficient_of_variation),
        format_float(metrics.spacing.median),
        format_float(metrics.spacing.percentile_10),
        format_float(metrics.spacing.percentile_90),
        format_float(metrics.spacing.coefficient_of_variation),
        format_float(beta0_density),
        format_float(beta1_density),
        format_float(beta2_density),
        format_float(metrics.geometry.bone_surface),
        format_float(metrics.geometry.bone_surface_to_bone_volume),
        format_float(metrics.geometry.bone_surface_to_total_volume),
        format_float(metrics.geometry.mean_breadth),
    ]
}

fn format_float(value: f64) -> String {
    format!("{value:.12}")
}
