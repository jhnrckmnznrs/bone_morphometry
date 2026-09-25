mod anisotropy;
mod edt;
mod ellipsoid_factor;
mod extended;
mod local_thickness;
mod linalg;
mod mechanical_graph;
mod minkowski;
mod orientation_proxy;
mod mesh;
mod metrics;
mod plate_rod;
mod skeleton;
mod sdf_curvature;
mod spacing;
mod tiff_io;
mod topology;
mod volume;

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use anisotropy::DaParameters;
use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use extended::{analyze_extended, ExtendedMetrics};
use metrics::{analyze, FeatureMode, Metrics};
use sdf_curvature::SdfCurvatureParameters;
use skeleton::{graph_stats, skeletonize};
use mechanical_graph::mechanical_graph_stats;
use minkowski::minkowski_w102_stats;
use orientation_proxy::orientation_proxy_stats;
use spacing::SpacingTable;
use tiff_io::{read_binary_tiff, read_isotropic_spacing_um, read_roi_tiff};

use volume::BinaryVolume;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum DescriptorFamily {
    /// Five-variable paper comparator: BV/TV, Tb.Th, Tb.Sp, Conn.D and MIL DA.
    PaperMorphometry,
    /// Eight-variable normalized skeleton descriptor block.
    Skeleton,
    /// Five-variable axial path/transport graph descriptor block.
    MechanicalGraph,
    /// Structural Organization Index proxy.
    SoiProxy,
    /// Three-variable spatial orientation heterogeneity block.
    OrientationField,
    /// W_1^{0,2} surface-normal Minkowski tensor invariants.
    MinkowskiW102,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum DescriptorPreset {
    /// Paper morphometry plus validated skeleton descriptors.
    Core,
    /// All non-topological scalar descriptor blocks used in the final manuscript.
    Manuscript,
}

#[derive(Debug, Parser)]
#[command(
    name = "bone-morphometry",
    version,
    about = "ROI-aware morphometry for 8-bit 3-D binary TIFF stacks"
)]
struct Cli {
    /// Root containing unmasked binary TIFFs, optionally grouped into subdirectories.
    #[arg(long)]
    binary_dir: PathBuf,

    /// Matching root containing true binary ROI masks (0 outside, 255 inside).
    #[arg(long)]
    roi_dir: Option<PathBuf>,

    /// Optional matching root used only for physical-spacing metadata.
    /// By default, calibration is read from each binarized TIFF, as in Fiji.
    #[arg(long, visible_alias = "original-dir")]
    spacing_dir: Option<PathBuf>,

    /// CSV with `filename` and isotropic `spacing_micrometers` columns.
    /// This is the authoritative calibration source when TIFF metadata is absent.
    #[arg(long)]
    spacing_csv: Option<PathBuf>,

    /// Comma-delimited subdirectories to process recursively. Empty means all of binary-dir.
    #[arg(long, value_delimiter = ',')]
    subdirs: Vec<String>,

    /// Output CSV file.
    #[arg(long, default_value = "rust_bonej_roi_morphometry.csv")]
    output: PathBuf,

    /// Select the CSV feature schema.
    #[arg(long, value_enum, default_value_t = FeatureMode::Standard)]
    feature_mode: FeatureMode,

    /// Add surface-curvature, skeleton-graph, and ITS-inspired plate/rod proxy descriptors.
    /// Valid only with --feature-mode standard.
    #[arg(long)]
    advanced: bool,

    /// Descriptor families for the consolidated CSV schema. Comma-delimited.
    #[arg(long, value_enum, value_delimiter = ',', conflicts_with = "preset")]
    descriptors: Vec<DescriptorFamily>,

    /// Named descriptor preset.
    #[arg(long, value_enum, conflicts_with = "descriptors")]
    preset: Option<DescriptorPreset>,

    /// Number of MIL directions for native DA. BoneJ comparison runs should use the same sampling budget.
    #[arg(long, default_value_t = 1024)]
    da_directions: usize,

    /// Target MIL line-length budget in image diagonals per direction.
    #[arg(long, default_value_t = 64)]
    da_lines: usize,

    /// DA sampling increment in voxels. BoneJ requires at least sqrt(3).
    #[arg(long, default_value_t = 3.0_f64.sqrt())]
    da_sampling_increment: f64,

    /// Number of independently rotated DA repetitions.
    #[arg(long, default_value_t = 5)]
    da_repetitions: usize,

    /// Deterministic seed for native DA sampling.
    #[arg(long, default_value_t = 4778124954572866625u64)]
    da_seed: u64,

    /// Radius in voxels for the native EF candidate / plate-rod local covariance window.
    #[arg(long, default_value_t = 4)]
    ef_window_radius: usize,

    /// Gaussian smoothing bandwidth for advanced signed-distance curvature, in micrometres.
    /// The outcome-blind synthetic calibration supports 20--40 µm at 20--25 µm voxel spacing;
    /// 30 µm is the prespecified single-scale compromise default.
    #[arg(long, default_value_t = 30.0)]
    curvature_sigma_um: f64,

    /// Override isotropic voxel size in micrometres for all files.
    #[arg(long)]
    voxel_size: Option<f64>,

    /// A sample is bone when its value is greater than this threshold.
    #[arg(long, default_value_t = 0)]
    threshold: u8,

    /// Reject any sample that is not exactly 0 or 255.
    #[arg(long)]
    strict_binary: bool,

    /// Number of Rayon worker threads. Omit to use Rayon's default.
    #[arg(long)]
    threads: Option<usize>,

    /// Log an error and continue with remaining files instead of failing immediately.
    #[arg(long)]
    continue_on_error: bool,
}

fn main() -> Result<()> {
    if std::env::args().nth(1).as_deref() == Some("list-descriptors") {
        print_descriptor_catalog();
        return Ok(());
    }
    let cli = Cli::parse();

    let calibration_sources = (cli.voxel_size.is_some() as usize)
        + (cli.spacing_dir.is_some() as usize)
        + (cli.spacing_csv.is_some() as usize);
    if calibration_sources > 1 {
        bail!("use only one calibration override: --spacing-csv, --spacing-dir, or --voxel-size");
    }

    if matches!(cli.feature_mode, FeatureMode::Bonej | FeatureMode::Standard)
        && cli.roi_dir.is_none()
    {
        bail!("--feature-mode bonej and standard require --roi-dir");
    }
    if cli.advanced && cli.feature_mode != FeatureMode::Standard {
        bail!("--advanced is valid only with --feature-mode standard");
    }
    let consolidated = resolved_descriptor_families(&cli);
    if consolidated.is_some() && cli.feature_mode != FeatureMode::Standard {
        bail!("--preset/--descriptors require --feature-mode standard");
    }
    if consolidated.is_some() && cli.advanced {
        bail!("do not combine --advanced with --preset/--descriptors");
    }
    if cli.da_directions < 9 || cli.da_lines == 0 || cli.da_repetitions == 0 {
        bail!("DA requires --da-directions >= 9 and positive --da-lines/--da-repetitions");
    }
    if cli.ef_window_radius == 0 {
        bail!("--ef-window-radius must be at least 1");
    }
    if !cli.curvature_sigma_um.is_finite() || cli.curvature_sigma_um <= 0.0 {
        bail!("--curvature-sigma-um must be finite and positive");
    }
    if cli.advanced && !(20.0..=40.0).contains(&cli.curvature_sigma_um) {
        eprintln!(
            "WARNING: --curvature-sigma-um={} lies outside the 20--40 µm outcome-blind synthetic calibration range; treat curvature outputs as experimental sensitivity results.",
            cli.curvature_sigma_um
        );
    }

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

    let spacing_table = cli
        .spacing_csv
        .as_deref()
        .map(SpacingTable::from_csv)
        .transpose()?;
    if let Some(table) = spacing_table.as_ref() {
        for (_, binary_path) in &files {
            let filename = binary_path
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("{} has no filename", binary_path.display()))?;
            table
                .isotropic_spacing_for(filename)
                .with_context(|| format!("validating calibration for {}", binary_path.display()))?;
        }
        eprintln!(
            "Loaded {} specimen spacing row(s) from {} and matched all {} input TIFF(s).",
            table.row_count(),
            cli.spacing_csv.as_ref().unwrap().display(),
            files.len()
        );
    }

    let mut writer = csv::Writer::from_path(&cli.output)
        .with_context(|| format!("creating {}", cli.output.display()))?;
    write_header(&mut writer, cli.feature_mode, cli.advanced, consolidated.as_deref())?;

    let mut failed = 0usize;
    for (relative_dir, binary_path) in files {
        eprintln!("Processing: {}", binary_path.display());
        match process_one(
            &cli,
            spacing_table.as_ref(),
            &relative_dir,
            &binary_path,
            &mut writer,
        ) {
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

fn write_header<W: std::io::Write>(
    writer: &mut csv::Writer<W>,
    mode: FeatureMode,
    advanced: bool,
    consolidated: Option<&[DescriptorFamily]>,
) -> Result<()> {
    if let Some(families) = consolidated {
        writer.write_record(consolidated_header(families))?;
        return Ok(());
    }
    match mode {
        FeatureMode::Bonej => writer.write_record([
            "filename",
            "BV",
            "TV",
            "BV/TV",
            "Euler characteristic",
            "Euler change/correction",
            "Connectivity",
            "Connectivity density",
            "Tb.Th Mean",
            "Tb.Th Std Dev",
            "Tb.Th Max",
            "Tb.Sp Mean",
            "Tb.Sp Std Dev",
            "Tb.Sp Max",
        ])?,
        FeatureMode::Standard => writer.write_record(standard_header(advanced))?,
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
            "Euler χ (ROI-clipped)",
            "Connectivity β1 (ROI-clipped)",
            "Connectivity Density (µm⁻³)",
        ])?,
        FeatureMode::Refined => writer.write_record([
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
            "Euler χ (ROI-clipped)",
            "Connectivity β1 (ROI-clipped)",
            "Connectivity Density (µm⁻³)",
            "BS (µm²)",
            "Mean Breadth (µm; ROI-clipped)",
        ])?,
        FeatureMode::Experimental => writer.write_record([
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
            "Mean Breadth (µm; ROI-clipped)",
        ])?,
    }
    Ok(())
}

fn standard_header(advanced: bool) -> Vec<&'static str> {
    let mut header = vec![
        "filename",
        "BV (µm³)",
        "TV (µm³)",
        "BV/TV",
        "Euler characteristic",
        "Euler change/correction",
        "BoneJ edge correction",
        "Connectivity",
        "Connectivity density (µm⁻³)",
        "Tb.Th Mean (µm)",
        "Tb.Th Std Dev (µm)",
        "Tb.Th Max (µm)",
        "Tb.Th P10 (µm; map-derived)",
        "Tb.Th P25 (µm; map-derived)",
        "Tb.Th Median (µm; map-derived)",
        "Tb.Th P75 (µm; map-derived)",
        "Tb.Th P90 (µm; map-derived)",
        "Tb.Th IQR (µm; map-derived)",
        "Tb.Th CV (map-derived)",
        "Tb.Sp Mean (µm)",
        "Tb.Sp Std Dev (µm)",
        "Tb.Sp Max (µm)",
        "Tb.Sp P10 (µm; map-derived)",
        "Tb.Sp P25 (µm; map-derived)",
        "Tb.Sp Median (µm; map-derived)",
        "Tb.Sp P75 (µm; map-derived)",
        "Tb.Sp P90 (µm; map-derived)",
        "Tb.Sp IQR (µm; map-derived)",
        "Tb.Sp CV (map-derived)",
        "BS mesh (µm²; BoneJ-comparable candidate)",
        "BS/BV mesh (µm⁻¹; BoneJ-comparable candidate)",
        "BS/TV mesh (µm⁻¹; BoneJ-comparable candidate)",
        "DA mean (native MIL candidate)",
        "DA SD (native MIL candidate)",
        "DA CV (native MIL candidate)",
        "DA min across repetitions (native MIL candidate)",
        "DA max across repetitions (native MIL candidate)",
        "DA principal angle to Z mean (deg)",
        "DA radius a mean (vox; shortest)",
        "DA radius b mean (vox)",
        "DA radius c mean (vox; longest)",
        "DA domain side (vox)",
        "DA domain depth (vox)",
        "EF mean (native candidate)",
        "EF SD (native candidate)",
        "EF median (native candidate)",
        "EF plate fraction (<-0.25; native candidate)",
        "EF rod fraction (>0.25; native candidate)",
        "EF intermediate fraction (native candidate)",
        "EF sampled skeleton seeds",
        "Cubical BS legacy (µm²)",
        "Mean Breadth legacy (µm; ROI-clipped)",
    ];
    if advanced {
        header.extend([
            "Curvature SDF sigma (µm)",
            "Curvature image-support margin (µm)",
            "Curvature selected surface area (µm²)",
            "Curvature H area-weighted mean (µm⁻¹)",
            "Curvature H area-weighted SD (µm⁻¹)",
            "Curvature |H| area-weighted mean (µm⁻¹)",
            "Curvature K area-weighted median (µm⁻²)",
            "Curvature K area-weighted IQR (µm⁻²)",
            "Curvature |K| area-weighted P90 (µm⁻²)",
            "Curvature |K| area-weighted P99 (µm⁻²)",
            "Curvature saddle area fraction",
            "Curvature convex area fraction",
            "Curvature concave area fraction",
            "Curvature H²-K<0 area fraction (QC)",
            "Curvature selected vertices",
            "Curvature ROI-excluded vertices",
            "Curvature image-support-excluded vertices",
            "Curvature invalid vertices",
            "Skeleton voxels",
            "Skeleton endpoint voxels",
            "Skeleton slab voxels",
            "Skeleton junction voxels",
            "Skeleton junction clusters",
            "Graph branches",
            "Graph components",
            "Graph cycle rank",
            "Graph mean branch length (µm)",
            "Graph max branch length (µm)",
            "Graph mean branch tortuosity",
            "Graph top-bottom spanning",
            "PR proxy plate-seed fraction",
            "PR proxy rod-seed fraction",
            "PR proxy intermediate-seed fraction",
            "PR proxy rod axial fraction",
            "PR proxy plate-normal axial fraction",
            "PR proxy mean |EF|",
            "PR proxy classified seeds",
        ]);
    }
    header
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
        collect_files_recursive(&directory, &relative_dir, &mut files)?;
    }
    files.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(files)
}

fn collect_files_recursive(
    directory: &Path,
    relative_dir: &Path,
    files: &mut Vec<(PathBuf, PathBuf)>,
) -> Result<()> {
    let entries = fs::read_dir(directory)
        .with_context(|| format!("reading directory {}", directory.display()))?;
    for entry in entries {
        let path = entry?.path();
        if path.is_dir() {
            let name = path
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("{} has no directory name", path.display()))?;
            collect_files_recursive(&path, &relative_dir.join(name), files)?;
        } else if path.is_file() && is_tiff(&path) {
            files.push((relative_dir.to_path_buf(), path));
        }
    }
    Ok(())
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
    spacing_table: Option<&SpacingTable>,
    relative_dir: &Path,
    binary_path: &Path,
    writer: &mut csv::Writer<std::fs::File>,
) -> Result<()> {
    let file_name = binary_path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("{} has no filename", binary_path.display()))?;
    let voxel_spacing = match (cli.voxel_size, spacing_table) {
        (Some(value), None) => [value; 3],
        (None, Some(table)) => table.isotropic_spacing_for(file_name)?,
        (None, None) => {
            let spacing_path = cli
                .spacing_dir
                .as_ref()
                .map(|root| root.join(relative_dir).join(file_name))
                .unwrap_or_else(|| binary_path.to_path_buf());
            read_isotropic_spacing_um(&spacing_path)?
        }
        (Some(_), Some(_)) => unreachable!("calibration-source exclusivity is checked in main"),
    };
    eprintln!(
        "  Voxel spacing: x={:.12} µm, y={:.12} µm, z={:.12} µm",
        voxel_spacing[0], voxel_spacing[1], voxel_spacing[2]
    );

    let require_bonej_binary =
        matches!(cli.feature_mode, FeatureMode::Bonej | FeatureMode::Standard);
    let bone = read_binary_tiff(
        binary_path,
        cli.threshold,
        cli.strict_binary || require_bonej_binary,
    )?;

    let (roi, roi_path) = match &cli.roi_dir {
        Some(roi_dir) => {
            let path = roi_dir.join(relative_dir).join(file_name);

            let roi = read_roi_tiff(&path)
                .with_context(|| format!("reading ROI mask {}", path.display()))?;

            (roi, Some(path))
        }

        None => {
            // No ROI supplied: include every voxel in the image.
            let roi =
                BinaryVolume::new(vec![255u8; bone.len()], bone.width, bone.height, bone.depth)?;

            (roi, None)
        }
    };

    let metrics =
        analyze(&bone, &roi, voxel_spacing, cli.feature_mode).with_context(|| match &roi_path {
            Some(path) => format!(
                "analyzing {} with ROI {}",
                binary_path.display(),
                path.display()
            ),
            None => format!(
                "analyzing {} using the full image as ROI",
                binary_path.display()
            ),
        })?;

    if metrics.roi_boundary_bone_fraction > 0.0 {
        eprintln!(
            "  ROI boundary intersects {:.3}% of included bone voxels. BoneJ compatibility mode applies BoneJ's rectangular-stack edge correction to the ROI-clipped structure; it cannot correct the curved ROI boundary.",
            100.0 * metrics.roi_boundary_bone_fraction,
        );
    }

    let filename = file_name.to_string_lossy().into_owned();
    if let Some(families) = resolved_descriptor_families(cli) {
        let row = consolidated_row(cli, filename, &bone, &roi, voxel_spacing, &metrics, &families)?;
        writer.write_record(&row)?;
        return Ok(());
    }
    let row = match cli.feature_mode {
        FeatureMode::Bonej => bonej_row(filename, &metrics),
        FeatureMode::Standard => {
            let da_parameters = DaParameters {
                directions: cli.da_directions,
                lines: cli.da_lines,
                sampling_increment_voxels: cli.da_sampling_increment,
                repetitions: cli.da_repetitions,
                seed: cli.da_seed,
            };
            let extended = analyze_extended(
                &bone,
                &roi,
                voxel_spacing,
                metrics.bone_volume,
                metrics.total_volume,
                da_parameters,
                cli.ef_window_radius,
                cli.advanced,
                SdfCurvatureParameters {
                    sigma_um: cli.curvature_sigma_um,
                    image_support_sigma: 4.0,
                },
            )?;
            standard_row(filename, &metrics, &extended, cli.advanced)
        }
        FeatureMode::Classic => classic_row(filename, &metrics),
        FeatureMode::Refined => refined_row(filename, &metrics),
        FeatureMode::Experimental => experimental_row(filename, &metrics),
    };
    writer.write_record(&row)?;
    Ok(())
}

fn bonej_row(filename: String, metrics: &Metrics) -> Vec<String> {
    vec![
        filename,
        format_float(metrics.bone_volume),
        format_float(metrics.total_volume),
        format_float(metrics.bone_volume_fraction),
        format_float(metrics.topology.raw_euler as f64),
        format_float(metrics.bonej_connectivity.delta_chi),
        format_float(metrics.bonej_connectivity.connectivity),
        format_float(metrics.bonej_connectivity_density),
        format_float(metrics.thickness.mean),
        format_float(metrics.thickness.standard_deviation),
        format_float(metrics.thickness.maximum),
        format_float(metrics.spacing.mean),
        format_float(metrics.spacing.standard_deviation),
        format_float(metrics.spacing.maximum),
    ]
}

fn standard_row(
    filename: String,
    metrics: &Metrics,
    extended: &ExtendedMetrics,
    advanced: bool,
) -> Vec<String> {
    let t = metrics.thickness;
    let sp = metrics.spacing;
    let da = extended.da;
    let ef = extended.ef_candidate;
    let mut row = vec![
        filename,
        format_float(metrics.bone_volume),
        format_float(metrics.total_volume),
        format_float(metrics.bone_volume_fraction),
        format_float(metrics.topology.raw_euler as f64),
        format_float(metrics.bonej_connectivity.delta_chi),
        format_float(metrics.bonej_connectivity.edge_correction),
        format_float(metrics.bonej_connectivity.connectivity),
        format_float(metrics.bonej_connectivity_density),
        format_float(t.mean),
        format_float(t.standard_deviation),
        format_float(t.maximum),
        format_float(t.percentile_10),
        format_float(t.percentile_25),
        format_float(t.median),
        format_float(t.percentile_75),
        format_float(t.percentile_90),
        format_float(t.interquartile_range),
        format_float(t.coefficient_of_variation),
        format_float(sp.mean),
        format_float(sp.standard_deviation),
        format_float(sp.maximum),
        format_float(sp.percentile_10),
        format_float(sp.percentile_25),
        format_float(sp.median),
        format_float(sp.percentile_75),
        format_float(sp.percentile_90),
        format_float(sp.interquartile_range),
        format_float(sp.coefficient_of_variation),
        format_float(extended.surface.mesh_surface_area),
        format_float(extended.surface.mesh_surface_to_bone_volume),
        format_float(extended.surface.mesh_surface_to_total_volume),
        format_float(da.mean),
        format_float(da.sd),
        format_float(da.cv),
        format_float(da.min),
        format_float(da.max),
        format_float(da.principal_angle_to_z_mean_deg),
        format_float(da.radius_a_mean),
        format_float(da.radius_b_mean),
        format_float(da.radius_c_mean),
        da.domain_side_voxels.to_string(),
        da.domain_depth_voxels.to_string(),
        format_float(ef.mean),
        format_float(ef.sd),
        format_float(ef.median),
        format_float(ef.plate_fraction),
        format_float(ef.rod_fraction),
        format_float(ef.intermediate_fraction),
        ef.sampled_seeds.to_string(),
        format_float(metrics.geometry.bone_surface),
        format_float(metrics.geometry.mean_breadth),
    ];

    if advanced {
        let a = extended
            .advanced
            .as_ref()
            .expect("advanced metrics must be present when --advanced is enabled");
        let c = a.curvature;
        let g = a.graph;
        let pr = a.plate_rod_proxy;
        row.extend([
            format_float(c.sigma_um),
            format_float(c.support_margin_um),
            format_float(c.selected_surface_area_um2),
            format_float(c.mean_curvature_mean),
            format_float(c.mean_curvature_sd),
            format_float(c.mean_curvature_abs_mean),
            format_float(c.gaussian_curvature_median),
            format_float(c.gaussian_curvature_iqr),
            format_float(c.gaussian_curvature_abs_q90),
            format_float(c.gaussian_curvature_abs_q99),
            format_float(c.saddle_fraction),
            format_float(c.convex_fraction),
            format_float(c.concave_fraction),
            format_float(c.discriminant_negative_fraction),
            c.selected_vertices.to_string(),
            c.roi_excluded_vertices.to_string(),
            c.support_excluded_vertices.to_string(),
            c.invalid_vertices.to_string(),
            g.skeleton_voxels.to_string(),
            g.endpoint_voxels.to_string(),
            g.slab_voxels.to_string(),
            g.junction_voxels.to_string(),
            g.junction_clusters.to_string(),
            g.branches.to_string(),
            g.graph_components.to_string(),
            g.cycle_rank.to_string(),
            format_float(g.mean_branch_length),
            format_float(g.max_branch_length),
            format_float(g.mean_branch_tortuosity),
            g.top_bottom_spanning.to_string(),
            format_float(pr.plate_seed_fraction),
            format_float(pr.rod_seed_fraction),
            format_float(pr.intermediate_seed_fraction),
            format_float(pr.rod_axial_fraction),
            format_float(pr.plate_normal_axial_fraction),
            format_float(pr.mean_abs_ef),
            pr.classified_seeds.to_string(),
        ]);
    }
    row
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
        metrics.topology.raw_euler.to_string(),
        format_float(metrics.connectivity),
        format_float(metrics.connectivity_density),
    ]
}

fn refined_row(filename: String, metrics: &Metrics) -> Vec<String> {
    let mut row = classic_row(filename, metrics);
    row.push(format_float(metrics.geometry.bone_surface));
    row.push(format_float(metrics.geometry.mean_breadth));
    row
}

fn experimental_row(filename: String, metrics: &Metrics) -> Vec<String> {
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


fn resolved_descriptor_families(cli: &Cli) -> Option<Vec<DescriptorFamily>> {
    if !cli.descriptors.is_empty() {
        let mut out = Vec::new();
        for &family in &cli.descriptors {
            if !out.contains(&family) { out.push(family); }
        }
        return Some(out);
    }
    cli.preset.map(|preset| match preset {
        DescriptorPreset::Core => vec![DescriptorFamily::PaperMorphometry, DescriptorFamily::Skeleton],
        DescriptorPreset::Manuscript => vec![
            DescriptorFamily::PaperMorphometry,
            DescriptorFamily::Skeleton,
            DescriptorFamily::MechanicalGraph,
            DescriptorFamily::SoiProxy,
            DescriptorFamily::OrientationField,
            DescriptorFamily::MinkowskiW102,
        ],
    })
}

fn consolidated_header(families: &[DescriptorFamily]) -> Vec<&'static str> {
    let mut h = vec!["filename"];
    for family in families {
        match family {
            DescriptorFamily::PaperMorphometry => h.extend([
                "paper_morphometry.BVTV",
                "paper_morphometry.TbTh_mean_um",
                "paper_morphometry.TbSp_mean_um",
                "paper_morphometry.ConnD_um_inv3",
                "paper_morphometry.DA_mean",
            ]),
            DescriptorFamily::Skeleton => h.extend([
                "skeleton.length_density_mm_per_mm3",
                "skeleton.branch_density_per_mm3",
                "skeleton.junction_cluster_density_per_mm3",
                "skeleton.endpoint_density_per_mm3",
                "skeleton.cycle_density_per_mm3",
                "skeleton.graph_component_density_per_mm3",
                "skeleton.mean_branch_length_um",
                "skeleton.mean_branch_tortuosity",
            ]),
            DescriptorFamily::MechanicalGraph => h.extend([
                "mechanical_graph.axial_shortest_path_tortuosity",
                "mechanical_graph.axial_edge_connectivity",
                "mechanical_graph.Cz_path_length_fraction",
                "mechanical_graph.axial_normalized_conductance",
                "mechanical_graph.axial_dissipation_gini",
            ]),
            DescriptorFamily::SoiProxy => h.extend([
                "soi_proxy.SOI_proxy",
                "soi_proxy.pO",
                "soi_proxy.rO",
                "soi_proxy.prO",
                "soi_proxy.n_plate",
                "soi_proxy.n_rod",
                "soi_proxy.n_classified",
                "soi_proxy.n_grid_representatives",
                "soi_proxy.t50_um",
            ]),
            DescriptorFamily::OrientationField => h.extend([
                "orientation_field.plate_local_misorientation_deg",
                "orientation_field.rod_local_misorientation_deg",
                "orientation_field.plate_rod_local_orthogonality_deviation_deg",
            ]),
            DescriptorFamily::MinkowskiW102 => h.extend([
                "minkowski_w102.W102_DA",
                "minkowski_w102.W102_mid_over_max",
                "minkowski_w102.selected_triangles",
                "minkowski_w102.selected_surface_area_um2",
            ]),
        }
    }
    h
}

#[allow(clippy::too_many_arguments)]
fn consolidated_row(
    cli: &Cli,
    filename: String,
    bone: &BinaryVolume,
    roi: &BinaryVolume,
    spacing: [f64; 3],
    metrics: &Metrics,
    families: &[DescriptorFamily],
) -> Result<Vec<String>> {
    let needs_skeleton = families.iter().any(|f| matches!(f,
        DescriptorFamily::Skeleton | DescriptorFamily::MechanicalGraph | DescriptorFamily::SoiProxy | DescriptorFamily::OrientationField));
    let needs_orientation = families.iter().any(|f| matches!(f, DescriptorFamily::SoiProxy | DescriptorFamily::OrientationField));
    let needs_masked_bone = needs_skeleton || families.contains(&DescriptorFamily::PaperMorphometry);
    let masked_bone = needs_masked_bone.then(|| bone.phase_inside(roi, true)).transpose()?;
    let skeleton = masked_bone.as_ref().filter(|_| needs_skeleton).map(skeletonize);

    let da_parameters = DaParameters {
        directions: cli.da_directions,
        lines: cli.da_lines,
        sampling_increment_voxels: cli.da_sampling_increment,
        repetitions: cli.da_repetitions,
        seed: cli.da_seed,
    };
    let da = if families.contains(&DescriptorFamily::PaperMorphometry) {
        Some(anisotropy::degree_of_anisotropy(masked_bone.as_ref().expect("paper morphometry bone dependency"), roi, da_parameters)?)
    } else { None };
    let graph = if families.contains(&DescriptorFamily::Skeleton) {
        Some(graph_stats(skeleton.as_ref().expect("skeleton dependency"), spacing))
    } else { None };
    let mech = if families.contains(&DescriptorFamily::MechanicalGraph) {
        Some(mechanical_graph_stats(skeleton.as_ref().expect("skeleton dependency"), spacing)?)
    } else { None };
    let orientation = if needs_orientation {
        Some(orientation_proxy_stats(masked_bone.as_ref().expect("bone dependency"), skeleton.as_ref().expect("skeleton dependency"), spacing)?)
    } else { None };
    let w102 = if families.contains(&DescriptorFamily::MinkowskiW102) {
        Some(minkowski_w102_stats(bone, roi, spacing)?)
    } else { None };

    let mut row=vec![filename];
    for family in families {
        match family {
            DescriptorFamily::PaperMorphometry => {
                let d=da.as_ref().unwrap();
                row.extend([
                    format_float(metrics.bone_volume_fraction),
                    format_float(metrics.thickness.mean),
                    format_float(metrics.spacing.mean),
                    format_float(metrics.bonej_connectivity_density),
                    format_float(d.mean),
                ]);
            }
            DescriptorFamily::Skeleton => {
                let g=graph.unwrap();
                let tv_mm3=metrics.total_volume*1e-9;
                let length_mm=(g.mean_branch_length * g.branches as f64)/1000.0;
                row.extend([
                    format_float(length_mm/tv_mm3),
                    format_float(g.branches as f64/tv_mm3),
                    format_float(g.junction_clusters as f64/tv_mm3),
                    format_float(g.endpoint_voxels as f64/tv_mm3),
                    format_float(g.cycle_rank as f64/tv_mm3),
                    format_float(g.graph_components as f64/tv_mm3),
                    format_float(g.mean_branch_length),
                    format_float(g.mean_branch_tortuosity),
                ]);
            }
            DescriptorFamily::MechanicalGraph => {
                let g=mech.unwrap();
                row.extend([
                    format_float(g.axial_shortest_path_tortuosity),
                    format_float(g.axial_edge_connectivity),
                    format_float(g.cz_path_length_fraction),
                    format_float(g.axial_normalized_conductance),
                    format_float(g.axial_dissipation_gini),
                ]);
            }
            DescriptorFamily::SoiProxy => {
                let s=orientation.as_ref().unwrap().soi;
                row.extend([
                    format_float(s.soi_proxy),format_float(s.plate_organization),format_float(s.rod_organization),format_float(s.plate_rod_overlap),
                    s.plate_samples.to_string(),s.rod_samples.to_string(),s.classified_samples.to_string(),s.grid_representatives.to_string(),format_float(s.median_local_thickness_um),
                ]);
            }
            DescriptorFamily::OrientationField => {
                let f=orientation.as_ref().unwrap().field;
                row.extend([format_float(f.plate_local_misorientation_deg),format_float(f.rod_local_misorientation_deg),format_float(f.plate_rod_local_orthogonality_deviation_deg)]);
            }
            DescriptorFamily::MinkowskiW102 => {
                let m=w102.unwrap();
                row.extend([format_float(m.degree_of_anisotropy),format_float(m.mid_over_max),m.selected_triangles.to_string(),format_float(m.selected_surface_area_um2)]);
            }
        }
    }
    Ok(row)
}

fn print_descriptor_catalog() {
    println!("bone-morphometry descriptor families\n");
    println!("paper-morphometry  BV/TV, Tb.Th, Tb.Sp, Conn.D, MIL DA");
    println!("skeleton          8 normalized skeleton descriptors");
    println!("mechanical-graph  5 axial path/transport graph descriptors");
    println!("soi-proxy         structural-organization proxy and QC fields");
    println!("orientation-field 3 local orientation-heterogeneity descriptors");
    println!("minkowski-w102    W_1^(0,2) surface-normal tensor invariants");
    println!("\nPreset: --preset manuscript enables all six families above.");
    println!("Persistent homology is intentionally implemented outside this crate.");
}

fn format_float(value: f64) -> String {
    // Rust's Display representation is the shortest decimal that round-trips
    // to the same f64. Fixed decimal places lose significant digits for small
    // quantities such as connectivity density (~1e-9).
    value.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{DistributionStats, GeometryMetrics};
    use crate::topology::{BonejConnectivity, Topology};

    fn sample_stats() -> DistributionStats {
        DistributionStats {
            mean: 1.0,
            standard_deviation: 2.0,
            maximum: 3.0,
            median: 4.0,
            percentile_10: 5.0,
            percentile_25: 5.5,
            percentile_75: 5.8,
            percentile_90: 6.0,
            interquartile_range: 0.3,
            coefficient_of_variation: 7.0,
        }
    }

    fn sample_metrics() -> Metrics {
        Metrics {
            thickness: sample_stats(),
            spacing: sample_stats(),
            bone_volume: 8.0,
            total_volume: 9.0,
            bone_volume_fraction: 10.0,
            topology: Topology {
                raw_euler: 11,
                beta0: 12,
                beta1: 13,
                beta2: 14,
            },
            bonej_connectivity: BonejConnectivity {
                edge_correction: 15.0,
                delta_chi: 16.0,
                connectivity: 17.0,
            },
            bonej_connectivity_density: 18.0,
            connectivity: 19.0,
            connectivity_density: 20.0,
            roi_boundary_bone_fraction: 0.25,
            geometry: GeometryMetrics {
                bone_surface: 21.0,
                bone_surface_to_bone_volume: 22.0,
                bone_surface_to_total_volume: 23.0,
                mean_breadth: 24.0,
            },
        }
    }

    fn header_for(mode: FeatureMode) -> csv::StringRecord {
        let mut writer = csv::Writer::from_writer(Vec::new());
        write_header(&mut writer, mode, false, None).unwrap();
        let bytes = writer.into_inner().unwrap();
        let mut reader = csv::Reader::from_reader(bytes.as_slice());
        reader.headers().unwrap().clone()
    }

    #[test]
    fn bonej_schema_matches_the_batch_measurement_columns() {
        let header = header_for(FeatureMode::Bonej);
        assert_eq!(header.len(), 14);
        assert_eq!(&header[1], "BV");
        assert_eq!(&header[5], "Euler change/correction");
        assert_eq!(&header[13], "Tb.Sp Max");

        let row = bonej_row("sample.tif".to_owned(), &sample_metrics());
        assert_eq!(row.len(), 14);
        assert_eq!(row[5], "16");
        assert_eq!(row[7], "18");
    }

    #[test]
    fn standard_schema_is_extended_without_changing_bonej_schema() {
        let header = standard_header(false);
        assert!(header.iter().any(|field| field.starts_with("BS mesh")));
        assert!(header.iter().any(|field| field.starts_with("DA mean")));
        assert!(header.iter().any(|field| field == &"BoneJ edge correction"));
        assert!(header.iter().any(|field| field.starts_with("DA min")));
        assert!(header.iter().any(|field| field == &"DA domain side (vox)"));
        assert!(header.iter().any(|field| field.starts_with("EF mean")));
        let advanced_header = standard_header(true);
        assert!(advanced_header.len() > header.len());
        assert!(advanced_header
            .iter()
            .any(|field| field == &"Graph cycle rank"));
        assert!(advanced_header
            .iter()
            .any(|field| field.starts_with("PR proxy")));
    }

    #[test]
    fn classic_schema_retains_thirteen_fields() {
        let header = header_for(FeatureMode::Classic);
        assert_eq!(header.len(), 13);
        assert_eq!(&header[12], "Connectivity Density (µm⁻³)");
        assert!(!header.iter().any(|field| field == "BS (µm²)"));
        assert_eq!(
            classic_row("sample.tif".to_owned(), &sample_metrics()).len(),
            13
        );
    }

    #[test]
    fn refined_schema_adds_only_bone_surface_and_mean_breadth() {
        let header = header_for(FeatureMode::Refined);
        assert_eq!(header.len(), 15);
        assert_eq!(&header[13], "BS (µm²)");
        assert_eq!(&header[14], "Mean Breadth (µm; ROI-clipped)");

        let row = refined_row("sample.tif".to_owned(), &sample_metrics());
        assert_eq!(row.len(), 15);
        assert_eq!(row[13], "21");
        assert_eq!(row[14], "24");
    }

    #[test]
    fn experimental_schema_keeps_the_previous_refined_features() {
        let header = header_for(FeatureMode::Experimental);
        assert_eq!(header.len(), 19);
        assert_eq!(&header[4], "Tb.Th Median (µm)");
        assert_eq!(&header[14], "Beta2 Density (µm⁻³)");
        assert_eq!(&header[18], "Mean Breadth (µm; ROI-clipped)");
        assert_eq!(
            experimental_row("sample.tif".to_owned(), &sample_metrics()).len(),
            19
        );
    }

    #[test]
    fn manuscript_preset_has_one_unique_column_per_descriptor() {
        let families = [
            DescriptorFamily::PaperMorphometry,
            DescriptorFamily::Skeleton,
            DescriptorFamily::MechanicalGraph,
            DescriptorFamily::SoiProxy,
            DescriptorFamily::OrientationField,
            DescriptorFamily::MinkowskiW102,
        ];
        let header = consolidated_header(&families);
        assert_eq!(header.len(), 35);
        let unique: std::collections::HashSet<_> = header.iter().copied().collect();
        assert_eq!(unique.len(), header.len());
        assert!(header.contains(&"mechanical_graph.Cz_path_length_fraction"));
        assert!(header.contains(&"soi_proxy.SOI_proxy"));
        assert!(header.contains(&"orientation_field.plate_local_misorientation_deg"));
        assert!(header.contains(&"minkowski_w102.W102_DA"));
    }

    #[test]
    fn csv_float_format_round_trips_small_density_values() {
        let value = 2.698_796_974_583_921e-9_f64;
        let encoded = format_float(value);
        let decoded: f64 = encoded.parse().unwrap();
        assert_eq!(decoded.to_bits(), value.to_bits());
    }
}
