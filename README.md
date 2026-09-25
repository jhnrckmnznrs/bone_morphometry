# bone-morphometry

`bone-morphometry` is a standalone Rust command-line program for ROI-aware 3-D
trabecular-bone morphometry from binary TIFF volumes. It provides conventional
morphometry together with the non-topological scalar descriptor families used
in the associated trabecular-bone strength analyses.

This archive is a cleaned packaging of **version 0.7.0-rc.3**. The numerical
algorithms are unchanged from that release candidate; the package organization,
documentation, validation-file names, and user-facing provenance wording have
been simplified.

## Package layout

```text
bone_morphometry_v0.7.0-rc.3_standalone/
├── Cargo.toml
├── Cargo.lock
├── LICENSE
├── README.md
├── examples/
│   └── spacing.csv
├── scripts/
│   └── check.sh
├── src/
│   └── ... Rust implementation ...
└── validation/
    ├── compare_reference_outputs.py
    ├── bonej/
    │   ├── bonej_batch_morphometry.js
    │   └── bonej_anisotropy_validation.js
    └── reference/
        ├── skeleton.csv
        ├── mechanical_graph.csv
        ├── soi_proxy.csv
        ├── orientation_field.csv
        └── minkowski_w102.csv
```

There is intentionally only one `README.md`. The Rust source is the production
implementation; validation assets are kept only where they support an executable
independent or reference comparison.

## Requirements

- a recent stable Rust toolchain (`cargo`, `rustc`, and `rustfmt`);
- Python 3 only if the bundled reference-output comparison is used;
- Fiji/BoneJ only if the optional independent BoneJ validation scripts are run.

No Python package is required for normal morphometry extraction.

## Build

From the package root:

```bash
cargo build --release --locked
```

The executable is then

```text
target/release/bone-morphometry
```

For formatting, tests, Clippy, and a release build in one command:

```bash
bash scripts/check.sh
```

## Inputs

The program expects binary 3-D TIFF volumes. For ROI-aware analyses, provide a
matching binary ROI mask for each image. Binary image and ROI files are matched
recursively by relative path and filename.

With `--strict-binary`, bone images must contain only 0 and 255. ROI masks may
use 0/1 or 0/255.

The ROI is the analysis domain:

- bone volume is bone inside the ROI;
- total volume is the physical volume of the ROI;
- trabecular thickness is evaluated on ROI-clipped bone;
- trabecular separation uses marrow inside the ROI while excluding exterior
  voxels from the separation phase;
- topology is evaluated on ROI-clipped bone.

### Voxel spacing

Calibration can come from one of three sources:

1. TIFF metadata (default);
2. `--spacing-csv` with columns `filename` and `spacing_micrometers`;
3. a common value supplied with `--voxel-size`.

An example CSV is provided in `examples/spacing.csv`.

## Descriptor families used by the manuscript

Run

```bash
./target/release/bone-morphometry list-descriptors
```

to print the available consolidated descriptor families.

| CLI family | Main outputs |
|---|---|
| `paper-morphometry` | BV/TV, Tb.Th, Tb.Sp, connectivity density, MIL degree of anisotropy |
| `skeleton` | skeleton length, branch, junction, endpoint, cycle and component densities; mean branch length and tortuosity |
| `mechanical-graph` | axial shortest-path tortuosity, axial edge connectivity, axial path fraction, normalized conductance, dissipation Gini |
| `soi-proxy` | structural-organization proxy, component terms and sampling QC |
| `orientation-field` | plate and rod local misorientation and plate-rod orthogonality deviation |
| `minkowski-w102` | two dimensionless invariants of the surface-normal `W_1^(0,2)` tensor plus support QC |

To compute every non-topological scalar family used in the manuscript:

```bash
./target/release/bone-morphometry \
  --binary-dir /path/to/binary \
  --roi-dir /path/to/roi \
  --spacing-csv /path/to/spacing.csv \
  --preset manuscript \
  --output manuscript_descriptors.csv \
  --strict-binary
```

Individual families can be requested instead:

```bash
./target/release/bone-morphometry \
  --binary-dir /path/to/binary \
  --roi-dir /path/to/roi \
  --spacing-csv /path/to/spacing.csv \
  --descriptors paper-morphometry,skeleton,mechanical-graph \
  --output selected_descriptors.csv \
  --strict-binary
```

Persistent homology is intentionally not implemented in this crate; it belongs
to the separate topological-analysis workflow.

## Conventional morphometry

The default `standard` schema provides ROI-aware conventional measurements,
including BV, TV, BV/TV, topology/connectivity, trabecular thickness and
separation distributions, marching-cubes surface measures, MIL anisotropy, and
the native ellipsoid-factor descriptor.

```bash
./target/release/bone-morphometry \
  --feature-mode standard \
  --binary-dir /path/to/binary \
  --roi-dir /path/to/roi \
  --spacing-csv /path/to/spacing.csv \
  --output morphometry.csv \
  --strict-binary
```

`--feature-mode bonej` emits the 13-column compatibility schema used for direct
comparison with the paired Fiji/BoneJ script.

The program also retains additional research schemas exposed by
`--feature-mode`; they are not required for the manuscript preset.

## Advanced curvature and graph outputs

With `--feature-mode standard`, `--advanced` additionally computes the
smoothed signed-distance curvature summaries, skeleton graph measurements, and
the plate/rod proxy block:

```bash
./target/release/bone-morphometry \
  --feature-mode standard \
  --advanced \
  --binary-dir /path/to/binary \
  --roi-dir /path/to/roi \
  --spacing-csv /path/to/spacing.csv \
  --output advanced.csv \
  --strict-binary
```

The primary curvature bandwidth is 30 micrometres. Values of 20 and 40
micrometres are the prespecified sensitivity settings. These measurements are
more computationally demanding than the conventional block.

## Main implementation modules

The source is separated by mathematical or computational responsibility:

- `main.rs`: CLI, file matching, descriptor selection, and CSV output;
- `metrics.rs`: conventional morphometry and distribution summaries;
- `local_thickness.rs`: local-thickness computation;
- `topology.rs`: cubical topology and connectivity quantities;
- `anisotropy.rs`: MIL degree of anisotropy;
- `sdf_curvature.rs`: smoothed signed-distance curvature;
- `skeleton.rs`: topology-preserving skeleton and compressed graph;
- `mechanical_graph.rs`: mechanically oriented graph descriptors;
- `orientation_proxy.rs`: SOI and orientation-field descriptors;
- `minkowski.rs`: `W_1^(0,2)` surface-normal tensor descriptors;
- `mesh.rs`: marching-cubes surface construction;
- `edt.rs`, `linalg.rs`, `spacing.rs`, `tiff_io.rs`, `volume.rs`: numerical and I/O support.

This organization is intended to make the implementation reviewable without
requiring any earlier development package.

## Validation

The package contains executable validation material for source checks and
independent/reference comparisons.

### Rust source checks

```bash
bash scripts/check.sh
```

This runs:

```text
cargo fmt --all -- --check
cargo test --all-targets --all-features --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo build --release --locked
```

### Manuscript reference-output comparison

After generating `manuscript_descriptors.csv` for the 24 blinded reference
specimens, run:

```bash
python3 validation/compare_reference_outputs.py manuscript_descriptors.csv
```

or combine it with the normal check script:

```bash
REFERENCE_CSV=manuscript_descriptors.csv bash scripts/check.sh
```

The bundled comparison checks the skeleton, mechanical-graph, SOI,
orientation-field, and `W_1^(0,2)` descriptor families. The first four use
strict floating-point tolerances. The `W_1^(0,2)` comparison uses absolute
`tolerance = 2e-4` and relative `tolerance = 5e-4`, because the independent
Python and Rust implementations use different marching-cubes backends.

### Independent BoneJ checks

`validation/bonej/` contains the Fiji/BoneJ scripts used for independent checks
of the conventional compatibility block and MIL anisotropy. They are optional
and are not required to run the Rust program.

## Validation status of version 0.7.0-rc.3

The conventional BoneJ-compatible measurements, local-thickness summaries,
mesh surface measures, MIL anisotropy implementation, and signed-distance
curvature field have recorded independent validation results. The previously
run 24-specimen Rust/reference comparison also showed numerical agreement for
the skeleton and mechanical-graph families and cross-backend agreement for the
retained `W_1^(0,2)` invariants.

Version `0.7.0-rc.3` contains small numerical-convention corrections to the SOI
and orientation-field implementations. The archived package did not contain a
record of a complete 24-specimen parity rerun after those corrections. For that
reason this cleaned package deliberately retains the `rc.3` version identifier.
A final release should be tagged only after `scripts/check.sh` and the complete
reference-output comparison pass on a Rust-equipped workstation.

No mechanical outcome, group label, or regression result is read by the Rust
morphometry executable.

## Reproducibility controls

MIL anisotropy exposes explicit controls for direction count, line budget,
sampling increment, number of repetitions, and random seed:

```text
--da-directions N
--da-lines N
--da-sampling-increment FLOAT
--da-repetitions N
--da-seed U64
```

For fixed input data, configuration, and seed, the native Rust MIL result is
deterministic.

Use `--threads N` to set the Rayon worker count. Use `--continue-on-error` only
when processing a collection in which failures should be recorded and skipped.

## License and source attribution

This package is distributed under **GPL-3.0-or-later**; see `LICENSE`.

The BoneJ-compatibility implementation was audited against BoneJ2, Fiji
LocalThickness, and ImageJ implementations of connectivity, local thickness,
stack statistics, and image statistics. The local-thickness implementation
follows the computational sequence used by Fiji LocalThickness. Users who
redistribute modified versions should also consult the upstream projects for
their respective copyright and license terms.
