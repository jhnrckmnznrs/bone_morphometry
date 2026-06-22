# bone-morphometry

A native, multithreaded Rust replacement for `bonej_features.py` for 3-D binary TIFF stacks where bone is 255 and background is 0.

## Feature modes

The program supports two CSV schemas through `--feature-mode`.

### `classic` (default)

Preserves the existing output:

- Tb.Th mean, population standard deviation, and maximum
- Tb.Sp mean, population standard deviation, and maximum
- BV, TV, and BV/TV
- Bone-phase Euler characteristic with BoneJ's stack-edge correction
- Connectivity and connectivity density

The classic connectivity definition is selected with `--connectivity-mode`:

```text
--connectivity-mode generalized
    beta0 + beta2 - corrected Euler

--connectivity-mode bonej
    1 - corrected Euler
```

### `refined`

Writes the compact parameter-light feature set:

- BV, TV, and BV/TV
- Tb.Th median, P10, P90, and coefficient of variation
- Tb.Sp median, P10, P90, and coefficient of variation
- Beta0/TV, Beta1/TV, and Beta2/TV
- BS, BS/BV, and BS/TV
- Mean breadth

Run it with:

```bash
./target/release/bone-morphometry \
  --feature-mode refined \
  --binary-dir images/binary/otsu2D_5_5 \
  --original-dir images/original \
  --subdirs ALN,OA,ELD \
  --output refined.csv \
  --strict-binary
```

`--connectivity-mode` is ignored by the refined schema. Its topology convention is fixed:

```text
Beta0 = number of 26-connected bone components
Beta2 = number of enclosed 6-connected background components
Beta1 = Beta0 + Beta2 - raw cubical Euler characteristic
Beta-k density = Beta-k / TV
```

The integer Betti numbers are used internally but only their densities are written because TV is already present in the CSV.

## Refined-feature definitions

### Thickness and spacing

The local-thickness maps are unchanged. Statistics are calculated from positive map values only.

- P10, median, and P90 use linear interpolation at `q * (n - 1)`, matching NumPy's default quantile method.
- CV is population standard deviation divided by the arithmetic mean (`ddof = 0`).
- The exact percentile calculation sorts the nonzero values only in refined mode; classic mode retains its linear-time summary pass.

### Cubical surface area and mean breadth

BS and mean breadth are measured on the closed union of foreground voxel cubes. Everything outside the TIFF is treated as background, so foreground cut by the ROI boundary contributes exposed surface.

For isotropic voxel side length `s`, with unique cubical-complex cell counts `n3` (cubes), `n2` (faces), `n1` (edges), and `n0` (vertices):

```text
raw Euler   = n0 - n1 + n2 - n3
BS          = 2 * (n2 - 3*n3) * s^2
mean breadth = 0.5 * (n1 - 2*n2 + 3*n3) * s
```

This is a deterministic voxel-union estimator, not a marching-cubes mesh area. Mean breadth is signed and can be negative for strongly concave or porous structures.

## Major correctness changes from the Python script

1. TIFF values are normalized to `0/1`, so corner occupancy can never accidentally contribute `255`.
2. Euler characteristic and boundary correction are both computed on the bone phase.
3. Foreground/background use a complementary connectivity pair: 26-connected bone and 6-connected background.
4. Refined Beta1 uses the raw integer Euler characteristic; the fractional boundary-corrected Euler is used only by classic connectivity.
5. The external CHUNKYEuler executable and temporary raw/euler files are eliminated.
6. Input order is deterministic.

## Algorithms and performance

- Exact separable Euclidean distance transform in `O(N)` time per axis.
- The same fast iterative weighted dilation used by Python's `localthickness.local_thickness(..., scale=1)`.
- Rayon parallelism in the distance transform and local-thickness propagation.
- Images are processed one at a time to bound peak memory.
- Cubical Euler, BS, and mean breadth share one cell-counting pass.
- Classic mode skips percentile sorting and refined mode skips BoneJ boundary correction.
- Release profile enables fat LTO and one codegen unit.

Peak working memory is roughly several floating-point volumes plus the 8-bit mask. The local-thickness method is the dominant cost, especially for structures with a large maximum inscribed radius.

## Build

Install a recent stable Rust toolchain, then:

```bash
cargo build --release
```

The binary will be at `target/release/bone-morphometry`.

## Input assumptions

- A 3-D stack is stored as multiple equal-sized pages in one TIFF.
- Pages are 8-bit grayscale.
- Each dimension is at least 2 voxels.
- A single isotropic voxel size is used, matching the Python script.
- Without `--voxel-size`, the original matching TIFF must contain a first-page `ImageDescription` line exactly beginning with `spacing=`.
- By default, values greater than `--threshold` are treated as bone. Add `--strict-binary` to require only 0 and 255.
- Each phase must contain at least one voxel; an all-bone or all-background image is rejected.

## Checks

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --release
cargo build --release
```

Unit tests cover EDT, local thickness, NumPy-compatible quantiles, cubical BS and mean breadth for simple boxes, 26-connected corner contact, separated components, and a hollow cube with one enclosed cavity.

## License note

The boundary-correction procedure follows BoneJ, and the local-thickness propagation follows the GPL-licensed `local-thickness` project. This package is therefore marked GPL-3.0-or-later.
