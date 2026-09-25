#@ String (visibility=MESSAGE, value="<html><b>BoneJ additional morphometry: DA convergence / final run</b><br/>Uses the final binary masks and ROI masks, but computes only the additional fabric descriptors.<br/>For DA, a central full-height square prism entirely inside the cylindrical ROI is constructed so BoneJ does not sample the zero-valued corners outside the cylinder.</html>", required=false) instructions
#@ File (label="Final binary-mask directory", style="directory") binaryDirectory
#@ File (label="ROI-mask directory", style="directory") roiDirectory
#@ File (label="QC spacing CSV (filename, spacing_micrometers)", style="file") spacingCsvFile
#@ File (label="Output directory", style="directory") outputDirectory
#@ String (label="Run mode", choices={"CONVERGENCE", "FINAL"}, value="CONVERGENCE", persist=false) runMode
#@ String (label="Pilot QC codes for convergence (comma-separated)", value="QC014,QC004,QC016", persist=false) pilotCodes
#@ String (label="Convergence directions (comma-separated)", value="64,128,256,512,1024,2048", persist=false) directionsList
#@ String (label="Convergence lines/direction (comma-separated)", value="8,16,32,64,128", persist=false) linesList
#@ Integer (label="Convergence repetitions per setting", value=3, min=1, persist=false) convergenceRepetitions
#@ Integer (label="FINAL directions", value=1024, min=9, persist=false) finalDirections
#@ Integer (label="FINAL lines per direction", value=64, min=1, persist=false) finalLines
#@ Integer (label="FINAL repetitions per specimen", value=5, min=1, persist=false) finalRepetitions
#@ Double (label="Sampling increment (voxels)", value=1.7320508075688772, min=1.7320508075688772, persist=false) samplingIncrement
#@ Boolean (label="Overwrite existing output CSVs", value=false, persist=false) overwriteOutput
#@ org.scijava.command.CommandService commandService
#@ org.scijava.convert.ConvertService convertService
#@ org.scijava.log.LogService logService

/*
 * BoneJ additional morphometry: degree of anisotropy (DA) and fabric direction.
 *
 * PURPOSE
 * -------
 * This script intentionally does NOT recompute BV, TV, BV/TV, Connectivity,
 * Tb.Th, or Tb.Sp. Those measurements are handled by the main BoneJ comparison script.
 *
 * It adds only:
 *   - Degree of anisotropy (DA), from BoneJ's MIL anisotropy command.
 *   - Fitted MIL ellipsoid radii, when BoneJ returns them (diagnostic).
 *   - Fitted eigensystem, when BoneJ returns it (diagnostic).
 *   - Principal fabric angle to image Z (loading axis), derived from the D1
 *     eigenvector when the eigensystem is available.
 *
 * WHY AN INSCRIBED PRISM?
 * -----------------------
 * BoneJ's Anisotropy command samples parallel lines through the rectangular
 * input stack. It does not consume the paired 3-D cylindrical mask. Treating
 * voxels outside our cylinder as ordinary marrow/background would bias MIL.
 * Therefore, for each specimen this script computes the largest CENTRAL,
 * AXIS-ALIGNED SQUARE in XY that is inside the ROI mask on EVERY z-slice,
 * crops that square through the full z extent, and runs DA on the resulting
 * full-height square prism. No thresholding, interpolation, resampling, or
 * morphology is performed.
 *
 * CALIBRATION
 * -----------
 * The scalar spacing_micrometers value from the supplied QC-keyed spacing CSV
 * is assigned isotropically to X,Y,Z. TIFF resolution metadata is ignored.
 *
 * CONVERGENCE MODE
 * ----------------
 * Runs a directions x lines-per-direction grid with repeated stochastic runs
 * on blinded pilot specimens. Defaults QC014/QC004/QC016 were selected only
 * to span low / middle / high BV/TV in the blinded calibration table; no
 * strength or group labels were used.
 *
 * FINAL MODE
 * ----------
 * After reviewing convergence externally, enter the selected directions and
 * lines values, then run all specimens with repeated stochastic evaluations.
 * The summary CSV reports mean, SD and CV of DA.
 *
 * BoneJ2's stochastic Anisotropy implementation does not expose a random seed
 * through the normal command parameters, so reproducibility is quantified by
 * repeated evaluations rather than by fixed seeds.
 */

importClass(Packages.ij.IJ);
importClass(Packages.ij.ImagePlus);
importClass(Packages.ij.ImageStack);
importClass(Packages.ij.macro.Interpreter);
importClass(Packages.ij.process.ByteProcessor);
importClass(Packages.java.io.BufferedReader);
importClass(Packages.java.io.BufferedWriter);
importClass(Packages.java.io.File);
importClass(Packages.java.io.FileInputStream);
importClass(Packages.java.io.FileOutputStream);
importClass(Packages.java.io.InputStreamReader);
importClass(Packages.java.io.OutputStreamWriter);
importClass(Packages.java.nio.charset.StandardCharsets);
importClass(Packages.java.util.LinkedHashMap);

var SCRIPT_VERSION = "2026-08-31-da-inscribed-prism-v2";
var SQRT3 = Math.sqrt(3.0);
var ByteArray = Java.type("byte[]");
var datasetClass = java.lang.Class.forName("net.imagej.Dataset");

function trimmed(v) {
    return String(v).replace(/^\s+|\s+$/g, "");
}

function boolValue(v) {
    return v === true || String(v).toLowerCase() === "true";
}

function parseCsvLine(line) {
    var cells = [];
    var cell = "";
    var quoted = false;
    var text = String(line);
    for (var i = 0; i < text.length; i++) {
        var ch = text.charAt(i);
        if (ch === '"') {
            if (quoted && i + 1 < text.length && text.charAt(i + 1) === '"') {
                cell += '"';
                i++;
            }
            else {
                quoted = !quoted;
            }
        }
        else if (ch === "," && !quoted) {
            cells.push(cell);
            cell = "";
        }
        else {
            cell += ch;
        }
    }
    if (quoted) {
        throw new Error("Unterminated quoted CSV field: " + text);
    }
    cells.push(cell);
    return cells;
}

function csvCell(v) {
    if (v === null || typeof v === "undefined") return "";
    var text = String(v);
    if (/[",\r\n]/.test(text)) {
        return '"' + text.replace(/"/g, '""') + '"';
    }
    return text;
}

function writeRows(file, headers, rows) {
    var writer = new BufferedWriter(new OutputStreamWriter(
        new FileOutputStream(file), StandardCharsets.UTF_8));
    try {
        var h = [];
        for (var i = 0; i < headers.length; i++) h.push(csvCell(headers[i]));
        writer.write(h.join(","));
        writer.newLine();
        for (var r = 0; r < rows.length; r++) {
            var cells = [];
            for (var c = 0; c < headers.length; c++) {
                cells.push(csvCell(rows[r][headers[c]]));
            }
            writer.write(cells.join(","));
            writer.newLine();
        }
    }
    finally {
        writer.close();
    }
}

function spacingKey(name) {
    return String(new File(String(name)).getName()).toLowerCase();
}

function readSpacingTable(file) {
    if (file === null || !file.isFile()) {
        throw new Error("Select an existing spacing CSV.");
    }
    var reader = new BufferedReader(new InputStreamReader(
        new FileInputStream(file), StandardCharsets.UTF_8));
    var table = new LinkedHashMap();
    try {
        var first = reader.readLine();
        if (first === null) throw new Error("Spacing CSV is empty.");
        var headers = parseCsvLine(first);
        var fn = -1;
        var sp = -1;
        for (var i = 0; i < headers.length; i++) {
            var x = trimmed(headers[i]).replace(/^\uFEFF/, "").toLowerCase();
            if (x === "filename") fn = i;
            if (x === "spacing_micrometers") sp = i;
        }
        if (fn < 0 || sp < 0) {
            throw new Error("Spacing CSV must contain filename and spacing_micrometers.");
        }
        var line;
        var lineNumber = 1;
        while ((line = reader.readLine()) !== null) {
            lineNumber++;
            if (trimmed(line).length === 0) continue;
            var cells = parseCsvLine(line);
            if (cells.length <= Math.max(fn, sp)) {
                throw new Error("Too few columns at spacing CSV line " + lineNumber + ".");
            }
            var name = trimmed(cells[fn]);
            var spacing = Number(trimmed(cells[sp]));
            if (!isFinite(spacing) || spacing <= 0) {
                throw new Error("Invalid spacing at line " + lineNumber + ".");
            }
            var key = spacingKey(name);
            if (table.containsKey(key)) {
                throw new Error("Duplicate spacing for " + name + ".");
            }
            table.put(key, spacing);
        }
    }
    finally {
        reader.close();
    }
    return table;
}

function requireSpacing(file, spacingTable) {
    var key = spacingKey(file.getName());
    if (!spacingTable.containsKey(key)) {
        throw new Error("No spacing_micrometers entry for " + file.getName() + ".");
    }
    return Number(spacingTable.get(key));
}

function isTiff(file) {
    var n = String(file.getName()).toLowerCase();
    return file.isFile() && (n.lastIndexOf(".tif") === n.length - 4 ||
        n.lastIndexOf(".tiff") === n.length - 5);
}

function collectImages(dir, out) {
    var children = dir.listFiles();
    if (children === null) return;
    for (var i = 0; i < children.length; i++) {
        if (children[i].isDirectory()) collectImages(children[i], out);
        else if (isTiff(children[i])) out.push(children[i]);
    }
}

function sampleCode(name) {
    return String(name).replace(/\.(?:tif|tiff)$/i, "")
        .replace(/(?:[_-](?:bone|binary|binarized|roi[_-]?mask|mask|roi))+$/i, "")
        .toUpperCase();
}

function buildFileIndex(dir) {
    var files = [];
    collectImages(dir, files);
    var index = new LinkedHashMap();
    for (var i = 0; i < files.length; i++) {
        var code = sampleCode(files[i].getName());
        if (index.containsKey(code)) {
            throw new Error("Ambiguous TIFFs for " + code + " in " + dir + ".");
        }
        index.put(code, files[i]);
    }
    return { files: files, index: index };
}

function binarySummary(image) {
    if (image.getBitDepth() !== 8 || image.getNChannels() !== 1 ||
        image.getNFrames() !== 1 || image.getNSlices() < 2) {
        throw new Error("Expected a single-channel, single-frame, 8-bit 3-D binary image.");
    }
    var stack = image.getStack();
    var zero = 0;
    var fore = 0;
    for (var z = 1; z <= stack.getSize(); z++) {
        var hist = stack.getProcessor(z).getHistogram();
        zero += Number(hist[0]);
        fore += Number(hist[255]);
        for (var v = 1; v < 255; v++) {
            if (hist[v] !== 0) throw new Error("Binary image contains intensity " + v + ".");
        }
    }
    if (zero === 0 || fore === 0) throw new Error("Binary image must contain both phases.");
}

function maskSummary(mask) {
    if (mask.getBitDepth() !== 8 || mask.getNChannels() !== 1 ||
        mask.getNFrames() !== 1 || mask.getNSlices() < 2) {
        throw new Error("Expected a single-channel, single-frame, 8-bit 3-D ROI mask.");
    }
    var stack = mask.getStack();
    var inside = 0;
    var outside = 0;
    for (var z = 1; z <= stack.getSize(); z++) {
        var hist = stack.getProcessor(z).getHistogram();
        outside += Number(hist[0]);
        inside += Number(hist[1]) + Number(hist[255]);
        for (var v = 2; v < 255; v++) {
            if (hist[v] !== 0) throw new Error("ROI mask contains intensity " + v + ".");
        }
    }
    if (inside === 0 || outside === 0) throw new Error("ROI mask must contain inside and outside voxels.");
}

function requireSameDimensions(a, b) {
    if (a.getWidth() !== b.getWidth() || a.getHeight() !== b.getHeight() ||
        a.getNSlices() !== b.getNSlices()) {
        throw new Error("Binary and ROI mask dimensions differ.");
    }
}

function applyCalibration(image, spacing) {
    var cal = image.getCalibration();
    cal.pixelWidth = spacing;
    cal.pixelHeight = spacing;
    cal.pixelDepth = spacing;
    cal.setUnit("um");
    image.setCalibration(cal);
}

function largestCentralSquareInsideAllSlices(mask) {
    var w = mask.getWidth();
    var h = mask.getHeight();
    var d = mask.getNSlices();
    var n = w * h;
    var common = [];
    for (var p = 0; p < n; p++) common[p] = true;

    var stack = mask.getStack();
    for (var z = 1; z <= d; z++) {
        var px = stack.getPixels(z);
        for (var p2 = 0; p2 < n; p2++) {
            if (Number(px[p2]) === 0) common[p2] = false;
        }
    }

    // 2-D integral image of OUTSIDE pixels in the intersection mask.
    var stride = w + 1;
    var prefix = [];
    for (var i = 0; i < (w + 1) * (h + 1); i++) prefix[i] = 0;
    for (var y = 0; y < h; y++) {
        var rowSum = 0;
        for (var x = 0; x < w; x++) {
            var outside = common[y * w + x] ? 0 : 1;
            rowSum += outside;
            prefix[(y + 1) * stride + (x + 1)] =
                prefix[y * stride + (x + 1)] + rowSum;
        }
    }

    function rectangleOutsideCount(x0, y0, x1, y1) {
        return prefix[y1 * stride + x1] - prefix[y0 * stride + x1] -
            prefix[y1 * stride + x0] + prefix[y0 * stride + x0];
    }

    var maxSide = Math.min(w, h);
    for (var side = maxSide; side >= 1; side--) {
        var x0 = Math.floor((w - side) / 2);
        var y0 = Math.floor((h - side) / 2);
        if (rectangleOutsideCount(x0, y0, x0 + side, y0 + side) === 0) {
            return { x0: x0, y0: y0, side: side };
        }
    }
    throw new Error("Could not find a non-empty central square inside the ROI.");
}

function cropFullHeightPrism(bone, mask, square, spacing) {
    var side = square.side;
    var depth = bone.getNSlices();
    var sourceW = bone.getWidth();
    var boneStack = bone.getStack();
    var maskStack = mask.getStack();
    var outStack = new ImageStack(side, side);
    var foreground = 0;
    var background = 0;

    for (var z = 1; z <= depth; z++) {
        var src = boneStack.getPixels(z);
        var m = maskStack.getPixels(z);
        var dst = new ByteArray(side * side);
        var q = 0;
        for (var yy = 0; yy < side; yy++) {
            var sy = square.y0 + yy;
            for (var xx = 0; xx < side; xx++) {
                var sx = square.x0 + xx;
                var idx = sy * sourceW + sx;
                if (Number(m[idx]) === 0) {
                    throw new Error("Internal error: selected DA prism contains an outside-ROI voxel.");
                }
                var value = Number(src[idx]);
                if (value !== 0) {
                    dst[q] = -1;
                    foreground++;
                }
                else {
                    dst[q] = 0;
                    background++;
                }
                q++;
            }
        }
        outStack.addSlice(new ByteProcessor(side, side, dst, null));
    }

    if (foreground === 0 || background === 0) {
        throw new Error("DA prism must contain both bone and marrow.");
    }
    var out = new ImagePlus(String(bone.getTitle()) + "__DA_PRISM", outStack);
    applyCalibration(out, spacing);
    return {
        image: out,
        side: side,
        depth: depth,
        foreground: foreground,
        background: background,
        x0: square.x0,
        y0: square.y0
    };
}

function normalizeDatasetUnits(dataset) {
    for (var d = 0; d < dataset.numDimensions(); d++) {
        var axis = dataset.axis(d);
        if (axis.type().isSpatial()) axis.setUnit("um");
    }
}

function loadClass(name) {
    try {
        return java.lang.Class.forName(name);
    }
    catch (e) {
        return java.lang.Thread.currentThread().getContextClassLoader().loadClass(name);
    }
}


function findField(commandClass, fieldName) {
    var current = commandClass;
    while (current !== null) {
        var fields = current.getDeclaredFields();
        for (var i = 0; i < fields.length; i++) {
            if (String(fields[i].getName()) === fieldName) {
                return fields[i];
            }
        }
        current = current.getSuperclass();
    }
    return null;
}

function commandInput(commandClass, imagePlus, dataset) {
    // BoneJ command input names have changed between wrappers/releases.
    // Resolve the actual field rather than hard-coding a name.
    var candidates = ["inputDataset", "inputImagePlus", "inputImage"];
    for (var i = 0; i < candidates.length; i++) {
        var field = findField(commandClass, candidates[i]);
        if (field === null) continue;

        var typeName = String(field.getType().getName());
        if (typeName === "net.imagej.Dataset") {
            return { name: candidates[i], value: dataset };
        }
        if (typeName === "ij.ImagePlus") {
            return { name: candidates[i], value: imagePlus };
        }
        if (typeName === "net.imagej.ImgPlus") {
            try {
                return { name: candidates[i], value: dataset.getImgPlus() };
            }
            catch (error) {
                var imgPlusClass = loadClass("net.imagej.ImgPlus");
                var imgPlus = convertService.convert(imagePlus, imgPlusClass);
                if (imgPlus === null) {
                    throw new Error("Could not convert the DA prism to ImgPlus.");
                }
                return { name: candidates[i], value: imgPlus };
            }
        }
    }
    throw new Error(
        "Could not determine the image-input parameter for BoneJ command " +
        commandClass.getName() + "."
    );
}

function commandFailureMessage(error) {
    var current = error;
    var text = String(error);
    var guard = 0;
    while (current !== null && guard < 20) {
        text = String(current);
        try { current = current.getCause(); }
        catch (ignored) { current = null; }
        guard++;
    }
    return text.replace(/[\r\n]+/g, " ");
}

function tableToMap(table) {
    if (table === null || typeof table === "undefined") {
        throw new Error("BoneJ Anisotropy did not return a results table.");
    }
    if (Number(table.getRowCount()) !== 1) {
        throw new Error("BoneJ Anisotropy returned " + table.getRowCount() +
            " rows; exactly one was expected.");
    }
    var out = {};
    var columns = [];
    for (var c = 0; c < Number(table.getColumnCount()); c++) {
        var column = table.get(c);
        var header = trimmed(column.getHeader());
        var value = column.get(0);
        var number = Number(value);
        if (value !== null && isFinite(number)) {
            out[header] = number;
            columns.push(header);
        }
    }
    out.__headers = columns;
    return out;
}

function findNumericByHeader(map, predicates) {
    for (var key in map) {
        if (!map.hasOwnProperty(key) || key === "__headers") continue;
        var lower = String(key).toLowerCase().replace(/^\s+|\s+$/g, "");
        for (var i = 0; i < predicates.length; i++) {
            if (predicates[i](String(key), lower)) return Number(map[key]);
        }
    }
    return NaN;
}

function exactHeader(map, name) {
    for (var key in map) {
        if (map.hasOwnProperty(key) && String(key).toLowerCase() === String(name).toLowerCase()) {
            return Number(map[key]);
        }
    }
    return NaN;
}

function parseAnisotropyResult(map) {
    var da = findNumericByHeader(map, [
        function (h, l) { return l === "da"; },
        function (h, l) { return l.indexOf("degree of anisotropy") >= 0; },
        function (h, l) { return l === "anisotropy"; }
    ]);
    if (!isFinite(da)) {
        throw new Error("Could not identify Degree of anisotropy in BoneJ result columns: " +
            map.__headers.join(" | "));
    }

    function radius(letter) {
        return findNumericByHeader(map, [
            function (h, l) { return l === letter; },
            function (h, l) { return l === "radius " + letter; },
            function (h, l) { return l === letter + " radius"; },
            function (h, l) { return l.indexOf("radius") >= 0 &&
                new RegExp("(^|[^a-z])" + letter + "([^a-z]|$)").test(l); }
        ]);
    }

    var result = {
        da: da,
        radius_a: radius("a"),
        radius_b: radius("b"),
        radius_c: radius("c")
    };

    for (var r = 0; r < 3; r++) {
        for (var c = 0; c < 3; c++) {
            result["m" + r + c] = exactHeader(map, "m" + r + c);
        }
    }
    result.D1 = exactHeader(map, "D1");
    result.D2 = exactHeader(map, "D2");
    result.D3 = exactHeader(map, "D3");

    result.da_from_eigens = NaN;
    result.da_eigen_abs_diff = NaN;
    if (isFinite(result.D1) && isFinite(result.D3) && result.D3 > 0) {
        result.da_from_eigens = 1.0 - result.D1 / result.D3;
        result.da_eigen_abs_diff = Math.abs(result.da - result.da_from_eigens);
    }

    result.da_from_radii = NaN;
    result.da_radii_abs_diff = NaN;
    if (isFinite(result.radius_a) && isFinite(result.radius_c) && result.radius_c > 0) {
        result.da_from_radii = 1.0 -
            (result.radius_a * result.radius_a) /
            (result.radius_c * result.radius_c);
        result.da_radii_abs_diff = Math.abs(result.da - result.da_from_radii);
    }

    // BoneJ docs define D1 = 1/c^2, where c is the longest MIL radius.
    // Eigenvectors are stored as matrix columns, so the D1 / c-axis vector is
    // (m00, m10, m20). Loading direction is image Z.
    result.principal_angle_to_Z_deg = NaN;
    if (isFinite(result.m00) && isFinite(result.m10) && isFinite(result.m20)) {
        var norm = Math.sqrt(result.m00 * result.m00 + result.m10 * result.m10 +
            result.m20 * result.m20);
        if (norm > 0) {
            var cosz = Math.abs(result.m20) / norm;
            cosz = Math.max(0.0, Math.min(1.0, cosz));
            result.principal_angle_to_Z_deg = Math.acos(cosz) * 180.0 / Math.PI;
        }
    }
    return result;
}

function runAnisotropy(commandClass, imagePlus, dataset, directions, lines,
    increment, sharedTable)
{
    // BoneJ writes to a process-wide SharedTable. Reset it before and after
    // every stochastic evaluation so convergence runs cannot contaminate one
    // another or inherit rows from a previous BoneJ command in this Fiji session.
    sharedTable.reset();
    try {
        var input = commandInput(commandClass, imagePlus, dataset);
        var module = commandService.run(
            commandClass,
            true,
            input.name, input.value,
            "directions", java.lang.Integer.valueOf(directions),
            "lines", java.lang.Integer.valueOf(lines),
            "samplingIncrement", java.lang.Double.valueOf(increment),
            "recommendedMin", java.lang.Boolean.FALSE,
            "printRadii", java.lang.Boolean.TRUE,
            "printEigens", java.lang.Boolean.TRUE,
            "displayMILVectors", java.lang.Boolean.FALSE,
            "printMILVectorsToTable", java.lang.Boolean.FALSE
        ).get();

        if (module !== null && module.isCanceled()) {
            var reason = module.getCancelReason();
            if (reason === null || String(reason).length === 0) {
                reason = "no reason was supplied";
            }
            throw new Error("BoneJ Anisotropy canceled: " + reason);
        }

        // Read the same SharedTable that AnisotropyWrapper populates.
        var table = sharedTable.getTable();
        return parseAnisotropyResult(tableToMap(table));
    }
    finally {
        sharedTable.reset();
    }
}

function parseIntList(text, minimum, label) {
    var parts = String(text).split(",");
    var values = [];
    var seen = {};
    for (var i = 0; i < parts.length; i++) {
        var x = Number(trimmed(parts[i]));
        if (!isFinite(x) || Math.floor(x) !== x || x < minimum) {
            throw new Error("Invalid " + label + " value: " + parts[i]);
        }
        if (!seen[String(x)]) {
            values.push(x);
            seen[String(x)] = true;
        }
    }
    values.sort(function (a, b) { return a - b; });
    if (values.length === 0) throw new Error("No " + label + " values supplied.");
    return values;
}

function parseCodes(text) {
    var parts = String(text).split(",");
    var values = [];
    var seen = {};
    for (var i = 0; i < parts.length; i++) {
        var code = sampleCode(trimmed(parts[i]));
        if (code.length === 0) continue;
        if (!seen[code]) { values.push(code); seen[code] = true; }
    }
    if (values.length === 0) throw new Error("No pilot QC codes supplied.");
    return values;
}

function closeQuietly(image) {
    if (image === null || typeof image === "undefined") return;
    try { image.changes = false; image.close(); }
    catch (ignored) {}
}

function finiteOrBlank(x) {
    return isFinite(Number(x)) ? Number(x) : "";
}

var RAW_HEADERS = [
    "filename", "blind_code", "run_mode", "script_version", "bonej_version",
    "spacing_micrometers", "domain", "domain_x0_px", "domain_y0_px",
    "domain_side_px", "domain_side_um", "domain_depth_slices", "domain_depth_um",
    "domain_bone_voxels", "domain_marrow_voxels",
    "directions", "lines_per_direction", "sampling_increment_voxels", "repetition",
    "DA", "radius_a", "radius_b", "radius_c",
    "D1", "D2", "D3",
    "m00", "m01", "m02", "m10", "m11", "m12", "m20", "m21", "m22",
    "principal_angle_to_Z_deg",
    "DA_from_eigens", "DA_eigen_abs_diff", "DA_from_radii", "DA_radii_abs_diff",
    "runtime_ms", "status", "error"
];

var SUMMARY_HEADERS = [
    "filename", "blind_code", "run_mode", "directions", "lines_per_direction",
    "sampling_increment_voxels", "n_requested", "n_success",
    "DA_mean", "DA_sd", "DA_cv", "DA_min", "DA_max",
    "principal_angle_to_Z_mean_deg", "principal_angle_to_Z_sd_deg",
    "runtime_mean_ms", "runtime_max_ms",
    "domain_side_px", "domain_side_um", "domain_depth_um", "status"
];

function mean(values) {
    if (values.length === 0) return NaN;
    var s = 0;
    for (var i = 0; i < values.length; i++) s += values[i];
    return s / values.length;
}

function sd(values) {
    if (values.length < 2) return values.length === 1 ? 0.0 : NaN;
    var m = mean(values);
    var s = 0;
    for (var i = 0; i < values.length; i++) {
        var d = values[i] - m;
        s += d * d;
    }
    return Math.sqrt(s / (values.length - 1));
}

function summarizeRows(rawRows, requestedRepetitions) {
    var groups = {};
    var order = [];
    for (var i = 0; i < rawRows.length; i++) {
        var r = rawRows[i];
        var key = r.blind_code + "|" + r.directions + "|" + r.lines_per_direction;
        if (!groups[key]) {
            groups[key] = [];
            order.push(key);
        }
        groups[key].push(r);
    }
    var out = [];
    for (var k = 0; k < order.length; k++) {
        var rows = groups[order[k]];
        var first = rows[0];
        var da = [];
        var angles = [];
        var times = [];
        for (var j = 0; j < rows.length; j++) {
            if (rows[j].status === "OK") {
                if (isFinite(Number(rows[j].DA))) da.push(Number(rows[j].DA));
                if (isFinite(Number(rows[j].principal_angle_to_Z_deg)))
                    angles.push(Number(rows[j].principal_angle_to_Z_deg));
                if (isFinite(Number(rows[j].runtime_ms))) times.push(Number(rows[j].runtime_ms));
            }
        }
        var m = mean(da);
        var s = sd(da);
        var row = {
            filename: first.filename,
            blind_code: first.blind_code,
            run_mode: first.run_mode,
            directions: first.directions,
            lines_per_direction: first.lines_per_direction,
            sampling_increment_voxels: first.sampling_increment_voxels,
            n_requested: requestedRepetitions,
            n_success: da.length,
            DA_mean: finiteOrBlank(m),
            DA_sd: finiteOrBlank(s),
            DA_cv: (isFinite(m) && m !== 0 && isFinite(s)) ? Math.abs(s / m) : "",
            DA_min: da.length ? Math.min.apply(null, da) : "",
            DA_max: da.length ? Math.max.apply(null, da) : "",
            principal_angle_to_Z_mean_deg: finiteOrBlank(mean(angles)),
            principal_angle_to_Z_sd_deg: finiteOrBlank(sd(angles)),
            runtime_mean_ms: finiteOrBlank(mean(times)),
            runtime_max_ms: times.length ? Math.max.apply(null, times) : "",
            domain_side_px: first.domain_side_px,
            domain_side_um: first.domain_side_um,
            domain_depth_um: first.domain_depth_um,
            status: da.length === requestedRepetitions ? "PASS" :
                (da.length > 0 ? "PARTIAL" : "FAIL")
        };
        out.push(row);
    }
    return out;
}

function boneJVersion(commandClass) {
    try {
        var v = commandClass.getPackage().getImplementationVersion();
        return v === null ? "unknown" : String(v);
    }
    catch (e) { return "unknown"; }
}

function main() {
    if (binaryDirectory === null || !binaryDirectory.isDirectory())
        throw new Error("Select the final binary-mask directory.");
    if (roiDirectory === null || !roiDirectory.isDirectory())
        throw new Error("Select the ROI-mask directory.");
    if (outputDirectory === null) throw new Error("Select an output directory.");
    if (!outputDirectory.exists() && !outputDirectory.mkdirs())
        throw new Error("Could not create output directory: " + outputDirectory);

    var mode = String(runMode).toUpperCase();
    if (mode !== "CONVERGENCE" && mode !== "FINAL")
        throw new Error("runMode must be CONVERGENCE or FINAL.");

    var increment = Number(samplingIncrement);
    if (!isFinite(increment) || increment + 1.0e-12 < SQRT3) {
        throw new Error("Sampling increment must be at least sqrt(3) voxels.");
    }

    var spacingTable = readSpacingTable(spacingCsvFile);
    var boneIndex = buildFileIndex(binaryDirectory);
    var maskIndex = buildFileIndex(roiDirectory);
    if (boneIndex.files.length === 0) throw new Error("No binary TIFFs found.");

    var codes = [];
    if (mode === "CONVERGENCE") {
        codes = parseCodes(pilotCodes);
    }
    else {
        var iterator = boneIndex.index.keySet().iterator();
        while (iterator.hasNext()) codes.push(String(iterator.next()));
        codes.sort();
    }

    for (var ci = 0; ci < codes.length; ci++) {
        if (!boneIndex.index.containsKey(codes[ci]))
            throw new Error("No binary TIFF found for " + codes[ci] + ".");
        if (!maskIndex.index.containsKey(codes[ci]))
            throw new Error("No ROI mask found for " + codes[ci] + ".");
        requireSpacing(boneIndex.index.get(codes[ci]), spacingTable);
    }

    var directions = mode === "CONVERGENCE" ?
        parseIntList(directionsList, 9, "directions") : [Number(finalDirections)];
    var lines = mode === "CONVERGENCE" ?
        parseIntList(linesList, 1, "lines") : [Number(finalLines)];
    var reps = mode === "CONVERGENCE" ? Number(convergenceRepetitions) : Number(finalRepetitions);
    if (!isFinite(reps) || reps < 1 || Math.floor(reps) !== reps)
        throw new Error("Repetitions must be a positive integer.");

    var rawFile = new File(outputDirectory,
        mode === "CONVERGENCE" ? "da_convergence_raw.csv" : "da_final_raw.csv");
    var summaryFile = new File(outputDirectory,
        mode === "CONVERGENCE" ? "da_convergence_summary.csv" : "da_final_summary.csv");
    if (!boolValue(overwriteOutput) && (rawFile.exists() || summaryFile.exists())) {
        throw new Error("Output CSV already exists. Choose a new directory or enable overwrite.");
    }

    var anisotropyClass = loadClass("org.bonej.wrapperPlugins.AnisotropyWrapper");
    var sharedTable = Java.type("org.bonej.utilities.SharedTable");
    var bjVersion = boneJVersion(anisotropyClass);
    var rawRows = [];

    IJ.log("BoneJ DA additional morphometry");
    IJ.log("Script: " + SCRIPT_VERSION);
    IJ.log("BoneJ: " + bjVersion);
    IJ.log("Mode: " + mode);
    IJ.log("Binary: " + binaryDirectory.getAbsolutePath());
    IJ.log("ROI:    " + roiDirectory.getAbsolutePath());
    IJ.log("Spacing: " + spacingCsvFile.getAbsolutePath());
    IJ.log("Output: " + outputDirectory.getAbsolutePath());
    IJ.log("Sampling increment: " + increment + " voxels");
    IJ.log("Specimens: " + codes.join(", "));
    IJ.log("Directions: " + directions.join(", "));
    IJ.log("Lines: " + lines.join(", "));
    IJ.log("Repetitions/setting: " + reps);

    var oldBatch = Interpreter.batchMode;
    Interpreter.batchMode = true;
    try {
        for (var si = 0; si < codes.length; si++) {
            var code = codes[si];
            var boneFile = boneIndex.index.get(code);
            var maskFile = maskIndex.index.get(code);
            var spacing = requireSpacing(boneFile, spacingTable);
            var bone = null;
            var mask = null;
            var prismImage = null;
            var dataset = null;
            try {
                IJ.log("");
                IJ.log("[" + (si + 1) + "/" + codes.length + "] " + code);
                bone = IJ.openImage(boneFile.getAbsolutePath());
                mask = IJ.openImage(maskFile.getAbsolutePath());
                if (bone === null || mask === null) throw new Error("Fiji could not open binary or mask TIFF.");
                bone.setTitle(String(boneFile.getName()));
                binarySummary(bone);
                maskSummary(mask);
                requireSameDimensions(bone, mask);
                applyCalibration(bone, spacing);
                applyCalibration(mask, spacing);

                var square = largestCentralSquareInsideAllSlices(mask);
                var prism = cropFullHeightPrism(bone, mask, square, spacing);
                prismImage = prism.image;
                dataset = convertService.convert(prismImage, datasetClass);
                if (dataset === null) throw new Error("Could not convert DA prism to Dataset.");
                dataset.setName(String(boneFile.getName()));
                normalizeDatasetUnits(dataset);

                IJ.log("  DA domain: " + prism.side + " x " + prism.side + " x " +
                    prism.depth + " voxels; " + (prism.side * spacing) + " x " +
                    (prism.side * spacing) + " x " + (prism.depth * spacing) + " um");

                for (var di = 0; di < directions.length; di++) {
                    for (var li = 0; li < lines.length; li++) {
                        for (var rep = 1; rep <= reps; rep++) {
                            var row = {
                                filename: String(boneFile.getName()),
                                blind_code: code,
                                run_mode: mode,
                                script_version: SCRIPT_VERSION,
                                bonej_version: bjVersion,
                                spacing_micrometers: spacing,
                                domain: "central_full_height_square_prism_inside_cylindrical_roi",
                                domain_x0_px: prism.x0,
                                domain_y0_px: prism.y0,
                                domain_side_px: prism.side,
                                domain_side_um: prism.side * spacing,
                                domain_depth_slices: prism.depth,
                                domain_depth_um: prism.depth * spacing,
                                domain_bone_voxels: prism.foreground,
                                domain_marrow_voxels: prism.background,
                                directions: directions[di],
                                lines_per_direction: lines[li],
                                sampling_increment_voxels: increment,
                                repetition: rep,
                                status: "FAILED",
                                error: ""
                            };
                            var start = java.lang.System.nanoTime();
                            try {
                                var result = runAnisotropy(anisotropyClass, prismImage, dataset,
                                    directions[di], lines[li], increment, sharedTable);
                                row.DA = result.da;
                                row.radius_a = finiteOrBlank(result.radius_a);
                                row.radius_b = finiteOrBlank(result.radius_b);
                                row.radius_c = finiteOrBlank(result.radius_c);
                                row.D1 = finiteOrBlank(result.D1);
                                row.D2 = finiteOrBlank(result.D2);
                                row.D3 = finiteOrBlank(result.D3);
                                for (var rr = 0; rr < 3; rr++) {
                                    for (var cc = 0; cc < 3; cc++) {
                                        row["m" + rr + cc] = finiteOrBlank(result["m" + rr + cc]);
                                    }
                                }
                                row.principal_angle_to_Z_deg = finiteOrBlank(result.principal_angle_to_Z_deg);
                                row.DA_from_eigens = finiteOrBlank(result.da_from_eigens);
                                row.DA_eigen_abs_diff = finiteOrBlank(result.da_eigen_abs_diff);
                                row.DA_from_radii = finiteOrBlank(result.da_from_radii);
                                row.DA_radii_abs_diff = finiteOrBlank(result.da_radii_abs_diff);
                                row.status = "OK";
                                row.error = "";
                            }
                            catch (error) {
                                row.status = "FAILED";
                                row.error = commandFailureMessage(error);
                                IJ.log("  ERROR " + code + " d=" + directions[di] +
                                    " l=" + lines[li] + " rep=" + rep + ": " + row.error);
                            }
                            finally {
                                row.runtime_ms = (java.lang.System.nanoTime() - start) / 1.0e6;
                                rawRows.push(row);
                                // Checkpoint after every run so a long convergence sweep is recoverable.
                                writeRows(rawFile, RAW_HEADERS, rawRows);
                            }
                        }
                    }
                }
            }
            catch (specimenError) {
                IJ.log("  SPECIMEN FAILED: " + commandFailureMessage(specimenError));
                throw specimenError;
            }
            finally {
                closeQuietly(prismImage);
                closeQuietly(mask);
                closeQuietly(bone);
                dataset = null;
                prismImage = null;
                mask = null;
                bone = null;
                java.lang.System.gc();
            }
        }
    }
    finally {
        Interpreter.batchMode = oldBatch;
    }

    var summaryRows = summarizeRows(rawRows, reps);
    writeRows(summaryFile, SUMMARY_HEADERS, summaryRows);

    var failures = 0;
    for (var r = 0; r < rawRows.length; r++) if (rawRows[r].status !== "OK") failures++;
    IJ.log("");
    IJ.log("Complete.");
    IJ.log("Raw:     " + rawFile.getAbsolutePath());
    IJ.log("Summary: " + summaryFile.getAbsolutePath());
    IJ.log("Runs: " + rawRows.length + "; failures: " + failures);
    IJ.showStatus("BoneJ DA " + mode + " complete");
}

try {
    main();
}
catch (error) {
    var message = commandFailureMessage(error);
    IJ.log("BoneJ DA script stopped: " + message);
    IJ.error("BoneJ DA additional morphometry", message);
}
