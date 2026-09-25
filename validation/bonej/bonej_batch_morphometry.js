#@ String (visibility=MESSAGE, value="<html><b>BoneJ batch morphometry</b><br/>Input: ~/Desktop/segmentation/images/original<br/>First run: use 1 image. Final run: use -1 for all images.</html>", required=false) instructions
#@ Integer (label="Maximum number of images (-1 = all)", value=1, persist=false) maximumImages
#@ Boolean (label="Overwrite bonej_morphometry.csv if it exists", value=false, persist=false) overwriteCsv
#@ org.scijava.command.CommandService commandService
#@ org.scijava.convert.ConvertService convertService
#@ org.scijava.log.LogService logService

/*
 * Batch BoneJ morphometry for 3D binary TIFF stacks.
 *
 * Input directory (searched recursively):
 *   ~/Desktop/segmentation/images/original
 *
 * Output CSV:
 *   ~/Desktop/segmentation/bonej_morphometry.csv
 *
 * Measurements requested from BoneJ:
 *   - Area/Volume fraction: BV, TV, BV/TV
 *   - Thickness: mean, standard deviation, and maximum for both Tb.Th and Tb.Sp
 *   - Connectivity: Euler characteristic, corrected Euler characteristic,
 *     connectivity, and connectivity density
 *
 * The CSV contains exactly 13 standardized measurement columns:
 *   BV, TV, BV/TV,
 *   Euler characteristic, Euler change/correction,
 *   Connectivity, Connectivity density,
 *   Tb.Th Mean, Tb.Th Std Dev, Tb.Th Max,
 *   Tb.Sp Mean, Tb.Sp Std Dev, Tb.Sp Max.
 *
 * Scientific safeguards:
 *   - No thresholding or conversion is performed.
 *   - Each input must be a single-channel, single-frame, 3D, 8-bit image
 *     whose only voxel values are 0 and 255.
 *   - 255 is treated as bone; 0 is treated as background/marrow.
 *   - Calibration is read from each image and written to the CSV.
 *   - BoneJ 7.2.2 cannot parse the Unicode micrometre symbols in its
 *     thickness-calibration check. After conversion to an ImageJ2 Dataset,
 *     spatial-axis units equal to "µm" or "μm" are changed in memory to
 *     the equivalent spelling "um". Voxel dimensions are not rescaled,
 *     source TIFF files are not modified, and the original unit remains
 *     recorded in the CSV.
 *   - The script refuses to run while the ROI Manager contains ROIs, because
 *     ROI handling differs among BoneJ commands. It measures the full cuboid
 *     represented by each input stack.
 *
 * The script supports the input parameter names used in recent BoneJ2
 * releases as well as the older BoneJ 7.1.x wrappers.
 */

importClass(Packages.ij.IJ);
importClass(Packages.ij.macro.Interpreter);
importClass(Packages.ij.plugin.frame.RoiManager);
importClass(Packages.java.io.BufferedWriter);
importClass(Packages.java.io.File);
importClass(Packages.java.io.FileOutputStream);
importClass(Packages.java.io.OutputStreamWriter);
importClass(Packages.java.nio.charset.StandardCharsets);
importClass(Packages.java.util.LinkedHashMap);
importClass(Packages.java.util.LinkedHashSet);

var USER_HOME = String(java.lang.System.getProperty("user.home"));
var INPUT_DIRECTORY = new File(USER_HOME,
    "Desktop/segmentation/images/original");
var OUTPUT_FILE = new File(USER_HOME,
    "Desktop/segmentation/bonej_morphometry.csv");
var SCRIPT_VERSION = "2026-08-21-unit-fix-v2";

function booleanValue(value) {
    return value === true || String(value).toLowerCase() === "true";
}

var OVERWRITE_CSV = booleanValue(overwriteCsv);

var MEASUREMENT_HEADERS = [
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
    "Tb.Sp Max"
];

var BASE_HEADERS = [
    "filename",
    "group",
    "relative_path",
    "status",
    "errors",
    "calibration_warning",
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
    "width_px",
    "height_px",
    "depth_slices",
    "pixel_width",
    "pixel_height",
    "pixel_depth",
    "spatial_unit",
    "foreground_voxels",
    "background_voxels",
    "imagej_version",
    "bonej_version"
];

var rows = [];
var headers = [];
var headerSet = new LinkedHashSet();

for (var baseIndex = 0; baseIndex < BASE_HEADERS.length; baseIndex++) {
    addHeader(BASE_HEADERS[baseIndex]);
}

function addHeader(header) {
    var text = String(header);
    if (headerSet.add(text)) {
        headers.push(text);
    }
}

function loadClass(className) {
    try {
        return java.lang.Class.forName(className);
    }
    catch (firstError) {
        return java.lang.Thread.currentThread().getContextClassLoader()
            .loadClass(className);
    }
}

function loadFirstAvailableClass(classNames, description) {
    var failures = [];
    for (var i = 0; i < classNames.length; i++) {
        try {
            return loadClass(classNames[i]);
        }
        catch (error) {
            failures.push(classNames[i]);
        }
    }
    throw new Error(
        "Could not find BoneJ's " + description + " command. Tried: " +
        failures.join(", ") + ". Update Fiji, enable the BoneJ update site, " +
        "restart Fiji, and run the script again."
    );
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
    // Prefer a native ImagePlus where a command explicitly accepts it.
    var candidates = ["inputImagePlus", "inputDataset", "inputImage"];
    for (var i = 0; i < candidates.length; i++) {
        var field = findField(commandClass, candidates[i]);
        if (field === null) {
            continue;
        }

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
                    throw new Error("Could not convert the image to ImgPlus.");
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

function checkModule(module, label) {
    if (module !== null && module.isCanceled()) {
        var reason = module.getCancelReason();
        if (reason === null || String(reason).length === 0) {
            reason = "no reason was supplied";
        }
        throw new Error(label + " was canceled: " + reason);
    }
}

function runElementFraction(commandClass, imagePlus, dataset) {
    var input = commandInput(commandClass, imagePlus, dataset);
    var module = commandService.run(
        commandClass,
        true,
        input.name,
        input.value
    ).get();
    checkModule(module, "Area/Volume fraction");
}

function runThickness(commandClass, imagePlus, dataset) {
    var input = commandInput(commandClass, imagePlus, dataset);
    var module = commandService.run(
        commandClass,
        true,
        input.name,
        input.value,
        "mapChoice",
        "Both",
        "showMaps",
        java.lang.Boolean.FALSE,
        "maskArtefacts",
        java.lang.Boolean.TRUE
    ).get();
    checkModule(module, "Thickness");
}

function runConnectivity(commandClass, imagePlus, dataset) {
    var input = commandInput(commandClass, imagePlus, dataset);
    var module = commandService.run(
        commandClass,
        true,
        input.name,
        input.value
    ).get();
    checkModule(module, "Connectivity");
}

function commandFailureMessage(error) {
    var current = error;
    var lastText = "Unknown error";
    var guard = 0;
    while (current !== null && guard < 20) {
        lastText = String(current);
        try {
            current = current.getCause();
        }
        catch (ignored) {
            current = null;
        }
        guard++;
    }
    return lastText.replace(/[\r\n]+/g, " ");
}

function runMetric(label, action, errors) {
    IJ.log("  " + label + "...");
    try {
        action();
        IJ.log("  " + label + ": done");
    }
    catch (error) {
        var message = label + ": " + commandFailureMessage(error);
        errors.push(message);
        IJ.log("  ERROR - " + message);
    }
}

function isSupportedImage(file) {
    var name = String(file.getName()).toLowerCase();
    return hasSuffix(name, ".tif") || hasSuffix(name, ".tiff");
}

function hasSuffix(text, suffix) {
    return text.length >= suffix.length &&
        text.lastIndexOf(suffix) === text.length - suffix.length;
}

function collectImages(directory, output) {
    var children = directory.listFiles();
    if (children === null) {
        return;
    }
    for (var i = 0; i < children.length; i++) {
        var child = children[i];
        if (child.isDirectory()) {
            collectImages(child, output);
        }
        else if (child.isFile() && isSupportedImage(child)) {
            output.push(child);
        }
    }
}

function binarySummary(imagePlus) {
    if (imagePlus.getBitDepth() !== 8) {
        return {
            ok: false,
            reason: "Image is not 8-bit (bit depth = " +
                imagePlus.getBitDepth() + ")."
        };
    }
    if (imagePlus.getNChannels() !== 1 || imagePlus.getNFrames() !== 1) {
        return {
            ok: false,
            reason: "Hyperstacks are not accepted by this script; expected " +
                "one channel and one time frame."
        };
    }
    if (imagePlus.getNSlices() < 2) {
        return {
            ok: false,
            reason: "Image is not a 3D stack (fewer than two z-slices)."
        };
    }

    var background = 0;
    var foreground = 0;
    var stack = imagePlus.getStack();
    for (var z = 1; z <= stack.getSize(); z++) {
        var histogram = stack.getProcessor(z).getHistogram();
        background += Number(histogram[0]);
        foreground += Number(histogram[255]);
        for (var value = 1; value < 255; value++) {
            if (histogram[value] !== 0) {
                return {
                    ok: false,
                    reason: "Found intensity " + value + " in slice " + z +
                        "; expected only 0 and 255."
                };
            }
        }
    }
    if (background === 0 || foreground === 0) {
        return {
            ok: false,
            reason: "The image is constant; both background (0) and bone " +
                "foreground (255) must be present."
        };
    }
    return {
        ok: true,
        background: background,
        foreground: foreground
    };
}

function calibrationWarning(imagePlus) {
    var calibration = imagePlus.getCalibration();
    var values = [
        Number(calibration.pixelWidth),
        Number(calibration.pixelHeight),
        Number(calibration.pixelDepth)
    ];
    for (var i = 0; i < values.length; i++) {
        if (!isFinite(values[i]) || values[i] <= 0) {
            return "Invalid non-positive or non-finite spatial calibration.";
        }
    }

    var maximum = Math.max(values[0], values[1], values[2]);
    var minimum = Math.min(values[0], values[1], values[2]);
    if ((maximum - minimum) / maximum > 1.0e-6) {
        return "Voxel calibration is anisotropic; verify that this is " +
            "intentional before interpreting the thickness measurements.";
    }
    var unit = String(calibration.getUnit());
    if (unit === "pixel" || unit === "pixels" || unit.length === 0) {
        return "Spatial calibration is in pixels rather than a physical unit.";
    }
    return "";
}

function boneJCompatibleUnit(unit) {
    var originalUnit = String(unit);
    var trimmedUnit = originalUnit.replace(/^\s+|\s+$/g, "");
    // U+00B5 is the micro sign and U+03BC is the Greek small letter mu.
    return /^[\u00b5\u03bc][mM]$/.test(trimmedUnit) ? "um" : originalUnit;
}

function normalizeDatasetUnitsForBoneJ(dataset) {
    var axisDescriptions = [];
    var changes = [];

    for (var d = 0; d < dataset.numDimensions(); d++) {
        var axis = dataset.axis(d);
        if (!axis.type().isSpatial()) {
            continue;
        }

        var axisLabel = String(axis.type().getLabel());
        var originalUnit = String(axis.unit());
        var processingUnit = boneJCompatibleUnit(originalUnit);
        if (processingUnit !== originalUnit) {
            axis.setUnit(processingUnit);
            if (String(axis.unit()) !== processingUnit) {
                throw new Error("Could not set the " + axisLabel +
                    " Dataset-axis unit to " + processingUnit + ".");
            }
            changes.push(axisLabel + ": " + originalUnit + " -> " +
                processingUnit);
        }

        axisDescriptions.push(axisLabel + "=" +
            Number(axis.averageScale(0, 1)) + " " + String(axis.unit()));
    }

    if (axisDescriptions.length !== 3) {
        throw new Error("Expected exactly three spatial Dataset axes, found " +
            axisDescriptions.length + ".");
    }

    return {
        descriptions: axisDescriptions,
        changes: changes
    };
}

function relativePath(root, file) {
    return String(root.toPath().relativize(file.toPath()).toString());
}

function groupFromRelativePath(path) {
    var slash = path.indexOf(File.separator);
    if (slash < 0) {
        return "";
    }
    return path.substring(0, slash);
}

function boneJVersion(commandClass) {
    try {
        var version = commandClass.getPackage().getImplementationVersion();
        return version === null ? "unknown" : String(version);
    }
    catch (error) {
        return "unknown";
    }
}

function standardizedMetricName(boneJHeader) {
    var header = String(boneJHeader).replace(/^\s+|\s+$/g, "");
    var lower = header.toLowerCase();

    if (header === "BV/TV") {
        return "BV/TV";
    }
    if (header === "BV" || header.indexOf("BV ") === 0 ||
        header.indexOf("BV(") === 0) {
        return "BV";
    }
    if (header === "TV" || header.indexOf("TV ") === 0 ||
        header.indexOf("TV(") === 0) {
        return "TV";
    }
    if (header.indexOf("Tb.Th Mean") === 0) {
        return "Tb.Th Mean";
    }
    if (header.indexOf("Tb.Th Std Dev") === 0) {
        return "Tb.Th Std Dev";
    }
    if (header.indexOf("Tb.Th Max") === 0) {
        return "Tb.Th Max";
    }
    if (header.indexOf("Tb.Sp Mean") === 0) {
        return "Tb.Sp Mean";
    }
    if (header.indexOf("Tb.Sp Std Dev") === 0) {
        return "Tb.Sp Std Dev";
    }
    if (header.indexOf("Tb.Sp Max") === 0) {
        return "Tb.Sp Max";
    }
    if (header === "Connectivity") {
        return "Connectivity";
    }
    if (header.indexOf("Conn.D") === 0 ||
        lower.indexOf("connectivity density") === 0 ||
        lower.indexOf("conn. density") === 0) {
        return "Connectivity density";
    }
    if (header.indexOf("Δ(χ)") === 0 ||
        lower.indexOf("corr. euler") === 0 ||
        lower.indexOf("corrected euler") === 0 ||
        lower.indexOf("delta euler") === 0) {
        return "Euler change/correction";
    }
    if (lower.indexOf("euler ch") === 0 ||
        lower.indexOf("euler char") === 0) {
        return "Euler characteristic";
    }
    return null;
}

function extractRequestedBoneJResults(row, sharedTable) {
    var table = sharedTable.getTable();
    var rowCount = table.getRowCount();
    var columnCount = table.getColumnCount();

    for (var c = 0; c < columnCount; c++) {
        var column = table.get(c);
        var header = String(column.getHeader());
        var standardizedHeader = standardizedMetricName(header);
        if (standardizedHeader === null) {
            continue;
        }
        for (var r = 0; r < rowCount; r++) {
            var value = column.get(r);
            if (value === null) {
                continue;
            }
            row.put(standardizedHeader, value);
        }
    }

    var missing = [];
    for (var i = 0; i < MEASUREMENT_HEADERS.length; i++) {
        if (!row.containsKey(MEASUREMENT_HEADERS[i])) {
            missing.push(MEASUREMENT_HEADERS[i]);
        }
    }
    return missing;
}

function putImageMetadata(row, file, relative, imagePlus, binary,
    elementFractionClass) {
    var calibration = imagePlus.getCalibration();
    row.put("filename", String(file.getName()));
    row.put("group", groupFromRelativePath(relative));
    row.put("relative_path", relative);
    row.put("width_px", imagePlus.getWidth());
    row.put("height_px", imagePlus.getHeight());
    row.put("depth_slices", imagePlus.getNSlices());
    row.put("pixel_width", calibration.pixelWidth);
    row.put("pixel_height", calibration.pixelHeight);
    row.put("pixel_depth", calibration.pixelDepth);
    row.put("spatial_unit", String(calibration.getUnit()));
    if (binary.ok) {
        row.put("foreground_voxels", binary.foreground);
        row.put("background_voxels", binary.background);
    }
    row.put("imagej_version", String(IJ.getFullVersion()));
    row.put("bonej_version", boneJVersion(elementFractionClass));
}

function csvCell(value) {
    if (value === null || typeof value === "undefined") {
        return "";
    }
    var text = String(value).replace(/"/g, "\"\"");
    return "\"" + text + "\"";
}

function writeCsv(file) {
    var parent = file.getParentFile();
    if (!parent.exists() && !parent.mkdirs()) {
        throw new Error("Could not create output directory: " + parent);
    }

    var writer = new BufferedWriter(new OutputStreamWriter(
        new FileOutputStream(file, false), StandardCharsets.UTF_8));
    try {
        var headerCells = [];
        for (var h = 0; h < headers.length; h++) {
            headerCells.push(csvCell(headers[h]));
        }
        writer.write(headerCells.join(","));
        writer.newLine();

        for (var r = 0; r < rows.length; r++) {
            var cells = [];
            for (var c = 0; c < headers.length; c++) {
                cells.push(csvCell(rows[r].get(headers[c])));
            }
            writer.write(cells.join(","));
            writer.newLine();
        }
    }
    finally {
        writer.close();
    }
}

function verifyRoiManagerIsEmpty() {
    var manager = RoiManager.getInstance2();
    if (manager !== null && manager.getCount() > 0) {
        throw new Error(
            "The ROI Manager contains " + manager.getCount() + " ROI(s). " +
            "Save them if needed, clear the ROI Manager, and rerun. The three " +
            "BoneJ commands do not all apply ROI Manager masks in the same way."
        );
    }
}

function main() {
    if (!INPUT_DIRECTORY.isDirectory()) {
        throw new Error("Input directory does not exist: " + INPUT_DIRECTORY);
    }
    if (OUTPUT_FILE.exists() && !OVERWRITE_CSV) {
        throw new Error(
            "Output already exists: " + OUTPUT_FILE + ". Check 'Overwrite " +
            "bonej_morphometry.csv' in the start dialog, rename the existing " +
            "file, or move it before rerunning."
        );
    }
    verifyRoiManagerIsEmpty();

    var elementFractionClass = loadFirstAvailableClass([
        "org.bonej.wrapperPlugins.ElementFractionWrapper"
    ], "Area/Volume fraction");
    var thicknessClass = loadFirstAvailableClass([
        "org.bonej.wrapperPlugins.ThicknessWrapper"
    ], "Thickness");
    var connectivityClass = loadFirstAvailableClass([
        "org.bonej.plugins.Connectivity",
        "org.bonej.wrapperPlugins.ConnectivityWrapper"
    ], "Connectivity");
    var datasetClass = loadClass("net.imagej.Dataset");
    var sharedTable = Java.type("org.bonej.utilities.SharedTable");

    var files = [];
    collectImages(INPUT_DIRECTORY, files);
    files.sort(function (left, right) {
        var a = String(left.getAbsolutePath());
        var b = String(right.getAbsolutePath());
        return a < b ? -1 : (a > b ? 1 : 0);
    });

    if (files.length === 0) {
        throw new Error("No .tif or .tiff files found under " + INPUT_DIRECTORY);
    }

    var requestedMaximum = Number(maximumImages);
    var fileCount = files.length;
    if (requestedMaximum >= 0) {
        fileCount = Math.min(fileCount, requestedMaximum);
    }
    if (fileCount === 0) {
        throw new Error("Maximum number of images was set to zero.");
    }

    IJ.log("BoneJ batch morphometry");
    IJ.log("Script version: " + SCRIPT_VERSION);
    IJ.log("Input:  " + INPUT_DIRECTORY.getAbsolutePath());
    IJ.log("Output: " + OUTPUT_FILE.getAbsolutePath());
    IJ.log("Images to process: " + fileCount + " of " + files.length);
    IJ.log("Requested measurements: 13 (anisotropy excluded)");

    var previousBatchMode = Interpreter.batchMode;
    Interpreter.batchMode = true;
    try {
        for (var index = 0; index < fileCount; index++) {
            var file = files[index];
            var relative = relativePath(INPUT_DIRECTORY, file);
            var row = new LinkedHashMap();
            var imagePlus = null;
            var dataset = null;
            var errors = [];

            IJ.showProgress(index, fileCount);
            IJ.showStatus("BoneJ " + (index + 1) + "/" + fileCount + ": " +
                file.getName());
            IJ.log("");
            IJ.log("[" + (index + 1) + "/" + fileCount + "] " + relative);

            try {
                imagePlus = IJ.openImage(file.getAbsolutePath());
                if (imagePlus === null) {
                    throw new Error("Fiji could not open the image.");
                }
                imagePlus.setTitle(String(file.getName()));

                var binary = binarySummary(imagePlus);
                putImageMetadata(row, file, relative, imagePlus, binary,
                    elementFractionClass);
                row.put("calibration_warning", calibrationWarning(imagePlus));

                if (!binary.ok) {
                    row.put("status", "SKIPPED_INVALID_INPUT");
                    row.put("errors", binary.reason);
                    IJ.log("  SKIPPED - " + binary.reason);
                }
                else {
                    dataset = convertService.convert(imagePlus, datasetClass);
                    if (dataset === null) {
                        throw new Error("Could not convert ImagePlus to Dataset.");
                    }
                    dataset.setName(String(file.getName()));

                    var datasetUnits = normalizeDatasetUnitsForBoneJ(dataset);
                    if (datasetUnits.changes.length > 0) {
                        IJ.log("  Dataset unit alias(es) for BoneJ: " +
                            datasetUnits.changes.join(", ") +
                            " (voxel dimensions unchanged)");
                    }
                    IJ.log("  BoneJ Dataset spatial axes: " +
                        datasetUnits.descriptions.join(", "));

                    sharedTable.reset();
                    runMetric("Area/Volume fraction", function () {
                        runElementFraction(elementFractionClass, imagePlus,
                            dataset);
                    }, errors);
                    runMetric("Thickness (Tb.Th and Tb.Sp)", function () {
                        runThickness(thicknessClass, imagePlus, dataset);
                    }, errors);
                    runMetric("Connectivity", function () {
                        runConnectivity(connectivityClass, imagePlus, dataset);
                    }, errors);
                    var missingMetrics = extractRequestedBoneJResults(row,
                        sharedTable);
                    if (missingMetrics.length > 0) {
                        errors.push("Missing expected BoneJ output(s): " +
                            missingMetrics.join(", "));
                    }
                    if (errors.length === 0) {
                        row.put("status",
                            String(row.get("calibration_warning")).length === 0 ?
                            "OK" : "OK_WITH_CALIBRATION_WARNING");
                        row.put("errors", "");
                    }
                    else {
                        row.put("status", "PARTIAL");
                        row.put("errors", errors.join(" | "));
                    }
                }
            }
            catch (error) {
                if (!row.containsKey("filename")) {
                    row.put("filename", String(file.getName()));
                    row.put("group", groupFromRelativePath(relative));
                    row.put("relative_path", relative);
                }
                row.put("status", "FAILED");
                row.put("errors", commandFailureMessage(error));
                IJ.log("  FAILED - " + commandFailureMessage(error));
            }
            finally {
                rows.push(row);
                writeCsv(OUTPUT_FILE);
                if (imagePlus !== null) {
                    imagePlus.changes = false;
                    imagePlus.close();
                }
                dataset = null;
                imagePlus = null;
                java.lang.System.gc();
            }
        }
    }
    finally {
        Interpreter.batchMode = previousBatchMode;
        IJ.showProgress(1.0);
    }

    IJ.showStatus("BoneJ batch complete");
    IJ.log("");
    IJ.log("Complete. Results saved to:");
    IJ.log(OUTPUT_FILE.getAbsolutePath());
    IJ.log("Check the status and errors columns before analysis.");
}

try {
    main();
}
catch (error) {
    var message = commandFailureMessage(error);
    IJ.log("BoneJ batch stopped: " + message);
    IJ.error("BoneJ batch morphometry", message);
}
