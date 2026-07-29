const formImage = document.getElementById("form-image");
const formVideo = document.getElementById("form-video");
const btnExtractCrops = document.getElementById("btn-extract-crops");
const btnExtractImageCrops = document.getElementById("btn-extract-image-crops");
const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("status-text");
const errorEl = document.getElementById("error");
const resultsEl = document.getElementById("results");
const resultsTitle = document.getElementById("results-title");
const resultsSummary = document.getElementById("results-summary");
const resultsHeadRow = document.getElementById("results-head-row");
const resultsBody = document.getElementById("results-body");
const videoMeta = document.getElementById("video-meta");
const downloadScryfallZipBtn = document.getElementById("download-scryfall-zip");
const downloadCropsZipBtn = document.getElementById("download-crops-zip");
const previewWrap = document.getElementById("preview-wrap");
const previewCanvas = document.getElementById("preview-canvas");

const MATCHED_COLOR = "#4ade80";
const UNMATCHED_COLOR = "#f97316";
const CROP_POLL_INTERVAL_MS = 2000;

let lastCards = [];
let lastMode = "image";
let lastCropJobId = null;
let lastImageCropsBlob = null;
let previewObjectUrl = null;
let cropPollTimer = null;

function getSettings() {
  return {
    conf: parseFloat(document.getElementById("conf").value),
    distThreshold: parseFloat(document.getElementById("dist-threshold").value),
    frameStride: parseInt(document.getElementById("frame-stride").value, 10),
    maxFrames: parseInt(document.getElementById("max-frames").value, 10),
    sampleIntervalSec: parseFloat(document.getElementById("sample-interval-sec").value),
    maxSamples: parseInt(document.getElementById("max-samples").value, 10),
    embeddingDedupThreshold: parseFloat(document.getElementById("embedding-dedup-threshold").value),
    cropIdentify: document.getElementById("crop-identify").checked,
  };
}

function setLoading(active, message = "Processing…") {
  statusText.textContent = message;
  statusEl.classList.toggle("hidden", !active);
  formImage.querySelector('button[type="submit"]').disabled = active;
  btnExtractImageCrops.disabled = active;
  formVideo.querySelector('button[type="submit"]').disabled = active;
  btnExtractCrops.disabled = active;
  downloadScryfallZipBtn.disabled = active;
  downloadCropsZipBtn.disabled = active;
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
}

function clearError() {
  errorEl.textContent = "";
  errorEl.classList.add("hidden");
}

function stopCropPolling() {
  if (cropPollTimer !== null) {
    clearTimeout(cropPollTimer);
    cropPollTimer = null;
  }
}

function hideResults() {
  resultsEl.classList.add("hidden");
  downloadScryfallZipBtn.classList.add("hidden");
  downloadCropsZipBtn.classList.add("hidden");
  videoMeta.classList.add("hidden");
  previewWrap.classList.add("hidden");
  lastCropJobId = null;
  lastImageCropsBlob = null;
  clearPreview();
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function buildImageCropFormData(fileInput) {
  const selectedFiles = Array.from(fileInput.files || []);
  if (selectedFiles.length === 0) {
    return { error: "Please choose one or more image files, or a ZIP archive." };
  }

  const zipFiles = selectedFiles.filter((file) => file.name.toLowerCase().endsWith(".zip"));
  if (zipFiles.length > 0 && selectedFiles.length > 1) {
    return { error: "Upload either one ZIP archive or multiple images, not both." };
  }

  const formData = new FormData();
  if (selectedFiles.length === 1 && zipFiles.length === 1) {
    formData.append("file", selectedFiles[0]);
    return { formData, mode: "zip" };
  }

  for (const file of selectedFiles) {
    formData.append("files", file);
  }
  return { formData, mode: "files", fileCount: selectedFiles.length };
}

async function readManifestFromZip(blob) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let offset = 0;

  while (offset + 30 <= bytes.length) {
    const signature =
      bytes[offset] |
      (bytes[offset + 1] << 8) |
      (bytes[offset + 2] << 16) |
      (bytes[offset + 3] << 24);

    if (signature === 0x06054b50) {
      break;
    }
    if (signature !== 0x04034b50) {
      break;
    }

    const compressionMethod = bytes[offset + 8] | (bytes[offset + 9] << 8);
    const compressedSize =
      bytes[offset + 18] |
      (bytes[offset + 19] << 8) |
      (bytes[offset + 20] << 16) |
      (bytes[offset + 21] << 24);
    const filenameLength = bytes[offset + 26] | (bytes[offset + 27] << 8);
    const extraLength = bytes[offset + 28] | (bytes[offset + 29] << 8);
    const nameStart = offset + 30;
    const nameEnd = nameStart + filenameLength;
    const dataStart = nameEnd + extraLength;
    const dataEnd = dataStart + compressedSize;

    if (dataEnd > bytes.length) {
      break;
    }

    const entryName = new TextDecoder().decode(bytes.subarray(nameStart, nameEnd));
    if (entryName === "manifest.json") {
      const compressed = bytes.subarray(dataStart, dataEnd);
      let raw = compressed;

      if (compressionMethod === 8) {
        const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
        raw = new Uint8Array(await new Response(stream).arrayBuffer());
      } else if (compressionMethod !== 0) {
        throw new Error(`Unsupported ZIP compression for manifest.json (method ${compressionMethod}).`);
      }

      return JSON.parse(new TextDecoder().decode(raw));
    }

    offset = dataEnd;
  }

  throw new Error("manifest.json was not found in the crop ZIP.");
}

function clearPreview() {
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
  const ctx = previewCanvas.getContext("2d");
  ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
}

async function parseErrorResponse(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") {
      return data.detail;
    }
    if (Array.isArray(data.detail)) {
      return data.detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
    }
    return JSON.stringify(data);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

function formatDistance(dist) {
  return typeof dist === "number" ? dist.toFixed(1) : dist;
}

function formatSeconds(value) {
  if (typeof value !== "number") {
    return "—";
  }
  return `${value.toFixed(1)}s`;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function drawDetections(file, detections) {
  clearPreview();

  const ctx = previewCanvas.getContext("2d");
  const img = new Image();

  img.onload = () => {
    previewCanvas.width = img.naturalWidth;
    previewCanvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);

    const lineWidth = Math.max(2, Math.round(previewCanvas.width / 400));
    const fontSize = Math.max(12, Math.round(previewCanvas.width / 50));
    ctx.font = `${fontSize}px sans-serif`;
    ctx.lineWidth = lineWidth;

    for (const det of detections) {
      const [x1, y1, x2, y2] = det.box;
      const width = x2 - x1;
      const height = y2 - y1;
      const color = det.identified ? MATCHED_COLOR : UNMATCHED_COLOR;

      ctx.strokeStyle = color;
      ctx.strokeRect(x1, y1, width, height);

      const label = det.identified
        ? `${det.name} (${formatDistance(det.dist)})`
        : `Unmatched (${formatDistance(det.dist)})`;

      const textWidth = ctx.measureText(label).width;
      const labelHeight = fontSize + 6;
      const labelY = Math.max(0, y1 - labelHeight);

      ctx.fillStyle = color;
      ctx.fillRect(x1, labelY, textWidth + 8, labelHeight);
      ctx.fillStyle = "#0f172a";
      ctx.fillText(label, x1 + 4, labelY + fontSize);
    }

    if (previewObjectUrl) {
      URL.revokeObjectURL(previewObjectUrl);
    }
    previewObjectUrl = null;
  };

  previewObjectUrl = URL.createObjectURL(file);
  img.src = previewObjectUrl;
  previewWrap.classList.remove("hidden");
}

function renderImageResults(data, imageFile) {
  lastMode = "image";
  lastCards = data.cards || [];
  lastCropJobId = null;
  const detections = data.detections || [];

  resultsTitle.textContent = "Image scan results";

  const detectedCount = data.detected_count ?? detections.length;
  const identifiedCount = data.identified_count ?? lastCards.length;

  if (detectedCount === 0) {
    resultsSummary.textContent = "No cards detected. Try a clearer photo or adjust advanced settings.";
  } else if (identifiedCount === 0) {
    resultsSummary.textContent = `Detected ${detectedCount} card${detectedCount === 1 ? "" : "s"}, but none were identified.`;
  } else if (identifiedCount === detectedCount) {
    resultsSummary.textContent = `Detected and identified ${identifiedCount} card${identifiedCount === 1 ? "" : "s"}.`;
  } else {
    resultsSummary.textContent = `Detected ${detectedCount} card${detectedCount === 1 ? "" : "s"}, identified ${identifiedCount}.`;
  }

  resultsHeadRow.innerHTML = `
    <th>Name</th>
    <th>Set</th>
    <th>Distance</th>
    <th>Status</th>
  `;

  resultsBody.innerHTML = detections
    .map((det) => {
      const name = det.identified ? escapeHtml(det.name) : "—";
      const set = det.identified ? escapeHtml(det.set) : "—";
      const distClass = det.identified ? "dist-good" : "dist-unmatched";
      const statusClass = det.identified ? "status-identified" : "status-unmatched";
      const status = det.identified ? "Identified" : "Unmatched";

      return `
    <tr>
      <td>${name}</td>
      <td class="set-code">${set}</td>
      <td class="${distClass}">${formatDistance(det.dist)}</td>
      <td class="${statusClass}">${status}</td>
    </tr>
  `;
    })
    .join("");

  videoMeta.classList.add("hidden");
  resultsEl.classList.remove("hidden");
  downloadScryfallZipBtn.classList.toggle("hidden", lastCards.length === 0);
  downloadCropsZipBtn.classList.add("hidden");

  if (imageFile && detections.length > 0) {
    drawDetections(imageFile, detections);
  } else {
    previewWrap.classList.add("hidden");
    clearPreview();
  }
}

function renderVideoResults(data) {
  lastMode = "video";
  lastCards = data.cards || [];
  lastCropJobId = null;

  resultsTitle.textContent = "Video scan results";
  resultsSummary.textContent =
    data.count === 0
      ? "No unique cards identified in the sampled frames."
      : `Found ${data.count} unique card${data.count === 1 ? "" : "s"}.`;

  resultsHeadRow.innerHTML = `
    <th>Name</th>
    <th>Set</th>
    <th>Best distance</th>
    <th>First seen</th>
    <th>Last seen</th>
  `;

  resultsBody.innerHTML = lastCards
    .map(
      (card) => `
    <tr>
      <td>${escapeHtml(card.name)}</td>
      <td class="set-code">${escapeHtml(card.set)}</td>
      <td class="dist-good">${formatDistance(card.best_dist)}</td>
      <td>${formatSeconds(card.first_seen_sec)}</td>
      <td>${formatSeconds(card.last_seen_sec)}</td>
    </tr>
  `
    )
    .join("");

  if (data.video) {
    const v = data.video;
    videoMeta.textContent = `Duration: ${formatSeconds(v.duration_sec)} · FPS: ${v.fps?.toFixed?.(1) ?? v.fps} · Processed frames: ${v.processed_frames}`;
    videoMeta.classList.remove("hidden");
  } else {
    videoMeta.classList.add("hidden");
  }

  resultsEl.classList.remove("hidden");
  downloadScryfallZipBtn.classList.toggle("hidden", lastCards.length === 0);
  downloadCropsZipBtn.classList.add("hidden");
  previewWrap.classList.add("hidden");
  clearPreview();
}

function renderImageCropResults(manifestPayload, imagesProcessed) {
  lastMode = "image-crops";
  lastCards = [];
  lastCropJobId = null;

  const manifest = manifestPayload.crops || [];
  const cropCount = manifestPayload.crop_count ?? manifest.length;
  const errors = manifestPayload.errors || [];
  const identified = manifest.some((entry) => entry.identified);
  const imageCount = imagesProcessed ?? new Set(manifest.map((entry) => entry.source_image).filter(Boolean)).size;

  resultsTitle.textContent = "Image crop extraction results";
  resultsSummary.textContent =
    cropCount === 0
      ? "No card crops were detected in the uploaded images."
      : `Saved ${cropCount} crop${cropCount === 1 ? "" : "s"} from ${imageCount} image${imageCount === 1 ? "" : "s"}.`;

  if (errors.length > 0) {
    resultsSummary.textContent += ` ${errors.length} image${errors.length === 1 ? "" : "s"} skipped (see manifest errors).`;
  }

  if (identified) {
    resultsHeadRow.innerHTML = `
      <th>Filename</th>
      <th>Source image</th>
      <th>Name</th>
      <th>Set</th>
      <th>Distance</th>
    `;

    resultsBody.innerHTML = manifest
      .map(
        (entry) => `
      <tr>
        <td>${escapeHtml(entry.filename)}</td>
        <td>${entry.source_image ? escapeHtml(entry.source_image) : "—"}</td>
        <td>${entry.name ? escapeHtml(entry.name) : "—"}</td>
        <td class="set-code">${entry.set ? escapeHtml(entry.set) : "—"}</td>
        <td class="${entry.identified ? "dist-good" : "dist-unmatched"}">${entry.dist != null ? formatDistance(entry.dist) : "—"}</td>
      </tr>
    `
      )
      .join("");
  } else {
    resultsHeadRow.innerHTML = `
      <th>Filename</th>
      <th>Source image</th>
    `;

    resultsBody.innerHTML = manifest
      .map(
        (entry) => `
      <tr>
        <td>${escapeHtml(entry.filename)}</td>
        <td>${entry.source_image ? escapeHtml(entry.source_image) : "—"}</td>
      </tr>
    `
      )
      .join("");
  }

  videoMeta.classList.add("hidden");
  resultsEl.classList.remove("hidden");
  downloadScryfallZipBtn.classList.add("hidden");
  downloadCropsZipBtn.classList.toggle("hidden", cropCount === 0);
  previewWrap.classList.add("hidden");
  clearPreview();
}

function renderCropResults(data) {
  lastMode = "crops";
  lastCards = [];
  lastCropJobId = data.job_id;

  const manifest = data.manifest || [];
  const cropCount = data.crop_count ?? manifest.length;
  const identified = manifest.some((entry) => entry.identified);
  const resultVideo = data.result?.video;

  resultsTitle.textContent = "Video crop extraction results";
  resultsSummary.textContent =
    cropCount === 0
      ? "No card crops were detected in the sampled frames."
      : `Saved ${cropCount} unique crop${cropCount === 1 ? "" : "s"} from camera footage.`;

  if (identified) {
    resultsHeadRow.innerHTML = `
      <th>Filename</th>
      <th>Name</th>
      <th>Set</th>
      <th>Timestamp</th>
      <th>Distance</th>
    `;

    resultsBody.innerHTML = manifest
      .map(
        (entry) => `
      <tr>
        <td>${escapeHtml(entry.filename)}</td>
        <td>${entry.name ? escapeHtml(entry.name) : "—"}</td>
        <td class="set-code">${entry.set ? escapeHtml(entry.set) : "—"}</td>
        <td>${formatSeconds(entry.timestamp_sec)}</td>
        <td class="${entry.identified ? "dist-good" : "dist-unmatched"}">${entry.dist != null ? formatDistance(entry.dist) : "—"}</td>
      </tr>
    `
      )
      .join("");
  } else {
    resultsHeadRow.innerHTML = `
      <th>Filename</th>
      <th>Timestamp</th>
      <th>Track</th>
    `;

    resultsBody.innerHTML = manifest
      .map(
        (entry) => `
      <tr>
        <td>${escapeHtml(entry.filename)}</td>
        <td>${formatSeconds(entry.timestamp_sec)}</td>
        <td>${entry.track_id}</td>
      </tr>
    `
      )
      .join("");
  }

  if (resultVideo) {
    const skipped = data.result?.skipped_embedding_duplicates ?? 0;
    videoMeta.textContent = `Duration: ${formatSeconds(resultVideo.duration_sec)} · Sample interval: ${resultVideo.sample_interval_sec}s · Samples processed: ${resultVideo.processed_samples}${skipped ? ` · Skipped duplicates: ${skipped}` : ""}`;
    videoMeta.classList.remove("hidden");
  } else if (data.progress?.processed_samples != null) {
    videoMeta.textContent = `Samples processed: ${data.progress.processed_samples} · Crops saved: ${data.progress.crops_saved ?? cropCount}`;
    videoMeta.classList.remove("hidden");
  } else {
    videoMeta.classList.add("hidden");
  }

  resultsEl.classList.remove("hidden");
  downloadScryfallZipBtn.classList.add("hidden");
  downloadCropsZipBtn.classList.toggle("hidden", cropCount === 0);
  previewWrap.classList.add("hidden");
  clearPreview();
}

function formatCropProgress(data) {
  const progress = data.progress || {};
  const samples = progress.processed_samples ?? 0;
  const crops = progress.crops_saved ?? data.crop_count ?? 0;
  const duration = progress.duration_sec;
  const durationText = typeof duration === "number" ? ` / ~${Math.ceil(duration / (progress.sample_interval_sec || 5))} samples` : "";
  return `Extracting crops… sampled ${samples}${durationText} · ${crops} crop${crops === 1 ? "" : "s"} saved`;
}

async function pollCropJob(jobId) {
  stopCropPolling();

  const poll = async () => {
    try {
      const response = await fetch(`/scan/video/crops/${jobId}`);
      if (!response.ok) {
        throw new Error(await parseErrorResponse(response));
      }

      const data = await response.json();

      if (data.status === "queued" || data.status === "running") {
        setLoading(true, formatCropProgress(data));
        cropPollTimer = setTimeout(poll, CROP_POLL_INTERVAL_MS);
        return;
      }

      stopCropPolling();
      setLoading(false);

      if (data.status === "failed") {
        showError(data.error || "Crop extraction failed.");
        return;
      }

      renderCropResults(data);
    } catch (err) {
      stopCropPolling();
      setLoading(false);
      showError(err.message || "Failed to poll crop job status.");
    }
  };

  await poll();
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.tab;

    tabs.forEach((t) => {
      const isActive = t === tab;
      t.classList.toggle("active", isActive);
      t.setAttribute("aria-selected", isActive ? "true" : "false");
    });

    panels.forEach((panel) => {
      const isImage = panel.id === "panel-image";
      const show = (target === "image" && isImage) || (target === "video" && !isImage);
      panel.classList.toggle("active", show);
      panel.hidden = !show;
    });
  });
});

formImage.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  hideResults();
  stopCropPolling();

  const fileInput = document.getElementById("image-file");
  const file = fileInput.files[0];
  if (!file) {
    showError("Please choose an image file.");
    return;
  }
  if (file.name.toLowerCase().endsWith(".zip")) {
    showError("Scan image expects a single photo. Use Extract card crops (ZIP) for ZIP archives.");
    return;
  }

  const { conf, distThreshold } = getSettings();
  const params = new URLSearchParams({ conf: String(conf), dist_threshold: String(distThreshold) });
  const formData = new FormData();
  formData.append("file", file);

  setLoading(true, "Scanning image…");

  try {
    const response = await fetch(`/scan?${params}`, { method: "POST", body: formData });
    if (!response.ok) {
      throw new Error(await parseErrorResponse(response));
    }
    renderImageResults(await response.json(), file);
  } catch (err) {
    showError(err.message || "Image scan failed.");
  } finally {
    setLoading(false);
  }
});

formVideo.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  hideResults();
  stopCropPolling();

  const fileInput = document.getElementById("video-file");
  const file = fileInput.files[0];
  if (!file) {
    showError("Please choose a video file.");
    return;
  }

  const { conf, distThreshold, frameStride, maxFrames } = getSettings();
  const params = new URLSearchParams({
    conf: String(conf),
    dist_threshold: String(distThreshold),
    frame_stride: String(frameStride),
    max_frames: String(maxFrames),
  });
  const formData = new FormData();
  formData.append("file", file);

  setLoading(true, "Processing video… this may take several minutes on CPU.");

  try {
    const response = await fetch(`/scan/video?${params}`, { method: "POST", body: formData });
    if (!response.ok) {
      throw new Error(await parseErrorResponse(response));
    }
    renderVideoResults(await response.json());
  } catch (err) {
    if (err.name === "TypeError") {
      showError("Request failed — the server may have timed out on a long video. Try a shorter clip or lower max frames.");
    } else {
      showError(err.message || "Video scan failed.");
    }
  } finally {
    setLoading(false);
  }
});

btnExtractImageCrops.addEventListener("click", async () => {
  clearError();
  hideResults();
  stopCropPolling();

  const fileInput = document.getElementById("image-file");
  const built = buildImageCropFormData(fileInput);
  if (built.error) {
    showError(built.error);
    return;
  }

  const { conf, distThreshold, cropIdentify } = getSettings();
  const params = new URLSearchParams({
    conf: String(conf),
    dist_threshold: String(distThreshold),
    identify: String(cropIdentify),
  });

  setLoading(true, built.mode === "zip" ? "Extracting crops from ZIP…" : "Extracting crops from images…");

  try {
    const response = await fetch(`/scan/images/crops-zip?${params}`, {
      method: "POST",
      body: built.formData,
    });
    if (!response.ok) {
      throw new Error(await parseErrorResponse(response));
    }

    const blob = await response.blob();
    lastImageCropsBlob = blob;
    downloadBlob(blob, "image_crops.zip");

    const manifestPayload = await readManifestFromZip(blob);
    const imagesProcessed =
      built.mode === "files" ? built.fileCount : new Set((manifestPayload.crops || []).map((entry) => entry.source_image)).size;
    renderImageCropResults(manifestPayload, imagesProcessed);
  } catch (err) {
    showError(err.message || "Image crop extraction failed.");
  } finally {
    setLoading(false);
  }
});

btnExtractCrops.addEventListener("click", async () => {
  clearError();
  hideResults();
  stopCropPolling();

  const fileInput = document.getElementById("video-file");
  const file = fileInput.files[0];
  if (!file) {
    showError("Please choose a video file.");
    return;
  }

  const {
    conf,
    distThreshold,
    sampleIntervalSec,
    maxSamples,
    embeddingDedupThreshold,
    cropIdentify,
  } = getSettings();

  const params = new URLSearchParams({
    conf: String(conf),
    dist_threshold: String(distThreshold),
    sample_interval_sec: String(sampleIntervalSec),
    max_samples: String(maxSamples),
    embedding_dedup_threshold: String(embeddingDedupThreshold),
    identify: String(cropIdentify),
  });
  const formData = new FormData();
  formData.append("file", file);

  setLoading(true, "Uploading video and starting crop job…");

  try {
    const response = await fetch(`/scan/video/crops?${params}`, { method: "POST", body: formData });
    if (!response.ok) {
      throw new Error(await parseErrorResponse(response));
    }

    const startPayload = await response.json();
    await pollCropJob(startPayload.job_id);
  } catch (err) {
    stopCropPolling();
    setLoading(false);
    showError(err.message || "Failed to start crop extraction.");
  }
});

downloadScryfallZipBtn.addEventListener("click", async () => {
  if (lastCards.length === 0) {
    return;
  }

  clearError();
  setLoading(true, "Building ZIP from Scryfall…");

  const payload = {
    cards: lastCards.map((card) => ({ name: card.name, set: card.set })),
  };

  try {
    const response = await fetch("/cards/images-zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(await parseErrorResponse(response));
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = lastMode === "video" ? "video_cards.zip" : "detected_cards.zip";
    anchor.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    showError(err.message || "Failed to download Scryfall ZIP.");
  } finally {
    setLoading(false);
  }
});

downloadCropsZipBtn.addEventListener("click", async () => {
  if (lastMode === "image-crops" && lastImageCropsBlob) {
    clearError();
    downloadBlob(lastImageCropsBlob, "image_crops.zip");
    return;
  }

  if (!lastCropJobId) {
    return;
  }

  clearError();
  setLoading(true, "Downloading crops ZIP…");

  try {
    const response = await fetch(`/scan/video/crops/${lastCropJobId}/download`);
    if (!response.ok) {
      throw new Error(await parseErrorResponse(response));
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `video_crops_${lastCropJobId}.zip`;
    anchor.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    showError(err.message || "Failed to download crops ZIP.");
  } finally {
    setLoading(false);
  }
});
