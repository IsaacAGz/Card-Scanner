const formImage = document.getElementById("form-image");
const formVideo = document.getElementById("form-video");
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
const downloadZipBtn = document.getElementById("download-zip");
const previewWrap = document.getElementById("preview-wrap");
const previewCanvas = document.getElementById("preview-canvas");

const MATCHED_COLOR = "#4ade80";
const UNMATCHED_COLOR = "#f97316";

let lastCards = [];
let lastMode = "image";
let previewObjectUrl = null;

function getSettings() {
  return {
    conf: parseFloat(document.getElementById("conf").value),
    distThreshold: parseFloat(document.getElementById("dist-threshold").value),
    frameStride: parseInt(document.getElementById("frame-stride").value, 10),
    maxFrames: parseInt(document.getElementById("max-frames").value, 10),
  };
}

function setLoading(active, message = "Processing…") {
  statusText.textContent = message;
  statusEl.classList.toggle("hidden", !active);
  formImage.querySelector("button").disabled = active;
  formVideo.querySelector("button").disabled = active;
  downloadZipBtn.disabled = active;
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
}

function clearError() {
  errorEl.textContent = "";
  errorEl.classList.add("hidden");
}

function hideResults() {
  resultsEl.classList.add("hidden");
  downloadZipBtn.classList.add("hidden");
  videoMeta.classList.add("hidden");
  previewWrap.classList.add("hidden");
  clearPreview();
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
  downloadZipBtn.classList.toggle("hidden", lastCards.length === 0);

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
  downloadZipBtn.classList.toggle("hidden", lastCards.length === 0);
  previewWrap.classList.add("hidden");
  clearPreview();
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

  const fileInput = document.getElementById("image-file");
  const file = fileInput.files[0];
  if (!file) {
    showError("Please choose an image file.");
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

downloadZipBtn.addEventListener("click", async () => {
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
    showError(err.message || "Failed to download ZIP.");
  } finally {
    setLoading(false);
  }
});
