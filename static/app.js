const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const variantInput = document.querySelector("#variant");
const topKInput = document.querySelector("#top-k");
const perVideoInput = document.querySelector("#per-video");
const searchButton = document.querySelector("#search-button");
const statusText = document.querySelector("#status");
const resultsRoot = document.querySelector("#results");
const actions = document.querySelector("#actions");

let currentResults = [];

function makeElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function renderResults(results) {
  resultsRoot.replaceChildren();
  currentResults = results;
  for (const row of results) {
    const card = makeElement("article", "card");
    const image = makeElement("img", "keyframe");
    image.src = `/keyframe/${encodeURIComponent(row.video_id)}/${row.keyframe_no}.jpg`;
    image.alt = `Rank ${row.rank}: ${row.video_id}`;
    image.loading = "lazy";

    const body = makeElement("div", "card-body");
    const headingRow = makeElement("div", "card-heading");
    headingRow.append(
      makeElement("span", "rank", `#${row.rank}`),
      makeElement("strong", "video-id", row.video_id)
    );

    const checkboxLabel = makeElement("label", "pick");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.dataset.rank = String(row.rank);
    checkboxLabel.append(checkbox, document.createTextNode(" chọn"));

    const metrics = makeElement("dl", "metrics");
    const values = [
      ["Frame", row.frame_idx],
      ["Time", `${Number(row.pts_time).toFixed(2)}s`],
      ["Score", Number(row.score).toFixed(4)],
      ["Keyframe", String(row.keyframe_no).padStart(3, "0")],
    ];
    for (const [label, value] of values) {
      metrics.append(
        makeElement("dt", "", label),
        makeElement("dd", "", String(value))
      );
    }

    const title = makeElement("p", "title", row.title || "Không có metadata title");
    body.append(headingRow, checkboxLabel, metrics, title);
    card.append(image, body);
    resultsRoot.append(card);
  }
  actions.hidden = results.length === 0;
}

function setChecked(value) {
  document.querySelectorAll(".pick input").forEach((input) => {
    input.checked = value;
  });
}

function csvCell(value) {
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function exportSelected() {
  const ranks = new Set(
    [...document.querySelectorAll(".pick input:checked")].map(
      (input) => Number(input.dataset.rank)
    )
  );
  const rows = currentResults.filter((row) => ranks.has(row.rank));
  if (!rows.length) {
    statusText.textContent = "Hãy chọn ít nhất một kết quả trước khi tải CSV.";
    return;
  }
  const csv = rows
    .map((row) => [row.video_id, row.frame_idx].map(csvCell).join(","))
    .join("\n") + "\n";
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "kis_submission.csv";
  link.click();
  URL.revokeObjectURL(link.href);
  statusText.textContent = `Đã xuất ${rows.length} đáp án KIS.`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  searchButton.disabled = true;
  actions.hidden = true;
  resultsRoot.replaceChildren();
  statusText.textContent = "Đang encode query và tìm trong FAISS…";
  const variant = variantInput.value.trim();
  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: queryInput.value.trim(),
        variants: variant ? [variant] : [],
        top_k: Number(topKInput.value),
        per_video: Number(perVideoInput.value),
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Search failed");
    renderResults(payload.results);
    statusText.textContent = `Tìm thấy ${payload.count} kết quả. Chọn các frame muốn nộp.`;
  } catch (error) {
    statusText.textContent = `Lỗi: ${error.message}`;
  } finally {
    searchButton.disabled = false;
  }
});

document.querySelector("#select-all").addEventListener("click", () => setChecked(true));
document.querySelector("#clear-all").addEventListener("click", () => setChecked(false));
document.querySelector("#export").addEventListener("click", exportSelected);
