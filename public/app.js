const fileInput = document.getElementById("file");
const player = document.getElementById("player");
const engineSelect = document.getElementById("engine");
const languageSelect = document.getElementById("language");
const button = document.getElementById("go");
const statusEl = document.getElementById("status");
const resultCard = document.getElementById("result");

let options = null; // engines + languages, loaded from the backend

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.className = isError ? "error" : "";
}

// Fill the language dropdown with the languages the chosen engine supports.
function updateLanguages() {
  const engine = options.engines[engineSelect.value];
  languageSelect.innerHTML = '<option value="auto">Auto-detect</option>';
  for (const code of engine.languages) {
    languageSelect.add(new Option(options.languages[code], code));
  }
  if (!engine.available) {
    setStatus("Sarvam is not available: set SARVAM_API_KEY in backend/.env and restart.", true);
  } else {
    setStatus("");
  }
  button.disabled = !fileInput.files[0] || !engine.available;
}

async function loadOptions() {
  try {
    const response = await fetch("/api/options");
    options = await response.json();
    updateLanguages();
  } catch {
    setStatus("Cannot reach the backend. Is `uvicorn app:app --port 8000` running?", true);
  }
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  player.src = URL.createObjectURL(file);
  player.hidden = false;
  if (options) updateLanguages();
});

engineSelect.addEventListener("change", updateLanguages);

button.addEventListener("click", async () => {
  const form = new FormData();
  form.append("file", fileInput.files[0]);
  form.append("engine", engineSelect.value);
  form.append("language", languageSelect.value);

  button.disabled = true;
  resultCard.hidden = true;
  setStatus("Transcribing… (the first Whisper request also loads the model)");

  try {
    const response = await fetch("/api/transcribe", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    showResult(data);
    setStatus("");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    button.disabled = false;
  }
});

function showResult(data) {
  const language = options.languages[data.language] || data.language;
  document.getElementById("info").textContent =
    `${language} · ${data.duration}s audio · ${data.processing_seconds}s to transcribe · ${data.engine} (${data.model})`;

  const text = document.getElementById("text");
  text.textContent = data.text || "(no speech detected)";
  text.lang = data.language; // helps the browser pick the right font for the script

  const segments = document.getElementById("segments");
  segments.innerHTML = "";
  for (const seg of data.segments) {
    const div = document.createElement("div");
    div.className = "seg";
    div.innerHTML = `<span>${seg.start.toFixed(1)}s</span>`;
    div.append(seg.text);
    segments.append(div);
  }
  resultCard.hidden = false;
}

loadOptions();
