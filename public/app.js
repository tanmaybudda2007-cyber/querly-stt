const fileInput = document.getElementById("file");
const player = document.getElementById("player");
const engineSelect = document.getElementById("engine");
const languageSelect = document.getElementById("language");
const button = document.getElementById("go");
const statusEl = document.getElementById("status");
const resultCard = document.getElementById("result");

const llmProvider = document.getElementById("llm-provider");
const llmModel = document.getElementById("llm-model");
const llmTemperature = document.getElementById("llm-temperature");
const llmMaxTokens = document.getElementById("llm-max-tokens");
const llmPrompt = document.getElementById("llm-prompt");
const llmAuto = document.getElementById("llm-auto");
const llmButton = document.getElementById("llm-go");
const llmStatus = document.getElementById("llm-status");

let options = null; // engines, languages and LLM settings, loaded from the backend
let transcription = null; // the latest transcription result

function setStatus(message, isError = false, el = statusEl) {
  el.textContent = message;
  el.className = isError ? "status error" : "status";
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
  let response;
  try {
    response = await fetch("/api/options");
  } catch {
    setStatus("Cannot reach the web server. Is `npm start` running?", true);
    return;
  }
  if (response.status === 502 || response.status === 504) {
    setStatus("The web server can't reach the Python backend. Is `uvicorn app:app --port 8000` running in backend/?", true);
    return;
  }
  if (!response.ok) {
    const detail = (await response.text()).slice(0, 300);
    setStatus(`The backend returned an error (HTTP ${response.status}): ${detail}. See the uvicorn terminal for details.`, true);
    return;
  }
  options = await response.json();
  if (!options.llm) {
    setStatus("The backend is running older code. Restart uvicorn after updating the backend files.", true);
    return;
  }
  updateLanguages();
  setupLlm();
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
    transcription = data;
    llmButton.disabled = false;
    if (llmAuto.checked) runLlm();
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

// ---------- LLM ----------

function setupLlm() {
  const { defaults, providers } = options.llm;
  for (const [name, info] of Object.entries(providers)) {
    const option = new Option(info.available ? name : `${name} (not set up)`, name);
    option.disabled = !info.available;
    option.title = info.note; // e.g. "set GROQ_API_KEY in backend/.env", shown on hover
    llmProvider.add(option);
  }
  // Use the default provider if it's set up, otherwise the first one that is.
  const usable = Object.keys(providers).filter((name) => providers[name].available);
  llmProvider.value = usable.includes(defaults.provider) ? defaults.provider : usable[0] || defaults.provider;
  llmModel.value = defaults.model;
  llmTemperature.value = defaults.temperature;
  llmMaxTokens.value = defaults.max_tokens;
  llmPrompt.value = defaults.prompt;
  updateModelPlaceholder();
}

// An empty model box means "use the provider's default model".
function updateModelPlaceholder() {
  llmModel.placeholder = options.llm.providers[llmProvider.value].default_model;
}
llmProvider.addEventListener("change", () => {
  llmModel.value = "";
  updateModelPlaceholder();
});

async function runLlm() {
  if (!transcription?.text) {
    setStatus("No transcript to process.", true, llmStatus);
    return;
  }
  llmButton.disabled = true;
  document.getElementById("llm-result").hidden = true;
  setStatus("Running LLM…", false, llmStatus);

  try {
    const response = await fetch("/api/llm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        transcript: transcription.text,
        language: transcription.language,
        prompt: llmPrompt.value,
        provider: llmProvider.value,
        model: llmModel.value || null,
        temperature: Number(llmTemperature.value),
        max_tokens: Number(llmMaxTokens.value),
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    document.getElementById("llm-info").textContent = `${data.provider} (${data.model}) · ${data.seconds}s`;
    document.getElementById("llm-output").textContent = data.output || "(empty response)";
    document.getElementById("llm-result").hidden = false;
    setStatus("", false, llmStatus);
  } catch (err) {
    setStatus(err.message, true, llmStatus);
  } finally {
    llmButton.disabled = false;
  }
}

llmButton.addEventListener("click", runLlm);

loadOptions();
