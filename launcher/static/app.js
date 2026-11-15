// Baja MV launcher page. Plain JS, no build step. Talks to launcher/server.py.
// The forms are filled once (and after each save) so a poll never overwrites what the user is typing; the
// status pills, logs and roster line are refreshed every 2 s.
"use strict";

let state = null;
let formsFilled = false;
let pin = localStorage.getItem("bajamv-pin") || "";
let cameraList = [];

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------------------------- HTTP helpers
async function api(method, url, body) {
  const options = { method, headers: { "X-Launcher-Pin": pin } };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(url, options);
  if (response.status === 401) {
    pin = await askPin();
    return api(method, url, body);
  }
  let data = {};
  try { data = await response.json(); } catch (e) { data = { ok: false, message: "resposta inválida" }; }
  data.httpOk = response.ok;
  return data;
}

function askPin() {
  return new Promise((resolve) => {
    $("pin-dialog").classList.remove("hidden");
    $("pin-input").focus();
    $("pin-ok").onclick = () => {
      const value = $("pin-input").value;
      localStorage.setItem("bajamv-pin", value);
      $("pin-dialog").classList.add("hidden");
      resolve(value);
    };
  });
}

function showMessage(text, isError) {
  const box = $("message");
  box.textContent = text;
  box.classList.toggle("error", !!isError);
  box.classList.remove("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---------------------------------------------------------------------------------------------- state
async function refresh() {
  try {
    const response = await fetch("/api/state");
    state = await response.json();
  } catch (e) {
    $("build").textContent = "sem conexão com o Baja MV";
    return;
  }
  render();
}

function render() {
  const build = state.build || {};
  $("build").textContent = "versão " + state.version + (build.built ? " (" + build.built + ")" : "") +
    " · modelos: " + (state.models_note || "?") + (build.dummy_models ? " · MODELOS DE TESTE (falsos)" : "");
  $("data-root").textContent = "Dados em " + state.data_root + " · este PC: " +
    (state.addresses.join(", ") || state.hostname) + ":" + state.launcher_port;
  for (const role of ["lapline", "zone"]) {
    $("card-" + role).classList.toggle("hidden", !state.launcher.roles.includes(role));
    renderProcess(role);
  }
  $("card-characterize").classList.toggle("hidden", !state.launcher.roles.includes("zone"));
  $("bench-warning").classList.toggle("hidden", !state.zone.bench);
  renderRoster(state.lapline.roster);
  $("char-log").textContent = (state.characterization.log || []).join("\n");
  if (!formsFilled) fillForms();
}

function renderProcess(role) {
  const process = state.processes[role];
  const pill = $("pill-" + role);
  pill.className = "pill";
  if (process.running) {
    pill.textContent = "rodando" + (process.adopted ? " (retomado)" : "") + " · " + formatUptime(process.uptime_s);
    pill.classList.add("running");
  } else if (process.exit_code !== null && process.exit_code !== undefined && !process.stopped_by_user) {
    pill.textContent = "parou (código " + process.exit_code + ")";
    pill.classList.add("failed");
  } else {
    pill.textContent = "parado";
  }
  const port = role === "lapline" ? state.lapline.settings.status_port : state.zone.settings.web_port;
  const link = $("link-" + role);
  link.href = location.protocol + "//" + location.hostname + ":" + port + "/";
  link.textContent = (role === "lapline" ? "Página de status" : "Página da zona") + " (:" + port + ")";
  $("log-" + role).textContent = process.log.join("\n");
}

function formatUptime(seconds) {
  if (seconds === null || seconds === undefined) return "";
  const minutes = Math.floor(seconds / 60);
  return minutes < 60 ? minutes + " min" : Math.floor(minutes / 60) + " h " + (minutes % 60) + " min";
}

function renderRoster(roster) {
  let text = "Lista: ";
  if (!roster.event) text += "informe o evento";
  else if (!roster.where) text += "ainda não há lista para " + roster.event + " (lê qualquer número)";
  else text += roster.cars + " carros (" + roster.where + ")";
  $("roster-status").textContent = text;
}

// ---------------------------------------------------------------------------------------------- forms
function fillSelect(select, values, labels) {
  select.innerHTML = "";
  values.forEach((value, i) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = labels ? labels[i] : value;
    select.appendChild(option);
  });
}

function fillForms() {
  formsFilled = true;
  $("role-lapline").checked = state.launcher.roles.includes("lapline");
  $("role-zone").checked = state.launcher.roles.includes("zone");
  $("autostart").checked = state.launcher.autostart;
  $("pin").placeholder = state.launcher.pin_set ? "PIN definido (deixe vazio para manter)" : "sem PIN";
  for (const role of ["lapline", "zone"]) {
    const form = $("form-" + role);
    fillSelect(form.elements.resolution, state.choices.resolutions);
    const settings = state[role].settings;
    for (const [name, value] of Object.entries(settings)) {
      if (form.elements[name] && name !== "ptz_report") form.elements[name].value = value;
    }
    form.querySelector(".config-path").textContent = state[role].config_path +
      (state[role].config_exists ? "" : " (ainda não criado)");
    showErrors(form, state[role].errors || {});
  }
  const reports = state.reports.map((r) => r.path);
  const current = state.zone.settings.ptz_report;
  if (current && !reports.includes(current)) reports.unshift(current);
  fillSelect($("form-zone").elements.ptz_report, [""].concat(reports),
    ["nenhum (Branch B, valores padrão)"].concat(reports));
  $("form-zone").elements.ptz_report.value = current;
  $("char-ip").value = state.zone.settings.ptz_host;
  updateBenchWarning();
  fillCameraSelects();
}

function showErrors(form, errors) {
  form.querySelectorAll(".field-error").forEach((el) => { el.textContent = errors[el.dataset.for] || ""; });
}

function formValues(form) {
  const values = {};
  for (const element of form.elements) {
    if (element.name && element.name !== "camera_choice") values[element.name] = element.value;
  }
  return values;
}

async function saveRole(role, event) {
  event.preventDefault();
  const form = $("form-" + role);
  const result = await api("POST", "/api/settings/" + role, formValues(form));
  showErrors(form, result.errors || {});
  if (result.ok) {
    showMessage("Configurações salvas. Reinicie o papel para aplicar.");
    formsFilled = false;
    refresh();
  } else {
    showMessage(result.message || "Corrija os campos marcados.", true);
  }
}

function updateBenchWarning() {
  const profile = $("form-zone").elements.profile.value;
  $("bench-warning-form").classList.toggle("hidden", !state.choices.bench_profiles.includes(profile));
}

// ---------------------------------------------------------------------------------------------- cameras
async function loadCameras() {
  const result = await api("GET", "/api/cameras");
  cameraList = result.cameras || [];
  if (result.error) showMessage("Lista de câmeras: " + result.error, true);
  fillCameraSelects();
}

function fillCameraSelects() {
  for (const role of ["lapline", "zone"]) {
    const form = $("form-" + role);
    const select = form.elements.camera_choice;
    const values = [""].concat(cameraList.map((c) => c.device));
    const labels = ["(escolha ou digite abaixo)"].concat(cameraList.map((c) => c.index + ": " + c.name));
    fillSelect(select, values, labels);
    select.value = values.includes(form.elements.camera_device.value) ? form.elements.camera_device.value : "";
    select.onchange = () => { if (select.value) form.elements.camera_device.value = select.value; };
  }
}

async function preview(role) {
  const form = $("form-" + role);
  const img = $("preview-" + role);
  const url = "/api/cameras/snapshot?device=" + encodeURIComponent(form.elements.camera_device.value) +
    "&resolution=" + encodeURIComponent(form.elements.resolution.value) + "&t=" + Date.now();
  const response = await fetch(url);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    showMessage(data.message || "Prévia falhou.", true);
    return;
  }
  img.src = URL.createObjectURL(await response.blob());
  img.classList.remove("hidden");
}

// ---------------------------------------------------------------------------------------------- wiring
function wire() {
  $("save-launcher").onclick = async () => {
    const roles = ["lapline", "zone"].filter((r) => $("role-" + r).checked);
    const newPin = $("pin").value;
    const body = { roles, autostart: $("autostart").checked };
    // empty field = keep the PIN (the key is left out); to remove a PIN type a single space
    if (newPin !== "") body.pin = newPin.trim();
    const result = await api("POST", "/api/settings/launcher", body);
    if (result.ok) {
      if (body.pin !== undefined) { pin = body.pin; localStorage.setItem("bajamv-pin", pin); }
      $("pin").value = "";
      showMessage("Salvo.");
      formsFilled = false;
      refresh();
    } else {
      showMessage(JSON.stringify(result.errors || result.message), true);
    }
  };
  $("form-lapline").onsubmit = (e) => saveRole("lapline", e);
  $("form-zone").onsubmit = (e) => saveRole("zone", e);
  $("form-zone").elements.profile.onchange = updateBenchWarning;
  document.querySelectorAll("button[data-role]").forEach((button) => {
    button.onclick = async () => {
      const { role, action } = button.dataset;
      if (action === "stop" && !confirm("Parar " + state.role_names[role] + "?")) return;
      button.disabled = true;
      const result = await api("POST", "/api/roles/" + role + "/" + action);
      button.disabled = false;
      showMessage(result.message || (result.ok ? "ok" : "falhou"), !result.ok);
      refresh();
    };
  });
  $("fetch-roster").onclick = async () => {
    const event = $("form-lapline").elements.event.value;
    showMessage("Buscando a lista de " + event + "...");
    const result = await api("POST", "/api/roster/fetch", { event });
    showMessage(result.message, !result.ok);
    renderRoster(result.roster);
  };
  document.querySelectorAll(".refresh-cameras").forEach((b) => { b.onclick = loadCameras; });
  document.querySelectorAll(".preview").forEach((b) => { b.onclick = () => preview(b.dataset.previewFor); });
  $("open-data").onclick = async () => {
    const result = await api("POST", "/api/open-data-folder");
    showMessage(result.message, !result.ok);
  };
  $("export-logs").onclick = async () => {
    const result = await api("POST", "/api/export");
    if (!result.ok) { showMessage(result.message || "falhou", true); return; }
    const link = $("export-link");
    link.href = "/api/export/" + encodeURIComponent(result.name);
    link.textContent = "Baixar " + result.name;
    link.classList.remove("hidden");
    showMessage("Logs exportados para " + result.path);
  };
  $("char-start").onclick = async () => {
    if (!confirm("A câmera PTZ vai se mover. Confirme na janela que abrir neste computador. Continuar?")) return;
    const result = await api("POST", "/api/characterize",
      { ip: $("char-ip").value, folder: $("char-folder").value, quick: $("char-quick").checked });
    showMessage(result.message, !result.ok);
  };
}

wire();
refresh().then(loadCameras);
setInterval(refresh, 2000);
