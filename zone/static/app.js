// Zona PTZ page: polls /api/state, refreshes /frame.jpg, sends teach / mode actions. No build step.
"use strict";
const $ = (s) => document.querySelector(s);
let S = null;              // last /api/state
let clickMode = null;      // "path" | "keyframe" | "move" | null
let speedLevel = 1;        // 0 slow, 1 medium, 2 fast
let jogTimer = null;

// why the director let a car go (zone/director.py, runtime): pt-BR for the header and the events list.
// ok = a normal end of a pass; the others are drops, shown in red
const DROP_REASONS = {
  exit: ["saiu da zona", true],
  lost: ["PERDIDO: sem detecção por mais de director.lost_timeout_s", false],
  stopped: ["PARADO: quase sem movimento por mais de director.stopped_timeout_s", false],
  off_corridor: ["FORA DO CORREDOR por mais de director.off_corridor_frames quadros", false],
  gone: ["o track sumiu (rastreador)", false],
  diverged: ["DIVERGIU: câmera longe demais do alvo (controller.divergence_deg)", false],
  wrong_direction: ["PTZ GIRANDO AO CONTRÁRIO: sinal de pan/tilt errado (ptz.pan_sign)", false],
};
const SIGN_HOW = { config: "site.toml", teach: "medido no ensino", report: "relatório da bancada",
                   default: "PADRÃO, não medido" };

function dropText(reason) {
  const r = DROP_REASONS[reason];
  return r ? r[0] : reason;
}

function msg(t, isErr) {
  const m = $("#msg");
  m.textContent = t || "";
  m.style.color = isErr ? "#ff6b5e" : "var(--amber)";
  if (t) setTimeout(() => { if (m.textContent === t) m.textContent = ""; }, 6000);
}

async function post(url, body) {
  try {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                                 body: JSON.stringify(body || {}) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { msg(j.detail || ("erro " + r.status), true); return null; }
    return j;
  } catch (e) { msg("sem conexão com o programa", true); return null; }
}

// ------------------------------------------------------------------ frame
function refreshFrame() {
  const img = $("#frame");
  const next = new Image();
  next.onload = () => { img.src = next.src; setTimeout(refreshFrame, 150); };
  next.onerror = () => setTimeout(refreshFrame, 1000);
  next.src = "/frame.jpg?t=" + Date.now();
}

$("#frame").addEventListener("click", async (ev) => {
  if (!clickMode || !S || !S.frame.size) return;
  const img = ev.target, rect = img.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * S.frame.size[0] / rect.width;
  const y = (ev.clientY - rect.top) * S.frame.size[1] / rect.height;
  if (clickMode === "move") {
    if (S.teach.selected === null || S.teach.selected === undefined) { msg("selecione um ponto na lista primeiro"); return; }
    await post("/api/teach/move", { index: S.teach.selected, x, y });
  } else {
    const r = await post("/api/teach/point", { x, y, keyframe: clickMode === "keyframe", zoom: typedZoom() });
    if (r && clickMode === "keyframe") msg("keyframe gravado no preset " + r.result.keyframe.preset);
  }
  setClickMode(null);
  poll();
});

function setClickMode(m) {
  clickMode = m;
  document.querySelectorAll("[data-click]").forEach((b) => b.classList.toggle("on", b.dataset.click === m));
  const h = $("#clickHint");
  h.style.display = m ? "block" : "none";
  h.textContent = { path: "clique no caminho", keyframe: "clique onde a PTZ está enquadrando",
                    move: "clique na nova posição" }[m] || "";
}
document.querySelectorAll("[data-click]").forEach((b) =>
  b.addEventListener("click", () => setClickMode(clickMode === b.dataset.click ? null : b.dataset.click)));

// ------------------------------------------------------------------ modes
document.querySelectorAll("[data-mode]").forEach((b) =>
  b.addEventListener("click", async () => { await post("/api/mode", { mode: b.dataset.mode }); poll(); }));
async function bigStop() { stopJog(); await post("/api/stop"); msg("PARADO"); poll(); }
$("#bigStop").addEventListener("click", bigStop);

// ------------------------------------------------------------------ jog
document.querySelectorAll("[data-speed]").forEach((b) => b.addEventListener("click", () => {
  speedLevel = +b.dataset.speed;
  document.querySelectorAll("[data-speed]").forEach((x) => x.classList.toggle("on", x === b));
}));

function startJog(dp, dt, dz) {
  if (!S) return;
  const t = S.teach;
  if (!t.velocity_jog) {           // absolute "nudges": one per press
    const u = t.nudge_units[speedLevel], zu = t.zoom_nudge_units[speedLevel];
    post("/api/teach/nudge", { pan: dp * u, tilt: dt * u, zoom: dz * zu, speed: t.jog_speeds[speedLevel] });
    return;
  }
  const step = t.jog_speeds[speedLevel];
  const body = { pan: dp * step, tilt: dt * step, zoom: dz * t.jog_zoom_speed };
  stopJog(true);
  post("/api/teach/jog", body);
  jogTimer = setInterval(() => post("/api/teach/jog", body), 150);   // the PTZ TTL stops it if this stops
}
function stopJog(silent) {
  if (jogTimer) { clearInterval(jogTimer); jogTimer = null; if (!silent) post("/api/teach/jog_stop"); }
}
function bindHold(el, dp, dt, dz) {
  el.addEventListener("pointerdown", (e) => { e.preventDefault(); startJog(dp, dt, dz); });
  ["pointerup", "pointerleave", "pointercancel"].forEach((k) => el.addEventListener(k, () => stopJog()));
}
document.querySelectorAll("[data-jog]").forEach((b) => {
  const [p, t] = b.dataset.jog.split(",").map(Number);
  bindHold(b, p, t, 0);
});
document.querySelectorAll("[data-zoom]").forEach((b) => bindHold(b, 0, 0, +b.dataset.zoom));
$("#jogStop").addEventListener("click", () => { stopJog(true); post("/api/teach/jog_stop"); });
$("#home").addEventListener("click", () => post("/api/teach/home"));
$("#teachMode").addEventListener("click", async () => {
  if (!S) return;
  stopJog(true);
  await post("/api/teach/mode", { on: !S.teach.teach_mode });
  poll();
});
$("#zeroHere").addEventListener("click", () => post("/api/teach/zero_here"));

const KEYS = { ArrowLeft: [-1, 0, 0], ArrowRight: [1, 0, 0], ArrowUp: [0, 1, 0], ArrowDown: [0, -1, 0],
               PageUp: [0, 0, 1], PageDown: [0, 0, -1] };
let keyDown = null;
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "Escape") { e.preventDefault(); bigStop(); return; }
  const k = KEYS[e.key];
  if (!k || keyDown === e.key || !S || !S.teach.teach_mode) return;
  e.preventDefault();
  keyDown = e.key;
  startJog(...k);
});
document.addEventListener("keyup", (e) => { if (e.key === keyDown) { keyDown = null; stopJog(); } });

// ------------------------------------------------------------------ teach list
// The corridor box is refreshed from the program twice a second. Clicking "Aplicar" moves the focus off the box
// first, so a refresh landing between that and the click used to put the old width (60) back and send it. Once
// the operator types in the box, the refresh leaves it alone until "Aplicar" succeeds.
let corridorEdited = false;
$("#corridor").addEventListener("input", () => { corridorEdited = true; });
$("#setCorridor").addEventListener("click", async () => {
  const r = await post("/api/teach/corridor", { half_width: +$("#corridor").value });
  if (r) { corridorEdited = false; msg("corredor aplicado (salve a zona para manter)"); }
});
$("#save").addEventListener("click", async () => {
  const r = await post("/api/teach/save");
  if (r) msg("zona salva" + (r.result ? " (backup: " + r.result.split(/[\\/]/).pop() + ")" : ""));
  poll();
});

async function act(url, index, confirmText) {
  if (confirmText && !confirm(confirmText)) return;
  await post(url, { index, zoom: typedZoom() });
  poll();
}

// cameras without a zoom inquiry: the operator types the zoom of each keyframe (null = not typed)
function typedZoom() {
  const v = $("#kfZoom").value;
  return S && S.teach.zoom_unknown && v !== "" ? +v : null;
}

function renderPoints(z, t) {
  const tb = $("#points tbody");
  tb.innerHTML = "";
  z.points.forEach((p) => {
    const tr = document.createElement("tr");
    if (t.selected === p.i) tr.className = "sel";
    const kf = p.keyframe;
    const kind = kf ? `keyframe P${kf.preset}${kf.approximate ? " ~" : ""}` : "caminho";
    tr.innerHTML = `<td>${p.i}</td><td title="${kf ? `pan ${kf.pan} tilt ${kf.tilt} zoom ${kf.zoom}` : ""}">${kind}</td>
      <td>${Math.round(p.x)},${Math.round(p.y)}</td><td></td>`;
    const cell = tr.lastElementChild;
    const add = (label, fn, title) => {
      const b = document.createElement("button"); b.textContent = label; b.title = title || "";
      b.addEventListener("click", (e) => { e.stopPropagation(); fn(); }); cell.appendChild(b);
    };
    add("↑", () => post("/api/teach/reorder", { index: p.i, delta: -1 }).then(poll), "subir");
    add("↓", () => post("/api/teach/reorder", { index: p.i, delta: 1 }).then(poll), "descer");
    if (kf) {
      add("testar", () => act("/api/teach/goto", p.i), "chamar o preset deste keyframe");
      add("re-ensinar", () => act("/api/teach/reteach", p.i, "Gravar o enquadramento atual neste keyframe?"),
          "gravar o enquadramento atual");
      add("→caminho", () => act("/api/teach/unkeyframe", p.i), "virar ponto de caminho");
      if (t.zoom_unknown || (kf.pose_source || "").includes("zoom_manual")) {
        add("zoom", async () => {
          const v = prompt(`Zoom do keyframe P${kf.preset} (${t.zoom_limits[0]}..${t.zoom_limits[1]})`, kf.zoom);
          if (v !== null && v !== "") { await post("/api/teach/keyframe_zoom", { index: p.i, zoom: +v }); poll(); }
        }, "digitar o zoom deste keyframe");
      }
    } else {
      add("→keyframe", () => act("/api/teach/make_keyframe", p.i, "Usar o enquadramento atual neste ponto?"),
          "usar o enquadramento atual");
    }
    add("✕", () => act("/api/teach/delete", p.i, `Excluir o ponto ${p.i}?`), "excluir");
    tr.addEventListener("click", () => post("/api/teach/select", { index: t.selected === p.i ? null : p.i }).then(poll));
    tb.appendChild(tr);
  });
}

// a "reacquired" event (zone/director.py): the followed car was seen again after a dropout
function reacquiredText(e) {
  const how = { same_id: "mesmo id", new_id: `novo id (era ${e.old_id})`, raw_detection: "detecção avulsa" }[e.how] || e.how;
  return `reencontrado após ${Number(e.dropout_s).toFixed(1)} s sem detecção (${how})`;
}

// ------------------------------------------------------------------ state
function kv(el, rows) {
  el.innerHTML = rows.map(([k, v, cls]) => `<div>${k}</div><div class="${cls || ""}">${v}</div>`).join("");
}

function render(s) {
  S = s;
  const d = s.runtime.director;
  const badge = $("#modeBadge");
  badge.textContent = { STOPPED: "PARADO", PASSIVE: "PASSIVO", AUTO: "AUTO" }[d.mode] || d.mode;
  badge.className = "state " + d.mode;
  $("#subState").textContent = d.sub ? ({ waiting: "aguardando carro", tracking: "seguindo",
                                           returning: "voltando à entrada" }[d.sub] || d.sub) : "";
  const L = s.runtime.locked;
  // while the car isn't detected its position is predicted (zone/progress.py): say for how long, whether the
  // prediction is holding at its cap (the camera waits there), and how much the zoom was opened meanwhile
  const pred = L && L.predicted ? `  prevendo (sem detecção) ${(L.age || 0).toFixed(1)} s` +
    (L.held ? ", aguardando no limite" : "") + (L.zoom_extra > 0 ? `, zoom +${Math.round(L.zoom_extra * 100)}% aberto` : "") : "";
  $("#locked").textContent = L ? `carro ${L.id}  s=${L.s.toFixed(2)}  v=${L.v.toFixed(2)}/s${pred}` : "";
  $("#locked").className = L && L.predicted ? "warn" : "";
  const ld = d.last_drop, ldEl = $("#lastDrop");
  ldEl.textContent = ld ? `último: ${dropText(ld.reason)} (s=${ld.s === null ? "?" : ld.s.toFixed(2)}, seguido ${ld.held_s} s)` : "";
  ldEl.className = ld && !(DROP_REASONS[ld.reason] || [0, false])[1] ? "bad" : "";
  document.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("on", b.dataset.mode === d.mode));
  document.querySelector('[data-mode="AUTO"]').disabled = !s.runtime.zone_ready;
  renderStudio(s.agent);

  const z = s.zone, t = s.teach;
  if (!corridorEdited && document.activeElement !== $("#corridor")) $("#corridor").value = z.corridor;
  $("#dirty").textContent = z.dirty ? "alterações não salvas" : "salvo";
  $("#dirty").className = z.dirty ? "bad" : "good";
  $("#poseSrc").textContent = { inquiry: "consulta à câmera", commanded: "movimento absoluto",
                                dead_reckoning: "estimada (aproximada)" }[t.pose_source] || t.pose_source;
  $("#zoneFile").textContent = z.file.split(/[\\/]/).pop();
  $("#poseButtons").style.display = t.velocity_jog && t.pose_source !== "dead_reckoning" ? "none" : "flex";
  $("#home").style.display = t.pose_source === "commanded" ? "" : "none";
  $("#zeroHere").style.display = t.pose_source === "dead_reckoning" ? "" : "none";
  document.querySelectorAll("[data-click],#save,#setCorridor,#teachMode").forEach((b) => b.disabled = !t.allowed);
  // moving the PTZ from the page needs teach mode (and never in AUTO)
  const canMove = t.allowed && t.teach_mode;
  document.querySelectorAll("[data-jog],[data-zoom],#jogStop,#home").forEach((b) => b.disabled = !canMove);
  if (!canMove) stopJog(true);
  const tm = $("#teachMode");
  tm.textContent = "Modo ensino: " + (t.teach_mode ? "LIGADO" : "desligado");
  tm.classList.toggle("on", t.teach_mode);
  $("#zoomWarn").style.display = t.zoom_unknown ? "flex" : "none";
  $("#kfZoom").min = t.zoom_limits[0]; $("#kfZoom").max = t.zoom_limits[1];
  renderPoints(z, t);

  const h = s.health;
  const gb = h.disk_free_gb;
  const sr = h.spotter_recording, pr = h.ptz_recording;
  // the bench-only COCO detector must not be left on at the race: a banner in the header + a red health line
  const det = h.detector;
  $("#benchWarn").style.display = det && det.bench ? "" : "none";
  kv($("#health"), [
    ["detector", det ? (det.bench ? `DETECTOR DE BANCADA (${det.kind}) – não usar na prova · ${det.desc}`
                                  : `${det.kind} · ${det.desc}`) : "–", det && det.bench ? "bad" : ""],
    ["processamento", `${h.fps} fps (descartados ${h.dropped})`, h.fps < 7 ? "bad" : "good"],
    ["último quadro", `${s.frame.age_s} s atrás`, s.frame.age_s > 2 ? "bad" : ""],
    ["PTZ", `${h.ptz_driver} · ${h.ptz_reachable === null || h.ptz_reachable === undefined
                                  ? "desconhecido (nada respondeu ainda)"
                                  : (h.ptz_reachable ? "respondendo" : "SEM RESPOSTA")}` +
            (h.ptz_reachable_how === "last_command" ? " (último comando)" : ""),
     h.ptz_reachable === false ? "bad" : (h.ptz_reachable ? "good" : "")],
    ["ramo de controle", `${h.branch} · ${h.report.reason || h.report.path || "sem relatório"}`],
    ["sinal pan / tilt", h.ptz_signs ? ["pan", "tilt"].map((a) => `${a} ${h.ptz_signs[a].sign > 0 ? "+1" : "-1"} (${SIGN_HOW[h.ptz_signs[a].how] || h.ptz_signs[a].how})`).join(" · ")
                         + (h.sign_warnings && h.sign_warnings.length ? " · " + h.sign_warnings.join(" · ") : "") : "–",
     h.sign_warnings && h.sign_warnings.length ? "bad" : ""],
    ["relógio (NTP)", h.ntp_server ? (h.ntp_offset_s === null ? "sem resposta de " + h.ntp_server
                                     : `${h.ntp_offset_s >= 0 ? "+" : ""}${h.ntp_offset_s} s vs ${h.ntp_server}`)
                                  : (h.clock_warnings.length ? h.clock_warnings.join(" ") : "não configurado"),
     (h.ntp_offset_s !== null && Math.abs(h.ntp_offset_s) > 0.5) || h.clock_warnings.length ? "bad" : ""],
    ["disco livre", gb === null ? "?" : gb + " GB", gb !== null && gb < 5 ? "bad" : ""],
    ["gravação spotter", sr ? `${sr.alive ? "gravando" : "PARADA"} ${sr.file} (${sr.written} quadros, ${sr.dropped} perdidos)` : "desligada",
     sr && !sr.alive ? "bad" : ""],
    ["gravação PTZ", pr.enabled ? (pr.file ? `${pr.growing ? "gravando" : "PARADA"} ${pr.file} ${pr.size_mb} MB` : "nenhum arquivo")
                                : "desligada", pr.enabled && !pr.growing ? "bad" : ""],
    ["perfil de velocidade", `${s.runtime.profile_passes} passagens`],
    ["fonte", h.source || "–"],
  ]);
  $("#events").innerHTML = s.events.slice().reverse().map((e) =>
    `<div class="${e.event === "release" && e.reason !== "exit" || e.type === "direction_mismatch" || e.type === "divergence" ? "bad" : ""}">${e.wall_iso} ${e.event || e.type} ${e.event === "release" ? dropText(e.reason) : (e.reason || e.why || "")} ${e.track_id !== undefined ? "carro " + e.track_id : ""}
     ${e.s !== undefined && e.s !== null ? "s=" + e.s : ""} ${e.held_s !== undefined ? "após " + e.held_s + " s" : ""}
     ${e.type === "divergence" ? `gap pan ${e.gap_pan} tilt ${e.gap_tilt}` : ""}
     ${e.event === "reacquired" ? reacquiredText(e) : ""}
     ${e.new ? "→ " + e.new : ""} ${e.msg || ""}</div>`).join("");
}

// the studio (baja-studio) drives this PTZ through the agent API: show what it is doing, and who
function renderStudio(a) {
  const el = $("#studioBadge");
  $("#releaseHold").style.display = a && a.enabled && a.mode === "hold" ? "" : "none";
  if (a && a.error) {      // the studio's API is down (port taken, websockets missing...): impossible to miss
    el.textContent = "API DO ESTÚDIO FORA DO AR: " + a.error;
    el.style.background = "var(--red)"; el.style.color = "#fff"; el.style.display = "";
    return;
  }
  if (!a || !a.enabled) { el.style.display = "none"; return; }
  const who = a.holder ? a.holder.replace(/^studio:/, "") : "estúdio";
  let text = "Estúdio: manual", bg = "#333", fg = "var(--text)";
  if (a.mode === "auto") { text = "Estúdio: AUTO"; bg = "var(--green)"; }
  else if (a.mode === "auto_paused") {
    text = `PAUSADO PELO ESTÚDIO (${who})` + (a.resume_in_s !== null ? ` · volta ao AUTO em ${a.resume_in_s.toFixed(1)} s` : "");
    bg = "var(--amber)"; fg = "#000";
  } else if (a.mode === "hold") {
    text = `ESTÚDIO SEGURA A CÂMERA (HOLD · ${who})`; bg = "var(--amber)"; fg = "#000";
  } else if (a.studio_moving) { text = `Estúdio movendo a câmera (${who})`; bg = "var(--blue)"; }
  el.textContent = text + (a.studios ? "" : " · nenhum estúdio conectado");
  el.style.background = bg; el.style.color = fg; el.style.display = "";
  // in HOLD only the studio may bring AUTO back (the contract): the page's AUTO is refused
  const auto = document.querySelector('[data-mode="AUTO"]');
  if (a.mode === "hold") { auto.disabled = true; auto.title = "O estúdio está segurando a câmera (HOLD): só o estúdio volta ao AUTO"; }
  else auto.title = "";
}

// "Liberar HOLD": the person at the laptop takes the camera back from a studio that holds it (a studio that is gone,
// or a HOLD remembered across a restart). Deliberate: a confirmation first.
$("#releaseHold").addEventListener("click", async () => {
  const who = S && S.agent && S.agent.holder ? S.agent.holder.replace(/^studio:/, "") : "o estúdio";
  if (!confirm(`O estúdio (${who}) está segurando esta câmera (HOLD).\n\nLiberar o HOLD? A câmera passa para manual ` +
               "e o AUTO desta página volta a funcionar. Avise o estúdio.")) return;
  if (await post("/api/agent/release_hold", { confirm: true })) msg("HOLD liberado: câmera em manual");
  poll();
});

async function poll() {
  try {
    const r = await fetch("/api/state");
    if (r.ok) render(await r.json());
  } catch (e) { $("#modeBadge").textContent = "SEM CONEXÃO"; }
}

async function pollPose() {
  if (S && S.teach.allowed) {
    try {
      const r = await fetch("/api/teach/pose");
      const j = await r.json();
      $("#poseNow").textContent = j.result ? `pan ${j.result[0]} · tilt ${j.result[1]} · zoom ${j.result[2] === null ? "?" : j.result[2]}` : "desconhecida";
    } catch (e) { /* ignore */ }
  }
  setTimeout(pollPose, 1000);
}

setInterval(poll, 500);
poll();
refreshFrame();
pollPose();
