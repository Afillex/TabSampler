// The tab view (ADR 0058). Draws ADR 0013's JSON document as six string lines, high e on top,
// notes placed by onset in proportion to time (ADR 0009), the same layout as the ASCII renderer.
//
// The posteriors are not calibrated (ADR 0057), so no posterior is ever printed as a number:
// a note is either marked uncertain (below the config's threshold) or not, and its tooltip
// gives its rank among the take's notes and the alternatives in rank order.
"use strict";

const SVG = "http://www.w3.org/2000/svg";  // a namespace name, never fetched
const LEFT = 60;     // room for the string labels and a note at 0 s
const TOP = 30;      // room for the time ticks
const ROW = 30;      // distance between strings
const RIGHT = 48;
const STANDARD = [40, 45, 50, 55, 59, 64];
const STANDARD_LABELS = ["E", "A", "D", "G", "B", "e"];
const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

const $ = (id) => document.getElementById(id);
let current = null;  // the last response, redrawn on zoom

function pitchName(midi) {
  return NAMES[midi - 12 * Math.floor(midi / 12)] + (Math.floor(midi / 12) - 1);
}

function seconds(t) {
  return t.toLocaleString("en", { maximumFractionDigits: 2 }) + " s";
}

function stringLabels(tuning) {
  const open = tuning.open_pitches;
  const standard = open.length === 6 && open.every((p, i) => p === STANDARD[i]);
  return standard ? STANDARD_LABELS : open.map((p) => pitchName(p));
}

function el(name, attrs, parent) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function setStatus(text, isError) {
  const status = $("status");
  status.textContent = text;
  status.classList.toggle("error", Boolean(isError));
}

// Rank 1 is the note the decoder is surest of. Ties share the better rank.
function ranks(notes) {
  const order = notes.map((n, i) => [n.posterior, i]).sort((a, b) => b[0] - a[0]);
  const rank = new Array(notes.length);
  order.forEach(([p, i], k) => {
    rank[i] = k > 0 && order[k - 1][0] === p ? rank[order[k - 1][1]] : k + 1;
  });
  return rank;
}

// The narrowest zoom at which at most one in twenty neighbouring notes on a string would overlap
// (a label box is about 30 px wide); the widest zoom when none is that sparse.
const BOX = 30;
function fitZoom(doc) {
  const options = [...$("zoom").options].map((o) => Number(o.value));
  const gaps = [];
  const last = new Map();
  for (const n of [...doc.notes].sort((a, b) => a.onset - b.onset)) {
    if (last.has(n.string)) gaps.push(n.onset - last.get(n.string));
    last.set(n.string, n.onset);
  }
  for (const pps of options) {
    if (gaps.filter((g) => g * pps < BOX).length <= gaps.length / 20) return pps;
  }
  return options[options.length - 1];
}

function draw(doc) {
  const svg = $("tab");
  svg.replaceChildren();
  const pps = Number($("zoom").value);  // pixels per second
  const labels = stringLabels(doc.tuning);
  const nStrings = labels.length;
  const end = doc.notes.reduce((m, n) => Math.max(m, n.offset), 1);
  const width = Math.ceil(LEFT + end * pps + RIGHT);
  const height = TOP + (nStrings - 1) * ROW + 24;
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const y = (string) => TOP + (nStrings - 1 - string) * ROW;  // string 0 is the low E
  const x = (t) => LEFT + t * pps;

  const step = pps >= 160 ? 1 : 2;  // a tick every one or two seconds
  for (let t = 0; t <= end; t += step) {
    el("line", { class: "tick", x1: x(t), x2: x(t), y1: TOP - 10, y2: y(0) + 10 }, svg);
    el("text", { class: "tick-label", x: x(t) + 3, y: TOP - 14 }, svg).textContent = `${t} s`;
  }
  for (let s = 0; s < nStrings; s++) {
    el("line", { class: "string", x1: LEFT - 8, x2: width - 8, y1: y(s), y2: y(s) }, svg);
    el("text", { class: "string-label", x: 16, y: y(s) + 4 }, svg).textContent = labels[s];
  }

  const rank = ranks(doc.notes);
  const threshold = doc.uncertainty_threshold;
  doc.notes.forEach((n, i) => {
    const unsure = n.posterior < threshold;
    const label = unsure ? `(${n.fret})` : String(n.fret);
    const w = 10 + 8 * label.length;
    const g = el("g", { class: unsure ? "note uncertain" : "note", tabindex: "0" }, svg);
    g.setAttribute(
      "aria-label",
      `${pitchName(n.pitch)} at ${seconds(n.onset)}: ${labels[n.string]} string, fret ${n.fret}` +
        (unsure ? ", uncertain" : ""),
    );
    el("rect", { x: x(n.onset) - w / 2, y: y(n.string) - 10, width: w, height: 20, rx: 4 }, g);
    el("text", { x: x(n.onset), y: y(n.string) + 5 }, g).textContent = label;
    const show = () => showTip(g, n, rank[i], doc.notes.length, labels, unsure);
    g.addEventListener("mouseenter", show);
    g.addEventListener("focus", show);
    g.addEventListener("mouseleave", hideTip);
    g.addEventListener("blur", hideTip);
  });
}

function showTip(target, n, rank, total, labels, unsure) {
  const tip = $("tip");
  tip.replaceChildren();
  const head = document.createElement("div");
  head.append(
    `${pitchName(n.pitch)} at ${seconds(n.onset)}, played ${labels[n.string]} string fret ${n.fret}.`,
  );
  const sure = document.createElement("div");
  sure.textContent =
    `Certainty rank ${rank} of ${total} (1 is the surest)` + (unsure ? "; marked uncertain." : ".");
  tip.append(head, sure);
  if (n.alternatives.length) {
    const intro = document.createElement("div");
    intro.textContent = "Other ways to play it, likeliest first:";
    const list = document.createElement("ol");
    for (const a of n.alternatives) {
      const item = document.createElement("li");
      item.textContent = `${labels[a.string]} string, fret ${a.fret}`;
      list.appendChild(item);
    }
    tip.append(intro, list);
  } else {
    const none = document.createElement("div");
    none.textContent = "No other way to play it was considered.";
    tip.append(none);
  }
  tip.hidden = false;
  const box = target.getBoundingClientRect();
  const left = Math.min(box.left, window.innerWidth - tip.offsetWidth - 8);
  const below = box.bottom + 8 + tip.offsetHeight < window.innerHeight;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${below ? box.bottom + 8 : box.top - tip.offsetHeight - 8}px`;
}

function hideTip() {
  $("tip").hidden = true;
}

// A recording far from A440 is reported, never corrected (ADR 0060): the transcriber assumes
// standard pitch. Same wording as the CLI's warning.
function tuningWarning(doc) {
  const offset = doc.metrics.tuning_offset;
  if (offset === null || Math.abs(offset) < doc.tuning_warning_threshold) return "";
  const size = Math.abs(offset).toLocaleString("en", { maximumFractionDigits: 2 });
  return (
    `The recording is about ${size} semitone ${offset > 0 ? "sharp" : "flat"} of standard ` +
    "pitch (A440). The transcriber assumes standard pitch, so many notes may be missed or " +
    "written a semitone off. Tune to A440 and record again for a better tab."
  );
}

function summarise(doc) {
  const warning = tuningWarning(doc);
  $("tuning").textContent = warning;
  $("tuning").hidden = !warning;

  const m = doc.metrics;
  const unsure = doc.notes.filter((n) => n.posterior < doc.uncertainty_threshold).length;
  $("summary").textContent =
    `${doc.notes.length} notes, ${unsure} marked uncertain` +
    (m.group_rate === 1 && m.transition_rate === 1
      ? "; every shape and hand move passes the playability rules."
      : "; some shapes or hand moves fail the playability rules.");

  const d = doc.degradation;
  const banner = $("degraded");
  banner.replaceChildren();
  const lost = d.n_notes_out_of_range + d.n_notes_dropped;
  if (d.is_clean) {
    banner.hidden = true;
    return;
  }
  const lines = [];
  if (d.n_notes_out_of_range) lines.push(`${d.n_notes_out_of_range} detected notes are outside the guitar's range and were left out.`);
  if (d.n_notes_dropped) lines.push(`${d.n_notes_dropped} notes were dropped to make an over-full chord playable.`);
  if (d.n_groups_dropped) lines.push(`${d.n_groups_dropped} chords or single notes were left out entirely, counting any made only of the notes above.`);
  if (d.n_groups_relaxed) lines.push(`${d.n_groups_relaxed} chords needed a wider stretch than usual (up to ${d.max_span_used} frets).`);
  const head = document.createElement("strong");
  head.textContent = lost ? `${lost} notes the transcriber heard are not in this tab.` : "Some chords needed a wider stretch.";
  const list = document.createElement("ul");
  for (const line of lines) {
    const item = document.createElement("li");
    item.textContent = line;
    list.appendChild(item);
  }
  banner.append(head, list);
  banner.hidden = false;
}

async function transcribe(file) {
  $("result").hidden = true;
  hideTip();
  setStatus(`Transcribing ${file.name}... a new file takes a few seconds per minute of audio.`);
  const body = new FormData();
  body.append("audio", file);
  let response;
  try {
    response = await fetch("/api/transcribe", { method: "POST", body });
  } catch (err) {
    setStatus(`Could not reach the server: ${err.message}. Is tabsampler serve still running?`, true);
    return;
  }
  let doc;
  try {
    doc = await response.json();
  } catch {
    setStatus(`The server answered ${response.status} without a readable reply.`, true);
    return;
  }
  if (!response.ok) {
    const detail = typeof doc.detail === "string" ? doc.detail : JSON.stringify(doc.detail);
    setStatus(`Could not transcribe ${file.name}: ${detail}`, true);
    return;
  }
  if (!doc.notes.length) {
    setStatus(`No notes were detected in ${file.name}. ${tuningWarning(doc)}`);
    return;
  }
  current = doc;
  setStatus(`${file.name}`);
  summarise(doc);
  $("zoom").value = String(fitZoom(doc));
  $("result").hidden = false;
  draw(doc);
}

// The exports carry the disclaimer inside the file (ADRs 0017, 0059).
async function download(fmt) {
  if (!current) return;
  const response = await fetch(`/api/export/${fmt}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(current),
  });
  if (!response.ok) {
    const doc = await response.json().catch(() => ({ detail: response.statusText }));
    setStatus(`Could not export: ${doc.detail}`, true);
    return;
  }
  const name = (response.headers.get("Content-Disposition") || "").split('filename="')[1];
  const link = document.createElement("a");
  link.href = URL.createObjectURL(await response.blob());
  link.download = name ? name.replace('"', "") : `tab.${fmt}`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function init() {
  const drop = $("drop");
  const input = $("file");
  input.addEventListener("change", () => input.files.length && transcribe(input.files[0]));
  for (const type of ["dragenter", "dragover"]) {
    drop.addEventListener(type, (e) => {
      e.preventDefault();
      drop.classList.add("over");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    drop.addEventListener(type, () => drop.classList.remove("over"));
  }
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) transcribe(e.dataTransfer.files[0]);
  });
  for (const button of document.querySelectorAll("[data-export]")) {
    button.addEventListener("click", () => download(button.dataset.export));
  }
  $("zoom").addEventListener("change", () => current && draw(current));
  $("scroller").addEventListener("scroll", hideTip);
}

init();
