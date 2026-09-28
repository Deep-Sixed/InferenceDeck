// Telemetry history charts: one small chart per measure (never two y-axes on one
// chart), a crosshair synced across all of them, lifecycle event markers, and a
// summary table that carries every number the charts show.
(() => {
  const SERIES_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500']; // validated on the #121922 card surface
  const GB = 1024 ** 3;
  const HEIGHT = 128, PAD = {l: 58, r: 10, t: 10, b: 20};
  const CHARTS = [
    {id: 'gpu-util', title: 'GPU utilization', unit: '%', match: /^gpu(\d+)\.utilization_percent$/, max: 100, fmt: v => `${Math.round(v)}%`},
    {id: 'gpu-vram', title: 'VRAM used', unit: 'GB', match: /^gpu(\d+)\.vram_used_bytes$/, scale: 1 / GB, limit: true, fmt: v => `${v.toFixed(1)} GB`},
    {id: 'gpu-temp', title: 'GPU temperature', unit: '°C', match: /^gpu(\d+)\.temperature_c$/, fmt: v => `${Math.round(v)}°C`},
    {id: 'gpu-power', title: 'GPU power', unit: 'W', match: /^gpu(\d+)\.power_watts$/, fmt: v => `${Math.round(v)} W`},
    {id: 'gpu-clock', title: 'GPU core clock', unit: 'MHz', match: /^gpu(\d+)\.sm_clock_mhz$/, fmt: v => `${Math.round(v)} MHz`},
    {id: 'server-vram', title: 'GPU memory by server', unit: 'GB', match: /^server:(.+)\.gpu_memory_bytes$/, scale: 1 / GB, fmt: v => `${v.toFixed(1)} GB`},
    // Speeds are measured only while a server works, so idle stretches are gaps, not zeros.
    {id: 'server-gen', title: 'Generation speed', unit: 'tok/s', match: /^server:(.+)\.tokens_per_second$/, fmt: v => `${v.toFixed(1)} tok/s`},
    {id: 'server-prompt', title: 'Prompt processing', unit: 'tok/s', match: /^server:(.+)\.prompt_tokens_per_second$/, fmt: v => `${Math.round(v)} tok/s`},
    {id: 'server-requests', title: 'Active requests', unit: '', match: /^server:(.+)\.requests_active$/, integer: true, fmt: v => Number.isInteger(v) ? String(v) : v.toFixed(1)},
    {id: 'server-kv', title: 'KV cache used', unit: '%', match: /^server:(.+)\.kv_cache_usage_percent$/, max: 100, fmt: v => `${Math.round(v)}%`},
    {id: 'cpu', title: 'CPU utilization', unit: '%', match: /^cpu_percent$/, max: 100, fmt: v => `${Math.round(v)}%`},
    {id: 'ram', title: 'RAM used', unit: 'GB', match: /^memory_used_bytes$/, scale: 1 / GB, limit: true, fmt: v => `${v.toFixed(1)} GB`},
  ];
  const EVENT_LABELS = {
    'server.started': 'started', 'server.ready': 'ready', 'server.start_failed': 'start failed',
    'server.stopped': 'stopped', 'server.stop_failed': 'stop failed', 'server.suspended': 'paused',
    'server.resumed': 'resumed', 'server.released': 'released GPU', 'server.restored': 'restored',
    'benchmark.completed': 'benchmark',
  };
  const FAILED = new Set(['server.start_failed', 'server.stop_failed']);
  const NS = 'http://www.w3.org/2000/svg';
  const $ = id => document.getElementById(id);
  let range = '1h', data = null, built = [], hoverIndex = null, timer = null;

  const el = (tag, attrs = {}, parent) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.appendChild(node);
    return node;
  };
  const html = (tag, cls, text, parent) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  };
  const clock = (t, withDay) => new Date(t * 1000).toLocaleString([], withDay
    ? {weekday: 'short', hour: '2-digit', minute: '2-digit'}
    : {hour: '2-digit', minute: '2-digit'});
  const niceMax = v => {
    if (!(v > 0)) return 1;
    const p = 10 ** Math.floor(Math.log10(v)), n = v / p;
    return [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find(step => n <= step) * p;
  };

  // Series for one chart, colored by a fixed order of their names so a filter or
  // a new server never repaints the others.
  function seriesFor(chart) {
    const out = [];
    for (const [key, values] of Object.entries(data.series)) {
      const m = key.match(chart.match);
      if (!m || !values.some(v => v != null)) continue;
      const id = m[1] ?? '';
      const name = chart.id.startsWith('gpu') ? (data.labels[`gpu${id}`] || `GPU ${id}`) : chart.id.startsWith('server-') ? id : chart.title;
      const scale = chart.scale || 1;
      const limit = chart.limit && data.limits[key] ? data.limits[key] * scale : null;
      out.push({key, name, values: values.map(v => v == null ? null : v * scale), limit});
    }
    out.sort((a, b) => a.key.localeCompare(b.key, undefined, {numeric: true}));
    out.forEach((s, i) => { s.color = SERIES_COLORS[i % SERIES_COLORS.length]; });
    return out.slice(0, SERIES_COLORS.length);
  }

  function summarize(values) {
    const present = values.filter(v => v != null);
    if (!present.length) return null;
    const last = [...values].reverse().find(v => v != null);
    return {last, avg: present.reduce((a, b) => a + b, 0) / present.length, max: Math.max(...present)};
  }

  function drawChart(chart, series, host) {
    const card = html('figure', 'tchart', null, host);
    const head = html('figcaption', null, null, card);
    html('span', 'ttitle', chart.title, head);
    const latest = series.length === 1 ? summarize(series[0].values) : null;
    if (latest) html('span', 'tvalue', chart.fmt(latest.last), head);
    if (series.length > 1) {
      const legend = html('span', 'tlegend', null, head);
      for (const s of series) {
        const item = html('span', null, null, legend);
        html('i', null, null, item).style.background = s.color;
        item.appendChild(document.createTextNode(s.name));
      }
    }
    const width = Math.max(260, card.clientWidth || 440);
    const svg = el('svg', {viewBox: `0 0 ${width} ${HEIGHT}`, width: '100%', height: HEIGHT, role: 'img', 'aria-label': `${chart.title} over the last ${range}`}, card);
    const ts = data.timestamps, n = ts.length;
    const peak = Math.max(...series.flatMap(s => [...s.values.filter(v => v != null), s.limit || 0]));
    // Counts get an even whole-number top so the midline tick is a whole number too.
    const yMax = chart.max || (chart.integer ? Math.max(2, Math.ceil(peak / 2) * 2) : niceMax(peak * 1.05));
    const x = i => PAD.l + (n > 1 ? i / (n - 1) : 0.5) * (width - PAD.l - PAD.r);
    const y = v => PAD.t + (1 - Math.min(v, yMax) / yMax) * (HEIGHT - PAD.t - PAD.b);

    for (const frac of [0, 0.5, 1]) {
      const v = yMax * frac, yy = y(v);
      el('line', {x1: PAD.l, x2: width - PAD.r, y1: yy, y2: yy, class: frac ? 'tgrid' : 'taxis'}, svg);
      el('text', {x: PAD.l - 6, y: yy + 4, 'text-anchor': 'end', class: 'ttick'}, svg).textContent = chart.fmt(v).replace(/\.0 /, ' ');
    }
    const withDay = ts.length && ts[n - 1] - ts[0] > 86400;
    for (const frac of width < 420 ? [0, 0.5, 1] : [0, 1 / 3, 2 / 3, 1]) {
      const i = Math.round(frac * (n - 1));
      if (ts[i] == null) continue;
      el('text', {x: x(i), y: HEIGHT - 5, 'text-anchor': frac === 0 ? 'start' : frac === 1 ? 'end' : 'middle', class: 'ttick'}, svg).textContent = clock(ts[i], withDay);
    }

    // Capacity (VRAM/RAM total) as a dashed reference, labelled once.
    const limit = series.find(s => s.limit)?.limit;
    if (limit) {
      el('line', {x1: PAD.l, x2: width - PAD.r, y1: y(limit), y2: y(limit), class: 'tlimit'}, svg);
      el('text', {x: width - PAD.r, y: y(limit) - 4, 'text-anchor': 'end', class: 'ttick'}, svg).textContent = `total ${chart.fmt(limit)}`;
    }

    // Lifecycle markers: a hairline per event, failures in the error color.
    const step = data.step_seconds || 1;
    for (const e of data.events) {
      if (!n || e.t < ts[0] || e.t > ts[n - 1] + step) continue;
      const i = Math.min(n - 1, Math.max(0, (e.t - ts[0]) / step));
      el('line', {x1: x(i), x2: x(i), y1: PAD.t, y2: HEIGHT - PAD.b, class: FAILED.has(e.type) ? 'tevent tfail' : 'tevent'}, svg);
    }

    // Lines bridge a missed sample or two but break across real gaps (the
    // control process was down), so an outage never looks like a flat line.
    for (const s of series) {
      let d = '', gap = Infinity, run = [];
      // A reading with no neighbour (a short burst) has no line to draw, so it gets a dot.
      const flush = () => { if (run.length === 1) el('circle', {cx: x(run[0][0]), cy: y(run[0][1]), r: 2, fill: s.color}, svg); run = []; };
      s.values.forEach((v, i) => {
        if (v == null) { gap++; return; }
        if (gap >= 3) flush();
        d += `${gap >= 3 ? 'M' : 'L'}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
        run.push([i, v]);
        gap = 0;
      });
      flush();
      if (d) el('path', {d: d.replace(/^L/, 'M'), class: 'tline', stroke: s.color}, svg);
    }

    const cross = el('line', {y1: PAD.t, y2: HEIGHT - PAD.b, class: 'tcross', visibility: 'hidden'}, svg);
    const dots = series.map(s => el('circle', {r: 4, fill: s.color, class: 'tdot', visibility: 'hidden'}, svg));
    const hit = el('rect', {x: PAD.l, y: 0, width: width - PAD.l - PAD.r, height: HEIGHT, fill: 'transparent', tabindex: 0}, svg);
    const indexAt = clientX => {
      const box = svg.getBoundingClientRect();
      const px = (clientX - box.left) * (width / box.width);
      return Math.max(0, Math.min(n - 1, Math.round((px - PAD.l) / (width - PAD.l - PAD.r) * (n - 1))));
    };
    hit.addEventListener('pointermove', ev => setHover(indexAt(ev.clientX), card));
    hit.addEventListener('pointerleave', () => setHover(null));
    hit.addEventListener('focus', () => setHover(n - 1, card));
    hit.addEventListener('blur', () => setHover(null));
    hit.addEventListener('keydown', ev => {
      if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
      ev.preventDefault();
      setHover(Math.max(0, Math.min(n - 1, (hoverIndex ?? n - 1) + (ev.key === 'ArrowLeft' ? -1 : 1))), card);
    });
    built.push({chart, series, card, width, x, y, cross, dots});
  }

  function setHover(index, source) {
    hoverIndex = index;
    const tip = $('telemetry-tip');
    for (const b of built) {
      const show = index != null;
      b.cross.setAttribute('visibility', show ? 'visible' : 'hidden');
      if (show) { b.cross.setAttribute('x1', b.x(index)); b.cross.setAttribute('x2', b.x(index)); }
      b.series.forEach((s, i) => {
        const v = show ? s.values[index] : null;
        b.dots[i].setAttribute('visibility', v == null ? 'hidden' : 'visible');
        if (v != null) { b.dots[i].setAttribute('cx', b.x(index)); b.dots[i].setAttribute('cy', b.y(v)); }
      });
    }
    const owner = built.find(b => b.card === source);
    if (index == null || !owner) { tip.hidden = true; return; }
    tip.replaceChildren();
    const t = data.timestamps[index], step = data.step_seconds || 1;
    html('div', 'meta', clock(t, true), tip);
    for (const s of owner.series) {
      const row = html('div', 'trow', null, tip);
      html('i', null, null, row).style.background = s.color;
      html('strong', null, s.values[index] == null ? '—' : owner.chart.fmt(s.values[index]), row);
      if (owner.series.length > 1) html('span', 'meta', s.name, row);
    }
    for (const e of data.events.filter(e => e.t >= t && e.t < t + step)) {
      html('div', FAILED.has(e.type) ? 'meta tfailtext' : 'meta', `${e.profile || ''} ${EVENT_LABELS[e.type] || e.type}${e.reason ? ` (${e.reason})` : ''}`.trim(), tip);
    }
    const box = owner.card.getBoundingClientRect(), host = $('telemetry-charts').getBoundingClientRect();
    const left = box.left - host.left + owner.x(index) * (box.width / owner.width);
    tip.hidden = false;
    tip.style.top = `${box.top - host.top + 28}px`;
    tip.style.left = `${Math.min(host.width - tip.offsetWidth, Math.max(0, left + 12))}px`;
  }

  function render() {
    const host = $('telemetry-charts'), table = $('telemetry-table');
    if (!host || !data) return;
    built = [];
    host.querySelectorAll('.tchart').forEach(n => n.remove());
    const rows = [];
    for (const chart of CHARTS) {
      const series = seriesFor(chart);
      if (!series.length) continue;
      drawChart(chart, series, host);
      for (const s of series) {
        const sum = summarize(s.values);
        if (sum) rows.push([chart.title + (series.length > 1 ? ` · ${s.name}` : ''), chart.fmt(sum.last), chart.fmt(sum.avg), chart.fmt(sum.max)]);
      }
    }
    const empty = $('telemetry-empty');
    empty.hidden = built.length > 0;
    empty.textContent = data.enabled === false
      ? 'Telemetry history is off (telemetry_sample_seconds is 0).'
      : 'No samples yet. The first points appear a few seconds after the control process starts.';
    table.replaceChildren();
    if (rows.length) {
      const head = html('tr', null, null, html('thead', null, null, table));
      for (const h of ['Measure', 'Latest', `Average (${range})`, `Peak (${range})`]) html('th', null, h, head);
      const body = html('tbody', null, null, table);
      for (const r of rows) {
        const tr = html('tr', null, null, body);
        r.forEach(c => html('td', null, c, tr));
      }
    }
    const events = $('telemetry-events');
    events.replaceChildren();
    for (const e of data.events.slice(-8).reverse()) {
      const row = html('div', 'meta', null, events);
      html('span', null, `${clock(e.t, true)} · `, row);
      html('span', FAILED.has(e.type) ? 'tfailtext' : null, `${e.profile || e.server_id || ''} ${EVENT_LABELS[e.type] || e.type}`.trim(), row);
      const extra = e.startup_seconds != null ? ` · ready in ${e.startup_seconds.toFixed(1)} s`
        : e.tokens_per_second != null ? ` · ${e.tokens_per_second} tok/s` : e.reason ? ` · ${e.reason}` : '';
      if (extra) html('span', null, extra, row);
    }
  }

  async function load() {
    try {
      const r = await fetch(`/api/telemetry/history?range=${encodeURIComponent(range)}`, {credentials: 'same-origin'});
      if (!r.ok) return;
      data = await r.json();
      render();
      if (hoverIndex != null) setHover(null);
    } catch (e) { /* the status card already reports a dead API */ }
  }

  function schedule() {
    clearInterval(timer);
    // Short ranges refresh with the samples; minute-averaged ranges once a minute.
    timer = setInterval(() => { if (!document.hidden) load(); }, range === '15m' || range === '1h' ? 10000 : 60000);
  }

  document.addEventListener('DOMContentLoaded', () => {
    const picker = $('telemetry-range');
    if (!picker) return;
    picker.addEventListener('click', ev => {
      const b = ev.target.closest('button[data-range]');
      if (!b) return;
      range = b.dataset.range;
      picker.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
      load(); schedule();
    });
    let resize;
    window.addEventListener('resize', () => { clearTimeout(resize); resize = setTimeout(render, 150); });
    load(); schedule();
  });
})();
