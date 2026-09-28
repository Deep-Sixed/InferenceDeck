// Machines card: every InferenceDeck machine in one table, plus advice on where
// to run a model. Hidden until fleet_peers lists at least one other machine.
(() => {
  const $ = id => document.getElementById(id);
  const GB = 1024 ** 3;
  const html = (tag, cls, text, parent) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  };
  const gb = v => v == null ? '—' : `${(v / GB).toFixed(1)}`;
  let timer = null;

  function gpuText(host) {
    if (!host.gpus.length) return '—';
    return host.gpus.map(g => `${g.utilization_percent == null ? '—' : Math.round(g.utilization_percent) + '%'}`).join(' / ');
  }

  function vramText(host) {
    const used = host.gpus.reduce((a, g) => a + (g.vram_used_bytes || 0), 0);
    const total = host.gpus.reduce((a, g) => a + (g.vram_total_bytes || 0), 0);
    return total ? `${gb(used)} / ${gb(total)} GB` : '—';
  }

  function render(data) {
    const card = $('fleet-card');
    const peers = data.peers || [];
    card.hidden = peers.length === 0 && !(data.config_errors || []).length;
    if (card.hidden) return;
    const body = $('fleet-rows');
    body.replaceChildren();
    const names = new Set();
    for (const host of data.hosts || []) {
      const tr = html('tr', host.reachable ? null : 'fleet-down', null, body);
      const name = html('td', null, null, tr);
      html('strong', null, host.name, name);
      if (host.local) html('span', 'meta', ' · this machine', name);
      if (!host.reachable) {
        const td = html('td', 'tfailtext', `⚠ unreachable: ${host.error || 'no answer'}`, tr);
        td.colSpan = 5;
        continue;
      }
      html('td', null, gpuText(host), tr);
      html('td', null, vramText(host), tr);
      const loaded = html('td', null, null, tr);
      const live = host.servers.filter(s => s.running);
      if (!live.length) html('span', 'muted', 'idle', loaded);
      for (const s of live) {
        names.add(s.profile);
        const line = html('div', null, null, loaded);
        html('span', null, s.profile, line);
        const bits = [s.suspended ? 'paused' : null, s.tokens_per_second ? `${s.tokens_per_second.toFixed(1)} tok/s` : null].filter(Boolean);
        if (bits.length) html('span', 'meta', ` · ${bits.join(' · ')}`, line);
      }
      for (const b of host.benchmarks || []) names.add(b.profile);
      html('td', null, String(live.reduce((a, s) => a + (s.requests_active || 0), 0)), tr);
      html('td', 'meta', host.local ? 'online' : `online · ${Math.round(host.latency_ms)} ms`, tr);
    }
    const list = $('fleet-profiles');
    list.replaceChildren(...[...names].filter(Boolean).sort().map(n => { const o = document.createElement('option'); o.value = n; return o; }));
    const errors = $('fleet-errors');
    errors.replaceChildren(...(data.config_errors || []).map(e => html('div', 'meta tfailtext', `⚠ ${e}`)));
    for (const p of peers.filter(p => p.token_env && !p.token_present)) {
      errors.appendChild(html('div', 'meta', `${p.name}: $${p.token_env} is not set, so its API will refuse the request if it needs a token.`));
    }
  }

  async function load() {
    try {
      const r = await fetch('/api/fleet', {credentials: 'same-origin'});
      if (r.ok) render(await r.json());
    } catch (e) { /* the status card already reports a dead API */ }
  }

  async function place(ev) {
    ev.preventDefault();
    const profile = $('fleet-profile').value.trim();
    const out = $('fleet-placement');
    out.replaceChildren();
    if (!profile) return;
    try {
      const r = await fetch(`/api/fleet/placement?profile=${encodeURIComponent(profile)}`, {credentials: 'same-origin'});
      const data = await r.json();
      if (!r.ok) { html('div', 'meta tfailtext', data.error || r.statusText, out); return; }
      html('div', 'meta', data.recommended ? `Best fit for ${profile}: ${data.recommended}` : `No machine has run ${profile} yet.`, out);
      data.candidates.forEach((c, i) => {
        const row = html('div', 'row', null, out);
        const left = html('div', null, null, row);
        html('strong', null, `${i + 1}. ${c.host}`, left);
        html('div', 'meta', c.reasons.join(' · '), left);
        html('span', c.tier === 0 ? 'pill' : 'meta', c.state, row);
      });
      for (const u of data.unreachable) html('div', 'meta', `${u.host} is unreachable: ${u.error || 'no answer'}`, out);
    } catch (e) { html('div', 'meta tfailtext', e.message, out); }
  }

  document.addEventListener('DOMContentLoaded', () => {
    if (!$('fleet-card')) return;
    $('fleet-form').addEventListener('submit', place);
    load();
    clearInterval(timer);
    timer = setInterval(() => { if (!document.hidden) load(); }, 10000);
  });
})();
