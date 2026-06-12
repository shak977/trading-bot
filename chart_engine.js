/* TradeChart — a reusable Capital IQ-style charting component.
 *
 * One instance renders a price panel (candles / line / area) plus optional
 * stacked sub-panels (volume, MACD, RSI, stochastics) and overlays
 * (SMA, EMA ribbon, Bollinger, VWAP, plan lines, S&P benchmark, multi-ticker
 * compare). It fetches OHLCV from the Cloudflare Worker (Yahoo) and falls back
 * to the data embedded in the page. Colors are read from CSS variables so it
 * flips with the light/dark theme.
 *
 * Usage:
 *   const tc = new TradeChart(rootEl, { app: window.__APP, compact:false });
 *   tc.setSymbol('AAPL', planObj);
 *   tc.onLive(pricesObj);   // push live quotes
 *   tc.applyTheme();        // after a theme switch
 */
(function () {
  'use strict';

  const RANGE_MAP = {
    '1D':  { range: '1d',  interval: '2m',  intraday: true,  unit: 'hour' },
    '5D':  { range: '5d',  interval: '15m', intraday: true,  unit: 'day' },
    '1M':  { range: '1mo', interval: '1d',  intraday: false, unit: 'week' },
    '3M':  { range: '3mo', interval: '1d',  intraday: false, unit: 'week' },
    '6M':  { range: '6mo', interval: '1d',  intraday: false, unit: 'month' },
    'YTD': { range: 'ytd', interval: '1d',  intraday: false, unit: 'month' },
    '1Y':  { range: '1y',  interval: '1d',  intraday: false, unit: 'month' },
    'Max': { range: 'max', interval: '1wk', intraday: false, unit: 'year' },
  };
  const RANGES = ['1D', '5D', '1M', '3M', '6M', 'YTD', '1Y', 'Max'];

  // ---------- indicator math ----------
  function sma(a, p) {
    const o = a.map(() => null); let s = 0, k = 0;
    for (let i = 0; i < a.length; i++) {
      if (a[i] == null) { o[i] = null; continue; }
      s += a[i]; k++;
      if (i >= p && a[i - p] != null) { s -= a[i - p]; k--; }
      if (k >= p) o[i] = s / p;
    }
    return o;
  }
  function ema(a, p) {
    const o = a.map(() => null); const m = 2 / (p + 1); let e = null;
    for (let i = 0; i < a.length; i++) {
      const v = a[i];
      if (v == null) { o[i] = e; continue; }
      e = (e == null) ? v : v * m + e * (1 - m);
      o[i] = e;
    }
    return o;
  }
  function boll(a, p) {
    const mid = sma(a, p), up = [], lo = [];
    for (let i = 0; i < a.length; i++) {
      if (mid[i] == null) { up[i] = null; lo[i] = null; continue; }
      let s = 0; for (let j = i - p + 1; j <= i; j++) s += Math.pow(a[j] - mid[i], 2);
      const sd = Math.sqrt(s / p); up[i] = mid[i] + 2 * sd; lo[i] = mid[i] - 2 * sd;
    }
    return { up, lo, mid };
  }
  function macd(a) {
    const e12 = ema(a, 12), e26 = ema(a, 26);
    const line = a.map((_, i) => (e12[i] != null && e26[i] != null) ? e12[i] - e26[i] : null);
    const sig = ema(line.map(v => v == null ? 0 : v), 9);
    const hist = line.map((v, i) => v != null ? v - sig[i] : null);
    return { line, sig, hist };
  }
  function rsi(a, p) {
    const o = a.map(() => null); let g = 0, l = 0;
    for (let i = 1; i < a.length; i++) {
      const ch = a[i] - a[i - 1], up = Math.max(ch, 0), dn = Math.max(-ch, 0);
      if (i <= p) { g += up; l += dn; if (i === p) { g /= p; l /= p; o[i] = 100 - 100 / (1 + (l === 0 ? 100 : g / l)); } }
      else { g = (g * (p - 1) + up) / p; l = (l * (p - 1) + dn) / p; o[i] = 100 - 100 / (1 + (l === 0 ? 100 : g / l)); }
    }
    return o;
  }
  function stoch(H, L, C, p, d) {
    const k = C.map(() => null);
    for (let i = 0; i < C.length; i++) {
      if (i < p - 1) continue;
      let hh = -Infinity, ll = Infinity;
      for (let j = i - p + 1; j <= i; j++) { if (H[j] > hh) hh = H[j]; if (L[j] < ll) ll = L[j]; }
      k[i] = hh === ll ? 50 : ((C[i] - ll) / (hh - ll)) * 100;
    }
    const dd = sma(k, d);
    return { k, d: dd };
  }
  function vwap(H, L, C, V, intraday, T) {
    // resets each session for intraday; cumulative otherwise
    const o = C.map(() => null); let pv = 0, vv = 0, day = null;
    for (let i = 0; i < C.length; i++) {
      const tp = (H[i] + L[i] + C[i]) / 3, vol = V[i] || 0;
      if (intraday) { const d = new Date(T[i]).toDateString(); if (d !== day) { pv = 0; vv = 0; day = d; } }
      pv += tp * vol; vv += vol;
      o[i] = vv > 0 ? pv / vv : C[i];
    }
    return o;
  }

  function volFmt(v) {
    v = Math.abs(v);
    return v >= 1e9 ? (v / 1e9).toFixed(2) + 'B' : v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(0) + 'K' : String(Math.round(v));
  }
  function money(v) { return v == null ? '—' : '$' + Number(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }); }

  // read a CSS variable off :root
  function cssv(name, fallback) {
    try { const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim(); return v || fallback; }
    catch (e) { return fallback; }
  }
  function theme() {
    return {
      txt: cssv('--txt', '#e6edf3'), muted: cssv('--muted', '#8b97a6'),
      grid: cssv('--grid', 'rgba(120,130,145,0.18)'), cross: cssv('--cross', 'rgba(120,130,145,0.5)'),
      buy: cssv('--buy', '#2ea043'), sell: cssv('--sell', '#f85149'),
      accent: cssv('--hold', '#388bfd'), card: cssv('--card', '#1a212b'), line: cssv('--line', '#2a3441'),
    };
  }

  // crosshair: vertical line + dot at the hovered point
  try {
    window.Chart && window.Chart.register({
      id: 'tcCrosshair',
      afterDatasetsDraw(chart) {
        if (!chart.options || !chart.options._crosshair) return;
        const act = chart.getActiveElements ? chart.getActiveElements() : [];
        if (!act || !act.length) return;
        const x = act[0].element.x, y = act[0].element.y;
        const a = chart.chartArea, c = chart.ctx;
        c.save();
        c.beginPath(); c.moveTo(x, a.top); c.lineTo(x, a.bottom);
        c.lineWidth = 1; c.strokeStyle = cssv('--cross', 'rgba(120,130,145,0.5)'); c.stroke();
        c.beginPath(); c.arc(x, y, 3.5, 0, Math.PI * 2); c.fillStyle = cssv('--txt', '#e6edf3'); c.fill();
        c.restore();
      },
    });
  } catch (e) {}

  const EMA_RIBBON = [[8, '#56d4dd'], [21, '#388bfd'], [50, '#f0883e'], [200, '#a371f7']];
  let _seq = 0;

  class TradeChart {
    constructor(root, opts) {
      opts = opts || {};
      this.root = root;
      this.app = opts.app || window.__APP || {};
      this.compact = !!opts.compact;
      this.id = 'tc' + (++_seq);
      this.symbol = null; this.plan = {};
      this.cache = {};            // "SYM:RANGE" -> bars
      this.compareCache = {};     // "SYM:RANGE" -> bars (for overlays)
      this.compare = [];          // extra symbols
      this.draw = { mode: null, items: {}, pending: null };  // items keyed by symbol
      this.charts = {};           // panel -> Chart instance
      this.cur = { bars: [], base: null, intraday: false };
      this.state = {
        range: opts.range || '6M', type: opts.type || 'candle', log: false,
        bench: false,
        ovl: { sma: true, ema: false, boll: false, vwap: false },
        panels: { vol: true, macd: false, rsi: false, stoch: false },
      };
      this._finOK = false;
      try { this._finOK = !!(window.Chart && Chart.registry.getController('candlestick')); } catch (e) {}
      this._build();
    }

    // ---------- DOM ----------
    _build() {
      const r = this.root;
      r.classList.add('tc');
      if (this.compact) r.classList.add('tc-compact');
      r.innerHTML = ''
        + '<div class="tc-bar"></div>'
        + '<div class="tc-readout"></div>'
        + '<div class="tc-wrap" style="position:relative;"><canvas class="tc-price"></canvas><canvas class="tc-draw" style="position:absolute;inset:0;pointer-events:none;"></canvas></div>'
        + '<div class="tc-sub tc-vol"  style="display:none;"><canvas></canvas></div>'
        + '<div class="tc-sub tc-macd" style="display:none;"><div class="tc-sublab">MACD</div><canvas></canvas></div>'
        + '<div class="tc-sub tc-rsi"  style="display:none;"><div class="tc-sublab">RSI (14)</div><canvas></canvas></div>'
        + '<div class="tc-sub tc-stoch" style="display:none;"><div class="tc-sublab">Stochastics</div><canvas></canvas></div>'
        + '<div class="tc-key"></div>';
      this.el = {
        bar: r.querySelector('.tc-bar'), readout: r.querySelector('.tc-readout'),
        price: r.querySelector('.tc-price'), drawc: r.querySelector('.tc-draw'),
        vol: r.querySelector('.tc-vol'), macd: r.querySelector('.tc-macd'),
        rsi: r.querySelector('.tc-rsi'), stoch: r.querySelector('.tc-stoch'),
        key: r.querySelector('.tc-key'),
      };
      this._buildBar();
      this._wireDraw();
    }
    _seg(label, items, active, cb, kind) {
      const wrap = document.createElement('span'); wrap.className = 'tc-seg';
      if (label) { const l = document.createElement('span'); l.className = 'tc-seglab'; l.textContent = label; wrap.appendChild(l); }
      items.forEach(([val, txt]) => {
        const b = document.createElement('button'); b.textContent = txt; b.dataset.v = val;
        if (active(val)) b.classList.add('on');
        b.onclick = () => { cb(val, b); if (kind === 'radio') wrap.querySelectorAll('button').forEach(x => x.classList.toggle('on', x.dataset.v === String(val))); else b.classList.toggle('on'); };
        wrap.appendChild(b);
      });
      return wrap;
    }
    _buildBar() {
      const bar = this.el.bar; bar.innerHTML = '';
      bar.appendChild(this._seg('', RANGES.map(r => [r, r]), v => this.state.range === v, v => { this.state.range = v; this.render(); }, 'radio'));
      bar.appendChild(this._seg('', [['candle', 'Candles'], ['line', 'Line'], ['area', 'Area']], v => this.state.type === v, v => { this.state.type = v; this.render(); }, 'radio'));
      bar.appendChild(this._seg('Overlays', [['sma', 'MA'], ['ema', 'EMA ribbon'], ['boll', 'Bollinger'], ['vwap', 'VWAP']], v => this.state.ovl[v], v => { this.state.ovl[v] = !this.state.ovl[v]; this.render(); }));
      bar.appendChild(this._seg('Panels', [['vol', 'Volume'], ['macd', 'MACD'], ['rsi', 'RSI'], ['stoch', 'Stoch']], v => this.state.panels[v], v => { this.state.panels[v] = !this.state.panels[v]; this.render(); }));
      bar.appendChild(this._seg('Scale', [['log', 'Log']], () => this.state.log, () => { this.state.log = !this.state.log; this.render(); }));
      bar.appendChild(this._seg('', [['bench', 'vs S&P']], () => this.state.bench, () => { this.state.bench = !this.state.bench; this.render(); }));
      // compare input
      const cmp = document.createElement('span'); cmp.className = 'tc-seg';
      const ci = document.createElement('input'); ci.className = 'tc-cmp'; ci.placeholder = '+ compare (e.g. MSFT)'; ci.size = 14;
      ci.onkeydown = (e) => { if (e.key === 'Enter') { const s = ci.value.trim().toUpperCase(); if (s && !this.compare.includes(s)) { this.compare.push(s); ci.value = ''; this.render(); } } };
      cmp.appendChild(ci); bar.appendChild(cmp);
      this.el.cmpInput = ci;
      // drawing tools
      bar.appendChild(this._seg('Draw', [['hline', '— line'], ['trend', '╱ trend']], v => this.draw.mode === v, (v) => { this.draw.mode = this.draw.mode === v ? null : v; this._syncDrawCursor(); this._refreshBarActive(); }, 'toggleDraw'));
      const clr = document.createElement('button'); clr.className = 'tc-clr'; clr.textContent = 'Clear'; clr.onclick = () => { this.draw.items[this.symbol] = []; this._redrawOverlay(); };
      bar.appendChild(clr);
    }
    _refreshBarActive() {
      this.el.bar.querySelectorAll('.tc-seg').forEach(seg => {
        seg.querySelectorAll('button').forEach(b => {
          const v = b.dataset.v;
          if (v === 'hline' || v === 'trend') b.classList.toggle('on', this.draw.mode === v);
        });
      });
    }

    // ---------- data ----------
    setSymbol(sym, plan) {
      this.symbol = sym; this.plan = plan || {};
      if (this.el.cmpInput) this.el.cmpInput.value = '';
      this.render();
    }
    _embWindow(c) {
      const n = c.t.length; let st = 0; const range = this.state.range;
      if (range === '1M') st = Math.max(0, n - 21);
      else if (range === '3M') st = Math.max(0, n - 63);
      else if (range === '6M') st = Math.max(0, n - 126);
      else if (range === 'YTD') { const y = new Date().getUTCFullYear(); const i = (c.dates || []).findIndex(d => parseInt(d.slice(0, 4), 10) === y); st = i < 0 ? 0 : i; }
      return c.t.slice(st).map((t, i) => ({ t, o: c.open[st + i], h: c.high[st + i], l: c.low[st + i], c: c.close[st + i], v: null }));
    }
    _fetch(sym, m) {
      const ck = sym + ':' + this.state.range;
      const store = sym === this.symbol ? this.cache : this.compareCache;
      if (store[ck]) return Promise.resolve(store[ck]);
      const url = this.app.LIVE_URL;
      if (!url) {
        const c = (this.app.DATA && this.app.DATA.charts) ? this.app.DATA.charts[sym] : null;
        return Promise.resolve(c ? this._embWindow(c) : []);
      }
      return fetch(url + '?chart=' + encodeURIComponent(sym) + '&range=' + m.range + '&interval=' + m.interval)
        .then(r => r.json())
        .then(d => { const bars = (d.bars || []).filter(b => b.c != null); if (bars.length) store[ck] = bars; return bars; })
        .catch(() => { const c = (this.app.DATA && this.app.DATA.charts) ? this.app.DATA.charts[sym] : null; return c ? this._embWindow(c) : []; });
    }

    render() {
      const sym = this.symbol; if (!sym) return;
      const m = RANGE_MAP[this.state.range] || RANGE_MAP['6M'];
      this.el.key.textContent = 'Loading ' + sym + ' …';
      const needSpy = this.state.bench && !m.intraday && this.compare.length === 0 && sym !== 'SPY';
      const jobs = [this._fetch(sym, m)]
        .concat(this.compare.map(s => this._fetch(s, m).then(b => ({ sym: s, bars: b }))))
        .concat(needSpy ? [this._fetch('SPY', m).then(b => ({ sym: 'SPY', bars: b, _bench: true }))] : []);
      Promise.all(jobs).then(res => {
        const bars = res[0];
        if (!bars || !bars.length) { this.el.key.textContent = 'No data for ' + sym + '.'; return; }
        const rest = res.slice(1);
        const cmp = rest.filter(x => x.bars && x.bars.length && !x._bench);
        const spy = rest.find(x => x._bench && x.bars && x.bars.length);
        if (this.symbol === sym && (RANGE_MAP[this.state.range] || {}).range === m.range) this._draw(bars, cmp, m, spy);
      });
    }

    // ---------- drawing the panels ----------
    _draw(bars, cmp, m, spy) {
      const t = theme(); const p = this.plan || {};
      this._destroy();
      const T = bars.map(b => b.t), O = bars.map(b => b.o), H = bars.map(b => b.h), L = bars.map(b => b.l), C = bars.map(b => b.c), V = bars.map(b => b.v);
      const live = this._live && this._live[this.symbol];
      if (live != null && C.length) { C[C.length - 1] = live; if (H[H.length - 1] != null && live > H[H.length - 1]) H[H.length - 1] = live; if (L[L.length - 1] != null && live < L[L.length - 1]) L[L.length - 1] = live; }
      const comparing = cmp.length > 0;
      const useCandle = this.state.type === 'candle' && this._finOK && O.some(v => v != null) && !comparing;
      const ds = [];

      if (comparing) {
        // percentage-rebased comparison mode
        const reb = (arr) => { const base = arr.find(v => v != null); return arr.map(v => v == null ? null : (v / base - 1) * 100); };
        ds.push({ type: 'line', label: this.symbol, data: T.map((x, i) => ({ x, y: reb(C)[i] })), borderColor: t.accent, borderWidth: 2, pointRadius: 0, fill: false, order: 1 });
        const pal = ['#2ea043', '#f0883e', '#a371f7', '#56d4dd', '#db61a2', '#f85149'];
        cmp.forEach((cc, i) => { const cc2 = cc.bars.map(b => b.c); ds.push({ type: 'line', label: cc.sym, data: cc.bars.map((b, j) => ({ x: b.t, y: reb(cc2)[j] })), borderColor: pal[i % pal.length], borderWidth: 1.5, pointRadius: 0, fill: false, order: 2 }); });
      } else if (useCandle) {
        ds.push({ type: 'candlestick', label: this.symbol, order: 3, data: bars.map((b, i) => ({ x: T[i], o: O[i], h: H[i], l: L[i], c: C[i] })), color: { up: t.buy, down: t.sell, unchanged: t.muted }, borderColor: { up: t.buy, down: t.sell, unchanged: t.muted } });
      } else {
        const pcol = this._dir(C) ? t.buy : t.sell;
        const area = this.state.type === 'area';
        ds.push({ type: 'line', label: 'Price', order: 3, borderColor: pcol, borderWidth: 2, pointRadius: 0, fill: area, backgroundColor: area ? this._grad(pcol) : undefined, data: T.map((x, i) => ({ x, y: C[i] })) });
      }

      if (!comparing && !m.intraday) {
        if (this.state.ovl.sma) {
          const f = sma(C, 20), s = sma(C, 50);
          ds.push(this._ovl('MA 20', f, T, 'rgba(56,139,253,0.8)'));
          ds.push(this._ovl('MA 50', s, T, 'rgba(240,136,62,0.8)'));
        }
        if (this.state.ovl.ema) EMA_RIBBON.forEach(([p2, col]) => ds.push(this._ovl('EMA ' + p2, ema(C, p2), T, col, 1)));
        if (this.state.ovl.boll) {
          const bb = boll(C, 20);
          ds.push({ type: 'line', label: 'BB up', order: 5, borderColor: 'rgba(163,113,247,0.5)', borderWidth: 1, borderDash: [3, 3], pointRadius: 0, fill: false, data: T.map((x, i) => ({ x, y: bb.up[i] })) });
          ds.push({ type: 'line', label: '_bblo', order: 5, borderColor: 'rgba(163,113,247,0.5)', borderWidth: 1, borderDash: [3, 3], pointRadius: 0, fill: '-1', backgroundColor: 'rgba(163,113,247,0.06)', data: T.map((x, i) => ({ x, y: bb.lo[i] })) });
        }
      }
      if (!comparing && this.state.ovl.vwap) ds.push(this._ovl('VWAP', vwap(H, L, C, V, m.intraday, T), T, 'rgba(232,200,120,0.95)', 1.4));

      // y range fit + plan lines (only in price mode, not %-compare)
      let y2 = false, benchNote = '';
      const opts = this._priceOpts(m, t, comparing);
      if (!comparing) {
        const ys = (useCandle ? [].concat(H, L) : C).filter(v => v != null);
        const lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys), spn = (hi - lo) || hi * 0.02, pad = Math.max(spn * 0.06, hi * 0.002);
        const within = v => v != null && v >= lo - spn * 0.5 && v <= hi + spn * 0.5;
        if (!m.intraday) {
          if (within(p.entry)) ds.push(this._hline(p.entry, t.muted, 'Buy near', T));
          if (within(p.target)) ds.push(this._hline(p.target, t.buy, 'Target', T));
          if (within(p.stop)) ds.push(this._hline(p.stop, t.sell, 'Stop', T));
        }
        if (!this.state.log) { opts.scales.y.min = lo - pad; opts.scales.y.max = hi + pad; }
        if (this.state.bench && spy && spy.bars.length && !m.intraday) {
          const sc = spy.bars.map(b => b.c), base = sc.find(v => v != null);
          if (base) {
            const bpts = spy.bars.map(b => ({ x: b.t, y: (b.c / base - 1) * 100 }));
            ds.push({ type: 'line', label: 'S&P 500 (%)', data: bpts, borderColor: t.accent, borderWidth: 1.4, pointRadius: 0, fill: false, yAxisID: 'y2', order: 2 });
            y2 = true; opts.scales.y2 = this._pctAxis(t); benchNote = ' · navy = S&P 500 % (left axis)';
          }
        }
      }
      this.cur = { bars, base: C.find(v => v != null), intraday: m.intraday };
      this.charts.price = new window.Chart(this.el.price, { data: { datasets: ds }, options: opts });
      this._redrawOverlay();

      this._panel('vol', this.state.panels.vol && V.some(v => v != null && v > 0), () => this._volCfg(T, O, C, V, m, t));
      this._panel('macd', this.state.panels.macd && !m.intraday, () => this._macdCfg(C, T, m, t));
      this._panel('rsi', this.state.panels.rsi, () => this._rsiCfg(C, T, m, t));
      this._panel('stoch', this.state.panels.stoch, () => this._stochCfg(H, L, C, T, m, t));

      this._updateReadout();
      const dfmt = ms => new Date(ms).toLocaleDateString([], { month: 'short', day: 'numeric', year: m.intraday ? undefined : 'numeric' });
      const span = m.intraday ? (this.state.range === '1D' ? 'Session of ' + dfmt(T[T.length - 1]) : dfmt(T[0]) + ' – ' + dfmt(T[T.length - 1])) : dfmt(T[0]) + ' – ' + dfmt(T[T.length - 1]);
      const cmpNote = comparing ? ' · comparing ' + [this.symbol].concat(this.compare).join(', ') + ' (%)' : '';
      this.el.key.innerHTML = span + ' · ' + bars.length + ' bars · live from Yahoo. Hover to scrub.' + benchNote + cmpNote
        + (this.compare.length ? '  ' + this.compare.map(s => '<span class="tc-chip" data-rm="' + s + '">' + s + ' ✕</span>').join(' ') : '');
      Array.prototype.forEach.call(this.el.key.querySelectorAll('[data-rm]'), el => el.onclick = () => { this.compare = this.compare.filter(x => x !== el.dataset.rm); this.render(); });
    }

    _ovl(label, arr, T, color, w) { return { type: 'line', label, order: 4, borderColor: color, borderWidth: w || 1.3, pointRadius: 0, fill: false, data: T.map((x, i) => ({ x, y: arr[i] })) }; }
    _hline(v, col, lab, T) { return { type: 'line', label: lab, borderColor: col, borderDash: [5, 4], borderWidth: 1, pointRadius: 0, order: 2, fill: false, data: [{ x: T[0], y: v }, { x: T[T.length - 1], y: v }] }; }
    _dir(arr) { const a = arr.find(v => v != null); let b = null; for (let i = arr.length - 1; i >= 0; i--) { if (arr[i] != null) { b = arr[i]; break; } } return !(b != null && a != null && b < a); }
    _grad(color) { return (ctx) => { const ch = ctx.chart, ar = ch.chartArea; if (!ar) return color; const g = ch.ctx.createLinearGradient(0, ar.top, 0, ar.bottom); g.addColorStop(0, color + '44'); g.addColorStop(1, color + '00'); return g; }; }

    _priceOpts(m, t, comparing) {
      const self = this;
      const yType = (this.state.log && !comparing) ? 'logarithmic' : 'linear';
      return {
        responsive: true, maintainAspectRatio: false, parsing: false, animation: false,
        _crosshair: true,
        interaction: { mode: 'index', intersect: false },
        onHover: (e, act) => self._updateReadout(act && act.length ? act[0].index : null),
        elements: { line: { tension: 0.15 } },
        plugins: {
          legend: { display: comparing, labels: { color: t.muted, usePointStyle: true, boxWidth: 8, filter: it => it.text && it.text[0] !== '_' } },
          tooltip: { enabled: false },
          zoom: { zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x' }, pan: { enabled: true, mode: 'x' } },
        },
        scales: {
          x: { type: 'time', time: { unit: m.unit, displayFormats: { hour: 'HH:mm', day: 'MMM d', week: 'MMM d', month: 'MMM yyyy', year: 'yyyy' } }, ticks: { color: t.muted, maxTicksLimit: 7 }, grid: { display: false } },
          y: { type: yType, position: 'right', ticks: { color: t.muted, callback: v => comparing ? (v > 0 ? '+' : '') + Math.round(v) + '%' : v }, grid: { color: t.grid } },
        },
      };
    }
    _pctAxis(t) { return { position: 'left', ticks: { color: t.accent, callback: v => v + '%' }, grid: { display: false } }; }
    _subOpts(m, t, extra) {
      const o = { responsive: true, maintainAspectRatio: false, parsing: false, animation: false, interaction: { mode: 'index', intersect: false }, plugins: { legend: { display: false }, tooltip: { enabled: false } }, scales: { x: { type: 'time', time: { unit: m.unit }, ticks: { display: false }, grid: { display: false } }, y: { position: 'right', ticks: { color: t.muted, maxTicksLimit: 3 }, grid: { color: t.grid } } } };
      if (extra) extra(o);
      return o;
    }
    _volCfg(T, O, C, V, m, t) {
      const cols = T.map((x, i) => { const o = O[i] != null ? O[i] : (C[i - 1] != null ? C[i - 1] : C[i]); return C[i] >= o ? t.buy + '66' : t.sell + '66'; });
      return { data: { datasets: [{ type: 'bar', data: T.map((x, i) => ({ x, y: V[i] || 0 })), backgroundColor: cols, borderWidth: 0, barPercentage: 1, categoryPercentage: 1 }] }, options: this._subOpts(m, t, o => { o.scales.y.beginAtZero = true; o.scales.y.ticks.callback = v => volFmt(v); }) };
    }
    _macdCfg(C, T, m, t) {
      const mc = macd(C);
      return { data: { datasets: [
        { type: 'bar', data: T.map((x, i) => ({ x, y: mc.hist[i] })), order: 3, borderWidth: 0, backgroundColor: T.map((x, i) => mc.hist[i] >= 0 ? t.buy + '99' : t.sell + '99') },
        { type: 'line', data: T.map((x, i) => ({ x, y: mc.line[i] })), borderColor: t.accent, borderWidth: 1.2, pointRadius: 0, order: 1 },
        { type: 'line', data: T.map((x, i) => ({ x, y: mc.sig[i] })), borderColor: '#f0883e', borderWidth: 1.2, pointRadius: 0, order: 1 },
      ] }, options: this._subOpts(m, t) };
    }
    _rsiCfg(C, T, m, t) {
      const rr = rsi(C, 14);
      const opts = this._subOpts(m, t, o => { o.scales.y.min = 0; o.scales.y.max = 100; o.scales.y.ticks.callback = v => v; });
      return { data: { datasets: [
        { type: 'line', data: T.map((x, i) => ({ x, y: rr[i] })), borderColor: '#a371f7', borderWidth: 1.4, pointRadius: 0 },
        { type: 'line', data: [{ x: T[0], y: 70 }, { x: T[T.length - 1], y: 70 }], borderColor: t.sell + '66', borderWidth: 1, borderDash: [4, 4], pointRadius: 0 },
        { type: 'line', data: [{ x: T[0], y: 30 }, { x: T[T.length - 1], y: 30 }], borderColor: t.buy + '66', borderWidth: 1, borderDash: [4, 4], pointRadius: 0 },
      ] }, options: opts };
    }
    _stochCfg(H, L, C, T, m, t) {
      const s = stoch(H, L, C, 14, 3);
      const opts = this._subOpts(m, t, o => { o.scales.y.min = 0; o.scales.y.max = 100; });
      return { data: { datasets: [
        { type: 'line', data: T.map((x, i) => ({ x, y: s.k[i] })), borderColor: t.accent, borderWidth: 1.3, pointRadius: 0 },
        { type: 'line', data: T.map((x, i) => ({ x, y: s.d[i] })), borderColor: '#f0883e', borderWidth: 1.2, pointRadius: 0 },
        { type: 'line', data: [{ x: T[0], y: 80 }, { x: T[T.length - 1], y: 80 }], borderColor: t.sell + '55', borderWidth: 1, borderDash: [4, 4], pointRadius: 0 },
        { type: 'line', data: [{ x: T[0], y: 20 }, { x: T[T.length - 1], y: 20 }], borderColor: t.buy + '55', borderWidth: 1, borderDash: [4, 4], pointRadius: 0 },
      ] }, options: opts };
    }
    _panel(name, show, cfgFn) {
      const box = this.el[name];
      if (this.charts[name]) { this.charts[name].destroy(); delete this.charts[name]; }
      if (!show) { box.style.display = 'none'; return; }
      box.style.display = 'block';
      const cv = box.querySelector('canvas');
      this.charts[name] = new window.Chart(cv, cfgFn());
    }

    // ---------- readout ----------
    _updateReadout(idx) {
      const r = this.el.readout; if (!r || !this.cur.bars.length) return;
      if (idx == null || !this.cur.bars[idx]) idx = this.cur.bars.length - 1;
      const b = this.cur.bars[idx]; if (!b || b.c == null) return;
      const live = this._live && this._live[this.symbol];
      const cl = (idx === this.cur.bars.length - 1 && live != null) ? live : b.c;
      const chg = this.cur.base ? (cl / this.cur.base - 1) * 100 : 0;
      const t = theme(), up = chg >= 0, col = up ? t.buy : t.sell;
      const d = new Date(b.t);
      const ds = this.cur.intraday ? d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : d.toLocaleDateString([], { year: 'numeric', month: 'short', day: 'numeric' });
      const ohlc = (b.o != null) ? '<span class="tc-ohlc">O ' + money(b.o) + ' · H ' + money(b.h) + ' · L ' + money(b.l) + ' · C ' + money(cl) + (b.v ? ' · Vol ' + volFmt(b.v) : '') + '</span>' : '';
      r.innerHTML = '<span class="tc-sym">' + (this.symbol || '') + '</span><span class="tc-price-lg">' + money(cl) + '</span>'
        + '<span class="tc-chg" style="color:' + col + ';">' + (up ? '+' : '') + chg.toFixed(2) + '% over range</span>' + ohlc
        + '<span class="tc-date">' + ds + '</span>';
    }

    // ---------- drawing tools ----------
    _wireDraw() {
      const c = this.el.price;
      const xyToData = (ev) => {
        const ch = this.charts.price; if (!ch) return null;
        const rect = c.getBoundingClientRect();
        const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
        return { x: ch.scales.x.getValueForPixel(px), y: ch.scales.y.getValueForPixel(py) };
      };
      c.addEventListener('click', (ev) => {
        if (this.draw.mode !== 'hline') return;
        const d = xyToData(ev); if (!d) return;
        (this.draw.items[this.symbol] = this.draw.items[this.symbol] || []).push({ type: 'h', y: d.y });
        this._redrawOverlay();
      });
      c.addEventListener('mousedown', (ev) => { if (this.draw.mode !== 'trend') return; this.draw.pending = xyToData(ev); });
      c.addEventListener('mouseup', (ev) => {
        if (this.draw.mode !== 'trend' || !this.draw.pending) return;
        const a = this.draw.pending, b = xyToData(ev); this.draw.pending = null;
        if (a && b) { (this.draw.items[this.symbol] = this.draw.items[this.symbol] || []).push({ type: 't', x1: a.x, y1: a.y, x2: b.x, y2: b.y }); this._redrawOverlay(); }
      });
    }
    _syncDrawCursor() { this.el.price.style.cursor = this.draw.mode ? 'crosshair' : 'default'; }
    _redrawOverlay() {
      const ch = this.charts.price, cv = this.el.drawc; if (!ch || !cv) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = this.el.price.getBoundingClientRect();
      cv.width = rect.width * dpr; cv.height = rect.height * dpr; cv.style.width = rect.width + 'px'; cv.style.height = rect.height + 'px';
      const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, rect.width, rect.height);
      const items = this.draw.items[this.symbol] || []; const t = theme();
      ctx.strokeStyle = t.accent; ctx.lineWidth = 1.4;
      const X = ch.scales.x, Y = ch.scales.y;
      items.forEach(it => {
        ctx.beginPath();
        if (it.type === 'h') { const py = Y.getPixelForValue(it.y); ctx.moveTo(X.left, py); ctx.lineTo(X.right, py); }
        else { ctx.moveTo(X.getPixelForValue(it.x1), Y.getPixelForValue(it.y1)); ctx.lineTo(X.getPixelForValue(it.x2), Y.getPixelForValue(it.y2)); }
        ctx.stroke();
      });
    }

    // ---------- live + theme ----------
    onLive(prices) { this._live = prices || {}; this._updateReadout(); if (this.charts.price) { const ds = this.charts.price.data.datasets.find(d => d.label === 'Price' || d.label === this.symbol); if (ds && ds.data && ds.data.length) { const last = ds.data[ds.data.length - 1], lp = this._live[this.symbol]; if (last && lp != null) { if ('c' in last) { last.c = lp; if (lp > last.h) last.h = lp; if (lp < last.l) last.l = lp; } else last.y = lp; this.charts.price.update('none'); this._redrawOverlay(); } } } }
    applyTheme() { if (this.symbol) this.render(); }
    resize() { Object.keys(this.charts).forEach(k => { try { this.charts[k].resize(); } catch (e) {} }); this._redrawOverlay(); }
    _destroy() { Object.keys(this.charts).forEach(k => { try { this.charts[k].destroy(); } catch (e) {} delete this.charts[k]; }); }
  }

  window.TradeChart = TradeChart;
})();
