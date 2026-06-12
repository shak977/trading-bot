/* TradeChart — thin wrapper around the TradingView Advanced Chart widget.
 *
 * Keeps the same interface the dashboard already uses (new TradeChart(el, opts),
 * setSymbol, applyTheme, onLive, resize) but renders a full TradingView chart —
 * native candlesticks, timeframes, indicators, drawing tools, comparisons. The
 * widget fetches its own data, so no Worker/Yahoo wiring is needed for display.
 *
 * Requires https://s3.tradingview.com/tv.js to be loaded on the page.
 */
(function () {
  'use strict';

  function curTheme() {
    try { return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'; }
    catch (e) { return 'light'; }
  }

  // run cb once tv.js is available (it loads async)
  function tvReady(cb) {
    if (window.TradingView && window.TradingView.widget) { cb(); return; }
    let n = 0;
    const t = setInterval(function () {
      if (window.TradingView && window.TradingView.widget) { clearInterval(t); cb(); }
      else if (++n > 120) { clearInterval(t); }
    }, 100);
  }

  let _seq = 0;

  class TradeChart {
    constructor(root, opts) {
      opts = opts || {};
      this.root = root;
      this.compact = !!opts.compact;
      this.symbol = null;
      this.plan = {};
      this.id = 'tv' + (++_seq);
      root.classList.add('tc');
      root.innerHTML =
        '<div class="tv-wrap' + (this.compact ? ' tv-compact' : '') + '">'
        + '<div id="' + this.id + '" style="height:100%;width:100%;"></div></div>'
        + '<div class="tc-key" style="margin-top:6px;"></div>';
      this.key = root.querySelector('.tc-key');
    }

    _tvSym(s) { return (s || '').toUpperCase(); }

    setSymbol(sym, plan) {
      this.symbol = sym;
      this.plan = plan || {};
      this._render();
    }

    _render() {
      const self = this;
      if (!this.symbol) return;
      if (this.key) this.key.textContent = 'Loading ' + this.symbol + ' …';
      tvReady(function () {
        const el = document.getElementById(self.id);
        if (!el) return;
        el.innerHTML = '';
        try {
          self._w = new window.TradingView.widget({
            container_id: self.id,
            autosize: true,
            symbol: self._tvSym(self.symbol),
            interval: 'D',
            timezone: 'Etc/UTC',
            theme: curTheme(),
            style: '1',
            locale: 'en',
            hide_side_toolbar: false,
            allow_symbol_change: true,
            withdateranges: true,
            details: !self.compact,
            calendar: false,
          });
          if (self.key) {
            self.key.innerHTML = '<b>' + self.symbol + '</b> · live TradingView chart — '
              + 'use the toolbar for timeframes, indicators, comparisons and drawing tools.';
          }
        } catch (e) {
          if (self.key) self.key.textContent = 'Chart unavailable right now.';
        }
      });
    }

    onLive() { /* TradingView self-updates; nothing to push */ }
    applyTheme() { if (this.symbol) this._render(); }   // theme change needs a re-mount
    resize() { try { window.dispatchEvent(new Event('resize')); } catch (e) {} }
  }

  window.TradeChart = TradeChart;
})();
