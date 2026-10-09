// skeebert.frgmt.xyz
// Every glyph on this page is drawn from /glyphs/glyphs.json (real stroke data from a
// checkpoint, written by scripts/render_site_glyphs.py). Every benchmark number comes from
// /bench.json and /trainlog.json (written by scripts/bench.py). Nothing here invents a
// glyph, a meaning or a number.
(function () {
  "use strict";

  var doc = document, root = doc.documentElement;
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var DPR = Math.min(2, window.devicePixelRatio || 1);
  var SVGNS = "http://www.w3.org/2000/svg";
  var RAMP = " .:-=+*#%@";
  var L = RAMP.length;
  var BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
  // skeebert/render.py presentation colours
  var BG = [14, 16, 24], VIG = [26, 30, 44], INK = [240, 232, 214], GLOW = [226, 150, 64];

  function dither(v, x, y) {
    var l = Math.floor(v * (L - 1) + (BAYER[((y & 3) << 2) | (x & 3)] + 0.5) / 16);
    return l < 0 ? 0 : l > L - 1 ? L - 1 : l;
  }
  function $(id) { return doc.getElementById(id); }
  function el(tag, attrs, text) {
    var n = doc.createElement(tag);
    if (attrs) for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  }
  function sv(tag, attrs, text) {
    var n = doc.createElementNS(SVGNS, tag);
    if (attrs) for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  }
  function cssVar(name) { return getComputedStyle(root).getPropertyValue(name).trim(); }
  function fmtInt(n) { return Number(n).toLocaleString("en-US"); }
  function pct(v, d) { return (v * 100).toFixed(d == null ? 1 : d) + "%"; }
  function usd(v, d) { return "$" + v.toFixed(d == null ? 4 : d); }
  function dur(s) { s = Math.round(s); var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60); return h ? h + "h" + String(m).padStart(2, "0") + "m" : m + "m" + String(s % 60).padStart(2, "0") + "s"; }
  function shuffle(a) { for (var i = a.length - 1; i > 0; i--) { var j = Math.floor(Math.random() * (i + 1)); var t = a[i]; a[i] = a[j]; a[j] = t; } return a; }
  function onVisible(node, fn, margin) {
    if (!("IntersectionObserver" in window)) { fn(true); return; }
    new IntersectionObserver(function (es) { es.forEach(function (e) { fn(e.isIntersecting); }); }, { rootMargin: margin || "0px" }).observe(node);
  }

  // ---------------------------------------------------------------- theme
  function currentTheme() {
    var t = root.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  var themeListeners = [];
  function setupTheme() {
    var btn = doc.querySelector(".theme");
    if (!btn) return;
    function label() { btn.setAttribute("aria-label", currentTheme() === "dark" ? "Switch to light theme" : "Switch to dark theme"); }
    label();
    btn.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) { /* storage blocked: theme lasts this page view */ }
      label();
      themeListeners.forEach(function (f) { f(); });
    });
    window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", function () { label(); themeListeners.forEach(function (f) { f(); }); });
  }

  // ---------------------------------------------------------------- glyph coverage (render.py, in JS)
  // Same soft distance field as skeebert/render.py: per pixel, sigmoid((half_width - d) / softness) * ink,
  // strokes combined as 1 - prod(1 - c). `progress` in [0, K] draws strokes in order; a partial
  // stroke is the curve from t = 0 to t = frac, sampled with the same 12 segments.
  var SEG = 12;
  function coverage(strokes, size, progress) {
    var n = size * size, empty = new Float32Array(n).fill(1);
    var soft = 0.6 * (2 / size), half = 1 / size, step = 2 / size;
    var px = new Float32Array(size);
    for (var i = 0; i < size; i++) px[i] = -1 + half + i * step;
    var K = strokes.length, P = progress == null ? K : progress;
    var ax = new Float32Array(SEG + 1), ay = new Float32Array(SEG + 1);
    for (var k = 0; k < K; k++) {
      var t1 = Math.max(0, Math.min(1, P - k));
      if (t1 <= 0) break;
      var s = strokes[k];
      for (var j = 0; j <= SEG; j++) {
        var t = (j / SEG) * t1, u = 1 - t;
        ax[j] = u * u * s[0] + 2 * u * t * s[2] + t * t * s[4];
        ay[j] = u * u * s[1] + 2 * u * t * s[3] + t * t * s[5];
      }
      var hw = s[6], ink = Math.max(0, Math.min(1, s[7]));
      // bounding box (+ margin where the sigmoid is still > ~1e-4)
      var minx = 9, maxx = -9, miny = 9, maxy = -9;
      for (j = 0; j <= SEG; j++) { if (ax[j] < minx) minx = ax[j]; if (ax[j] > maxx) maxx = ax[j]; if (ay[j] < miny) miny = ay[j]; if (ay[j] > maxy) maxy = ay[j]; }
      var m = hw + soft * 10;
      var x0 = Math.max(0, Math.floor((minx - m + 1) / step)), x1 = Math.min(size - 1, Math.ceil((maxx + m + 1) / step));
      var y0 = Math.max(0, Math.floor((miny - m + 1) / step)), y1 = Math.min(size - 1, Math.ceil((maxy + m + 1) / step));
      for (var yi = y0; yi <= y1; yi++) {
        var py = px[yi];
        for (var xi = x0; xi <= x1; xi++) {
          var pxv = px[xi], best = 1e9;
          for (j = 0; j < SEG; j++) {
            var bx = ax[j + 1] - ax[j], by = ay[j + 1] - ay[j];
            var qx = pxv - ax[j], qy = py - ay[j];
            var den = bx * bx + by * by; if (den < 1e-8) den = 1e-8;
            var tt = (qx * bx + qy * by) / den; tt = tt < 0 ? 0 : tt > 1 ? 1 : tt;
            var dx = qx - tt * bx, dy = qy - tt * by, d2 = dx * dx + dy * dy;
            if (d2 < best) best = d2;
          }
          var dist = Math.sqrt(best + 1e-10);
          var c = ink / (1 + Math.exp(-(hw - dist) / soft));
          empty[yi * size + xi] *= 1 - c;
        }
      }
    }
    for (i = 0; i < n; i++) empty[i] = 1 - empty[i];
    return empty;
  }
  // render.colorize's glow: Gaussian blur, sigma = size / 48, times 0.85
  function haloOf(cov, size) {
    var sigma = Math.max(1, size / 48), r = Math.ceil(sigma * 3), ker = [], sum = 0, i;
    for (i = -r; i <= r; i++) { var w = Math.exp(-(i * i) / (2 * sigma * sigma)); ker.push(w); sum += w; }
    for (i = 0; i < ker.length; i++) ker[i] /= sum;
    var tmp = new Float32Array(size * size), out = new Float32Array(size * size);
    for (var y = 0; y < size; y++) for (var x = 0; x < size; x++) {
      var a = 0; for (var k = -r; k <= r; k++) { var xx = x + k; if (xx >= 0 && xx < size) a += cov[y * size + xx] * ker[k + r]; } tmp[y * size + x] = a;
    }
    for (y = 0; y < size; y++) for (x = 0; x < size; x++) {
      var b = 0; for (k = -r; k <= r; k++) { var yy = y + k; if (yy >= 0 && yy < size) b += tmp[yy * size + x] * ker[k + r]; } out[y * size + x] = Math.min(1, b * 0.85);
    }
    return out;
  }
  function pixelRGB(x, y, size, c, h) {
    var rx = (x + 0.5) / size - 0.5, ry = (y + 0.5) / size - 0.5;
    var r = Math.sqrt(rx * rx + ry * ry) / 0.7071, centre = Math.pow(Math.max(0, 1 - r), 1.5);
    var out = [0, 0, 0];
    for (var i = 0; i < 3; i++) {
      var bg = BG[i] * (1 - centre) + VIG[i] * centre;
      var v = bg * (1 - h) + GLOW[i] * h;
      out[i] = Math.round(v * (1 - c) + INK[i] * c);
    }
    return out;
  }

  // character atlas: RAMP drawn once per colour, blitted per cell
  function atlas(cell, color) {
    var cs = Math.max(4, Math.round(cell)), cv = doc.createElement("canvas");
    cv.width = cs * L; cv.height = cs;
    var cx = cv.getContext("2d");
    cx.fillStyle = color;
    cx.font = "700 " + Math.round(cs * 1.08) + "px 'JetBrains Mono', ui-monospace, monospace";
    cx.textAlign = "center"; cx.textBaseline = "middle";
    for (var i = 1; i < L; i++) cx.fillText(RAMP[i], i * cs + cs / 2, cs / 2 + cs * 0.04);
    return { cv: cv, cs: cs };
  }

  // ---------------------------------------------------------------- glyph screen
  var N = 64; // one ASCII cell per pixel of the model's 64 x 64 training render
  function GlyphScreen(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.mode = "ascii";
    this.glyph = null;
    this.progress = 0;
    this.lens = null;
    this.raf = 0;
    this.onStroke = null;
    this.resize();
  }
  GlyphScreen.prototype.resize = function () {
    var w = Math.max(64, Math.round(this.canvas.clientWidth * DPR));
    if (w === this.size) return;
    this.size = w; this.canvas.width = w; this.canvas.height = w;
    this.cell = w / N;
    this.aInk = atlas(this.cell, "rgb(240,232,214)");
    this.aGlow = atlas(this.cell, "rgb(226,150,64)");
    this.aDim = atlas(this.cell, "rgba(240,232,214,0.13)");
    var g = this.ctx.createRadialGradient(w / 2, w / 2, 0, w / 2, w / 2, w * 0.7071);
    g.addColorStop(0, "rgb(26,30,44)"); g.addColorStop(1, "rgb(14,16,24)");
    this.bg = g;
    this.px = doc.createElement("canvas"); this.px.width = N; this.px.height = N;
    this.draw();
  };
  GlyphScreen.prototype.set = function (glyph, progress) {
    this.glyph = glyph;
    this.progress = progress == null ? glyph.strokes.length : progress;
    this.compute();
    this.draw();
  };
  GlyphScreen.prototype.compute = function () {
    if (!this.glyph) return;
    this.cov = coverage(this.glyph.strokes, N, this.progress);
    this.halo = haloOf(this.cov, N);
  };
  GlyphScreen.prototype.draw = function () {
    var ctx = this.ctx, w = this.size, cell = this.cell;
    if (!w) return;
    ctx.fillStyle = this.bg; ctx.fillRect(0, 0, w, w);
    if (!this.cov) return;
    var cov = this.cov, halo = this.halo, x, y, i;
    if (this.mode === "px64") {
      var px = this.px.getContext("2d"), img = px.createImageData(N, N);
      for (y = 0; y < N; y++) for (x = 0; x < N; x++) {
        i = y * N + x; var c = pixelRGB(x, y, N, cov[i], halo[i]);
        img.data[i * 4] = c[0]; img.data[i * 4 + 1] = c[1]; img.data[i * 4 + 2] = c[2]; img.data[i * 4 + 3] = 255;
      }
      px.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(this.px, 0, 0, w, w);
      return;
    }
    var aI = this.aInk, aG = this.aGlow, aD = this.aDim, cs = aI.cs, lens = this.lens, R2 = 81;
    for (y = 0; y < N; y++) {
      var ty = Math.round(y * cell);
      for (x = 0; x < N; x++) {
        i = y * N + x;
        var tx = Math.round(x * cell);
        if (lens) {
          var lx = x - lens[0], ly = y - lens[1];
          if (lx * lx + ly * ly < R2) {
            var col = pixelRGB(x, y, N, cov[i], halo[i]);
            ctx.fillStyle = "rgb(" + col[0] + "," + col[1] + "," + col[2] + ")";
            ctx.fillRect(tx, ty, Math.ceil(cell), Math.ceil(cell));
            continue;
          }
        }
        var l = dither(Math.pow(Math.min(1, cov[i] * 1.35), 0.7), x, y);
        if (l > 0) { ctx.drawImage(aI.cv, l * cs, 0, cs, cs, tx, ty, cell, cell); continue; }
        var hl = dither(Math.min(1, halo[i] * 1.7), x, y);
        if (hl > 0) { ctx.drawImage(aG.cv, hl * cs, 0, cs, cs, tx, ty, cell, cell); continue; }
        if (!(x & 1) && !(y & 1)) ctx.drawImage(aD.cv, cs, 0, cs, cs, tx, ty, cell, cell);
      }
    }
  };
  // Draw strokes in order. Calls onStroke(k) as stroke k starts and onStroke(K) when done.
  GlyphScreen.prototype.play = function (glyph, delay) {
    var self = this, K = glyph.strokes.length;
    cancelAnimationFrame(this.raf);
    this.glyph = glyph;
    if (reduce) { this.set(glyph, K); if (this.onStroke) this.onStroke(K); return; }
    // visible strokes take longer than near-invisible ones
    var durs = glyph.strokes.map(function (s) { return s[7] > 0.15 ? 620 : 240; });
    var start = performance.now() + (delay || 0), last = -1;
    function frame(now) {
      var t = now - start, p = 0, acc = 0;
      if (t < 0) { self.raf = requestAnimationFrame(frame); return; }
      for (var k = 0; k < K; k++) {
        if (t < acc + durs[k]) { var f = (t - acc) / durs[k]; p = k + (1 - Math.pow(1 - f, 2.2)); break; }
        acc += durs[k]; p = k + 1;
      }
      var cur = Math.min(K, Math.floor(p));
      if (cur !== last) { last = cur; if (self.onStroke) self.onStroke(cur); }
      self.progress = p; self.compute(); self.draw();
      if (p < K) self.raf = requestAnimationFrame(frame);
    }
    this.progress = 0; this.cov = null; this.draw();
    this.raf = requestAnimationFrame(frame);
  };
  GlyphScreen.prototype.enableLens = function () {
    var self = this, c = this.canvas, pending = 0;
    if (!window.matchMedia("(pointer: fine)").matches) return;
    c.addEventListener("pointermove", function (e) {
      var r = c.getBoundingClientRect();
      self.lens = [((e.clientX - r.left) / r.width) * N, ((e.clientY - r.top) / r.height) * N];
      if (!pending) pending = requestAnimationFrame(function () { pending = 0; self.draw(); });
    });
    c.addEventListener("pointerleave", function () { self.lens = null; self.draw(); });
  };

  // ---------------------------------------------------------------- hero
  function strokeText(s, k, K) {
    return "stroke " + (k + 1) + "/" + K + "  " + s.slice(0, 6).map(function (v) { return (v >= 0 ? " " : "") + v.toFixed(3); }).join(" ") + "  w " + s[6].toFixed(3) + "  ink " + s[7].toFixed(2);
  }
  function hero(m) {
    var g = m.glyphs[0], canvas = $("hero-canvas");
    if (!g || !canvas) return;
    canvas.setAttribute("aria-label", "A real skeebert glyph meaning “" + g.gloss + "”, drawn stroke by stroke from its stroke data by " + m.label + ".");
    var scr = new GlyphScreen(canvas), K = g.strokes.length;
    var r1 = $("hero-r1"), r2 = $("hero-r2");
    r1.textContent = g.intent.join(" + ") + " · “" + g.gloss + "”";
    scr.onStroke = function (k) {
      if (k >= K) { r2.textContent = K + " strokes · " + (K * 8) + " numbers · ckpt " + m.model_version; return; }
      r2.textContent = strokeText(g.strokes[k], k, K);
    };
    scr.enableLens();
    var started = false;
    onVisible(canvas, function (vis) { if (vis && !started) { started = true; scr.play(g, 200); } });
    $("hero-replay").addEventListener("click", function () { scr.play(g, 0); });
    window.addEventListener("resize", function () { scr.resize(); });
  }

  // ---------------------------------------------------------------- decoder (Fig. 1)
  function decoder(m) {
    var glyphs = m.glyphs, ui = $("decoder-ui");
    if (!ui || glyphs.length < 2) return;
    var canvas = $("dec-canvas"), img = $("dec-img"), scr = new GlyphScreen(canvas);
    var opts = $("dec-options"), result = $("dec-result"), strip = $("dec-strip"), tbody = doc.querySelector("#dec-strokes tbody");
    var modes = doc.querySelectorAll("#decoder .modes button");
    var nOpts = Math.min(4, glyphs.length), answered = {}, current = -1, thumbs = [];
    var order = shuffle(glyphs.map(function (_, i) { return i; }));
    function atoms(g) { return g.intent.join(" + "); }

    glyphs.forEach(function (g, i) {
      var li = el("li"), b = el("button", { type: "button", "aria-pressed": "false" });
      var t = el("img", { src: "/glyphs/" + (g.thumb || g.webp), alt: "Glyph " + (i + 1) + ", meaning hidden", width: "58", height: "58", decoding: "async", loading: "lazy" });
      b.appendChild(t); b.addEventListener("click", function () { show(i); });
      li.appendChild(b); strip.appendChild(li); thumbs.push({ b: b, img: t });
    });

    function fillStrokes(g) {
      tbody.textContent = "";
      g.strokes.forEach(function (s, k) {
        var tr = el("tr");
        tr.appendChild(el("td", null, String(k + 1)));
        s.forEach(function (v, j) { tr.appendChild(el("td", null, j >= 6 ? v.toFixed(3) : v.toFixed(3))); });
        tbody.appendChild(tr);
      });
    }
    scr.onStroke = function (k) {
      var rows = tbody.querySelectorAll("tr");
      rows.forEach(function (tr, j) { tr.className = j === k ? "on" : j > k ? "pending" : ""; });
    };
    function setMode(mode) {
      modes.forEach(function (b) { b.setAttribute("aria-checked", b.dataset.mode === mode ? "true" : "false"); });
      scr.mode = mode;
      var png = mode === "png";
      img.hidden = !png; canvas.hidden = png;
      if (png && current >= 0) img.src = "/glyphs/" + glyphs[current].webp;
      if (!png) { scr.resize(); scr.draw(); }
    }
    modes.forEach(function (b, i) {
      b.addEventListener("click", function () { setMode(b.dataset.mode); });
      b.addEventListener("keydown", function (e) {
        var d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
        if (!d) return;
        e.preventDefault();
        var nb = modes[(i + d + modes.length) % modes.length]; nb.focus(); setMode(nb.dataset.mode);
      });
    });
    modes.forEach(function (b) { b.tabIndex = b.getAttribute("aria-checked") === "true" ? 0 : -1; });
    var obs = new MutationObserver(function () { modes.forEach(function (b) { b.tabIndex = b.getAttribute("aria-checked") === "true" ? 0 : -1; }); });
    modes.forEach(function (b) { obs.observe(b, { attributes: true, attributeFilter: ["aria-checked"] }); });

    function tally() {
      var keys = Object.keys(answered);
      if (!keys.length) { $("dec-tally").textContent = ""; return; }
      var right = keys.filter(function (k) { return answered[k]; }).length;
      $("dec-tally").textContent = right + " right out of " + keys.length;
    }
    function show(i, quiet) {
      current = i;
      var g = glyphs[i];
      thumbs.forEach(function (t, k) { t.b.setAttribute("aria-pressed", k === i ? "true" : "false"); });
      canvas.setAttribute("aria-label", "Glyph " + (i + 1) + ". Its meaning is hidden until you guess.");
      img.alt = "Glyph " + (i + 1) + ", the exported 512 pixel image.";
      if (scr.mode === "png") img.src = "/glyphs/" + g.webp;
      fillStrokes(g);
      if (quiet) { scr.set(g); scr.onStroke(g.strokes.length); } else scr.play(g, 0);
      var pool = [], seen = {}; seen[atoms(g)] = true;
      shuffle(glyphs.slice()).forEach(function (o) { var k = atoms(o); if (!seen[k] && pool.length < nOpts - 1) { seen[k] = true; pool.push(o); } });
      var choices = shuffle(pool.concat([g]));
      opts.textContent = ""; result.textContent = "";
      choices.forEach(function (c) {
        var li = el("li"), b = el("button", { type: "button" });
        b.appendChild(el("span", null, atoms(c)));
        b.appendChild(el("span", { class: "mark", "aria-hidden": "true" }));
        b.addEventListener("click", function () { pick(c, b); });
        li.appendChild(b); opts.appendChild(li);
      });
    }
    function pick(choice, button) {
      var g = glyphs[current], right = choice === g;
      if (!(current in answered)) answered[current] = right;
      opts.querySelectorAll("button").forEach(function (b) {
        b.disabled = true;
        var label = b.firstChild.textContent;
        if (label === atoms(g)) { b.dataset.state = "right"; b.lastChild.textContent = "meant"; }
        else if (b === button) { b.dataset.state = "wrong"; b.lastChild.textContent = "your pick"; }
        else b.dataset.state = "dim";
      });
      result.textContent = "";
      result.appendChild(el("strong", null, right ? "Right. " : "Not quite. "));
      result.appendChild(doc.createTextNode("It meant " + atoms(g) + ": “" + g.gloss + "”."));
      canvas.setAttribute("aria-label", "Glyph " + (current + 1) + ", meaning " + atoms(g) + ".");
      thumbs[current].img.alt = "Glyph " + (current + 1) + ", meaning " + atoms(g);
      tally();
      $("dec-next").focus({ preventScroll: true });
    }
    $("dec-next").addEventListener("click", function () {
      var pos = order.indexOf(current);
      for (var s = 1; s <= order.length; s++) {
        var cand = order[(pos + s) % order.length];
        if (!(cand in answered) || s === order.length) { show(cand); return; }
      }
    });
    ui.hidden = false;
    scr.resize();
    var first = order[0], started = false;
    show(first, true);
    onVisible(canvas, function (vis) { if (vis && !started) { started = true; scr.play(glyphs[first], 150); } });
    window.addEventListener("resize", function () { if (scr.mode !== "png") scr.resize(); });
  }

  // ---------------------------------------------------------------- sample exchange
  function chat(m) {
    var list = $("chat");
    if (!list || !m.examples) return;
    m.examples.forEach(function (ex) {
      var s = ex.situation;
      if (s.charAt(0) !== "'") return; // only examples that are literal messages
      var li = el("li");
      li.appendChild(el("img", { src: "/glyphs/" + ex.webp, alt: "skeebert's glyph for " + ex.intent.join(" + "), width: "76", height: "76", loading: "lazy", decoding: "async" }));
      var d = el("div");
      d.appendChild(el("p", { class: "said" }, "“" + s.slice(1, -1) + "”"));
      var meant = el("p", { class: "meant" }, "skeebert: ");
      meant.appendChild(el("b", null, ex.intent.join(" + ")));
      d.appendChild(meant);
      li.appendChild(d);
      list.appendChild(li);
    });
  }

  // ---------------------------------------------------------------- ASCII fields
  // Glyph silhouettes (real coverage of the curated glyphs) drifting along the page edges,
  // ordered-dithered into characters, densest at the outer edge.
  function fields(m) {
    var cvs = doc.querySelectorAll("canvas.edge");
    if (!cvs.length || !m.glyphs.length) return;
    var tiles = m.glyphs.map(function (g) { var c = coverage(g.strokes, N); return { c: c, h: haloOf(c, N) }; });
    var CELL = 11, colA, colB, atlasA, atlasB, running = false, last = 0, t0 = performance.now(), sized = false;
    var state = cvs.length ? [] : null;
    function colors() {
      colA = "rgba(" + cssVar("--field-b") + "," + cssVar("--field-b-a") + ")";
      colB = "rgba(" + cssVar("--field") + "," + cssVar("--field-a") + ")";
      atlasA = atlas(CELL * DPR, colA); atlasB = atlas(CELL * DPR, colB);
    }
    function size() {
      var margin = (window.innerWidth - Math.min(window.innerWidth, 1240)) / 2;
      var w = Math.max(0, Math.min(230, margin + 30));
      state = [];
      cvs.forEach(function (c, idx) {
        var vis = getComputedStyle(c).display !== "none" && w > 40;
        c.style.width = w + "px";
        c.width = Math.round(w * DPR); c.height = Math.round(window.innerHeight * DPR);
        state.push({ c: c, ctx: c.getContext("2d"), left: idx === 0, vis: vis, cols: Math.ceil(w / CELL), rows: Math.ceil(window.innerHeight / CELL) + 1 });
      });
      sized = true;
    }
    function draw(now) {
      var drift = reduce ? 0 : (now - t0) * 0.006, sc = reduce ? 0 : window.scrollY * 0.22;
      state.forEach(function (s, idx) {
        if (!s.vis) return;
        var ctx = s.ctx, cs = atlasA.cs, cell = CELL * DPR;
        ctx.clearRect(0, 0, s.c.width, s.c.height);
        var tile = s.cols * 1.15; // tile edge, in cells
        var off = (sc + drift) / CELL + idx * tile * 0.5;
        var r0 = Math.floor(off), frac = off - r0;
        for (var ry = 0; ry < s.rows; ry++) {
          var wy = ry + r0, ti = Math.floor(wy / tile), gy = (wy - ti * tile) / tile;
          var T = tiles[((ti * 5 + idx * 7) % tiles.length + tiles.length) % tiles.length];
          var sy = Math.min(N - 1, Math.floor(gy * N));
          var y = Math.round((ry - frac) * cell);
          for (var cx = 0; cx < s.cols; cx++) {
            var dEdge = s.left ? cx : s.cols - 1 - cx;
            var fade = Math.max(0, 1 - dEdge / s.cols);
            fade = fade * fade;
            var gx = cx / s.cols, sx = Math.min(N - 1, Math.floor((s.left ? gx : 1 - gx) * N));
            var k = sy * N + sx, ink = T.c[k], h = T.h[k];
            var x = Math.round((s.left ? cx : cx) * cell);
            var l = dither(Math.min(1, ink * 1.6) * (0.35 + 0.65 * fade), cx, wy);
            if (l > 0) { ctx.drawImage(atlasA.cv, l * cs, 0, cs, cs, x, y, cell, cell); continue; }
            var base = 0.10 * fade + h * 1.2 * fade;
            var lb = dither(Math.min(1, base), cx + 1, wy + 2);
            if (lb > 0) ctx.drawImage(atlasB.cv, lb * cs, 0, cs, cs, x, y, cell, cell);
          }
        }
      });
    }
    function loop(now) {
      if (!running) return;
      if (now - last > 90) { last = now; draw(now); }
      requestAnimationFrame(loop);
    }
    function start() {
      if (reduce) { draw(performance.now()); return; }
      if (running || doc.hidden) return;
      running = true; requestAnimationFrame(loop);
    }
    function stop() { running = false; }
    colors(); size(); start();
    if (reduce) draw(performance.now());
    var rt = 0;
    window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(function () { size(); draw(performance.now()); }, 150); });
    doc.addEventListener("visibilitychange", function () { if (doc.hidden) stop(); else start(); });
    themeListeners.push(function () { colors(); draw(performance.now()); });
    void sized;
  }

  // ---------------------------------------------------------------- dithered rules and figure halos
  function ruleCanvas(host) {
    var c = el("canvas"); host.appendChild(c);
    function paint() {
      var w = host.clientWidth - parseFloat(getComputedStyle(host).paddingLeft) * 2, h = host.clientHeight;
      if (w <= 0) return;
      c.style.width = w + "px"; c.style.height = h + "px";
      c.width = Math.round(w * DPR); c.height = Math.round(h * DPR);
      var ctx = c.getContext("2d"), cell = 10 * DPR, cols = Math.floor(c.width / cell), rows = Math.floor(c.height / cell);
      var a = atlas(cell, "rgba(" + cssVar("--field") + "," + (parseFloat(cssVar("--field-a")) + 0.2) + ")");
      for (var y = 0; y < rows; y++) for (var x = 0; x < cols; x++) {
        var u = x / Math.max(1, cols - 1);
        var v = Math.pow(1 - u, 2.4) * (y === 1 ? 1 : 0.65);
        var l = dither(v, x, y);
        if (l > 0) ctx.drawImage(a.cv, l * a.cs, 0, a.cs, a.cs, x * cell, y * cell, cell, cell);
      }
    }
    paint();
    return paint;
  }
  function haloCanvas(fig) {
    var c = el("canvas", { class: "halo", "aria-hidden": "true" });
    fig.insertBefore(c, fig.firstChild);
    function paint() {
      var w = fig.clientWidth, h = fig.clientHeight;
      if (!w || !h) return;
      c.width = Math.round(w * DPR); c.height = Math.round(h * DPR);
      var ctx = c.getContext("2d"), cell = 7 * DPR, cols = Math.ceil(c.width / cell), rows = Math.ceil(c.height / cell);
      var a = atlas(cell, "rgba(" + cssVar("--field") + "," + cssVar("--field-a") + ")");
      var band = Math.min(4, Math.floor(Math.min(cols, rows) / 6));
      for (var y = 0; y < rows; y++) for (var x = 0; x < cols; x++) {
        var d = Math.min(x, y, cols - 1 - x, rows - 1 - y);
        if (d >= band) { x = Math.max(x, cols - band - 1); continue; }
        var v = Math.pow(1 - d / band, 1.6) * 0.62;
        var l = dither(v, x, y);
        if (l > 0) ctx.drawImage(a.cv, l * a.cs, 0, a.cs, a.cs, x * cell, y * cell, cell, cell);
      }
    }
    paint();
    if ("ResizeObserver" in window) { var t = 0; new ResizeObserver(function () { clearTimeout(t); t = setTimeout(paint, 80); }).observe(fig); }
    return paint;
  }
  function ornaments() {
    var painters = [];
    doc.querySelectorAll(".rule").forEach(function (r) { painters.push(ruleCanvas(r)); });
    doc.querySelectorAll(".fig").forEach(function (f) { painters.push(haloCanvas(f)); });
    var t = 0;
    window.addEventListener("resize", function () { clearTimeout(t); t = setTimeout(function () { painters.forEach(function (p) { p(); }); }, 150); });
    themeListeners.push(function () { painters.forEach(function (p) { p(); }); });
  }

  // ---------------------------------------------------------------- benchmark rendering
  function sweepRow(b, k) { return b.sweep.candidate_sweep.filter(function (r) { return r.candidates === k; })[0]; }
  function pci(p) { return pct(p.value) + " "; }
  function ciText(p) { return "[" + (p.ci95[0] * 100).toFixed(1) + ", " + (p.ci95[1] * 100).toFixed(1) + "]"; }
  function cellProp(td, p, hi) {
    td.className = "n" + (hi ? " hi" : "");
    td.appendChild(doc.createTextNode(pci(p)));
    td.appendChild(el("span", { class: "ci" }, ciText(p)));
  }
  function table(id, head, rows) {
    var t = $(id);
    if (!t) return;
    var thead = el("thead"), tr = el("tr");
    head.forEach(function (h) { var th = el("th", { scope: "col" }, h.t != null ? h.t : h); if (h.n) th.className = "n"; tr.appendChild(th); });
    thead.appendChild(tr); t.appendChild(thead);
    var tb = el("tbody");
    rows.forEach(function (r) {
      var tr2 = el("tr");
      r.forEach(function (c, i) {
        var td = el(i === 0 ? "th" : "td", i === 0 ? { scope: "row" } : null);
        if (c && c.prop) cellProp(td, c.prop, c.hi);
        else { td.textContent = c && c.t != null ? c.t : c; if (c && c.n) td.className = "n" + (c.hi ? " hi" : ""); }
        tr2.appendChild(td);
      });
      tb.appendChild(tr2);
    });
    t.appendChild(tb);
  }

  function bars(b) {
    var box = $("gb-bars"), gb = b.glyphbench;
    if (!box) return;
    var W = 40;
    var rows = [];
    gb.entrants.slice().sort(function (x, y) { return (y.name === "trained") - (x.name === "trained"); }).forEach(function (e) {
      var name = e.name === "trained" ? "skeebert-1" : e.name === "untrained" ? "skeebert, untrained" : e.name + " †";
      var v = e.eligible ? e.score.value : 0;
      rows.push({ name: name, v: v, label: e.eligible ? pct(v) : "0% (ineligible)", cls: e.name === "trained" ? "" : e.eligible ? "dim" : "inel" });
    });
    rows.splice(2, 0, { name: "chance", v: gb.chance, label: pct(gb.chance, 2), cls: "dim" });
    rows.forEach(function (r) {
      var row = el("div", { class: "row " + r.cls });
      var n = Math.round(r.v * W);
      row.appendChild(el("span", { class: "name" }, r.name));
      var bar = el("span", { class: "bar", "aria-hidden": "true" });
      bar.appendChild(doc.createTextNode(new Array(n + 1).join("#")));
      bar.appendChild(el("span", { class: "rest" }, new Array(W - n + 1).join(".")));
      row.appendChild(bar);
      row.appendChild(el("span", { class: "val" }, r.label));
      box.appendChild(row);
    });
    var t = gb.entrants.filter(function (e) { return e.name === "trained"; })[0].score;
    var u = gb.entrants.filter(function (e) { return e.name === "untrained"; })[0].score;
    $("gb-note").textContent = "n = " + fmtInt(gb.n) + " intents, seed " + gb.seed + ", " + gb.candidates + " candidates (chance " + pct(gb.chance, 2) + "). Reader: " + gb.reader + ", images augmented as in training. skeebert-1 " + pct(t.value) + " " + ciText(t) + "; untrained " + pct(u.value) + " " + ciText(u) + ", 95% Wilson. † Not run: see the rule below.";
    $("gb-rule").textContent = "GlyphBench v1, the rule. " + gb.rule;
  }

  function trainChart(t) {
    var box = $("train-chart");
    if (!box) return;
    var W = Math.max(340, Math.min(900, Math.round(box.clientWidth || 760))), H = W < 560 ? 250 : 300, ml = 40, mr = 12, mt = 14, mb = 34, iw = W - ml - mr, ih = H - mt - mb;
    var svg = sv("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Per-batch training accuracy over " + fmtInt(t.steps) + " steps, with " + t.listener_reset_steps.length + " listener resets visible as drops to near zero." });
    function X(s) { return ml + (s / t.steps) * iw; }
    function Y(v) { return mt + (1 - v) * ih; }
    var g = sv("g", { class: "grid" }), ax = sv("g", { class: "ax" });
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      g.appendChild(sv("line", { x1: ml, x2: W - mr, y1: Y(v), y2: Y(v) }));
      ax.appendChild(sv("text", { x: ml - 8, y: Y(v) + 4, "text-anchor": "end" }, (v * 100) + "%"));
    });
    for (var s = 0; s <= t.steps; s += (W < 560 ? 10000 : 5000)) ax.appendChild(sv("text", { x: X(s), y: H - 12, "text-anchor": "middle" }, s === 0 ? "0" : (s / 1000) + "k"));
    svg.appendChild(g); svg.appendChild(ax);
    t.listener_reset_steps.forEach(function (s, i) {
      svg.appendChild(sv("line", { class: "reset", x1: X(s), x2: X(s), y1: mt, y2: mt + ih }));
      svg.appendChild(sv("text", { class: "lbl", x: X(s) + 4, y: mt + 11 }, (W < 560 ? "r" : "reset ") + (i + 1)));
    });
    var pts = t.points, raw = "", sm = "", win = t.rolling_window_points || 10, q = [];
    pts.forEach(function (p, i) {
      raw += (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[5]).toFixed(1);
      q.push(p[5]); if (q.length > win) q.shift();
      var mean = q.reduce(function (a, c) { return a + c; }, 0) / q.length;
      sm += (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(mean).toFixed(1);
    });
    svg.appendChild(sv("path", { class: "s-raw", d: raw }));
    var line = sv("path", { class: "s-trained", d: sm });
    svg.appendChild(line);
    svg.appendChild(sv("line", { class: "s-chance", x1: ml, x2: W - mr, y1: Y(1 / 16), y2: Y(1 / 16) }));
    box.appendChild(svg);
    var lg = el("ul", { class: "chart-legend" });
    var a = el("li"); a.appendChild(el("i")); a.appendChild(doc.createTextNode("rolling mean, " + win + " logs")); lg.appendChild(a);
    var c = el("li"); c.appendChild(el("i", { class: "c" })); c.appendChild(doc.createTextNode("chance, 1 in 16")); lg.appendChild(c);
    box.appendChild(lg);
    drawIn(line, box);
    $("tc-note").textContent = "From " + t.source + ": " + fmtInt(t.points.length) + " logged points, one per " + t.config.log_every + " steps; each is one batch of " + t.batch_size + " augmented intents, 16 candidates. Faint line: raw batch accuracy. Device " + t.device + ", " + dur(t.seconds) + " total, " + t.seconds_per_step.toFixed(3) + " s/step. Final eval on 512 clean intents: " + pct(t.final_eval.eval_candidate_acc) + ".";
  }

  function drawIn(path, box) {
    if (reduce || !path.getTotalLength) return;
    var len = path.getTotalLength();
    path.style.strokeDasharray = len; path.style.strokeDashoffset = len;
    var done = false;
    onVisible(box, function (vis) {
      if (!vis || done) return;
      done = true;
      path.style.transition = "stroke-dashoffset 2.4s cubic-bezier(.45,.05,.2,1)";
      requestAnimationFrame(function () { path.style.strokeDashoffset = 0; });
    }, "-15% 0px");
  }

  function sweepChart(b) {
    var box = $("sweep-chart");
    if (!box) return;
    var rows = b.sweep.candidate_sweep, W = Math.max(340, Math.min(900, Math.round(box.clientWidth || 760))), H = W < 560 ? 260 : 300, ml = 40, mr = 70, mt = 14, mb = 40, iw = W - ml - mr, ih = H - mt - mb;
    var svg = sv("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Listener accuracy against number of candidates, trained versus untrained versus chance." });
    function X(i) { return ml + (i / (rows.length - 1)) * iw; }
    function Y(v) { return mt + (1 - v) * ih; }
    var g = sv("g", { class: "grid" }), ax = sv("g", { class: "ax" });
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      g.appendChild(sv("line", { x1: ml, x2: ml + iw, y1: Y(v), y2: Y(v) }));
      ax.appendChild(sv("text", { x: ml - 8, y: Y(v) + 4, "text-anchor": "end" }, (v * 100) + "%"));
    });
    rows.forEach(function (r, i) { ax.appendChild(sv("text", { x: X(i), y: H - 18, "text-anchor": "middle" }, r.candidates)); });
    ax.appendChild(sv("text", { x: ml + iw / 2, y: H - 2, "text-anchor": "middle" }, "candidates (log scale)"));
    svg.appendChild(g); svg.appendChild(ax);
    function path(key, cls) {
      var d = rows.map(function (r, i) { var v = key === "chance" ? r.chance : r[key].value; return (i ? "L" : "M") + X(i).toFixed(1) + " " + Y(v).toFixed(1); }).join("");
      var p = sv("path", { class: cls, d: d }); svg.appendChild(p); return p;
    }
    path("chance", "s-chance");
    path("untrained", "s-untrained");
    var tp = path("trained", "s-trained");
    rows.forEach(function (r, i) {
      svg.appendChild(sv("circle", { class: "dot-u", cx: X(i), cy: Y(r.untrained.value), r: 3 }));
      svg.appendChild(sv("circle", { class: "dot-t", cx: X(i), cy: Y(r.trained.value), r: 3.5 }));
      svg.appendChild(sv("text", { class: "lbl lbl-t", x: X(i), y: Y(r.trained.value) - 10, "text-anchor": "middle" }, pct(r.trained.value)));
    });
    var lastR = rows[rows.length - 1];
    svg.appendChild(sv("text", { class: "lbl", x: X(rows.length - 1) + 8, y: Y(lastR.untrained.value) - 4 }, "untrained"));
    box.appendChild(svg);
    var lg = el("ul", { class: "chart-legend" });
    [["", "skeebert-1"], ["u", "untrained (init.pt)"], ["c", "chance, 1 / candidates"]].forEach(function (x) { var li = el("li"); li.appendChild(el("i", x[0] ? { class: x[0] } : null)); li.appendChild(doc.createTextNode(x[1])); lg.appendChild(li); });
    box.appendChild(lg);
    drawIn(tp, box);
    var s = b.sweep;
    $("sw-note").textContent = "Same " + fmtInt(s.n) + " intents (seed " + s.seed_intents + ") at every point; distractor sets seeded per candidate count (base seed " + s.seed_distractors + ") and shared by both models; " + pct(s.hard_distractor_frac, 0) + " one-concept-off distractors. Clean 64 px renders, " + b.device.toUpperCase() + ". With training-time augmentation, skeebert-1 scores " + rows.map(function (r) { return pct(r.trained_augmented.value); }).join(", ") + ".";
  }

  function renderBench(b, t) {
    var s16 = sweepRow(b, 16), s32 = sweepRow(b, 32), ex = b.sweep.exact_set_by_size;
    var exAll = ex.filter(function (r) { return r.atoms === "all"; })[0];
    var gbT = b.glyphbench.entrants.filter(function (e) { return e.name === "trained"; })[0].score;
    var gbU = b.glyphbench.entrants.filter(function (e) { return e.name === "untrained"; })[0].score;
    var le = b.logged_eval_rerun, ts = b.topsim, di = b.distinctness;
    var H = [{ t: "Metric" }, { t: "skeebert-1", n: 1 }, { t: "untrained", n: 1 }, { t: "chance", n: 1 }, { t: "Notes" }];
    table("std-table", H, [
      ["Listener accuracy, 16 candidates", { prop: s16.trained, hi: 1 }, { prop: s16.untrained }, { t: pct(s16.chance, 2), n: 1 }, "n = " + fmtInt(b.sweep.n) + " intents, clean renders"],
      ["Same, augmented", { prop: s16.trained_augmented, hi: 1 }, { prop: s16.untrained_augmented }, { t: pct(s16.chance, 2), n: 1 }, "rotation, scale, shift and noise as in training"],
      ["Listener accuracy, 32 candidates", { prop: s32.trained, hi: 1 }, { prop: s32.untrained }, { t: pct(s32.chance, 2), n: 1 }, "31 distractors"],
      ["Exact set, all 218 decisions", { prop: exAll.trained, hi: 1 }, { prop: exAll.untrained }, { t: "≈0", n: 1 }, "every concept present and absent right"],
      ["Single concept, top-1 of 218", { prop: b.atoms.trained.top1, hi: 1 }, { prop: b.atoms.untrained.top1 }, { t: pct(b.atoms.chance_top1, 2), n: 1 }, "all " + b.atoms.n + " one-concept glyphs"],
      ["Topographic similarity", { t: ts.trained.mean.toFixed(3) + " ± " + ts.trained.sd.toFixed(3), n: 1, hi: 1 }, { t: ts.untrained.mean.toFixed(3) + " ± " + ts.untrained.sd.toFixed(3), n: 1 }, { t: "≈0", n: 1 }, ts.seeds.length + " seeds × " + ts.n_per_seed + " intents. Higher is more compositional. Untrained is higher."],
      ["Distinctness, between ÷ within", { t: di.trained.ratio.toFixed(1) + "×", n: 1, hi: 1 }, { t: di.untrained.ratio.toFixed(2) + "×", n: 1 }, { t: "1×", n: 1 }, "pixel distance between meanings ÷ between re-draws of one meaning; " + di.intents + " intents × " + di.variations_per_intent],
      ["train.py's own eval, re-run", { t: pct(le.trained.eval_candidate_acc), n: 1, hi: 1 }, { t: pct(le.untrained.eval_candidate_acc), n: 1 }, { t: pct(1 / 16, 2), n: 1 }, "seed 10000, n = 512, " + le.device.toUpperCase() + "; logged at the end of the run on " + t.device.toUpperCase() + ": " + pct(t.final_eval.eval_candidate_acc)],
      ["GlyphBench v1", { prop: gbT, hi: 1 }, { prop: gbU }, { t: pct(b.glyphbench.chance, 2), n: 1 }, "Fig. 2"]
    ]);
    table("exact-table", [{ t: "Concepts" }, { t: "n", n: 1 }, { t: "untrained", n: 1 }, { t: "skeebert-1", n: 1 }],
      ex.map(function (r) { return [String(r.atoms), { t: fmtInt(r.n), n: 1 }, { t: pct(r.untrained.value), n: 1 }, { t: pct(r.trained.value), n: 1, hi: 1 }]; }));
    table("cat-table", [{ t: "Category" }, { t: "n", n: 1 }, { t: "untrained", n: 1 }, { t: "skeebert-1", n: 1 }],
      b.atoms.categories.map(function (r) { return [r.category, { t: String(r.n), n: 1 }, { t: pct(r.untrained.value), n: 1 }, { t: pct(r.trained.value), n: 1, hi: 1 }]; }));
    if (b.scoring && b.scoring.available) {
      table("scoring-table", [{ t: "Guess, for " + b.scoring.intent.join(" + ") }, { t: "score", n: 1 }],
        b.scoring.rows.map(function (r) { return ["“" + r.guess + "”", { t: r.score.toFixed(3), n: 1, hi: 1 }]; }));
      $("scoring-note").textContent = "Computed by scripts/bench.py with skeebert.scoring.score_guess and " + b.scoring.embedder + ". The meaning's gloss is “" + b.scoring.gloss + "”. Naming one of two concepts scores more than an unrelated guess and less than a full one.";
    } else if ($("scoring-note")) {
      $("scoring-note").textContent = "The embedder was not available when the benchmarks ran, so no example scores are shown.";
    }
    var c = t.config;
    table("config-table", [{ t: "Setting" }, { t: "Value", n: 1 }], [
      ["steps", { t: fmtInt(c.steps), n: 1 }], ["batch size", { t: String(c.batch_size), n: 1 }], ["learning rate (Adam)", { t: String(c.lr), n: 1 }],
      ["distractors", { t: c.n_distractors + " (" + pct(c.hard_distractor_frac, 0) + " one-concept-off)", n: 1 }],
      ["intent sizes 1 / 2 / 3", { t: c.atom_count_probs.map(function (p) { return pct(p, 0); }).join(" / "), n: 1 }],
      ["loss weights game / BCE / ink", { t: c.game_weight + " / " + c.bce_weight + " / " + c.ink_weight, n: 1 }],
      ["speaker noise", { t: String(c.speaker_noise), n: 1 }],
      ["augment: rotate / scale / shift / noise", { t: "±" + c.aug_rotate_deg + "° / ±" + c.aug_scale + " / ±" + c.aug_translate + " / " + c.aug_noise, n: 1 }],
      ["listener reset every", { t: fmtInt(c.listener_reset_every) + " steps", n: 1 }],
      ["seed", { t: String(c.seed), n: 1 }],
      ["device, time", { t: t.device + ", " + dur(t.seconds) + " (" + t.seconds_per_step.toFixed(3) + " s/step)", n: 1 }]
    ]);
    table("gen-table", [{ t: "Listener" }, { t: "Steps", n: 1 }, { t: "acc, first 500", n: 1 }, { t: "acc, last 500", n: 1 }, { t: "steps to 50%", n: 1 }],
      t.generations.map(function (g) { return ["#" + g.generation, { t: fmtInt(g.from_step) + "–" + fmtInt(g.to_step), n: 1 }, { t: pct(g.acc_first_500), n: 1 }, { t: pct(g.acc_last_500), n: 1, hi: 1 }, { t: g.steps_to_rolling_0_5 == null ? "never" : fmtInt(g.steps_to_rolling_0_5), n: 1 }]; }));
    $("gen-note").textContent = "Mean logged batch accuracy over the first and last 500 steps of each listener's life, and steps until a rolling mean of " + t.rolling_window_points + " logs first reaches 50%. From " + t.source + ".";
    var r = b.reported, lg = r.ledger;
    table("ledger-table", [{ t: "Calls", n: 1 }, { t: "Cost", n: 1 }, { t: "Input", n: 1 }, { t: "Output", n: 1 }, { t: "Cache read", n: 1 }, { t: "Cache write", n: 1 }],
      []);
    var lt = $("ledger-table");
    if (lt) {
      var tr = el("tr");
      [fmtInt(lg.calls), usd(lg.usd), fmtInt(lg.tokens_in), fmtInt(lg.tokens_out), fmtInt(lg.cache_read), fmtInt(lg.cache_write)].forEach(function (v, i) { var td = el("td", { class: "n" + (i === 1 ? " hi" : "") }, v); tr.appendChild(td); });
      lt.querySelector("tbody").appendChild(tr);
    }
    var per = lg.usd / lg.calls, perT = r.persona_tuning.usd / r.persona_tuning.calls;
    table("cost-table", [{ t: "Item" }, { t: "Cost", n: 1 }, { t: "Basis" }], [
      ["Per message, live", { t: usd(per), n: 1, hi: 1 }, usd(lg.usd) + " over " + lg.calls + " calls"],
      ["Per message, persona tuning", { t: usd(perT, 6), n: 1 }, usd(r.persona_tuning.usd) + " over " + r.persona_tuning.calls + " calls"],
      ["Daily budget", { t: usd(r.daily_budget_usd, 2), n: 1 }, "then asleep until the next UTC day"],
      ["Messages per day before sleep", { t: "≈" + fmtInt(Math.floor(r.daily_budget_usd / per)), n: 1 }, "budget ÷ live average; n = " + lg.calls],
      ["Price to you", { t: "$0", n: 1, hi: 1 }, "every plan"]
    ]);
    $("cost-note").textContent = "Prices configured as Claude Haiku 5.5's per-million-token rates: input $" + r.prices_per_mtok_usd.input.toFixed(2) + ", output $" + r.prices_per_mtok_usd.output.toFixed(2) + ", cache read $" + r.prices_per_mtok_usd.cache_read.toFixed(2) + ", cache write $" + r.prices_per_mtok_usd.cache_write.toFixed(3) + ". Four calls is not a distribution; the per-message figure will move.";
    var leads = doc.querySelectorAll("#compare tbody tr[data-lead]").length, all = doc.querySelectorAll("#compare tbody tr").length;
    if ($("lead-count")) $("lead-count").textContent = "skeebert leads in " + leads + " of " + all + " categories. It trails only in writing, range, being a Claude model and human evaluation, which we consider out of scope.";
  }

  // ---------------------------------------------------------------- bound numbers
  function bindings(b, t, m) {
    var p = b.model.params, s16 = sweepRow(b, 16), ex = b.sweep.exact_set_by_size, r = b.reported;
    function exa(a) { return ex.filter(function (x) { return x.atoms === a; })[0].trained.value; }
    var gens = t.generations, sc = b.scoring && b.scoring.available ? b.scoring : null;
    var V = {
      params_total: fmtInt(p.total), params_speaker: fmtInt(p.speaker), params_listener: fmtInt(p.listener), params_proxy: fmtInt(p.proxy),
      numbers_per_glyph: String(b.model.n_strokes * b.model.params_per_stroke), possible_messages: fmtInt(b.model.possible_messages),
      n_concepts: String(b.model.n_concepts), n_categories: String(b.model.n_categories), n_strokes: String(b.model.n_strokes),
      trained_version: b.checkpoints.trained.version, untrained_version: b.checkpoints.untrained.version,
      git_short: (b.git.short || "unknown") + (b.git.dirty ? " (uncommitted changes)" : ""),
      bench_runtime: Math.round(b.runtime_s) + " s", bench_machine: b.machine + ", torch " + b.torch,
      reset_every: fmtInt(t.config.listener_reset_every),
      gen1_steps: fmtInt(gens[0].steps_to_rolling_0_5), gen6_steps: fmtInt(gens[gens.length - 1].steps_to_rolling_0_5),
      untrained_random16: pct(b.sweep.distractor_type_16.filter(function (x) { return x.hard_frac === 0; })[0].untrained.value),
      atom_chance: pct(b.atoms.chance_top1, 2),
      human_guesses: String(r.human_eval.guesses), human_people: String(r.human_eval.people), human_mean: r.human_eval.mean_score.toFixed(3),
      human_unlock: String(r.human_eval.unlock_threshold), human_short: String(r.human_eval.unlock_threshold - r.human_eval.guesses),
      gb_trained_pct: pct(b.glyphbench.entrants.filter(function (e) { return e.name === "trained"; })[0].score.value),
      train_hours: dur(t.seconds), train_steps: fmtInt(t.steps),
      speak_ms: b.latency.speak.median_ms.toFixed(1), speak_n: String(b.latency.speak.n), threads: String(b.latency.torch_threads),
      render_ms: String(Math.round(b.latency.render_512_tensor.median_ms)), png_ms: String(Math.round(b.latency.png_512_export.median_ms)),
      png_kb: String(Math.round(b.latency.png_bytes_median / 1000)),
      d_model: String(b.model.d_model), enc_layers: String(b.model.enc_layers), dec_layers: String(b.model.dec_layers), n_heads: String(b.model.n_heads),
      render_segments: String(b.model.render_segments), image_size: String(b.model.image_size_train), export_size: String(b.model.export_size),
      channels: b.model.channels.join("-"), hidden: String(b.model.hidden),
      p1: pct(t.config.atom_count_probs[0], 0), p2: pct(t.config.atom_count_probs[1], 0), p3: pct(t.config.atom_count_probs[2], 0),
      n_distractors: String(t.config.n_distractors), val_frac: pct(t.config.proxy_val_frac, 0), align_w: String(t.config.align_game_weight),
      topsim_trained: b.topsim.trained.mean.toFixed(3), topsim_untrained: b.topsim.untrained.mean.toFixed(3),
      exact1: pct(exa(1)), exact3: pct(exa(3)), exact_all: pct(exa("all")),
      acc16_trained: pct(s16.trained.value), acc16_logged: pct(b.logged_eval_rerun.trained.eval_candidate_acc, 0),
      ledger_calls: String(r.ledger.calls), tuning_calls: String(r.persona_tuning.calls)
    };
    if (sc) { V.whole_w = String(sc.whole_weight); V.whole_w2 = String(1 - sc.whole_weight); V.floor = sc.floor.toFixed(2); V.ceil = sc.ceil.toFixed(2); }
    doc.querySelectorAll("[data-v]").forEach(function (n) {
      var v = V[n.getAttribute("data-v")];
      if (v != null) n.textContent = v;
    });
    void m;
  }

  // ---------------------------------------------------------------- nav state, read time
  function navState() {
    var groups = [".sections", ".toc"].map(function (sel) {
      var links = Array.prototype.slice.call(doc.querySelectorAll(sel + " a")).filter(function (a) { return $(a.getAttribute("href").slice(1)); });
      return links.map(function (a) { return { a: a, t: $(a.getAttribute("href").slice(1)) }; });
    }).filter(function (g) { return g.length; });
    if (!groups.length) return;
    var tk = 0;
    function update() {
      tk = 0;
      var line = window.innerHeight * 0.3;
      groups.forEach(function (g) {
        var cur = null;
        g.forEach(function (x) { if (x.t.getBoundingClientRect().top < line) cur = x.a; });
        g.forEach(function (x) { if (x.a === cur) x.a.setAttribute("aria-current", "true"); else x.a.removeAttribute("aria-current"); });
      });
    }
    window.addEventListener("scroll", function () { if (!tk) tk = requestAnimationFrame(update); }, { passive: true });
    update();
  }
  function readTime() {
    var n = $("readtime"), main = $("main");
    if (!n || !main) return;
    var words = (main.innerText || main.textContent).split(/\s+/).filter(Boolean).length;
    n.textContent = Math.max(1, Math.round(words / 230)) + " min read";
  }

  // ---------------------------------------------------------------- init
  function getJSON(url) { return fetch(url).then(function (r) { if (!r.ok) throw new Error(url + " " + r.status); return r.json(); }); }
  function init() {
    setupTheme();
    ornaments();
    var isHome = !!$("hero-canvas");
    var gp = getJSON("/glyphs/glyphs.json");
    gp.then(function (m) {
      if (!m || !m.glyphs || !m.glyphs.length) throw new Error("empty manifest");
      fields(m);
      if (isHome) { hero(m); decoder(m); chat(m); }
    }).catch(function () {
      if ($("hero-r1")) $("hero-r1").textContent = "The glyphs didn't load. Reload the page to try again.";
    });
    if (!isHome) return;
    Promise.all([getJSON("/bench.json"), getJSON("/trainlog.json")]).then(function (res) {
      var b = res[0], t = res[1];
      bindings(b, t);
      bars(b);
      trainChart(t);
      sweepChart(b);
      renderBench(b, t);
      readTime();
    }).catch(function (e) {
      doc.querySelectorAll("#benchmarks .lead").forEach(function (n) { n.textContent = "The benchmark data didn't load (" + e.message + "). The raw numbers are at /bench.json."; });
      readTime();
    });
    navState();
  }
  if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", init); else init();
})();
