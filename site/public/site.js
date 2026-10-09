// skeebert.frgmt.xyz: everything glyph-related on the page comes from
// /glyphs/glyphs.json, written by scripts/render_site_glyphs.py from a real
// checkpoint. Nothing here invents a glyph, a meaning or a number.
(function () {
  "use strict";

  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var GLYPH_BASE = "/glyphs/";
  var SVGNS = "http://www.w3.org/2000/svg";

  // ---------- scroll reveals ----------
  function setupReveals() {
    var items = document.querySelectorAll("[data-reveal]");
    if (reduce || !("IntersectionObserver" in window) || !items.length) return;
    document.documentElement.classList.add("reveal-ready");
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
    items.forEach(function (el) { io.observe(el); });
  }

  // ---------- helpers ----------
  function el(tag, attrs, text) {
    var n = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (text != null) n.textContent = text;
    return n;
  }
  function svgEl(tag, attrs) {
    var n = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    return n;
  }
  function atoms(g) { return g.intent.join(" + "); }
  function fmt(n) { return Number(n).toLocaleString("en-US"); }
  function shuffle(a) {
    for (var i = a.length - 1; i > 0; i--) { var j = Math.floor(Math.random() * (i + 1)); var t = a[i]; a[i] = a[j]; a[j] = t; }
    return a;
  }
  function setText(id, text) { var n = document.getElementById(id); if (n) n.textContent = text; }

  // ---------- spec sheet: overwrite static numbers with the checkpoint's own ----------
  function fillSpecs(m) {
    var md = m.model || {}, p = md.params || {};
    var values = {
      total: p.total, speaker: p.speaker, listener: p.listener, proxy: p.proxy,
      speaker_short: p.speaker != null ? (p.speaker / 1e6).toFixed(1) + "M" : null,
      n_strokes: md.n_strokes, params_per_stroke: md.params_per_stroke,
      n_concepts: md.n_concepts, n_categories: md.n_categories,
      max_atoms: md.max_atoms, possible_messages: md.possible_messages
    };
    document.querySelectorAll("[data-spec]").forEach(function (node) {
      var v = values[node.getAttribute("data-spec")];
      if (v == null) return;
      node.textContent = typeof v === "number" ? fmt(v) : v;
    });
  }

  // ---------- a glyph as SVG, from its real stroke parameters ----------
  // Canvas is [-1, 1] x [-1, 1], x right, y down (skeebert/render.py), which is
  // exactly SVG's orientation, so the viewBox maps one to one.
  function glyphSVG(g, label) {
    var svg = svgEl("svg", { viewBox: "-1 -1 2 2", class: "glyph-svg", role: "img", "aria-label": label });
    var defs = svgEl("defs");
    var f = svgEl("filter", { id: "halo-blur", filterUnits: "userSpaceOnUse", x: "-1.4", y: "-1.4", width: "2.8", height: "2.8" });
    f.appendChild(svgEl("feGaussianBlur", { stdDeviation: "0.032" }));
    defs.appendChild(f);
    svg.appendChild(defs);
    var halo = svgEl("g", { class: "halo", filter: "url(#halo-blur)" });
    var ink = svgEl("g", { class: "ink" });
    var pairs = [];
    g.strokes.forEach(function (s) {
      var d = "M" + s[0] + " " + s[1] + "Q" + s[2] + " " + s[3] + " " + s[4] + " " + s[5];
      var w = 2 * s[6], a = Math.max(0, Math.min(1, s[7]));
      var hp = svgEl("path", { d: d, "stroke-width": (w * 1.6).toFixed(5), opacity: a.toFixed(4) });
      var ip = svgEl("path", { d: d, "stroke-width": w.toFixed(5), opacity: a.toFixed(4) });
      halo.appendChild(hp); ink.appendChild(ip);
      pairs.push([hp, ip]);
    });
    svg.appendChild(halo); svg.appendChild(ink);
    return { svg: svg, pairs: pairs };
  }

  function drawIn(pairs, startDelay) {
    if (reduce || !Element.prototype.animate) return;
    pairs.forEach(function (pair, k) {
      var len = pair[1].getTotalLength() + 0.01;
      var delay = startDelay + k * 260;
      pair.forEach(function (path, which) {
        path.style.strokeDasharray = len + " " + len;
        path.style.strokeDashoffset = len;
        path.style.visibility = "hidden";
        var anim = path.animate(
          [{ strokeDashoffset: len, visibility: "visible" }, { strokeDashoffset: 0, visibility: "visible" }],
          { duration: 1100, delay: delay + (which === 0 ? 120 : 0), easing: "cubic-bezier(.65,.05,.36,1)", fill: "forwards" }
        );
        anim.onfinish = function () { path.style.strokeDasharray = ""; path.style.strokeDashoffset = ""; path.style.visibility = ""; anim.cancel(); };
      });
    });
  }

  function tilt(container, svg) {
    if (reduce || !window.matchMedia("(pointer: fine)").matches) return;
    var tx = 0, ty = 0, x = 0, y = 0, raf = 0;
    function step() {
      x += (tx - x) * 0.07; y += (ty - y) * 0.07;
      svg.style.transform = "rotateX(" + (-y * 7).toFixed(3) + "deg) rotateY(" + (x * 9).toFixed(3) + "deg) translate3d(" + (x * 6).toFixed(2) + "px," + (y * 6).toFixed(2) + "px,0)";
      raf = Math.abs(tx - x) + Math.abs(ty - y) > 0.001 ? requestAnimationFrame(step) : 0;
    }
    window.addEventListener("pointermove", function (e) {
      var r = container.getBoundingClientRect();
      if (r.bottom < 0) return;
      tx = Math.max(-1, Math.min(1, (e.clientX - (r.left + r.width / 2)) / (window.innerWidth / 2)));
      ty = Math.max(-1, Math.min(1, (e.clientY - (r.top + r.height / 2)) / (window.innerHeight / 2)));
      if (!raf) raf = requestAnimationFrame(step);
    }, { passive: true });
  }

  function hero(m) {
    var g = m.glyphs[0];
    var box = document.getElementById("hero-glyph");
    if (!g || !box) return;
    var built = glyphSVG(g, "A real Skeebert glyph meaning “" + g.gloss + "”, drawn from its stroke data, " + m.label + ".");
    box.appendChild(built.svg);
    drawIn(built.pairs, 250);
    tilt(box, built.svg);

    var copy = m.untrained
      ? "The one up there means “" + g.gloss + "”. Your browser just drew it from the model's actual output: " + g.strokes.length + " quadratic strokes, eight numbers each. The model that drew it is " + m.label + ", which is why it looks like a scribble. Every glyph is a scribble for now. That's the starting line."
      : "The one up there means “" + g.gloss + "”. Your browser just drew it from the model's actual output: " + g.strokes.length + " quadratic strokes, eight numbers each. The model that drew it: " + m.label + ".";
    setText("real-copy", copy);
  }

  // ---------- decode toy ----------
  function decode(m) {
    var glyphs = m.glyphs;
    var ui = document.getElementById("decode-ui");
    if (!ui || glyphs.length < 2) return;
    var nOpts = Math.min(4, glyphs.length);
    setText("decode-intro", m.untrained
      ? "Pick a glyph, then pick what you think it means. These are the actual images Skeebert sends to Discord, from the same untrained model. So for now this is a 1-in-" + nOpts + " guess wearing a lab coat."
      : "Pick a glyph, then pick what you think it means. These are the actual images Skeebert sends to Discord, from the same model.");

    var img = document.getElementById("decode-img");
    var opts = document.getElementById("decode-options");
    var result = document.getElementById("decode-result");
    var strip = document.getElementById("decode-strip");
    var png = document.getElementById("decode-png");
    var answered = {}; // glyph index -> true/false (first answer only)
    var order = shuffle(glyphs.map(function (_, i) { return i; }));
    var current = -1;
    var thumbs = [];

    glyphs.forEach(function (g, i) {
      var li = el("li");
      var b = el("button", { type: "button", "aria-pressed": "false" });
      var t = el("img", { src: GLYPH_BASE + (g.thumb || g.webp), alt: "Glyph " + (i + 1) + ", meaning hidden", width: "72", height: "72", decoding: "async" });
      b.appendChild(t);
      b.addEventListener("click", function () { show(i); });
      li.appendChild(b); strip.appendChild(li);
      thumbs.push({ b: b, img: t });
    });

    function tally() {
      var keys = Object.keys(answered);
      if (!keys.length) { setText("decode-tally", ""); return; }
      var right = keys.filter(function (k) { return answered[k]; }).length;
      setText("decode-tally", right + " right out of " + keys.length);
    }

    function show(i) {
      current = i;
      var g = glyphs[i];
      thumbs.forEach(function (t, k) { t.b.setAttribute("aria-pressed", k === i ? "true" : "false"); });
      var done = function () {
        img.src = GLYPH_BASE + g.webp;
        img.alt = "Glyph " + (i + 1) + ". Its meaning is hidden until you guess.";
        img.classList.remove("swap");
      };
      if (reduce || !img.getAttribute("src")) done();
      else { img.classList.add("swap"); setTimeout(done, 220); }
      png.href = GLYPH_BASE + g.png;

      var pool = [];
      var seen = {}; seen[atoms(g)] = true;
      shuffle(glyphs.slice()).forEach(function (o) {
        var key = atoms(o);
        if (!seen[key] && pool.length < nOpts - 1) { seen[key] = true; pool.push(o); }
      });
      var choices = shuffle(pool.concat([g]));
      opts.textContent = "";
      result.textContent = "";
      choices.forEach(function (c) {
        var li = el("li");
        var b = el("button", { type: "button" });
        b.appendChild(el("span", null, atoms(c)));
        b.appendChild(el("span", { class: "mark", "aria-hidden": "true" }));
        b.addEventListener("click", function () { pick(c, b); });
        li.appendChild(b); opts.appendChild(li);
      });
    }

    function pick(choice, button) {
      var g = glyphs[current];
      var right = choice === g;
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
      result.appendChild(document.createTextNode("It meant " + atoms(g) + ": “" + g.gloss + "”."));
      img.alt = "Glyph " + (current + 1) + ", meaning " + atoms(g) + ".";
      thumbs[current].img.alt = "Glyph " + (current + 1) + ", meaning " + atoms(g);
      tally();
      document.getElementById("decode-next").focus({ preventScroll: true });
    }

    document.getElementById("decode-next").addEventListener("click", function () {
      var pos = order.indexOf(current);
      for (var step = 1; step <= order.length; step++) {
        var cand = order[(pos + step) % order.length];
        if (!(cand in answered) || step === order.length) { show(cand); return; }
      }
    });

    ui.hidden = false;
    show(order[0]);
  }

  function failed() {
    setText("real-copy", "The glyphs didn't load, so there's nothing drawn up there right now. Reload the page to try again.");
    setText("decode-intro", "The glyphs didn't load. Reload the page to try again.");
  }

  function init() {
    setupReveals();
    if (!document.getElementById("hero-glyph")) return;
    fetch(GLYPH_BASE + "glyphs.json")
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (m) {
        if (!m || !Array.isArray(m.glyphs) || !m.glyphs.length) throw new Error("empty manifest");
        fillSpecs(m);
        hero(m);
        decode(m);
      })
      .catch(failed);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
