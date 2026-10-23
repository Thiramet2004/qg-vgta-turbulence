/* main.js — Leaflet map, WebGL2 ping-pong pipeline, live time stepping on a unified
 * slow-motion clock, VGTA uniforms, and the astronomical dashboard (time panel, Julian Day,
 * Moon phase, day/night terminator, planets, ecliptic). Planets and the terminator are
 * display-only: the fluid is forced by the Sun and Moon VGTA alone. */
'use strict';
(function () {
  const SH = window.QGShaders;
  const Sky = window.Celestial;
  const $ = id => document.getElementById(id);
  const DEG = Math.PI / 180;

  // ───────────────────────── grid & physical units ─────────────────────────
  const qs = new URLSearchParams(location.search);
  // Basemap theme: light (default) or ?basemap=dark
  const LIGHT = qs.get('basemap') !== 'dark';
  document.documentElement.dataset.basemap = LIGHT ? 'light' : 'dark';
  const NX = Math.max(256, Math.min(2048, parseInt(qs.get('nx'), 10) || 1024)) & ~1; // even: red-black SOR
  const LAT_EDGE_DEG = 78;
  const D_RAD = 2 * Math.PI / NX;
  const NY = 2 * Math.round(LAT_EDGE_DEG * DEG / D_RAD);
  const LAT_MAX = NY * D_RAD / 2;

  const R_EARTH = 6.371e6, OMEGA = 7.2921e-5, GRAV = 9.81;
  const H_OCEAN = 4000;            // m, full depth felt by the barotropic equilibrium tide
  const C_BAROCLINIC = 2.5;        // m/s, first-mode gravity-wave speed (sets L_d before exaggeration)
  const T_UNIT = 3600;             // s, model time unit
  const L_CELL = R_EARTH * D_RAD;  // m, meridional cell size
  const KM_PER_CELL = L_CELL / 1000;
  const R_CELLS = 1 / D_RAD;       // Earth radius in cells
  const F0 = 2 * OMEGA * T_UNIT;
  // Equilibrium-tide stretching: Δq = T·(2Ω sinφ)·R/(gH) · Δ(Σ γ_n V_n/R), with V_n/R in µm/s²
  // (the shader's tidePotBody includes the Love-number factors γ_n and the degree-n weights 1/n).
  const KS = T_UNIT * 2 * OMEGA * R_EARTH / (GRAV * H_OCEAN) * 1e-6;
  const A_REF = 1.0;               // µm/s², reference VGTA for the eddy-forcing modulation
  const NU4 = 0.003;               // hyperviscosity coefficient (× h⁴, stable for < 1/32)
  const SOR_OMEGA = 1.7;

  // Unified clock: fluid and Sun/Moon both advance by timeScale × BASE_HOURS_PER_SEC model
  // hours per real second. 0.01× → 14.4 model min/s · 0.1× → 2.4 h/s · 1× → 1 day/s · 5× → 5 days/s.
  const BASE_HOURS_PER_SEC = 24;
  const DT_MAX = 1;                // h, largest fluid sub-step (resolves the semidiurnal tide)
  const MAX_STEPS_PER_FRAME = 12;  // on slow GPUs the model falls behind rather than freezing
  const MAX_FRAME_SEC = 0.1;       // longest real interval one frame may advance (tab switch, stall)

  // ───────────────────────── tunable state ─────────────────────────
  const state = {
    field: 0, colorLog: 0, contours: true, markers: true, paused: false,
    night: true, planets: true, ecliptic: true,      // display overlays
    hfr: true, glyphPx: 26, arrowScale: 80,          // HF radar overlay (arrow px per m/s)
    hfrSource: 'bundles',                            // 'bundles' or an NCEI live network key
    speedLog: -1, timeScale: 0.1,   // simulation speed; the slider is log10(timeScale)
    sorSweeps: 8,
    forcing: 0.8,       // ×10⁻³ PV units per √hour
    forcingKm: 200,     // forcing wavelength
    dragDays: 45,       // bottom-drag spin-down time
    ldExagg: 10,        // Rossby-radius exaggeration so mesoscale eddies are resolved
    stretchGain: 100,   // VGTA vortex-stretching gain (1 = physical)
    modGain: 2,         // VGTA modulation of eddy generation
    simTime: Date.now(),  // model clock [ms UTC] — fractional, never rounded
  };
  const cNd = () => C_BAROCLINIC * state.ldExagg * T_UNIT / L_CELL;
  function invLd2(latRad) {
    const f = F0 * Math.sin(latRad), b = F0 * Math.cos(latRad) * D_RAD, c = cNd();
    return (f * f + 2 * b * c) / (c * c);
  }
  const ldKm = latDeg => KM_PER_CELL / Math.sqrt(invLd2(latDeg * DEG));

  // ───────────────────────── colour maps & display fields ─────────────────────────
  // Dark basemap: colour maps fade to near-black at zero. Light basemap: they fade to near-white,
  // so weak values blend into the light-grey ocean instead of veiling it.
  const DIVERGING = LIGHT ? [
    [0.02, 0.19, 0.38], [0.13, 0.40, 0.67], [0.26, 0.58, 0.77], [0.57, 0.77, 0.87], [0.97, 0.97, 0.97],
    [0.96, 0.65, 0.51], [0.84, 0.38, 0.30], [0.70, 0.09, 0.17], [0.40, 0.00, 0.12],
  ] : [
    [0.75, 0.93, 0.96], [0.33, 0.70, 0.90], [0.20, 0.42, 0.80], [0.13, 0.19, 0.42], [0.07, 0.07, 0.09],
    [0.42, 0.11, 0.18], [0.80, 0.25, 0.18], [0.96, 0.56, 0.25], [1.00, 0.89, 0.66],
  ];
  const SEQUENTIAL = LIGHT ? [
    [0.99, 0.99, 0.75], [0.99, 0.85, 0.46], [0.99, 0.70, 0.30], [0.99, 0.51, 0.24], [0.94, 0.30, 0.20],
    [0.80, 0.11, 0.17], [0.60, 0.00, 0.20], [0.40, 0.00, 0.20], [0.20, 0.00, 0.15],
  ] : [
    [0.00, 0.00, 0.02], [0.10, 0.04, 0.25], [0.30, 0.05, 0.42], [0.51, 0.12, 0.42], [0.71, 0.21, 0.33],
    [0.87, 0.35, 0.20], [0.97, 0.55, 0.04], [0.98, 0.78, 0.15], [0.99, 1.00, 0.64],
  ];
  const vortUnit = v => v / T_UNIT * 1e6;
  const FIELDS = [
    { name: 'Relative vorticity ζ (red = anticlockwise)', unit: '10⁻⁶ s⁻¹', base: 2e-2, div: true, phys: vortUnit },
    { name: 'Streamfunction ψ', unit: '10⁴ m²/s', base: 0.5, div: true, phys: v => v * L_CELL * L_CELL / T_UNIT / 1e4 },
    { name: 'Current speed |u|', unit: 'm/s', base: 0.06, div: false, phys: v => v * L_CELL / T_UNIT },
    { name: 'Potential vorticity q', unit: '10⁻⁶ s⁻¹', base: 2.5e-2, div: true, phys: vortUnit },
    { name: 'VGTA a_z (Sun + Moon)', unit: 'µm/s²', base: 1.6, div: true, phys: v => v },
    { name: 'Rossby radius L_d(φ)', unit: 'km', base: 600, div: false, phys: v => v },
  ];
  const flat = pal => new Float32Array(pal.flat());
  const CMAP = { div: flat(DIVERGING), seq: flat(SEQUENTIAL) };

  // ───────────────────────── map ─────────────────────────
  const map = L.map('map', {
    center: [15, -40], zoom: 3, minZoom: 2, maxZoom: 13,   // ≥ 12 needed for the 1 km SF Bay radar grid
    zoomAnimation: false,           // canvas is re-projected every frame; no CSS zoom transforms
    zoomControl: false, worldCopyJump: false,
  });
  L.control.zoom({ position: 'bottomright' }).addTo(map);
  const qgPane = map.createPane('qg');
  qgPane.style.zIndex = 350;        // above tiles (200), below labels/overlays
  qgPane.style.pointerEvents = 'none';
  const labelPane = map.createPane('labels');
  labelPane.style.zIndex = 450;
  labelPane.style.pointerEvents = 'none';
  // CARTO basemaps require a (free) key: https://carto.com/basemaps/apikey
  // Set CARTO_KEY here or pass ?cartokey=… ; without one, Esri's keyless dark canvas is used.
  const CARTO_KEY = qs.get('cartokey') || '';
  if (CARTO_KEY) {
    const CARTO = 'https://{s}.basemaps.cartocdn.com/rastertiles/{style}/{z}/{x}/{y}{r}.png?key={key}';
    const attr = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>';
    const tone = LIGHT ? 'light' : 'dark';
    L.tileLayer(CARTO, { style: tone + '_nolabels', key: CARTO_KEY, subdomains: 'abcd', maxZoom: 20, attribution: attr }).addTo(map);
    L.tileLayer(CARTO, { style: tone + '_only_labels', key: CARTO_KEY, subdomains: 'abcd', maxZoom: 20, pane: 'labels', opacity: 0.8 }).addTo(map);
  } else {
    const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/{style}/MapServer/tile/{z}/{y}/{x}';
    const attr = 'Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors';
    const tone = LIGHT ? 'World_Light_Gray' : 'World_Dark_Gray';
    L.tileLayer(ESRI, { style: tone + '_Base', maxZoom: 16, attribution: attr }).addTo(map);
    L.tileLayer(ESRI, { style: tone + '_Reference', maxZoom: 16, pane: 'labels', opacity: 0.8 }).addTo(map);
  }

  const canvas = document.createElement('canvas');
  canvas.className = 'qg-canvas';
  qgPane.appendChild(canvas);

  const bodyIcon = (cls, glyph) => L.divIcon({ className: 'body-marker ' + cls, html: glyph, iconSize: [24, 24], iconAnchor: [12, 12] });
  const moonMk = L.marker([0, 0], { icon: bodyIcon('moon', '☾'), interactive: false, keyboard: false });
  const sunMk = L.marker([0, 0], { icon: bodyIcon('sun', '☀'), interactive: false, keyboard: false });

  // Planets (display only). Colours follow the dashboard legend.
  const PLANET_STYLE = {
    Mercury: { color: '#4ade80', glyph: '☿' }, Venus: { color: '#22d3ee', glyph: '♀' },
    Mars: { color: '#e040fb', glyph: '♂' }, Jupiter: { color: '#fb923c', glyph: '♃' },
    Saturn: { color: '#a78bfa', glyph: '♄' }, Uranus: { color: '#bef264', glyph: '♅' },
    Neptune: { color: '#3b82f6', glyph: '♆' },
  };
  const planetMk = Object.fromEntries(Sky.PLANETS.map(name => [name, L.marker([0, 0], {
    icon: L.divIcon({
      className: 'planet-marker', iconSize: [18, 18], iconAnchor: [9, 9],
      html: `<span style="--c:${PLANET_STYLE[name].color}">${PLANET_STYLE[name].glyph}</span>`,
    }),
    title: name, keyboard: false,
  })]));
  const eclipticLine = L.polyline([], {
    color: LIGHT ? '#9a6b00' : '#f5d76e', weight: 1.3, opacity: LIGHT ? 0.75 : 0.6, dashArray: '6 6', interactive: false,
  });

  // ───────────────────────── WebGL2 setup ─────────────────────────
  const gl = canvas.getContext('webgl2', {
    alpha: true, premultipliedAlpha: false, antialias: false, depth: false, stencil: false,
    powerPreference: 'high-performance',
  });
  if (!gl) return fatal('WebGL2 is not available in this browser.');
  let FMT;
  if (gl.getExtension('EXT_color_buffer_float')) {
    FMT = { r: gl.R32F, rgba: gl.RGBA32F };
  } else if (gl.getExtension('EXT_color_buffer_half_float')) {
    FMT = { r: gl.R16F, rgba: gl.RGBA16F };
    console.warn('Float32 render targets unavailable; using half floats (reduced accuracy).');
  } else {
    return fatal('This GPU cannot render to floating-point textures.');
  }

  let contextLost = false;
  canvas.addEventListener('webglcontextlost', e => {
    e.preventDefault();
    contextLost = true;
    fatal('The GPU context was lost (driver reset or memory pressure). Reload the page, or try ?nx=512.');
  });

  function compile(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const listing = src.split('\n').map((l, i) => (i + 1) + ': ' + l).join('\n');
      throw new Error(gl.getShaderInfoLog(s) + '\n' + listing);
    }
    return s;
  }
  const vs = compile(gl.VERTEX_SHADER, SH.VERT);
  function program(fsSrc) {
    const p = gl.createProgram();
    gl.attachShader(p, vs);
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveUniform(p, i);
      u[info.name.replace(/\[0\]$/, '')] = { loc: gl.getUniformLocation(p, info.name), type: info.type };
    }
    return { p, u };
  }
  let P;
  try {
    P = {
      vel: program(SH.VEL), advect: program(SH.ADVECT), sor: program(SH.SOR),
      init: program(SH.INIT), reduce: program(SH.REDUCE), display: program(SH.DISPLAY),
    };
  } catch (e) {
    console.error(e);
    return fatal('Shader compilation failed — see the console.');
  }

  function setUniforms(prog, vals) {
    for (const k in vals) {
      const e = prog.u[k];
      if (!e) continue;          // unused uniforms are optimised away
      const v = vals[k];
      switch (e.type) {
        case gl.FLOAT: gl.uniform1f(e.loc, v); break;
        case gl.FLOAT_VEC2: gl.uniform2fv(e.loc, v); break;
        case gl.FLOAT_VEC3: gl.uniform3fv(e.loc, v); break;
        case gl.FLOAT_VEC4: gl.uniform4fv(e.loc, v); break;
        case gl.INT: case gl.BOOL: gl.uniform1i(e.loc, +v); break;
        case gl.INT_VEC2: gl.uniform2iv(e.loc, v); break;
      }
    }
  }

  function renderTarget(w, h, internal) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texStorage2D(gl.TEXTURE_2D, 1, internal, w, h);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('Incomplete framebuffer');
    return { tex, fb, w, h };
  }
  function pingPong(internal) {
    const pp = { read: renderTarget(NX, NY, internal), write: renderTarget(NX, NY, internal) };
    pp.swap = () => { [pp.read, pp.write] = [pp.write, pp.read]; };
    return pp;
  }

  const q = pingPong(FMT.r);         // potential vorticity
  const psi = pingPong(FMT.r);       // streamfunction
  const vel = renderTarget(NX, NY, FMT.rgba);  // u, v, ζ, ∇²q
  const RW = Math.ceil(NX / 16), RH = Math.ceil(NY / 16);
  const reduceT = renderTarget(RW, RH, FMT.rgba);
  const reduceBuf = new Float32Array(RW * RH * 4);
  const maskT = { tex: gl.createTexture() };

  const common = () => ({ u_grid: [NX, NY], u_dRad: D_RAD, u_latMax: LAT_MAX, u_F0: F0, u_c: cNd() });

  // Every sampler a program uses must be listed in `tex` (avoids stale-unit feedback loops).
  function run(prog, target, tex, vals) {
    gl.useProgram(prog.p);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target ? target.fb : null);
    gl.viewport(0, 0, target ? target.w : canvas.width, target ? target.h : canvas.height);
    let unit = 0;
    for (const k in tex) {
      const e = prog.u[k];
      if (!e) continue;
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex[k].tex);
      gl.uniform1i(e.loc, unit++);
    }
    setUniforms(prog, common());
    setUniforms(prog, vals);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // ───────────────────────── land mask ─────────────────────────
  const LAND_URL = 'https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-110m.json';

  async function buildLandMask() {
    const c = document.createElement('canvas');
    c.width = NX; c.height = NY;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#fff';                       // white = ocean
    ctx.fillRect(0, 0, NX, NY);
    let ok = true;
    try {
      if (!window.topojson) throw new Error('topojson-client not loaded');
      const topo = await (await fetch(LAND_URL)).json();
      const land = topojson.feature(topo, topo.objects.land);
      const geoms = land.type === 'FeatureCollection' ? land.features.map(f => f.geometry) : [land.geometry];
      const dDeg = 360 / NX, top = LAT_MAX / DEG;
      ctx.fillStyle = '#000';
      ctx.beginPath();
      for (const g of geoms) {
        const polys = g.type === 'Polygon' ? [g.coordinates] : g.type === 'MultiPolygon' ? g.coordinates : [];
        for (const poly of polys) for (const ring of poly) {
          // Unwrap rings that jump across the dateline (e.g. Fiji, Chukotka); otherwise the
          // even-odd fill inverts a one-row strip around the whole planet. Draw wrapped copies.
          let prev = ring[0][0], shift = 0, lo = Infinity, hi = -Infinity;
          const pts = ring.map(([lon, lat]) => {
            if (lon - prev > 180) shift -= 360; else if (prev - lon > 180) shift += 360;
            prev = lon;
            lo = Math.min(lo, lon + shift); hi = Math.max(hi, lon + shift);
            return [lon + shift, lat];
          });
          const offsets = [0];
          if (hi > 180) offsets.push(-360);
          if (lo < -180) offsets.push(360);
          for (const off of offsets) {
            pts.forEach(([lon, lat], k) => {
              const x = (lon + off + 180) / dDeg, y = (top - lat) / dDeg;
              if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y);
            });
            ctx.closePath();
          }
        }
      }
      ctx.fill('evenodd');
    } catch (e) {
      console.warn('Land mask unavailable, running an all-ocean planet:', e);
      ok = false;
    }
    const px = ctx.getImageData(0, 0, NX, NY).data;
    const m = new Uint8Array(NX * NY);
    for (let j = 0; j < NY; j++) {
      const row = NY - 1 - j;                     // texture row 0 = southern edge
      for (let i = 0; i < NX; i++) m[j * NX + i] = px[(row * NX + i) * 4] > 127 ? 1 : 0;
    }
    for (let i = 0; i < NX; i++) { m[i] = 0; m[(NY - 1) * NX + i] = 0; }  // walls at ±78°
    removeSmallBasins(m, 200);
    return { m, ok };
  }

  // Lakes and tiny enclosed seas would only collect noise: drop basins under minCells.
  function removeSmallBasins(m, minCells) {
    const seen = new Uint8Array(NX * NY), queue = new Int32Array(NX * NY);
    for (let s = 0; s < NX * NY; s++) {
      if (!m[s] || seen[s]) continue;
      let head = 0, tail = 0;
      queue[tail++] = s; seen[s] = 1;
      while (head < tail) {
        const c = queue[head++], i = c % NX, row = c - i;
        const nbs = [row + (i + 1) % NX, row + (i + NX - 1) % NX, c + NX, c - NX];
        for (const n of nbs) {
          if (n >= 0 && n < NX * NY && m[n] && !seen[n]) { seen[n] = 1; queue[tail++] = n; }
        }
      }
      if (tail < minCells) for (let k = 0; k < tail; k++) m[queue[k]] = 0;
    }
  }

  function uploadMask(m) {
    gl.bindTexture(gl.TEXTURE_2D, maskT.tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, NX, NY, 0, gl.RED, gl.UNSIGNED_BYTE, m.map(v => v * 255));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  }

  // ───────────────────────── simulation ─────────────────────────
  const bodyVec = b => [b.u[0], b.u[1], b.u[2], b.k2];
  let bodies = Sky.tidalBodies(state.simTime);   // Sun/Moon at exactly state.simTime
  let kick = null;
  let startTime = state.simTime;
  const forcingModes = new Float32Array(4 * SH.N_MODES);

  // Random 3-D wave vector of magnitude k (uniform direction) and random phase.
  function randomMode(arr, n, k) {
    const z = 2 * Math.random() - 1, az = 2 * Math.PI * Math.random(), s = Math.sqrt(1 - z * z);
    arr[4 * n] = k * s * Math.cos(az);
    arr[4 * n + 1] = k * s * Math.sin(az);
    arr[4 * n + 2] = k * z;
    arr[4 * n + 3] = 2 * Math.PI * Math.random();
  }

  function solvePsi(sweeps) {
    for (let s = 0; s < sweeps; s++) {
      for (let parity = 0; parity < 2; parity++) {
        run(P.sor, psi.write, { u_psi: psi.read, u_q: q.read, u_mask: maskT }, { u_parity: parity, u_omega: SOR_OMEGA });
        psi.swap();
      }
    }
  }

  // Advance fluid and celestial clock together by dt hours (0 < dt ≤ DT_MAX). Every term is
  // dt-aware: advection displacement (offset-exact, see shaders.js), drag and hyperviscosity
  // scale with dt, stochastic forcing with √dt, and tidal stretching uses the exact Δa_z over
  // [t, t+dt], which telescopes across sub-steps — so tiny steps are as stable as large ones.
  function step(dt) {
    const t1 = state.simTime + dt * T_UNIT * 1000;
    const next = Sky.tidalBodies(t1);
    const kf = 2 * Math.PI * L_CELL / (state.forcingKm * 1000);
    for (let n = 0; n < SH.N_MODES; n++) randomMode(forcingModes, n, kf);

    run(P.vel, vel, { u_psi: psi.read, u_q: q.read, u_mask: maskT }, {});
    run(P.advect, q.write, { u_q: q.read, u_vel: vel, u_mask: maskT }, {
      u_dt: dt, u_drag: 1 / (state.dragDays * 24), u_nu4: NU4,
      u_forceAmp: state.forcing * 1e-3, u_modGain: state.modGain, u_aRef: A_REF,
      u_stretchGain: state.stretchGain, u_Ks: KS, u_Rcells: R_CELLS, u_modes: forcingModes,
      u_moon0: bodyVec(bodies.moon), u_sun0: bodyVec(bodies.sun),
      u_moon1: bodyVec(next.moon), u_sun1: bodyVec(next.sun),
      u_k3: [bodies.moon.k3, bodies.sun.k3, next.moon.k3, next.sun.k3],
      u_kick: kick || [0, 0, 1, 0],
    });
    q.swap();
    solvePsi(state.sorSweeps);
    kick = null;
    bodies = next;
    state.simTime = t1;
  }

  // Split `hours` of model time into n equal sub-steps no larger than DT_MAX.
  // 5× at 60 FPS: 2 h/frame → two 1-hour steps. 0.01× at 60 FPS: one 14.4-second step.
  function advance(hours) {
    if (!(hours > 0)) return;
    const n = Math.min(MAX_STEPS_PER_FRAME, Math.ceil(hours / DT_MAX - 1e-9));
    const dt = Math.min(DT_MAX, hours / n);
    for (let i = 0; i < n; i++) step(dt);
  }

  function initField(amp, lamMinKm, lamMaxKm) {
    const modes = new Float32Array(4 * SH.N_INIT_MODES);
    for (let n = 0; n < SH.N_INIT_MODES; n++) {
      const lamCells = (lamMinKm + Math.random() * (lamMaxKm - lamMinKm)) * 1000 / L_CELL;
      randomMode(modes, n, 2 * Math.PI / lamCells);
    }
    run(P.init, q.write, { u_mask: maskT }, { u_modes: modes, u_amp: amp, u_Rcells: R_CELLS });
    q.swap();
    gl.clearColor(0, 0, 0, 0);
    for (const t of [psi.read, psi.write]) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    solvePsi(150);
    run(P.vel, vel, { u_psi: psi.read, u_q: q.read, u_mask: maskT }, {});
    history.length = 0;
    startTime = state.simTime;
  }

  // ───────────────────────── diagnostics ─────────────────────────
  const history = [];
  const diag = { urms: NaN, lambdaKm: NaN };
  function diagnostics() {
    run(P.reduce, reduceT, { u_vel: vel, u_mask: maskT }, {});
    gl.readPixels(0, 0, RW, RH, gl.RGBA, gl.FLOAT, reduceBuf);
    let ke = 0, ens = 0, area = 0;
    for (let i = 0; i < reduceBuf.length; i += 4) { ke += reduceBuf[i]; ens += reduceBuf[i + 1]; area += reduceBuf[i + 2]; }
    if (!(area > 0)) return;
    const keMean = ke / area;
    diag.urms = Math.sqrt(2 * keMean) * L_CELL / T_UNIT;
    // Energy-weighted wavelength 2π·sqrt(E/Z): grows as the inverse cascade proceeds.
    diag.lambdaKm = ens > 0 ? 2 * Math.PI * Math.sqrt(ke / ens) * KM_PER_CELL : NaN;
    history.push({ ke: keMean, lam: diag.lambdaKm });
    if (history.length > 240) history.shift();
    drawSparkline();
  }

  function drawSparkline() {
    const c = $('spark'), ctx = c.getContext('2d');
    const w = c.width, h = c.height;
    ctx.clearRect(0, 0, w, h);
    if (history.length < 2) return;
    for (const [key, color] of [['lam', '#f5a35c'], ['ke', '#6cc4ee']]) {
      const vals = history.map(p => p[key]).filter(Number.isFinite);
      if (vals.length < 2) continue;
      const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
      ctx.beginPath();
      history.forEach((p, i) => {
        const x = (i / (history.length - 1)) * (w - 4) + 2;
        const y = h - 3 - ((p[key] - lo) / span) * (h - 6);
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  // ───────────────────────── rendering ─────────────────────────
  let pxRatio = 1;
  function resize() {
    const s = map.getSize();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(s.x * dpr));
    canvas.height = Math.max(1, Math.round(s.y * dpr));
    canvas.style.width = s.x + 'px';
    canvas.style.height = s.y + 'px';
    pxRatio = canvas.width / s.x;
  }
  map.on('resize', resize);
  resize();

  // Fields plus VGTA contours evaluated in the shader for the Sun/Moon at exactly state.simTime.
  function render() {
    // Pin the canvas to the viewport while the map pane is translated by panning.
    const origin = map.containerPointToLayerPoint([0, 0]);
    L.DomUtil.setPosition(canvas, origin);
    const p0 = origin.add(map.getPixelOrigin());
    const F = FIELDS[state.field];
    const H = hfr.active;
    run(P.display, null, {
      u_q: q.read, u_psi: psi.read, u_vel: vel, u_mask: maskT,
      u_hfr0: H ? H.tex[0] : hfr.dummy, u_hfr1: H ? H.tex[1] : hfr.dummy, u_coast: coast,
    }, {
      u_canvas: [canvas.width, canvas.height], u_pxRatio: pxRatio,
      u_p0: [p0.x, p0.y], u_world: map.options.crs.scale(map.getZoom()),
      u_field: state.field, u_range: F.base * Math.pow(10, state.colorLog), u_alpha: 0.9,
      u_diverging: F.div, u_contours: state.contours, u_cmap: F.div ? CMAP.div : CMAP.seq,
      u_moon1: bodyVec(bodies.moon), u_sun1: bodyVec(bodies.sun),
      u_k3: [0, 0, bodies.moon.k3, bodies.sun.k3], u_kmPerCell: KM_PER_CELL,
      u_night: state.night ? (LIGHT ? 0.35 : 0.55) : 0, u_light: LIGHT,
      ...coastUniforms(p0),
      u_hfrOn: !!(H && state.hfr && hfr.inWindow), u_hfrMix: hfr.mix,
      u_hfrGeo: H ? H.geo : [0, 0, 1, 1], u_hfrSize: H ? [H.meta.nx, H.meta.ny] : [1, 1],
      u_glyphPx: state.glyphPx, u_arrowScale: state.arrowScale, u_time: performance.now() / 1000,
    });
  }

  // ───────────────────────── fine coastline for display clipping ─────────────────────────
  // The physics land mask is the model grid (≈39 km). For display, Natural Earth 50 m (10 m from
  // zoom 6) land polygons are rasterised in screen space, in the basemap's Web Mercator projection,
  // so the field is clipped exactly along the map's coastline. Redrawn after every pan/zoom; while
  // dragging, the last mask is shifted with the map (it is drawn with a margin around the view).
  const COAST_URL = { '50m': 'https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-50m.json',
                      '10m': 'https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-10m.json' };
  const COAST_10M_ZOOM = 6;
  const COAST_MARGIN = 0.25;                       // extra fraction of the view drawn on each side
  const coast = { tex: gl.createTexture(), canvas: document.createElement('canvas'), sets: {}, loading: {},
                  drawn: null, pending: false };
  coast.ctx = coast.canvas.getContext('2d', { willReadFrequently: false });
  gl.bindTexture(gl.TEXTURE_2D, coast.tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([255]));

  // Rings with longitudes unwrapped across the dateline, plus bounding boxes for culling.
  function prepareRings(topo) {
    const land = topojson.feature(topo, topo.objects.land);
    const geoms = land.type === 'FeatureCollection' ? land.features.map(f => f.geometry) : [land.geometry];
    const rings = [];
    for (const g of geoms) {
      const polys = g.type === 'Polygon' ? [g.coordinates] : g.type === 'MultiPolygon' ? g.coordinates : [];
      for (const poly of polys) for (const ring of poly) {
        const pts = new Float64Array(ring.length * 2);
        let prev = ring[0][0], shift = 0, x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
        ring.forEach(([lon, lat], i) => {
          if (lon - prev > 180) shift -= 360; else if (prev - lon > 180) shift += 360;
          prev = lon;
          const L2 = lon + shift, la = Math.max(-85, Math.min(85, lat));
          pts[2 * i] = L2; pts[2 * i + 1] = la;
          x0 = Math.min(x0, L2); x1 = Math.max(x1, L2); y0 = Math.min(y0, la); y1 = Math.max(y1, la);
        });
        rings.push({ pts, bbox: [x0, x1, y0, y1] });
      }
    }
    return rings;
  }

  async function loadCoast(level) {
    if (coast.sets[level] || coast.loading[level] || !window.topojson) return;
    coast.loading[level] = true;
    try {
      coast.sets[level] = prepareRings(await (await fetch(COAST_URL[level])).json());
      scheduleCoast();
    } catch (e) {
      console.warn(`Coastline ${level} unavailable; using the model mask for clipping.`, e);
    }
  }

  function scheduleCoast() {
    if (coast.pending) return;
    coast.pending = true;
    requestAnimationFrame(() => { coast.pending = false; drawCoast(); });
  }

  function drawCoast() {
    const zoom = map.getZoom();
    if (zoom >= COAST_10M_ZOOM) loadCoast('10m');
    const rings = (zoom >= COAST_10M_ZOOM && coast.sets['10m']) || coast.sets['50m'];
    if (!rings) return;
    const W = map.options.crs.scale(zoom);                       // world width [CSS px]
    const size = map.getSize();
    const mx = Math.round(size.x * COAST_MARGIN), my = Math.round(size.y * COAST_MARGIN);
    const p0 = map.containerPointToLayerPoint([0, 0]).add(map.getPixelOrigin()).subtract([mx, my]);
    const cw = Math.round((size.x + 2 * mx) * pxRatio), ch = Math.round((size.y + 2 * my) * pxRatio);
    if (coast.canvas.width !== cw || coast.canvas.height !== ch) { coast.canvas.width = cw; coast.canvas.height = ch; }
    const ctx = coast.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#fff';                                      // ocean
    ctx.fillRect(0, 0, cw, ch);
    // Visible longitude/latitude window (unwrapped, so it can span several world copies).
    const lonOf = x => (x / W) * 360 - 180;
    const latOf = y => Math.atan(Math.sinh(Math.PI * (1 - 2 * y / W))) / DEG;
    const west = lonOf(p0.x), east = lonOf(p0.x + cw / pxRatio);
    const north = latOf(Math.max(0, p0.y)), south = latOf(Math.min(W, p0.y + ch / pxRatio));
    const X = lon => ((lon + 180) / 360 * W - p0.x) * pxRatio;
    const Y = lat => ((1 - Math.log(Math.tan(Math.PI / 4 + lat * DEG / 2)) / Math.PI) / 2 * W - p0.y) * pxRatio;
    ctx.fillStyle = '#000';                                      // land
    ctx.beginPath();
    for (const { pts, bbox } of rings) {
      if (bbox[3] < south || bbox[2] > north) continue;
      for (let off = 360 * Math.floor((west - bbox[1]) / 360); bbox[0] + off <= east; off += 360) {
        if (bbox[1] + off < west) continue;
        ctx.moveTo(X(pts[0] + off), Y(pts[1]));
        for (let i = 2; i < pts.length; i += 2) ctx.lineTo(X(pts[i] + off), Y(pts[i + 1]));
        ctx.closePath();
      }
    }
    ctx.fill('evenodd');
    gl.bindTexture(gl.TEXTURE_2D, coast.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, gl.RED, gl.UNSIGNED_BYTE, coast.canvas);
    coast.drawn = { zoom, p0x: p0.x, p0y: p0.y, w: cw, h: ch };
  }

  // Shift between the current view and the view the mask was drawn for (device px).
  function coastUniforms(p0) {
    const d = coast.drawn;
    if (!d || d.zoom !== map.getZoom()) return { u_coastOn: false };
    return { u_coastOn: true, u_coastShift: [(p0.x - d.p0x) * pxRatio, (p0.y - d.p0y) * pxRatio], u_coastSize: [d.w, d.h] };
  }

  map.on('moveend zoomend resize', scheduleCoast);

  // ───────────────────────── HF radar observations (display only) ─────────────────────────
  // Bundles written by tools/hfr_preprocess.py register themselves in window.HFR_DATA when loaded
  // through a <script> tag, which works from file:// as well as from a web server. Months listed
  // in the manifest (tools/process_all_hfr.py) are fetched only when the model clock reaches them,
  // and at most HFR_CACHE decoded months are kept. Extra bundles: HFR_SOURCES or ?hfr=a.js,b.js
  const HFR_MANIFEST = 'data/hfr_manifest.js';
  const HFR_SOURCES = [];
  const HFR_CACHE = 3;
  const hfr = { catalog: [], active: null, entry: null, inWindow: false, mix: 0, decoded: 0, dummy: { tex: gl.createTexture() } };
  gl.bindTexture(gl.TEXTURE_2D, hfr.dummy.tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, new Float32Array(4));

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const el = document.createElement('script');
      el.src = src;
      el.onload = () => { el.remove(); resolve(); };
      el.onerror = () => { el.remove(); reject(new Error('could not load ' + src)); };
      document.head.appendChild(el);
    });
  }

  // Decode one bundle: int16 (u, v) in mm/s → per-frame RGBA32F (u, v, valid, 0) in m/s.
  function decodeHFR({ meta, data }) {
    const bin = atob(data);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const raw = new Int16Array(bytes.buffer);                  // little-endian, as written by numpy '<i2'
    const cells = meta.nx * meta.ny;
    const frames = [];
    for (let f = 0; f < meta.nFrames; f++) {
      const out = new Float32Array(cells * 4);
      for (let c = 0; c < cells; c++) {
        const u = raw[(f * cells + c) * 2], v = raw[(f * cells + c) * 2 + 1];
        if (u === meta.missing || v === meta.missing) continue;
        out[c * 4] = u * meta.scale;
        out[c * 4 + 1] = v * meta.scale;
        out[c * 4 + 2] = 1;                                   // valid
      }
      frames.push(out);
    }
    const tex = [0, 1].map(() => {
      const t = { tex: gl.createTexture() };
      gl.bindTexture(gl.TEXTURE_2D, t.tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, meta.nx, meta.ny);   // sampled only (texelFetch)
      return t;
    });
    const t0 = Date.parse(meta.t0), stepMs = meta.stepHours * 3600e3;
    return {
      meta, frames, tex, t0, stepMs, t1: t0 + (meta.nFrames - 1) * stepMs, loaded: [-1, -1],
      // grid-cell centre of (0,0) and spacing, in radians
      geo: [meta.lon0 * DEG, meta.lat0 * DEG, meta.dlon * DEG, meta.dlat * DEG],
    };
  }

  function uploadFrame(set, slot, k) {
    if (set.loaded[slot] === k) return;
    gl.bindTexture(gl.TEXTURE_2D, set.tex[slot].tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, set.meta.nx, set.meta.ny, gl.RGBA, gl.FLOAT, set.frames[k]);
    set.loaded[slot] = k;
  }

  const covers = (e, t) => t >= e.t0 - e.stepMs / 2 && t <= e.t1 + e.stepMs / 2;

  function catalogEntry(name, file, t0, nFrames, stepHours, bbox, sizeMB) {
    const stepMs = stepHours * 3600e3;
    return { name, file, t0, t1: t0 + (nFrames - 1) * stepMs, stepMs, bbox, sizeMB, set: null, loading: null, lastUsed: 0 };
  }

  // Decode bundles that have arrived in window.HFR_DATA since the last call, then drop the base64.
  function absorbBundles() {
    const list = window.HFR_DATA || [];
    for (; hfr.decoded < list.length; hfr.decoded++) {
      const bundle = list[hfr.decoded];
      try {
        const set = decodeHFR(bundle);
        bundle.data = null;
        const m = set.meta;
        let e = hfr.catalog.find(c => c.name === m.name);
        if (!e) {
          e = catalogEntry(m.name, null, set.t0, m.nFrames, m.stepHours,
            [m.lon0, m.lat0, m.lon0 + m.dlon * (m.nx - 1), m.lat0 + m.dlat * (m.ny - 1)], null);
          hfr.catalog.push(e);
        }
        e.set = set;
        e.lastUsed = performance.now();
      } catch (err) {
        hfr.error = 'Bad HF radar bundle: ' + err.message;
        console.error(err);
      }
    }
    hfr.catalog.sort((x, y) => x.t0 - y.t0);
    // Keep at most HFR_CACHE decoded months (least recently used go first).
    const loaded = hfr.catalog.filter(e => e.set).sort((x, y) => x.lastUsed - y.lastUsed);
    while (loaded.length > HFR_CACHE) {
      const e = loaded.shift();
      if (e === hfr.entry) continue;
      for (const t of e.set.tex) gl.deleteTexture(t.tex);
      e.set = null;
    }
  }

  function requestEntry(e) {
    if (e.set || e.loading || !e.file) return;
    e.loading = loadScript(e.file).then(absorbBundles)
      .catch(err => { hfr.error = err.message; console.warn(err); })
      .finally(() => { e.loading = null; });
  }

  // ───────────────────────── NOAA NCEI radials, fetched live by the browser ─────────────────────────
  // NCEI serves IOOS CODAR radial files with CORS enabled, so the page reads them directly:
  // {base}/{YYYY}/{YYYYMM}/CODAR/{SITE}/RDL_[m|i]_CODAR_{SITE}_{YYYY_MM_DD_HHMM}.ruv
  // Each hour, every site's file is fetched and parsed, and radials are combined into totals by
  // unweighted least squares (u·sinH + v·cosH = VELO), with the same QC as tools/hfr_ncei.py:
  // VFLG = 0, |VELO| ≤ 1.5 m/s, ≥ 3 radials from ≥ 2 sites, GDOP ≤ 2/sin 30°, a radial within 0.75 r.
  const NCEI_BASE = 'https://www.ncei.noaa.gov/data/oceans/ioos/hfradar/radial';
  const NCEI_NETWORKS = {
    ccal: { label: 'Central California, 6 km', sites: ['BIGC', 'PILR', 'PSLR', 'FORT', 'GCYN', 'MONT', 'PPNS', 'SCRZ'], resKm: 6, searchKm: 9 },
    mbay: { label: 'Monterey Bay, 2 km', sites: ['MLML', 'NPGS', 'SCRZ', 'PPNS', 'GCYN'], resKm: 2, searchKm: 3 },
    sfb: { label: 'San Francisco Bay, 1 km', sites: ['ANGL', 'CRIS', 'EXPL', 'RTC1', 'SAUS'], resKm: 1, searchKm: 1.5 },
  };
  const live = { key: null, listings: new Map(), frames: new Map(), pending: new Map(), grid: null, set: null,
                 origins: {}, info: '', error: '' };
  const LIVE_CACHE_HOURS = 48, LIVE_PREFETCH = 3;
  const RUV_NAME = /RDL_([im])_CODAR_(\w+?)_(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})\.ruv/g;

  // Hour → file URL for one site and month (measured pattern preferred); cached per site-month.
  function siteListing(site, y, m) {
    const ym = `${y}${String(m).padStart(2, '0')}`, key = `${site}/${ym}`;
    if (!live.listings.has(key)) {
      const dir = `${NCEI_BASE}/${y}/${ym}/CODAR/${site}/`;
      live.listings.set(key, fetch(dir).then(r => (r.ok ? r.text() : '')).then(html => {
        const best = new Map();
        for (const mm of html.matchAll(RUV_NAME)) {
          const [name, kind, , yy, mo, dd, hh, mi] = mm;
          if (mi !== '00') continue;
          const t = Date.UTC(+yy, +mo - 1, +dd, +hh);
          if (!best.has(t) || (kind === 'm' && best.get(t).kind === 'i')) best.set(t, { kind, url: dir + name });
        }
        return best;
      }).catch(() => new Map()));
    }
    return live.listings.get(key);
  }

  // LLUV radial table → {lon, lat, velo [m/s], head [rad]} of QC-passed radials.
  function parseRuv(text) {
    let cols = null, origin = null;
    const rows = [];
    for (const line of text.split('\n')) {
      if (line[0] === '%') {
        if (!cols && line.startsWith('%TableColumnTypes:')) cols = line.slice(18).trim().split(/\s+/);
        else if (!origin && line.startsWith('%Origin:')) origin = line.slice(8).trim().split(/\s+/).map(Number);
        else if (cols && rows.length && line.startsWith('%TableEnd')) break;
        continue;
      }
      if (cols && line.trim()) rows.push(line.trim().split(/\s+/));
    }
    if (!cols) return null;
    const c = name => cols.indexOf(name);
    const iLon = c('LOND'), iLat = c('LATD'), iV = c('VELO'), iH = c('HEAD'), iF = c('VFLG');
    const out = { lon: [], lat: [], velo: [], head: [], origin };
    for (const r of rows) {
      const v = +r[iV];
      if (!(Math.abs(v) <= 150) || (iF >= 0 && +r[iF] !== 0)) continue;
      out.lon.push(+r[iLon]); out.lat.push(+r[iLat]); out.velo.push(v / 100); out.head.push(+r[iH] * DEG);
    }
    return out;
  }

  // Fixed Cartesian grid for the session, from the first hour's radial extent (as in hfr_ncei.py).
  function makeGrid(parts, net) {
    const lon = parts.flatMap(p => p.lon).sort((a, b) => a - b), lat = parts.flatMap(p => p.lat).sort((a, b) => a - b);
    const q = (arr, f) => arr[Math.min(arr.length - 1, Math.max(0, Math.round(f * (arr.length - 1))))];
    const lon0 = q(lon, 0.002), lon1 = q(lon, 0.998), lat0 = q(lat, 0.002), lat1 = q(lat, 0.998);
    const latc = (lat0 + lat1) / 2, lonc = (lon0 + lon1) / 2;
    const kx = 111.32 * Math.cos(latc * DEG), ky = 110.57;
    const x0 = (lon0 - lonc) * kx, y0 = (lat0 - latc) * ky;
    const nx = Math.floor(((lon1 - lonc) * kx - x0) / net.resKm) + 1, ny = Math.floor(((lat1 - latc) * ky - y0) / net.resKm) + 1;
    return { nx, ny, x0, y0, kx, ky, lonc, latc, res: net.resKm,
             lonFirst: lonc + x0 / kx, latFirst: latc + y0 / ky, dlon: net.resKm / kx, dlat: net.resKm / ky };
  }

  // Least-squares totals on the grid from all radials within the search radius → RGBA32F frame.
  function combineRadials(parts, grid, net) {
    const { nx, ny, x0, y0, kx, ky, lonc, latc, res } = grid, G = nx * ny, r = net.searchKm, r2 = r * r;
    const Sss = new Float64Array(G), Scc = new Float64Array(G), Ssc = new Float64Array(G), Ssv = new Float64Array(G),
          Scv = new Float64Array(G), n = new Float64Array(G), sites = new Uint32Array(G), near = new Float64Array(G).fill(Infinity);
    parts.forEach((p, si) => {
      for (let i = 0; i < p.lon.length; i++) {
        const x = (p.lon[i] - lonc) * kx, y = (p.lat[i] - latc) * ky, s = Math.sin(p.head[i]), c = Math.cos(p.head[i]), v = p.velo[i];
        const i0 = Math.max(0, Math.ceil((x - r - x0) / res)), i1 = Math.min(nx - 1, Math.floor((x + r - x0) / res));
        const j0 = Math.max(0, Math.ceil((y - r - y0) / res)), j1 = Math.min(ny - 1, Math.floor((y + r - y0) / res));
        for (let j = j0; j <= j1; j++) for (let k = i0; k <= i1; k++) {
          const dx = x0 + k * res - x, dy = y0 + j * res - y, d2 = dx * dx + dy * dy;
          if (d2 > r2) continue;
          const g = j * nx + k;
          Sss[g] += s * s; Scc[g] += c * c; Ssc[g] += s * c; Ssv[g] += s * v; Scv[g] += c * v; n[g]++;
          sites[g] |= 1 << si;
          if (d2 < near[g]) near[g] = d2;
        }
      }
    });
    const out = new Float32Array(G * 4), gdopMax = 2 / Math.sin(30 * DEG), nearMax = (0.75 * r) ** 2;
    let count = 0;
    for (let g = 0; g < G; g++) {
      let ns = 0;
      for (let b = sites[g]; b; b &= b - 1) ns++;
      const det = Sss[g] * Scc[g] - Ssc[g] * Ssc[g];
      if (n[g] < 3 || ns < 2 || !(det > 0) || near[g] > nearMax) continue;
      if (Math.sqrt(n[g] * (Sss[g] + Scc[g]) / det) > gdopMax) continue;
      const u = (Scc[g] * Ssv[g] - Ssc[g] * Scv[g]) / det, v = (Sss[g] * Scv[g] - Ssc[g] * Ssv[g]) / det;
      if (Math.hypot(u, v) > 1.5) continue;
      out[g * 4] = u; out[g * 4 + 1] = v; out[g * 4 + 2] = 1; count++;
    }
    return { frame: out, count };
  }

  // Fetch, parse and combine one hour (deduplicated; resolves to a frame record or null).
  function liveHour(t) {
    if (live.frames.has(t)) return Promise.resolve(live.frames.get(t));
    if (live.pending.has(t)) return live.pending.get(t);
    const key = live.key, net = NCEI_NETWORKS[key], d = new Date(t);
    const job = (async () => {
      const lists = await Promise.all(net.sites.map(s => siteListing(s, d.getUTCFullYear(), d.getUTCMonth() + 1)));
      const texts = await Promise.all(net.sites.map((s, i) => {
        const f = lists[i].get(t);
        return f ? fetch(f.url).then(r => (r.ok ? r.text() : null)).catch(() => null) : null;
      }));
      if (live.key !== key) return null;                       // network changed meanwhile
      const parts = [];
      texts.forEach((txt, i) => {
        const p = txt && parseRuv(txt);
        if (p && p.lon.length) { parts.push(p); if (p.origin) live.origins[net.sites[i]] = p.origin; }
      });
      let rec = { sites: parts.length, total: net.sites.length, count: 0, frame: null };
      if (parts.length >= 2) {
        if (!live.grid) { live.grid = makeGrid(parts, net); live.set = liveSet(live.grid); }
        Object.assign(rec, combineRadials(parts, live.grid, net));
      }
      live.frames.set(t, rec);
      for (const k of [...live.frames.keys()]) if (Math.abs(k - t) > LIVE_CACHE_HOURS * 3600e3) live.frames.delete(k);
      return rec;
    })().catch(e => { live.error = e.message; return null; }).finally(() => live.pending.delete(t));
    live.pending.set(t, job);
    return job;
  }

  // Texture pair and geometry for the live grid, shaped like a decoded bundle.
  function liveSet(g) {
    const tex = [0, 1].map(() => {
      const t = { tex: gl.createTexture() };
      gl.bindTexture(gl.TEXTURE_2D, t.tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, g.nx, g.ny);
      return t;
    });
    return { meta: { name: 'NCEI ' + NCEI_NETWORKS[live.key].label, nx: g.nx, ny: g.ny, dxKm: g.res },
             tex, loaded: [null, null], geo: [g.lonFirst * DEG, g.latFirst * DEG, g.dlon * DEG, g.dlat * DEG] };
  }

  function setLiveSource(key) {
    if (live.set) for (const t of live.set.tex) gl.deleteTexture(t.tex);
    Object.assign(live, { key, frames: new Map(), pending: new Map(), grid: null, set: null, error: '' });
    state.hfrSource = key || 'bundles';
  }

  function updateLive() {
    const t = state.simTime, h0 = Math.floor(t / 3600e3) * 3600e3, h1 = h0 + 3600e3;
    for (let k = 0; k <= LIVE_PREFETCH; k++) if (live.pending.size < 3 || k < 2) liveHour(h0 + k * 3600e3);
    const r0 = live.frames.get(h0), r1 = live.frames.get(h1);
    hfr.entry = null;
    hfr.inWindow = !!(r0 && r0.frame && live.set);
    hfr.active = hfr.inWindow ? live.set : null;
    if (!hfr.inWindow) return;
    const f1 = r1 && r1.frame ? r1 : r0;
    for (const [slot, rec, key] of [[0, r0, h0], [1, f1, f1 === r0 ? h0 : h1]]) {
      if (live.set.loaded[slot] === key) continue;
      gl.bindTexture(gl.TEXTURE_2D, live.set.tex[slot].tex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, live.grid.nx, live.grid.ny, gl.RGBA, gl.FLOAT, rec.frame);
      live.set.loaded[slot] = key;
    }
    hfr.mix = f1 === r0 ? 0 : (t - h0) / 3600e3;
    live.info = `${r0.sites}/${r0.total} sites · ${r0.count} vectors`;
  }

  // Pick the month covering the model clock (loading it if needed) and the two bracketing frames.
  function updateHFR() {
    if (live.key) return updateLive();
    const t = state.simTime;
    const e = hfr.catalog.find(c => covers(c, t)) || null;
    hfr.entry = e;
    hfr.inWindow = !!e;
    hfr.active = e && e.set ? e.set : null;
    if (!e) return;
    if (!e.set) { requestEntry(e); return; }
    e.lastUsed = performance.now();
    const set = e.set;
    const f = Math.min(Math.max((t - set.t0) / set.stepMs, 0), set.meta.nFrames - 1);
    const k0 = Math.floor(f), k1 = Math.min(k0 + 1, set.meta.nFrames - 1);
    uploadFrame(set, 0, k0);
    uploadFrame(set, 1, k1);
    hfr.mix = f - k0;                                          // linear in time between hourly frames
    hfr.frame = f;
  }

  // The catalogue month closest in time to the model clock.
  function nearestEntry() {
    const t = state.simTime;
    let best = null, bestD = Infinity;
    for (const e of hfr.catalog) {
      const d = t < e.t0 ? e.t0 - t : t > e.t1 ? t - e.t1 : 0;
      if (d < bestD) { best = e; bestD = d; }
    }
    return best;
  }

  function hfrStatus() {
    const el = $('hfr-status');
    if (live.key) {
      const net = NCEI_NETWORKS[live.key], h0 = Math.floor(state.simTime / 3600e3) * 3600e3, r0 = live.frames.get(h0);
      $('b-hfr-go').hidden = hfr.inWindow;
      el.textContent = `NOAA NCEI live · ${net.label} · `
        + (hfr.inWindow ? live.info
          : live.pending.size ? 'fetching radials…'
          : r0 ? `${r0.sites}/${r0.total} sites have data for this hour (need ≥ 2)` : live.error || 'waiting…');
      return;
    }
    const e = hfr.entry;
    $('b-hfr-go').hidden = !hfr.catalog.length || hfr.inWindow;
    if (!hfr.catalog.length) {
      el.textContent = hfr.error || 'No HF radar data (run tools/process_all_hfr.py — see README).';
    } else if (e && !e.set) {
      el.textContent = `Loading ${e.name}${e.sizeMB ? ` (${e.sizeMB} MB)` : ''}…`;
    } else if (e) {
      const m = e.set.meta;
      el.textContent = `${m.name}: frame ${(hfr.frame + 1).toFixed(1)} / ${m.nFrames} · ${m.nx}×${m.ny} @ `
        + (m.dxKm ? m.dxKm + ' km' : m.dlon.toFixed(3) + '°');
    } else {
      const first = hfr.catalog[0], last = hfr.catalog[hfr.catalog.length - 1];
      el.textContent = `No radar data at this time · ${hfr.catalog.length} month(s) available, `
        + `${fmtClock(first.t0).slice(0, 10)} → ${fmtClock(last.t1).slice(0, 10)}`;
    }
  }

  async function loadHFR() {
    try {
      await loadScript(HFR_MANIFEST);
      for (const m of window.HFR_MANIFEST || []) {
        hfr.catalog.push(catalogEntry(m.name, m.file, Date.parse(m.t0), Math.round((Date.parse(m.t1) - Date.parse(m.t0))
          / (m.stepHours * 3600e3)) + 1, m.stepHours, m.bbox, m.sizeMB));
      }
    } catch (err) {
      console.info('No HF radar manifest:', err.message);
    }
    const extra = [...HFR_SOURCES, ...(qs.get('hfr') || '').split(',').filter(Boolean)];
    for (const src of extra) {
      try { await loadScript(src); } catch (err) { hfr.error = err.message; console.warn(err); }
    }
    absorbBundles();
    $('hfr-section').hidden = false;
    updateHFR();
    hfrStatus();
  }

  // ───────────────────────── sky overlays (display only) ─────────────────────────
  // Unwrap a lat/lon path so consecutive points never jump by more than 180° of longitude.
  function unwrapPath(pts) {
    const out = [];
    let prev = null;
    for (const [lat, lon] of pts) {
      let L2 = lon;
      if (prev !== null) L2 += 360 * Math.round((prev - L2) / 360);
      out.push([lat, L2]);
      prev = L2;
    }
    return out;
  }

  // Planets and ecliptic for state.simTime; called every frame so they glide with the clock.
  let skyCache = null;
  function updateSky() {
    if (state.planets) {
      skyCache = Sky.planets(state.simTime);
      for (const p of skyCache) planetMk[p.name].setLatLng([p.lat, nearestCopy(p.lon)]);
    }
    if (state.ecliptic) {
      // The ecliptic spans 360° of longitude; three copies keep it continuous across world wraps.
      const path = unwrapPath(Sky.eclipticPath(state.simTime, 4));
      const base = 360 * Math.round((map.getCenter().lng - path[0][1]) / 360);
      eclipticLine.setLatLngs([-360, 0, 360].map(k => path.map(([la, lo]) => [la, lo + base + k])));
    }
  }

  function setOverlay(layers, on) {
    for (const l of layers) on ? l.addTo(map) : l.remove();
  }

  // Moon disc: lit limb on the right while waxing, left while waning (northern-hemisphere view).
  function drawMoon(elongDeg) {
    const c = $('moon-canvas'), ctx = c.getContext('2d');
    const cx = c.width / 2, cy = c.height / 2, r = c.width / 2 - 3;
    const waxing = elongDeg < 180, k = Math.cos(elongDeg * DEG);   // +1 new … −1 full
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = '#1c2231';
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = '#ece6d6';
    ctx.beginPath();
    ctx.arc(cx, cy, r, -Math.PI / 2, Math.PI / 2, !waxing);                    // lit limb
    ctx.ellipse(cx, cy, r * Math.abs(k), r, 0, Math.PI / 2, -Math.PI / 2, waxing ? k > 0 : k <= 0); // terminator
    ctx.fill();
    ctx.strokeStyle = 'rgba(236,230,214,0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, 2 * Math.PI); ctx.stroke();
  }

  // ───────────────────────── clock, HUD, legend ─────────────────────────
  const fmtLat = v => Math.abs(v).toFixed(1) + '°' + (v >= 0 ? 'N' : 'S');
  const fmtLon = v => Math.abs(v).toFixed(1) + '°' + (v >= 0 ? 'E' : 'W');
  function phaseName(e) {
    const names = ['New', 'Waxing crescent', 'First quarter', 'Waxing gibbous', 'Full', 'Waning gibbous', 'Last quarter', 'Waning crescent'];
    return names[Math.round(e / 45) % 8];
  }
  const nearestCopy = lon => lon + 360 * Math.round((map.getCenter().lng - lon) / 360);
  const pad = n => String(n).padStart(2, '0');
  function fmtClock(ms) {
    const d = new Date(ms);
    return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} `
      + `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
  }
  const fmtScale = v => (v < 0.1 ? v.toFixed(3) : v < 1 ? v.toFixed(2) : v.toFixed(1));
  const fmtRate = hps => (hps >= 24 ? (hps / 24).toFixed(2) + ' model days/s'
    : hps >= 1 ? hps.toFixed(2) + ' model h/s' : (hps * 60).toFixed(1) + ' model min/s');

  // Time panel: a native datetime-local picker, transparent over our own "YYYY-MM-DD HH:mm"
  // display (locale-independent, never clipped). Its value is always read and written as UTC
  // ("YYYY-MM-DDTHH:mm"); the browser's local-time interpretation is never used.
  const picker = $('t-datetime');
  const toPickerValue = ms => {
    const d = new Date(ms);
    return `${String(d.getUTCFullYear()).padStart(4, '0')}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())}`
      + `T${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}`;
  };
  const fromPickerValue = v => (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(v) ? Date.parse(v.slice(0, 16) + ':00Z') : NaN);

  // Mirrors the clock every frame, but never while the picker has focus (the user is choosing).
  function updateClock() {
    const d = new Date(state.simTime);
    if (document.activeElement !== picker) {
      const v = toPickerValue(state.simTime);
      if (picker.value !== v) picker.value = v;   // only rewrite when the minute changes
    }
    // While choosing, show the pending selection; otherwise the live clock.
    const shown = document.activeElement === picker && picker.value ? picker.value : toPickerValue(state.simTime);
    $('t-display').textContent = shown.replace('T', ' ');
    $('t-sec').textContent = ':' + pad(d.getUTCSeconds());
    $('t-jd').textContent = Sky.julianDay(state.simTime).toFixed(5);
    $('clock-rate').textContent = state.paused ? 'paused'
      : '×' + fmtScale(state.timeScale) + ' · ' + fmtRate(state.timeScale * BASE_HOURS_PER_SEC);
  }

  // Move the clock (and every celestial position) to t. The fluid state is kept: the turbulence
  // is statistically stationary, and re-anchoring the Sun/Moon means Δa_z has no spurious jump.
  function jumpTo(t) {
    if (!Number.isFinite(t)) return;
    startTime += t - state.simTime;
    state.simTime = t;
    bodies = Sky.tidalBodies(t);
    rate = { t: performance.now(), sim: t, hoursPerSec: rate.hoursPerSec };
    updateClock();
    updateMarkers();
    updateSky();
    updateHUD();
  }

  let rate = { t: performance.now(), sim: state.simTime, hoursPerSec: 0 };
  function updateHUD() {
    const b = bodies;
    $('r-elapsed').textContent = ((state.simTime - startTime) / 864e5).toFixed(2) + ' days';
    const now = performance.now();
    if (now - rate.t > 1000) {
      rate = { t: now, sim: state.simTime, hoursPerSec: (state.simTime - rate.sim) / 3600e3 / ((now - rate.t) / 1000) };
    }
    // Measured rate: drops below the requested rate only if the GPU hits MAX_STEPS_PER_FRAME.
    $('r-rate').textContent = fmtRate(rate.hoursPerSec);
    $('m-phase').textContent = phaseName(b.elong);
    $('m-illum').textContent = (b.illum * 100).toFixed(1) + '%';
    $('m-age').textContent = (b.elong / 360 * 29.530589).toFixed(1) + ' d';
    $('m-dist').textContent = Math.round(b.moon.dist / 1000).toLocaleString('en-US') + ' km';
    drawMoon(b.elong);
    if (skyCache) {
      $('planet-list').innerHTML = skyCache.map(p => `<li><i style="--c:${PLANET_STYLE[p.name].color}"></i>`
        + `<span class="pn">${PLANET_STYLE[p.name].glyph} ${p.name}</span>`
        + `<span class="pv">${fmtLat(p.lat)} ${fmtLon(p.lon)}</span><span class="pv">${p.distAU.toFixed(2)} AU</span></li>`).join('');
    }
    $('r-spring').textContent = b.spring > 0.8 ? 'Spring' : b.spring < 0.55 ? 'Neap' : b.elong % 180 < 90 ? 'Spring → neap' : 'Neap → spring';
    $('r-spring-bar').style.width = (b.spring * 100).toFixed(0) + '%';
    $('r-sublunar').textContent = fmtLat(b.moon.lat) + ', ' + fmtLon(b.moon.lon);
    $('r-subsolar').textContent = fmtLat(b.sun.lat) + ', ' + fmtLon(b.sun.lon);
    const peak = Sky.vgta(b.moon.u, b.moon) + Sky.vgta(b.moon.u, b.sun);
    $('r-vgta').textContent = peak.toFixed(2) + ' µm/s² at sub-lunar point';
    $('r-urms').textContent = Number.isFinite(diag.urms) ? diag.urms.toFixed(3) + ' m/s' : '—';
    $('r-lambda').textContent = Number.isFinite(diag.lambdaKm) ? diag.lambdaKm.toFixed(0) + ' km' : '—';
  }

  function updateMarkers() {
    if (!state.markers) return;
    moonMk.setLatLng([bodies.moon.lat, nearestCopy(bodies.moon.lon)]);
    sunMk.setLatLng([bodies.sun.lat, nearestCopy(bodies.sun.lon)]);
  }

  function updateLegend() {
    const F = FIELDS[state.field];
    const r = F.base * Math.pow(10, state.colorLog);
    const pal = F.div ? DIVERGING : SEQUENTIAL;
    const stops = pal.map((c, i) => `rgb(${c.map(x => Math.round(x * 255)).join(',')}) ${(i / 8 * 100).toFixed(1)}%`);
    $('legend-bar').style.background = `linear-gradient(to right, ${stops.join(', ')})`;
    $('legend-title').textContent = F.name + ' [' + F.unit + ']';
    const fmt = v => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toPrecision(2));
    $('legend-min').textContent = fmt(F.phys(F.div ? -r : 0));
    $('legend-mid').textContent = fmt(F.phys(F.div ? 0 : r / 2));
    $('legend-max').textContent = fmt(F.phys(r));
  }

  // ───────────────────────── UI bindings ─────────────────────────
  function bindRange(id, key, fmt, onChange) {
    const el = $(id), out = $(id + '-val');
    const update = () => {
      state[key] = parseFloat(el.value);
      out.textContent = fmt(state[key]);
      if (onChange) onChange();
    };
    el.addEventListener('input', update);
    update();
  }
  function setRange(id, value) {
    const el = $(id);
    el.value = value;
    el.dispatchEvent(new Event('input'));
  }

  bindRange('s-color', 'colorLog', v => '×' + Math.pow(10, v).toFixed(2), updateLegend);
  // Simulation speed: slider is log10(timeScale), so 0.01×–5× gets equal room per decade.
  bindRange('s-speed', 'speedLog', v => {
    state.timeScale = Math.pow(10, v);
    return '×' + fmtScale(state.timeScale);
  });
  for (const b of document.querySelectorAll('[data-speed]')) {
    b.addEventListener('click', () => setRange('s-speed', Math.log10(parseFloat(b.dataset.speed))));
  }
  bindRange('s-sor', 'sorSweeps', v => String(v));
  bindRange('s-forcing', 'forcing', v => v.toFixed(2) + ' ×10⁻³');
  bindRange('s-kf', 'forcingKm', v => v + ' km');
  bindRange('s-drag', 'dragDays', v => v + ' days');
  bindRange('s-ld', 'ldExagg', v => '×' + v.toFixed(1) + ' → ' + ldKm(30).toFixed(0) + ' km @30°');
  bindRange('s-stretch', 'stretchGain', v => '×' + v);
  bindRange('s-mod', 'modGain', v => v.toFixed(1));
  bindRange('s-glyph', 'glyphPx', v => v + ' px');
  bindRange('s-arrow', 'arrowScale', v => `1 m/s = ${v} px`);
  $('c-hfr').addEventListener('change', e => { state.hfr = e.target.checked; });
  $('hfr-source').addEventListener('change', e => {
    setLiveSource(e.target.value === 'bundles' ? null : e.target.value);
    updateHFR();
    hfrStatus();
  });
  // Jump the clock to the nearest month with radar data and fly to its footprint.
  $('b-hfr-go').addEventListener('click', () => {
    if (live.key) {                                              // NCEI: a month known to have data
      const t = state.simTime, ref = Date.UTC(2020, 11, 15, 0);
      const rec = live.frames.get(Math.floor(t / 3600e3) * 3600e3);
      if (!rec || rec.sites < 2) jumpTo(ref);
      setRange('s-speed', Math.log10(0.01));
      const pts = NCEI_NETWORKS[live.key].sites.map(s => live.origins[s]).filter(Boolean);
      const box = { ccal: [[35.4, -124.6], [38.3, -121.6]], mbay: [[36.45, -122.25], [37.05, -121.75]],
                    sfb: [[37.75, -122.55], [37.92, -122.32]] }[live.key];
      map.fitBounds(pts.length > 1 && live.grid ? [[live.grid.latFirst, live.grid.lonFirst],
        [live.grid.latFirst + live.grid.dlat * (live.grid.ny - 1), live.grid.lonFirst + live.grid.dlon * (live.grid.nx - 1)]] : box,
        { padding: [40, 40] });
      return;
    }
    const e = nearestEntry();
    if (!e) return;
    const t = state.simTime;                                   // land with room to keep playing
    jumpTo(t < e.t0 ? e.t0 : t > e.t1 ? Math.max(e.t0, e.t1 - 7 * 864e5) : t);
    setRange('s-speed', Math.log10(0.01));
    const [lon0, lat0, lon1, lat1] = e.bbox;
    map.fitBounds([[lat0, lon0], [lat1, lon1]], { padding: [60, 60] });
  });

  $('field').addEventListener('change', e => { state.field = +e.target.value; updateLegend(); });
  $('c-contours').addEventListener('change', e => { state.contours = e.target.checked; });
  $('c-night').addEventListener('change', e => { state.night = e.target.checked; });
  $('c-planets').addEventListener('change', e => {
    state.planets = e.target.checked;
    setOverlay(Object.values(planetMk), state.planets);
    if (state.planets) updateSky();
  });
  $('c-ecliptic').addEventListener('change', e => {
    state.ecliptic = e.target.checked;
    setOverlay([eclipticLine], state.ecliptic);
    if (state.ecliptic) updateSky();
  });
  $('c-markers').addEventListener('change', e => {
    state.markers = e.target.checked;
    for (const mk of [moonMk, sunMk]) state.markers ? mk.addTo(map) : mk.remove();
  });
  function setPaused(paused) {
    state.paused = paused;
    $('b-pause').textContent = paused ? 'Resume' : 'Pause';
    $('t-play').textContent = paused ? '▶' : '❚❚';
    $('t-play').title = paused ? 'Resume (Space)' : 'Pause (Space)';
  }
  const togglePause = () => { pickerPaused = false; setPaused(!state.paused); };
  $('b-pause').addEventListener('click', togglePause);
  $('t-play').addEventListener('click', togglePause);

  // Date-time picker: opening it pauses the clock; every complete value picked jumps the
  // simulation there at once; closing it (blur / Enter / Escape) resumes if it was playing.
  const RESUME_AFTER_PICK = true;  // false: stay paused on the chosen moment
  let pickerPaused = false;
  function pausePickerClock() {
    if (!state.paused) { pickerPaused = true; setPaused(true); }
  }
  picker.addEventListener('focus', pausePickerClock);
  picker.addEventListener('pointerdown', () => {
    pausePickerClock();
    try { picker.showPicker(); } catch (e) { /* older browsers: native focus behaviour */ }
  });
  picker.addEventListener('change', () => {
    const t = fromPickerValue(picker.value);
    if (Number.isFinite(t)) jumpTo(t);
  });
  picker.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === 'Escape') picker.blur(); });
  picker.addEventListener('blur', () => {
    picker.value = toPickerValue(state.simTime);   // drop any half-typed value
    if (pickerPaused && RESUME_AFTER_PICK) setPaused(false);
    pickerPaused = false;
  });
  $('b-reset').addEventListener('click', () => {
    if (state.forcing === 0) setRange('s-forcing', 0.8);
    initField(0.004, 150, 300);
  });
  $('b-decay').addEventListener('click', () => {
    setRange('s-forcing', 0);
    setRange('s-mod', 0);
    initField(0.025, 120, 200);
  });
  $('b-now').addEventListener('click', () => jumpTo(Date.now()));
  $('t-now').addEventListener('click', () => jumpTo(Date.now()));
  $('panel-toggle').addEventListener('click', () => $('panel').classList.toggle('collapsed'));
  if (window.innerWidth < 720) $('panel').classList.add('collapsed');
  window.addEventListener('keydown', e => {
    if (e.code === 'Space' && !/INPUT|SELECT|BUTTON/.test(document.activeElement.tagName)) {
      e.preventDefault();
      togglePause();
    }
  });

  // Click: cyclone (shift: anticyclone) with radius ≈ local Rossby radius.
  map.on('click', e => {
    const lat = e.latlng.lat * DEG;
    if (Math.abs(lat) > LAT_MAX - 2 * D_RAD) return;
    const lon = ((e.latlng.lng + 540) % 360 - 180) * DEG;
    const radius = Math.min(14, Math.max(3, 1 / Math.sqrt(invLd2(lat))));
    const sense = e.originalEvent.shiftKey ? -1 : 1;
    kick = [lat, lon, radius, 0.02 * sense * Math.sign(lat || 1)];
  });

  // ───────────────────────── main loop ─────────────────────────
  let ready = false, frameNo = 0, lastFrame = null;

  function frame(now) {
    requestAnimationFrame(frame);
    // Real time since the last frame, from the rAF timestamp (sub-millisecond resolution).
    const realSec = lastFrame === null ? 0 : Math.min((now - lastFrame) / 1000, MAX_FRAME_SEC);
    lastFrame = now;
    if (!ready || contextLost) return;
    if (!state.paused) advance(state.timeScale * BASE_HOURS_PER_SEC * realSec);
    // Everything below reads state.simTime / bodies, which the sub-steps just advanced,
    // so fields, VGTA contours, markers, radar frames and clock stay in lock-step on every frame.
    updateHFR();
    render();          // fields, VGTA contours and day/night terminator
    updateMarkers();
    updateSky();       // planets + ecliptic
    updateClock();
    frameNo++;
    if (!state.paused && frameNo % 20 === 0) diagnostics();
    if (frameNo % 6 === 0) { updateHUD(); hfrStatus(); }
  }

  (async function start() {
    $('status').textContent = 'Loading coastlines…';
    const { m, ok } = await buildLandMask();
    uploadMask(m);
    $('status').textContent = ok ? '' : 'Coastlines unavailable — running an all-ocean planet.';
    initField(0.004, 150, 300);
    // Optional ?spinup=<days>: integrate before the first frame so the view opens equilibrated.
    const spinHours = Math.min(365, parseFloat(qs.get('spinup')) || 0) * 24;
    if (spinHours > 0) {
      for (let i = 0; i < spinHours; i++) step(DT_MAX);
      diagnostics();
    }
    for (const mk of [moonMk, sunMk]) mk.addTo(map);
    setOverlay(Object.values(planetMk), state.planets);
    setOverlay([eclipticLine], state.ecliptic);
    updateMarkers();
    updateSky();
    updateLegend();
    updateHUD();
    loadCoast('50m');
    await loadHFR();
    ready = true;
  })();
  requestAnimationFrame(frame);

  function fatal(msg) {
    $('status').textContent = msg;
    $('status').classList.add('error');
  }
})();
