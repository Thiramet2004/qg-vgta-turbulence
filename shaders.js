/* shaders.js — GLSL ES 3.00 programs for the equivalent-barotropic QG + VGTA model.
 *
 * Grid: equirectangular, NX × NY cells, longitude periodic, walls at ±latMax.
 * Model units: length = 1 meridional cell (R·Δφ ≈ 39 km at NX = 1024), time = 1 hour.
 * Texture row 0 is the southern edge.
 *
 * Prognostic variable: q = ∇²ψ − ψ/L_d²(φ)         (relative PV, f excluded)
 *   ∂q/∂t + J(ψ, q + f) = F_VGTA + F_eddy − r ∇²ψ − ν₄∇⁴q
 */
'use strict';
window.QGShaders = (function () {
  const N_MODES = 12;       // plane waves in the stochastic forcing (per step)
  const N_INIT_MODES = 32;  // plane waves in the initial condition

  // Attribute-less full-screen triangle.
  const VERT = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

  const COMMON = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

uniform ivec2 u_grid;     // NX, NY
uniform float u_dRad;     // grid spacing [rad]
uniform float u_latMax;   // northern grid edge [rad]
uniform float u_F0;       // 2Ω in model units [1/h]
uniform float u_c;        // gravity-wave speed setting L_d [cells/h]

const float PI = 3.141592653589793;

// GLSL ES leaves % undefined for negative operands, so shift into range first.
int wrapX(int i) { return (i + 4 * u_grid.x) % u_grid.x; }
ivec2 cell(ivec2 p) { return ivec2(wrapX(p.x), clamp(p.y, 0, u_grid.y - 1)); }
float f1(sampler2D t, ivec2 p) { return texelFetch(t, cell(p), 0).r; }
vec4 f4(sampler2D t, ivec2 p) { return texelFetch(t, cell(p), 0); }

float latOfRow(float j) { return -u_latMax + (j + 0.5) * u_dRad; }
float lonOfCol(float i) { return -PI + (i + 0.5) * u_dRad; }
// Zonal cell width in units of the meridional cell height.
float metric(float lat) { return max(cos(lat), 0.05); }
float coriolis(float lat) { return u_F0 * sin(lat); }
float betaND(float lat) { return u_F0 * cos(lat) * u_dRad; }
// 1/L_d² with L_d = c/|f| at mid-latitudes, blending to the equatorial radius sqrt(c/2β).
float invLd2(float lat) {
  float f = coriolis(lat);
  return (f * f + 2.0 * betaND(lat) * u_c) / (u_c * u_c);
}
vec3 unitVec(float lat, float lon) { return vec3(cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)); }

// Vertical tidal acceleration [µm/s²] from one body (degree 2 + 3 Legendre terms).
// b.xyz = Earth-fixed unit vector to the body, b.w = GM R/d³, k3 = GM R²/d⁴.
float vgtaBody(vec3 X, vec4 b, float k3) {
  float c = dot(X, b.xyz);
  return b.w * (3.0 * c * c - 1.0) + k3 * 1.5 * c * (5.0 * c * c - 3.0);
}

// Bilinear sample at cell p plus offset d (cell-index units, centres at integers).
// Keeping the base cell and the offset separate preserves sub-ulp displacements: at 0.01×
// speed a step moves fluid ~1e-4 cells, which float(p) + d would round away near i ≈ 1000.
vec4 bilinearAt(sampler2D t, ivec2 p, vec2 d) {
  vec2 b = floor(d), w = d - b;
  ivec2 i = p + ivec2(b);
  return mix(mix(f4(t, i), f4(t, i + ivec2(1, 0)), w.x),
             mix(f4(t, i + ivec2(0, 1)), f4(t, i + ivec2(1, 1)), w.x), w.y);
}
vec4 bilinear4(sampler2D t, vec2 x) { return bilinearAt(t, ivec2(floor(x)), fract(x)); }
`;

  // Pass 1: geostrophic velocity, relative vorticity and ∇²q from ψ and q.
  const VEL = COMMON + `
uniform sampler2D u_psi, u_q, u_mask;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  if (texelFetch(u_mask, p, 0).r < 0.5) { o = vec4(0.0); return; }
  float h = metric(latOfRow(float(p.y))), ih2 = 1.0 / (h * h);
  ivec2 E = ivec2(1, 0), N = ivec2(0, 1);
  float c = f1(u_psi, p), e = f1(u_psi, p + E), w = f1(u_psi, p - E), n = f1(u_psi, p + N), s = f1(u_psi, p - N);
  float u = -(n - s) * 0.5;               // u = -∂ψ/∂y
  float v = (e - w) * 0.5 / h;            // v =  ∂ψ/∂x
  float zeta = (e + w - 2.0 * c) * ih2 + (n + s - 2.0 * c);
  float qc = f1(u_q, p);
  float lapq = (f1(u_q, p + E) + f1(u_q, p - E) - 2.0 * qc) * ih2 + (f1(u_q, p + N) + f1(u_q, p - N) - 2.0 * qc);
  o = vec4(u, v, zeta, lapq);
}`;

  // Pass 2: advection of total PV, dissipation, VGTA and eddy forcing.
  const ADVECT = COMMON + `
#define NMODES ${N_MODES}
uniform sampler2D u_q, u_vel, u_mask;
uniform float u_dt, u_drag, u_nu4, u_forceAmp, u_modGain, u_aRef, u_stretchGain, u_Ks, u_Rcells;
uniform vec4 u_modes[NMODES];                   // k-vector [1/cells] (xyz) + phase (w)
uniform vec4 u_moon0, u_sun0, u_moon1, u_sun1;  // bodies at t and t+dt
uniform vec4 u_k3;                              // degree-3 coefficients: moon0, sun0, moon1, sun1
uniform vec4 u_kick;                            // user vortex: lat, lon [rad], radius [cells], amplitude
out vec4 o;

// Love-number diminishing factors γ_n = 1 + k_n − h_n (k₂ 0.298, h₂ 0.604; k₃ 0.093, h₃ 0.292):
// the solid Earth rises with the tide, so only γ_n of the equilibrium tide changes water depth.
const float GAMMA2 = 0.694, GAMMA3 = 0.801;
// Effective tide-generating potential at the surface divided by R [µm/s²]:
// Σ γ_n V_n/R with V_n/R = a_z,n / n, i.e. γ₂·K₂·P₂(c) + γ₃·K₃·P₃(c).
float tidePotBody(vec3 X, vec4 b, float k3) {
  float c = dot(X, b.xyz);
  return GAMMA2 * b.w * 0.5 * (3.0 * c * c - 1.0) + GAMMA3 * k3 * 0.5 * c * (5.0 * c * c - 3.0);
}

vec4 crWeights(float t) {
  float t2 = t * t, t3 = t2 * t;
  return vec4(-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2);
}
// Catmull-Rom bicubic at cell p + offset d, clamped to the surrounding 2×2 cells (no new extrema).
float catmullRom(sampler2D t, ivec2 p, vec2 d) {
  vec2 b = floor(d), w = d - b;
  ivec2 i0 = p + ivec2(b);
  vec4 wx = crWeights(w.x), wy = crWeights(w.y);
  float acc = 0.0, lo = 1e30, hi = -1e30;
  for (int m = -1; m <= 2; m++) {
    float row = 0.0;
    for (int n = -1; n <= 2; n++) {
      float v = f1(t, i0 + ivec2(n, m));
      row += v * wx[n + 1];
      if (m >= 0 && m <= 1 && n >= 0 && n <= 1) { lo = min(lo, v); hi = max(hi, v); }
    }
    acc += row * wy[m + 1];
  }
  return clamp(acc, lo, hi);
}

void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  if (texelFetch(u_mask, p, 0).r < 0.5) { o = vec4(0.0); return; }
  float lat = latOfRow(float(p.y)), lon = lonOfCol(float(p.x));
  float h = metric(lat);

  // 1) Semi-Lagrangian advection of total PV Q = q + f (midpoint departure point).
  //    Carrying f along the trajectory is the β-effect: Rossby waves, westward eddy drift.
  //    Positions are offsets from this cell, so arbitrarily small dt stays exact.
  vec4 V = texelFetch(u_vel, p, 0);
  vec2 dm = -0.5 * u_dt * vec2(V.x / h, V.y);
  vec4 Vm = bilinearAt(u_vel, p, dm);
  vec2 dd = -u_dt * vec2(Vm.x / metric(lat + dm.y * u_dRad), Vm.y);
  float dlat = dd.y * u_dRad;
  // f(φ+δ) − f(φ) = 2 F0 cos(φ + δ/2) sin(δ/2): no cancellation between two nearly equal sines.
  float q = catmullRom(u_q, p, dd) + 2.0 * u_F0 * cos(lat + 0.5 * dlat) * sin(0.5 * dlat);

  // 2) Linear bottom drag on ζ (halts the inverse cascade at large scale)
  //    + hyperviscosity −ν₄∇⁴q (removes the forward enstrophy cascade at the grid scale).
  ivec2 E = ivec2(1, 0), N = ivec2(0, 1);
  float bih = (f4(u_vel, p + E).w + f4(u_vel, p - E).w - 2.0 * V.w) / (h * h)
            + (f4(u_vel, p + N).w + f4(u_vel, p - N).w - 2.0 * V.w);
  q -= u_dt * (u_drag * V.z + u_nu4 * h * h * h * h * bih);

  // 3) VGTA forcing.
  vec3 X = unitVec(lat, lon);
  float az1 = vgtaBody(X, u_moon1, u_k3.z) + vgtaBody(X, u_sun1, u_k3.w);
  //  (a) Equilibrium-tide vortex stretching (mass divergence):
  //      F = (f/H) ∂η_eq/∂t,  η_eq = (R/g) Σ γ_n V_n/R   ⇒   Δq = Ks · sinφ · Δ(Σ γ_n V_n/R)
  //      Exact over the step, and it telescopes across sub-steps.
  float pot0 = tidePotBody(X, u_moon0, u_k3.x) + tidePotBody(X, u_sun0, u_k3.y);
  float pot1 = tidePotBody(X, u_moon1, u_k3.z) + tidePotBody(X, u_sun1, u_k3.w);
  q += u_stretchGain * u_Ks * sin(lat) * (pot1 - pot0);
  //  (b) Stochastic small-scale eddy generation (ring of 3-D plane waves restricted to
  //      the sphere: seamless, isotropic), amplified under the tidal bulges.
  float xi = 0.0;
  for (int m = 0; m < NMODES; m++) xi += cos(dot(u_modes[m].xyz, X) * u_Rcells + u_modes[m].w);
  xi *= sqrt(2.0 / float(NMODES));
  float tidalMod = 1.0 + u_modGain * max(az1, 0.0) / u_aRef;
  q += u_forceAmp * sqrt(u_dt) * tidalMod * xi;

  // 4) Vortex injected by a map click.
  if (u_kick.w != 0.0) {
    float d = acos(clamp(dot(X, unitVec(u_kick.x, u_kick.y)), -1.0, 1.0)) * u_Rcells;
    q += u_kick.w * exp(-d * d / (u_kick.z * u_kick.z));
  }
  o = vec4(q, 0.0, 0.0, 1.0);
}`;

  // Pass 3: red-black SOR sweep for the PV inversion (∇² − 1/L_d²) ψ = q, ψ = 0 on land.
  const SOR = COMMON + `
uniform sampler2D u_psi, u_q, u_mask;
uniform int u_parity;
uniform float u_omega;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  if (texelFetch(u_mask, p, 0).r < 0.5) { o = vec4(0.0); return; }
  float c = f1(u_psi, p);
  if (((p.x + p.y) & 1) != u_parity) { o = vec4(c, 0.0, 0.0, 1.0); return; }
  float lat = latOfRow(float(p.y)), ih2 = 1.0 / (metric(lat) * metric(lat));
  ivec2 E = ivec2(1, 0), N = ivec2(0, 1);
  float gs = (ih2 * (f1(u_psi, p + E) + f1(u_psi, p - E)) + f1(u_psi, p + N) + f1(u_psi, p - N) - f1(u_q, p))
           / (2.0 * ih2 + 2.0 + invLd2(lat));
  o = vec4(mix(c, gs, u_omega), 0.0, 0.0, 1.0);
}`;

  // Initial condition: random small-scale PV.
  const INIT = COMMON + `
#define NM ${N_INIT_MODES}
uniform sampler2D u_mask;
uniform vec4 u_modes[NM];
uniform float u_amp, u_Rcells;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  if (texelFetch(u_mask, p, 0).r < 0.5) { o = vec4(0.0); return; }
  vec3 X = unitVec(latOfRow(float(p.y)), lonOfCol(float(p.x)));
  float s = 0.0;
  for (int m = 0; m < NM; m++) s += cos(dot(u_modes[m].xyz, X) * u_Rcells + u_modes[m].w);
  o = vec4(u_amp * sqrt(2.0 / float(NM)) * s, 0.0, 0.0, 1.0);
}`;

  // Diagnostics: area-weighted block sums of KE, enstrophy and ocean area over 16×16 cells.
  const REDUCE = COMMON + `
uniform sampler2D u_vel, u_mask;
out vec4 o;
void main() {
  ivec2 b = ivec2(gl_FragCoord.xy) * 16;
  vec4 s = vec4(0.0);
  for (int j = 0; j < 16; j++) for (int i = 0; i < 16; i++) {
    ivec2 p = b + ivec2(i, j);
    if (p.x >= u_grid.x || p.y >= u_grid.y) continue;
    float w = step(0.5, texelFetch(u_mask, p, 0).r) * cos(latOfRow(float(p.y)));
    vec4 V = texelFetch(u_vel, p, 0);
    s += w * vec4(0.5 * dot(V.xy, V.xy), 0.5 * V.z * V.z, 1.0, 0.0);
  }
  o = s;
}`;

  // Screen pass: Web-Mercator pixel → lat/lon → simulation grid → colour.
  const DISPLAY = COMMON + `
uniform sampler2D u_q, u_psi, u_vel, u_mask;
uniform vec2 u_canvas, u_p0;          // canvas size [device px], world-pixel of top-left corner
uniform float u_pxRatio, u_world;     // device px per CSS px, world size [CSS px] at current zoom
uniform float u_range, u_alpha, u_kmPerCell;
uniform int u_field;
uniform bool u_diverging, u_contours;
uniform vec3 u_cmap[9];
uniform vec4 u_moon1, u_sun1, u_k3;
uniform float u_night;                // day/night shading strength (0 = off)
uniform bool u_light;                 // light basemap: darker contour colours
// High-resolution coastline mask in screen space (drawn by main.js in the basemap's projection):
// R = 1 ocean, 0 land. u_coastShift maps this frame's pixels into the mask drawn for an earlier view.
uniform sampler2D u_coast;
uniform bool u_coastOn;
uniform vec2 u_coastShift, u_coastSize;   // device px
// HF radar observations: two hourly frames as RGBA32F (u, v [m/s], valid, 0), blended in time.
uniform sampler2D u_hfr0, u_hfr1;
uniform bool u_hfrOn;
uniform float u_hfrMix;
uniform vec4 u_hfrGeo;                // lon0, lat0 (centre of cell 0,0), dlon, dlat [rad]
uniform ivec2 u_hfrSize;
uniform float u_glyphPx, u_arrowScale, u_time;   // lattice [CSS px], arrow px per m/s, seconds
out vec4 o;

// Composite a straight-alpha layer top over col.
vec4 over(vec4 top, vec4 col) {
  float A = top.a + col.a * (1.0 - top.a);
  return vec4((top.rgb * top.a + col.rgb * col.a * (1.0 - top.a)) / max(A, 1e-4), A);
}

// Gap-aware bilinear sample of one radar frame at radar-grid coordinates g:
// returns (u, v, w) where w is the summed weight of valid corners (0 = no data nearby).
vec3 hfrFrame(sampler2D t, vec2 g) {
  vec2 b = floor(g), w = g - b;
  ivec2 i = ivec2(b);
  vec3 acc = vec3(0.0);
  for (int k = 0; k < 4; k++) {
    ivec2 off = ivec2(k & 1, k >> 1);
    ivec2 c = i + off;
    if (any(lessThan(c, ivec2(0))) || any(greaterThanEqual(c, u_hfrSize))) continue;
    vec4 s = texelFetch(t, c, 0);
    float wt = (off.x == 1 ? w.x : 1.0 - w.x) * (off.y == 1 ? w.y : 1.0 - w.y) * s.z;
    acc += vec3(s.xy * wt, wt);
  }
  return acc.z > 0.0 ? vec3(acc.xy / acc.z, acc.z) : vec3(0.0);
}
vec3 hfrSample(float lat, float lon) {
  vec2 g = vec2((lon - u_hfrGeo.x) / u_hfrGeo.z, (lat - u_hfrGeo.y) / u_hfrGeo.w);
  vec3 a = hfrFrame(u_hfr0, g), b = hfrFrame(u_hfr1, g);
  if (a.z > 0.0 && b.z > 0.0) return vec3(mix(a.xy, b.xy, u_hfrMix), min(a.z, b.z));
  return a.z > 0.0 ? a : b;
}

// One arrow per screen-space lattice cell, sampled at the cell centre. Web Mercator is
// conformal, so the current's direction maps to the screen without distortion: east = +x,
// north = −y in CSS pixels. Returns straight-alpha colour.
vec4 hfrGlyph(vec2 css) {
  vec2 c = (floor(css / u_glyphPx) + 0.5) * u_glyphPx;
  vec2 wp = u_p0 + c;
  float lon = fract(wp.x / u_world) * 2.0 * PI - PI;
  float lat = atan(sinh(PI * (1.0 - 2.0 * wp.y / u_world)));
  vec3 s = hfrSample(lat, lon);
  float speed = length(s.xy);
  if (s.z < 0.5 || speed < 1e-3) return vec4(0.0);          // need data near the cell centre
  vec2 dir = vec2(s.x, -s.y) / speed;
  vec2 p = css - c;
  float len = clamp(speed * u_arrowScale, 3.0, 0.95 * u_glyphPx);
  float along = dot(p, dir) + 0.5 * len;                     // 0 at tail, len at tip
  float across = abs(dot(p, vec2(-dir.y, dir.x)));
  float head = min(0.38 * len, 7.0), halfW = 0.6 * head;
  float dShaft = (along >= -0.5 && along <= len - head) ? across - 0.7 : 1e3;
  float dHead = (along > len - head && along <= len) ? across - halfW * (len - along) / head : 1e3;
  float d = min(dShaft, dHead) * u_pxRatio;                 // signed distance in device px
  float cover = clamp(0.5 - d, 0.0, 1.0);
  float halo = clamp(0.5 - (d - 1.6 * u_pxRatio), 0.0, 1.0) * 0.8;
  // colour by speed (0 → 1 m/s: ice blue → amber); a pulse runs along the current
  vec3 colr = mix(vec3(0.72, 0.93, 1.0), vec3(1.0, 0.78, 0.28), clamp(speed, 0.0, 1.0));
  colr *= 0.78 + 0.22 * sin(along * 0.7 - u_time * 6.0);
  vec4 outline = vec4(0.02, 0.03, 0.05, halo);
  return over(vec4(colr, cover), outline);
}

// Cubic B-spline weights (positive, C² smooth).
vec4 bsplineWeights(float t) {
  float t2 = t * t, t3 = t2 * t, s = 1.0 - t;
  return vec4(s * s * s, 3.0 * t3 - 6.0 * t2 + 4.0, -3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0, t3) / 6.0;
}
// Smooth, coast-aware sample of a model field: B-spline weights × physics ocean mask, renormalised,
// so pixels near the coast take values only from ocean cells (land cells hold zeros).
vec4 fieldSample(sampler2D t, vec2 g) {
  vec2 b = floor(g), f = g - b;
  ivec2 i0 = ivec2(b);
  vec4 wx = bsplineWeights(f.x), wy = bsplineWeights(f.y);
  vec4 acc = vec4(0.0);
  float wsum = 0.0;
  for (int m = -1; m <= 2; m++) {
    for (int n = -1; n <= 2; n++) {
      ivec2 c = i0 + ivec2(n, m);
      float w = wx[n + 1] * wy[m + 1] * f4(u_mask, c).r;
      acc += w * f4(t, c);
      wsum += w;
    }
  }
  return wsum > 1e-4 ? acc / wsum : vec4(0.0);
}

// Ocean fraction at this pixel: the fine screen-space coastline when available, else the
// (coarse) physics mask.
float oceanAt(vec2 g) {
  if (u_coastOn) {
    vec2 cp = vec2(gl_FragCoord.x, u_canvas.y - gl_FragCoord.y) + u_coastShift;
    if (all(greaterThanEqual(cp, vec2(0.0))) && all(lessThan(cp, u_coastSize)))
      return texelFetch(u_coast, ivec2(cp), 0).r;
  }
  return smoothstep(0.35, 0.65, bilinear4(u_mask, g).r);
}

vec3 palette(float t) {
  float x = clamp(t, 0.0, 1.0) * 8.0;
  int i = min(int(x), 7);
  return mix(u_cmap[i], u_cmap[i + 1], x - float(i));
}

void main() {
  vec2 wp = u_p0 + vec2(gl_FragCoord.x, u_canvas.y - gl_FragCoord.y) / u_pxRatio;
  float lon = fract(wp.x / u_world) * 2.0 * PI - PI;
  float lat = atan(sinh(PI * (1.0 - 2.0 * wp.y / u_world)));
  vec3 X = unitVec(lat, lon);
  float az = vgtaBody(X, u_moon1, u_k3.z) + vgtaBody(X, u_sun1, u_k3.w);

  vec4 col = vec4(0.0);
  if (abs(lat) < u_latMax - 0.5 * u_dRad) {
    vec2 g = vec2((lon + PI) / u_dRad - 0.5, (lat + u_latMax) / u_dRad - 0.5);
    float ocean = oceanAt(g);
    float v;
    bool overLand = false;
    if (u_field == 0) v = fieldSample(u_vel, g).z;
    else if (u_field == 1) v = fieldSample(u_psi, g).r;
    else if (u_field == 2) v = length(fieldSample(u_vel, g).xy);
    else if (u_field == 3) v = fieldSample(u_q, g).r;
    else if (u_field == 4) { v = az; overLand = true; }
    else v = u_kmPerCell / sqrt(invLd2(lat));
    float t = u_diverging ? 0.5 + 0.5 * v / u_range : v / u_range;
    float strength = u_diverging ? abs(v) / u_range : v / u_range;
    float a = u_alpha * (0.35 + 0.65 * smoothstep(0.0, 0.5, strength));
    a *= overLand ? mix(0.55, 1.0, ocean) : ocean;
    col = vec4(palette(t), a);
  }

  // Day/night terminator: solar elevation from the sub-solar direction, with a twilight
  // ramp from the horizon (incl. refraction, +0.8°) down to nautical twilight (−12°).
  if (u_night > 0.0) {
    float sinElev = dot(X, u_sun1.xyz);
    float night = 1.0 - smoothstep(-0.2079, 0.0145, sinElev);
    col = over(vec4(0.0, 0.015, 0.05, u_night * night), col);
  }

  if (u_contours && abs(lat) < 1.5) {
    // VGTA isolines every 0.25 µm/s²; the zero line bounds the tidal bulges.
    float cl = az / 0.25;
    float d = abs(fract(cl + 0.5) - 0.5) / max(fwidth(cl), 1e-6);
    float line = (1.0 - smoothstep(0.6, 1.6, d)) * 0.4;
    float d0 = abs(az) / max(fwidth(az), 1e-6);
    line = max(line, (1.0 - smoothstep(0.8, 2.2, d0)) * 0.75);
    vec3 lc = u_light ? (az > 0.0 ? vec3(0.62, 0.30, 0.0) : vec3(0.05, 0.25, 0.60))
                      : (az > 0.0 ? vec3(1.0, 0.92, 0.72) : vec3(0.62, 0.82, 1.0));
    col = over(vec4(lc, line), col);
  }

  // Observed HF radar currents on top of the simulated field.
  if (u_hfrOn) {
    vec2 css = vec2(gl_FragCoord.x, u_canvas.y - gl_FragCoord.y) / u_pxRatio;
    col = over(hfrGlyph(css), col);
  }
  o = col;
}`;

  return { N_MODES, N_INIT_MODES, VERT, VEL, ADVECT, SOR, INIT, REDUCE, DISPLAY };
})();
