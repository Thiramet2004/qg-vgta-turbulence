/* celestial.js — low-precision Sun/Moon ephemeris and Vertical Gravitational Tidal
 * Acceleration (VGTA), the browser-side stand-in for armilla's astrotide forcing, plus
 * planetary sub-points and the ecliptic for the map overlays (display only — the planets
 * do not enter the tidal forcing).
 *
 * VGTA here is the radial ("up") component of the lunisolar tide-generating acceleration:
 *   a_z = n·GM·R^(n-1)/d^(n+1) · P_n(cos ψ), summed over degree n = 2, 3
 * where ψ is the angular distance from the sub-body point. Units returned: µm/s².
 */
'use strict';
window.Celestial = (function () {
  const DEG = Math.PI / 180;
  const GM_MOON = 4.902800e12;       // m³/s²
  const GM_SUN = 1.32712440018e20;   // m³/s²
  const R_EARTH = 6.371e6;           // m, mean radius used in the tidal expansion
  const RE_EQ = 6378.14e3;           // m, equatorial radius (defines lunar parallax)
  const AU = 1.495978707e11;         // m

  const sinD = d => Math.sin(d * DEG);
  const cosD = d => Math.cos(d * DEG);
  const norm360 = d => ((d % 360) + 360) % 360;

  function julianDay(ms) { return ms / 86400000 + 2440587.5; }

  // Astronomical Almanac low-precision lunar theory (~0.3° longitude, ~0.2 % distance).
  function moonEcliptic(T) {
    const lam = 218.32 + 481267.881 * T
      + 6.29 * sinD(134.9 + 477198.85 * T) - 1.27 * sinD(259.2 - 413335.38 * T)
      + 0.66 * sinD(235.7 + 890534.23 * T) + 0.21 * sinD(269.9 + 954397.70 * T)
      - 0.19 * sinD(357.5 + 35999.05 * T) - 0.11 * sinD(186.6 + 966404.05 * T);
    const bet = 5.13 * sinD(93.3 + 483202.03 * T) + 0.28 * sinD(228.2 + 960400.87 * T)
      - 0.28 * sinD(318.3 + 6003.18 * T) - 0.17 * sinD(217.6 - 407332.20 * T);
    const par = 0.9508 + 0.0518 * cosD(134.9 + 477198.85 * T) + 0.0095 * cosD(259.2 - 413335.38 * T)
      + 0.0078 * cosD(235.7 + 890534.23 * T) + 0.0028 * cosD(269.9 + 954397.70 * T);
    return { lam: norm360(lam), bet, dist: RE_EQ / sinD(par) };
  }

  // Low-precision solar theory (~0.01°).
  function sunEcliptic(n) {
    const g = 357.528 + 0.9856003 * n;
    const lam = 280.460 + 0.9856474 * n + 1.915 * sinD(g) + 0.020 * sinD(2 * g);
    const r = 1.00014 - 0.01671 * cosD(g) - 0.00014 * cosD(2 * g);
    return { lam: norm360(lam), bet: 0, dist: r * AU };
  }

  // Earth orientation: days since J2000, centuries, obliquity and Greenwich sidereal angle [deg].
  function earthFrame(ms) {
    const n = julianDay(ms) - 2451545.0;
    return { n, T: n / 36525, eps: 23.439 - 0.0000004 * n, gmst: norm360(280.46061837 + 360.98564736629 * n) };
  }

  // Ecliptic Cartesian vector → equatorial (tilt by ε) → Earth-fixed unit vector (rotate by GMST).
  function eclVecToECEF(v, eps, gmst) {
    const ce = cosD(eps), se = sinD(eps);
    const x = v[0], y = v[1] * ce - v[2] * se, z = v[1] * se + v[2] * ce;
    const r = Math.hypot(x, y, z), cg = cosD(gmst), sg = sinD(gmst);
    return [(x * cg + y * sg) / r, (-x * sg + y * cg) / r, z / r];
  }

  function eclipticToECEF(lam, bet, eps, gmst) {
    const cb = cosD(bet);
    return eclVecToECEF([cb * cosD(lam), cb * sinD(lam), sinD(bet)], eps, gmst);
  }

  const subPoint = u => [Math.asin(u[2]) / DEG, Math.atan2(u[1], u[0]) / DEG];

  function body(u, gm, d) {
    return {
      u,                                                   // unit vector to the body (ECEF)
      k2: gm * R_EARTH / (d * d * d) * 1e6,                // degree-2 coefficient [µm/s²]
      k3: gm * R_EARTH * R_EARTH / (d * d * d * d) * 1e6,  // degree-3 coefficient [µm/s²]
      lat: Math.asin(u[2]) / DEG,                          // sub-body point
      lon: Math.atan2(u[1], u[0]) / DEG,
      dist: d,
    };
  }

  // Vertical tidal acceleration [µm/s²] at Earth-fixed unit vector X from one body.
  // Mirrors vgtaBody() in shaders.js.
  function vgta(X, b) {
    const c = X[0] * b.u[0] + X[1] * b.u[1] + X[2] * b.u[2];
    return b.k2 * (3 * c * c - 1) + b.k3 * 1.5 * c * (5 * c * c - 3);
  }

  function tidalBodies(ms) {
    const { n, T, eps, gmst } = earthFrame(ms);
    const m = moonEcliptic(T), s = sunEcliptic(n);
    const moon = body(eclipticToECEF(m.lam, m.bet, eps, gmst), GM_MOON, m.dist);
    const sun = body(eclipticToECEF(s.lam, 0, eps, gmst), GM_SUN, s.dist);
    const elong = norm360(m.lam - s.lam);
    // Degree-2 bulges add in phase at syzygy (spring) and in quadrature at quarter moon (neap).
    const spring = Math.hypot(moon.k2 + sun.k2 * cosD(2 * elong), sun.k2 * sinD(2 * elong)) / (moon.k2 + sun.k2);
    return { moon, sun, elong, illum: (1 - cosD(elong)) / 2, spring };
  }

  // ── Planets (display only) ──────────────────────────────────────────────
  // JPL "Approximate Positions of the Planets" (E. M. Standish), Table 1, valid 1800–2050
  // (~1′ for the inner planets, a few ′ for the outer ones).
  // Each row: [a (AU), e, I, L, ϖ, Ω] at J2000 (degrees), then rates per Julian century.
  const ELEMENTS = {
    Mercury: [[0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593],
              [0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081]],
    Venus:   [[0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255],
              [0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418]],
    EMBary:  [[1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0],
              [0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0]],
    Mars:    [[1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891],
              [0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343]],
    Jupiter: [[5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909],
              [-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106]],
    Saturn:  [[9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448],
              [-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794]],
    Uranus:  [[19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503],
              [-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589]],
    Neptune: [[30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574],
              [0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.01262724]],
  };
  const PLANETS = ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'];

  // Heliocentric ecliptic (J2000) position [AU] from osculating Keplerian elements.
  function heliocentric(name, T) {
    const [el, rate] = ELEMENTS[name];
    const [a, e, I, L, wbar, node] = el.map((v, k) => v + rate[k] * T);
    const M = (norm360(L - wbar + 180) - 180) * DEG;     // mean anomaly in (−π, π]
    let E = M + e * Math.sin(M);
    for (let k = 0; k < 6; k++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
    const xp = a * (Math.cos(E) - e), yp = a * Math.sqrt(1 - e * e) * Math.sin(E);
    const w = (wbar - node) * DEG, O = node * DEG, i = I * DEG;
    const cw = Math.cos(w), sw = Math.sin(w), cO = Math.cos(O), sO = Math.sin(O), ci = Math.cos(i), si = Math.sin(i);
    return [
      (cw * cO - sw * sO * ci) * xp + (-sw * cO - cw * sO * ci) * yp,
      (cw * sO + sw * cO * ci) * xp + (-sw * sO + cw * cO * ci) * yp,
      sw * si * xp + cw * si * yp,
    ];
  }

  const PRECESSION_DEG_PER_CENTURY = 1.396971;
  const rotZ = (v, deg) => [v[0] * cosD(deg) - v[1] * sinD(deg), v[0] * sinD(deg) + v[1] * cosD(deg), v[2]];

  // Geocentric planets: sub-point (where the planet is overhead), distance and solar elongation.
  function planets(ms) {
    const { T, eps, gmst } = earthFrame(ms);
    const earth = heliocentric('EMBary', T);
    const sunLon = Math.atan2(-earth[1], -earth[0]) + PRECESSION_DEG_PER_CENTURY * T * DEG;
    return PLANETS.map(name => {
      const p = heliocentric(name, T);
      // J2000 → equinox of date (general precession in longitude) to match the Sun/Moon theories.
      const g = rotZ([p[0] - earth[0], p[1] - earth[1], p[2] - earth[2]], PRECESSION_DEG_PER_CENTURY * T);
      const [lat, lon] = subPoint(eclVecToECEF(g, eps, gmst));
      const eclLon = Math.atan2(g[1], g[0]);
      const elong = Math.acos(Math.max(-1, Math.min(1, Math.cos(Math.asin(g[2] / Math.hypot(...g))) * Math.cos(eclLon - sunLon)))) / DEG;
      return { name, lat, lon, distAU: Math.hypot(...g), eclLon: norm360(eclLon / DEG), elong };
    });
  }

  // Sub-points of the ecliptic great circle (the Sun's apparent path) every stepDeg of longitude.
  function eclipticPath(ms, stepDeg) {
    const { eps, gmst } = earthFrame(ms);
    const pts = [];
    for (let lam = 0; lam <= 360; lam += stepDeg) pts.push(subPoint(eclipticToECEF(lam, 0, eps, gmst)));
    return pts;
  }

  return { tidalBodies, vgta, julianDay, planets, eclipticPath, PLANETS };
})();
