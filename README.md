# An Interactive WebGL Framework for Real-Time Simulation of Tidally Forced Global Quasi-Geostrophic Turbulence

A live, real-time WebGL2 simulation of ocean mesoscale turbulence and the inverse energy cascade
on a latitude-dependent β-plane. It is forced by the lunisolar **Vertical Gravitational Tidal
Acceleration (VGTA)**, which is computed in the browser from Sun and Moon ephemerides. Everything
runs client-side, so the site works on GitHub Pages as-is.

```
index.html     page structure, CDN imports (Leaflet 1.9.4, topojson-client 3.1.0)
style.css      full-screen map + HUD / control-panel styling
celestial.js   Sun/Moon ephemeris → Earth-fixed directions, distances, VGTA coefficients
shaders.js     GLSL ES 3.00: velocity, advection+forcing, SOR inversion, diagnostics, display
main.js        Leaflet map, WebGL2 context, ping-pong FBOs, unified clock + sub-stepping, VGTA uniforms, UI
```

## Run / deploy

* Locally: `python3 -m http.server` in this folder, then open <http://localhost:8000>.
  (Opening `index.html` directly also works, because every script is a classic script.)
* GitHub Pages: push the folder to a repository, then go to **Settings → Pages → Deploy from branch**.
* **Basemap:** CARTO Dark Matter now requires a free API key (<https://carto.com/basemaps/apikey>).
  Pass it as `?cartokey=YOUR_KEY`, or hard-code it at `CARTO_KEY` in `main.js`. Without a key,
  the page uses Esri's keyless *World Dark Gray* canvas, so you never see watermarked tiles.
  A key in a public repository is visible to everyone, so restrict it to your Pages domain in the
  CARTO dashboard.
* **Theme and colour maps.** The default is a light basemap (Esri *World Light Gray*) with
  [cmocean](https://matplotlib.org/cmocean/) colour maps (Thyng et al. 2016, *Oceanography* 29(3)),
  one per quantity: `curl` for relative and potential vorticity, `balance` for streamfunction,
  `diff` for VGTA, `speed` for current speed and `deep` for the Rossby radius. They are
  perceptually uniform and colour-blind safe. `?basemap=dark` restores the dark theme and its
  original colour maps.
* URL options: `?nx=512|1024|1536` sets the grid resolution (default 1024 × 444, ≈ 39 km cells).
  `?spinup=60` integrates 60 model days before the first frame, so the page opens on an
  equilibrated field.

## Simulation speed and the model clock

A single clock, `state.simTime` (ms UTC, fractional and never rounded), drives the fluid, the
Sun/Moon ephemeris, the VGTA contours and the on-screen datetime.

* **Speed.** Each frame advances the clock by `timeScale × BASE_HOURS_PER_SEC × (real seconds since
  the last frame)`. **Simulation speed** spans 0.01× (14.4 model min/s) to 5× (5 model days/s) on a
  log scale, with presets. `BASE_HOURS_PER_SEC = 24` defines 1×.
* **Sub-stepping.** That interval is split into equal fluid steps of at most `DT_MAX = 1 h`:
  * At 5×: two 1-hour steps per frame at 60 FPS.
  * At 0.01×: one ~14-second step per frame.
  * At most `MAX_STEPS_PER_FRAME` steps run per frame. If the GPU can't keep up, the model runs
    slower than requested instead of freezing; the HUD's *Measured* line shows the real rate.
* **Small-dt accuracy.** Very small steps stay exact:
  * The semi-Lagrangian advection samples at *cell + offset*, so 10⁻⁴-cell displacements aren't
    rounded away.
  * The β-term is evaluated as `2F₀ cos(φ+δ/2) sin(δ/2)` rather than a difference of two sines.
  * The tidal stretching increment `Δa_z` telescopes exactly across sub-steps.
* **Every frame.** After the sub-steps, the fields, VGTA contours, Sun/Moon markers and the
  `YYYY-MM-DD HH:mm:ss` clock are redrawn from the same `simTime`, so they glide together at any speed.

## Astronomical dashboard (display only)

These overlays never enter the physics: the fluid is forced by the Sun + Moon VGTA alone.

* **Time panel.**
  * A native `datetime-local` picker sits transparently over a locale-independent
    `YYYY-MM-DD HH:mm` display, with ticking seconds and the Julian Day (5 decimals). It follows
    the model clock every frame while playing.
  * Clicking it pauses the clock and opens the browser's calendar. Each date/time you pick jumps
    the clock, and every celestial position, there at once. Closing the picker (click away, Enter
    or Escape) resumes playback if it was playing before. Set `RESUME_AFTER_PICK = false` in
    `main.js` to stay paused on the chosen moment instead.
  * Values are read and written as UTC. The range is 1800–2050, the validity window of the
    planetary elements.
  * The fluid state carries over across a jump, because the turbulence is statistically
    stationary. The Sun/Moon are re-anchored so the tidal-stretching increment has no spurious spike.
* **Day/night terminator.** It is drawn per pixel in the display shader from the solar elevation
  `dot(X, ŝ)`, with a twilight ramp from +0.8° (refraction) down to −12°. It moves with the clock
  at any speed.
* **Planets.** `Celestial.planets(ms)` computes them from JPL Keplerian elements (Standish, valid
  1800–2050). Positions are geocentric, precessed to the equinox of date, and rotated to Earth-fixed
  coordinates with GMST. Each marker sits at the planet's sub-point, where it is overhead.
  * Checked against almanac events: Venus/Mercury greatest elongations match to 0.03°, and
    oppositions of Mars–Neptune to within 0.06° in longitude.
  * Colours: Mercury green, Venus cyan, Mars magenta, Jupiter orange, Saturn purple, Uranus light
    green, Neptune blue.
* **Ecliptic.** The great circle of the Sun's apparent path, recomputed every frame as the Earth
  rotates under it. The Sun lies on it, and the planets lie within a few degrees.
* **Moon phase.** A canvas disc is drawn from the Sun–Moon elongation, alongside illumination,
  age and distance.

## HF radar observations (Gulf of Thailand)

GISTDA HF radar total vectors (upper Gulf / Bight of Bangkok, 2 km grid) are processed in bulk,
shown in the viewer as animated arrows synced to the model clock, and analysed spectrally.
Requirements: Python ≥ 3.9 with numpy, scipy and matplotlib.

### 1. Bulk preprocessing

```bash
python tools/process_all_hfr.py /path/to/dumps --jobs 6
```

* **What it runs.** `tools/hfr_preprocess.py` is called on every `YYYYMM_current.sql`, in parallel,
  with the GISTDA settings:
  * columns `time_stamp` (already UTC), `latitude`, `longtitude` (sic), `ucomp`, `vcomp` (cm/s);
  * `--where sitename=GULF`;
  * `--native-xy xdistance,ydistance`, so each vector sits on its exact 2 km node with no
    interpolation smoothing.
* **Time windows.** Each dump holds one *local* (UTC+7) month, which is converted to exact UTC
  bounds.
* **Outputs.**
  * `data/hfr_YYYYMM.js`: viewer bundles, int16 mm/s.
  * `data/hfr_manifest.js`: index of the bundles.
  * `analysis/npz/hfr_YYYYMM.npz`: float32 fields for analysis.
* **Run time.** All 18 months (Feb 2025 – Jul 2026, 2.55 GB of SQL) take about 1 minute.
* **Single files.** `hfr_preprocess.py` can still be used on one file. It reads any MySQL or
  PostgreSQL dump, or a live database via `--db`; see `--help`.

### 2. Viewer

The viewer reads `data/hfr_manifest.js`. It loads a month's bundle only when the model clock
enters that month, and keeps at most 3 months decoded. **Go to radar period & area** jumps to the
nearest month with data and zooms to its footprint. Extra bundles can be added with `HFR_SOURCES`
in `main.js` or `?hfr=…`.

### 3. Spectral analysis (`tools/hfr_spectra.py`)

```bash
python tools/hfr_spectra.py analysis/npz/hfr_202603.npz --segment 26 --segment-meridional 26 --plot s.png
```

* **Inertial filtering.** A Lanczos low-pass with a 72 h half-power cutoff and ±120 h window.
  Gain is 0.000 at 12.42 h and 24 h, and 0.016 at the 53.3 h inertial period, so 99.97 % of
  near-inertial energy is removed (Godin leaves it in at 13°N); periods beyond 96 h pass at
  ≥ 0.9. Five days are lost at each end of a month.
* **Spectra.** Gap-tolerant segments: interior gaps of ≤ 2 nodes are linearly filled. Segments
  are de-meaned and Tukey(0.5)-tapered. Linear detrending with a Hann taper was found, on synthetic
  data, to bias the lowest bins unequally for C_L and C_T.
* **Helmholtz decomposition** of the 1-D spectra into rotational D_ψ and divergent D_φ parts
  (Bühler, Callies & Ferrari 2014). It assumes isotropy; **a rotational share above 1 flags
  anisotropic flow**, where the split is not valid.
* **Scale-masked fits.** Power laws are fitted only for λ ≥ 20 km (`--min-wavelength-km`), because
  shorter scales are shaped by the least-squares combination of radials into totals.
* **Footprint limit.** The radar footprint changed in Nov 2025 and now spans only 54–64 km. With
  26-node (52 km) segments, the longest that fit every month, **only two bins (52 and 26 km)
  exceed 20 km**. Slopes are therefore meaningful as multi-month averages.

### 4. Validation and report

```bash
python tools/hfr_validate.py analysis/npz/hfr_202603.npz --segment 26 --realisations 4 \
    --out analysis/report/validation.json        # known-answer test of the whole method
python tools/hfr_report.py                        # → analysis/report/hfr_report.md + figures + CSV
```

* **Validation.** `hfr_validate.py` builds synthetic months with a prescribed k⁻⁵/³ or k⁻³
  spectrum and rotational share, adds tide, inertial oscillation and noise, imposes the real gap
  mask, and runs the identical analysis. At 26-node segments the single-month slope bias is
  +0.06 (k⁻⁵/³) / −0.30 (k⁻³) with scatter ±0.56 / ±0.70, and the rotational-share RMS error
  is 0.19.
* **Report.** `hfr_report.py` analyses every month and builds wet (SW monsoon, May–Oct) vs dry
  (Nov–Apr) seasonal mean spectra with bootstrap 95 % intervals over months. It also compares
  seasons within each radar configuration, because the footprint change is confounded with
  season.
* **Notes.** The curated interpretation lives in `analysis/report/notes.md` and is included in the
  regenerated report.

## NOAA NCEI radials and the rotating-turbulence (QNSE) test

* **In the browser.** HF radar panel → **Source**: NOAA NCEI live (central California 6 km,
  Monterey Bay 2 km, SF Bay 1 km). The page fetches the CODAR radial files for the model clock's
  hour directly from NCEI (CORS is open) and combines them into totals in JavaScript.
  * Combination: least squares, u·sinH + v·cosH = VELO.
  * QC: VFLG = 0, ≥ 3 radials from ≥ 2 sites, crossing angle ≥ 30°.
  * Any month in the archive works; set the datetime picker. The first hour takes ~20 s while
    each site's directory listing loads.
* **Bulk, for analysis.** `tools/hfr_ncei.py <month URL> --sites … --res-km … --search-km …`
  streams a whole month into memory, with no raw files stored, and writes the usual
  `data/*.js` and `analysis/npz/*.npz`. December 2020 covers all 15 central-California sites as
  three networks.
* **Per-network report.** `tools/hfr_ncei_report.py` runs the same spectral method and a
  known-answer validation on each network's own gap mask.
* **QNSE test.** `tools/hfr_rotation_test.py` tests QNSE's 1-D spectra
  E_L = (18/55)C_KΠ^(2/3)k^(−5/3) + 0.0926 f²k⁻³ and E_T = (24/55)C_KΠ^(2/3)k^(−5/3) + 0.24 f²k⁻³
  (Galperin et al., IOVWST 2022). It compares 13°N (GISTDA) with 36°N (NOAA):
  * the absolute Coriolis-range amplitudes E k³/f²;
  * the energy flux Π from the third-order structure function, with a null-calibrated error;
  * the latitude below which the Coriolis range falls outside the radar's resolved scales.
* **Cross-latitude test.** `tools/hfr_rtv.py 2020/202012` streams the official HFRNet total-vector
  products (`rtv/`, NetCDF, UWLS, 10 km search radius; the same class of product as CODAR `.tuv`
  totals) for every US region, from Puerto Rico to Alaska, and cuts them into well-observed areas.
  `tools/hfr_latitude_test.py` then tests QNSE across all areas plus the GISTDA Gulf of Thailand:
  * Coriolis-range amplitudes and slopes in each area;
  * the exponent p in E(50 km) ∝ f^p (QNSE predicts p = 2), with robustness cases.
  * Run monthly with `hfr_rtv.py 2020/2020MM`; areas are grouped into sites across months (≥ 3 months)
    and their spectra averaged.
  * 2020 result (15 sites, 13–45°N): p = +0.22 (95 % CI −1.20 … +1.31) with tides removed, and
    −0.35 (−1.73 … +0.85) subtidal; p = 2 lies outside the interval in every robustness case.
* **Results page.** All results are on **`analysis.html`**, linked from the viewer.

## Model

Equivalent-barotropic (1½-layer) quasi-geostrophic PV equation:

```
∂q/∂t + J(ψ, q + f) = F_VGTA + F_eddy − r ∇²ψ − ν₄ ∇⁴q
q = ∇²ψ − ψ / L_d(φ)²,        u = −∂ψ/∂y,  v = ∂ψ/∂x
```

* **Coriolis and β.** The full f = 2Ω sin φ is used, not a linear β-plane fit. It is carried
  along trajectories: total PV `q + f` is advected, so `f(φ_departure) − f(φ)` is the β-term.
  This gives westward Rossby-wave propagation, north-west drift of cyclones, and zonal jets at
  large scales.
* **Latitude-dependent Rossby radius.** `L_d = c / √(f² + 2βc)`. This is `c/|f|` at mid-latitudes
  and blends smoothly into the equatorial radius `√(c/2β)`, a smooth version of the piecewise
  form in Chelton et al. (1998). At true scale (c = 2.5 m/s) that is 34 km at 30° and 234 km at
  the equator. It equals ≈ 315 km at 30°, ≈ 740 km at
  the equator, and ≈ 195 km at 60° (with the default ×10 exaggeration of c = 2.5 m/s). The
  exaggeration lets a 39 km grid resolve the eddies. Set **Rossby radius** to ×1 for true
  scales, then raise `nx`.
* **Inverse cascade.** Energy is injected at the forcing wavelength (default 200 km), below L_d.
  Like-signed vortices merge, and energy moves to larger scales until bottom drag and the
  Rhines/L_d scales arrest it. The HUD's **λ_E = 2π√(E/Z)** sparkline tracks the energy-containing
  scale; E is kinetic energy only, with the available-potential-energy term ψ²/2L_d² excluded.
  Press **Decay run** (unforced, starts from small-scale noise) to watch the merger sequence on
  its own.
* **Numerics.**
  * Grid: equirectangular, periodic in longitude, walls at ±78°, with metric factor cos φ.
  * Coastlines: rasterised Natural Earth 110m land with ψ = 0 on land. Lakes and basins under
    200 cells are removed.
  * Time step: 1 hour, which resolves the semidiurnal tide.
  * Advection: semi-Lagrangian with a midpoint departure point and limited Catmull-Rom
    interpolation (unconditionally stable).
  * PV inversion: red-black SOR (ω = 1.7), warm-started from the previous ψ.
  * Dissipation: scale-selective hyperviscosity plus linear (Rayleigh) drag on ζ. A 1½-layer
    model has no bottom, so the drag is a stand-in for bottom friction and eddy energy loss.
  * Passes per model step: 1 velocity + 1 advection + 2 × *sweeps* SOR passes. SOR runs a fixed
    number of sweeps per *step*, so the PV inversion is better converged at slow speeds (many
    small steps) than at 5×. Raise the sweeps if you compare runs at different speeds.
* **Assumptions and limits**, to state in any write-up:
  * **QG validity.** QG requires a small Rossby number and f ≠ 0, so it is not valid within
    about ±5° of the equator. The equatorial L_d blend keeps the model well behaved there; it
    does not make equatorial dynamics correct.
  * **Geometry.** Spherical metric (cos φ) is applied to zonal derivatives. The curvature term
    −tan φ ∂ψ/∂y in the Laplacian is neglected, which is under ~2% for mesoscale eddies even at
    78°.
  * **Coasts.** ψ = 0 on every coast and island, so no net transport passes between landmasses.
    There is no Antarctic Circumpolar Current through Drake Passage and no Indonesian
    Throughflow. The model represents the mesoscale eddy field, not the basin-scale circulation.
  * **Conservation.** Semi-Lagrangian advection is stable but not exactly energy- or
    enstrophy-conserving; its interpolation adds a small numerical dissipation.

## How VGTA enters the QG equation

`celestial.js` computes the Sun and Moon positions for the model clock. It uses the Astronomical
Almanac low-precision lunar theory and the standard low-precision solar formula, then rotates them
into Earth-fixed coordinates with GMST. The shader evaluates the radial tide-generating
acceleration at every grid cell:

```
a_z = Σ_bodies  K₂ (3cos²ψ − 1) + K₃ (3/2) cosψ (5cos²ψ − 3),
K₂ = GM R / d³,  K₃ = GM R² / d⁴        (µm/s²; ≈ 1.1 Moon + 0.5 Sun at the sub-body point)
```

This is the "up" component that armilla's `astrotide` module produces. armilla's forcing code is
JavaScript (`astrotide/astrotide.js`), so you can swap in its function directly if you want
bit-identical values.

**(a) Mass divergence / vortex stretching (physically derived).** VGTA is *not* treated as a wind
stress curl. The tidal force is the gradient of the tide-generating potential, `∇V`, so it cannot
spin up a fluid column directly. What it does change is the equilibrium sea-surface height.

* **Potential from acceleration.** `V = Σ V_n` with `V_n ∝ rⁿ`, so `a_z,n = ∂V_n/∂r = n V_n / R`.
  Each degree therefore contributes `V_n = R a_z,n / n` (R/2 for degree 2, R/3 for degree 3).
* **Love numbers.** The solid Earth deforms with the tide, raising the seafloor and adding
  self-gravitation. Only `γ_n = 1 + k_n − h_n` of the equilibrium tide changes water-column
  thickness: γ₂ ≈ 0.69 and γ₃ ≈ 0.80. So `η_eq = Σ γ_n V_n / g`, giving ≈ 0.25 m for the Moon,
  the textbook M₂ equilibrium amplitude.
* **Stretching source.** Writing the shallow-water equations with the tidal potential and
  taking the QG limit, `ζ_t = −f ∇·u` and `H ∇·u = −∂η_eq/∂t` give the PV source

  ```
  F_VGTA = (f / H) ∂η_eq/∂t = (f / gH) Σ γ_n ∂V_n/∂t
  ```

* **Why the full depth H.** H is the full depth, 4000 m, not the upper-layer thickness. The
  barotropic tide's divergence is depth-uniform, so every layer, including the 1½-layer model's
  active layer, is stretched at the rate `(1/H)∂η/∂t`.
* **Integration.** The shader adds `Δq = Ks · sinφ · Δ(Σ γ_n V_n / R)` each step, using the
  potential at the step's start and end. That is exact over the step and telescopes across
  sub-steps.
* **Effect.** Columns are squeezed and stretched as the bulges sweep past, creating PV of
  opposite sign in the two hemispheres. Because ψ = 0 at coasts, the large-scale tidal PV is
  expressed as coastal boundary vorticity.
* **Magnitude.** At ×1 the term is ≈ 6 × 10⁻¹³ s⁻² at 30°N. Over half an M₂ cycle that builds up
  ≈ 10⁻⁸ s⁻¹ of PV, about 0.2% of typical eddy vorticity. The **Tidal stretching gain** therefore
  defaults to ×100 for visibility; set it to ×1 for the physical magnitude.
* **Equilibrium vs dynamic tide.** Real ocean tides are *dynamic* (amphidromes, basin resonance),
  and their ∂η/∂t differs from the equilibrium tide by factors of order 1–3 and in phase.
  Prescribing η_eq is the leading-order, analytically tractable choice. A tide model
  (e.g. TPXO/FES elevations) could replace it through the same source term.

**(b) Bulge-modulated eddy generation (parameterisation).** Tidal flow over topography generates
internal tides and mixing, which are an energy source for submesoscale and mesoscale variability.
The stochastic small-scale forcing is therefore multiplied by `1 + G_m · max(a_z, 0) / 1 µm/s²`.
Eddy generation is enhanced inside the two moving tidal bulges and follows the spring–neap cycle.
This is a modelling choice, not a first-principles result. Set **Bulge eddy modulation** to 0
to switch it off.

The VGTA isolines (0.25 µm/s² spacing; the bold line marks a_z = 0) and the ☾/☀ sub-body markers
show the forcing pattern moving westward at about 15° per hour.

## Tuning the VGTA strength

| What | Where | Default |
|---|---|---|
| Stretching gain G_s (1 = physical) | slider **Tidal stretching gain**, `state.stretchGain` in `main.js` | 100 |
| Bulge modulation G_m | slider **Bulge eddy modulation**, `state.modGain` | 2 |
| Reference VGTA for modulation | `A_REF` in `main.js` | 1.0 µm/s² |
| Ocean depth in the stretching coefficient | `H_OCEAN` in `main.js` (`KS` is derived from it) | 4000 m |
| Shape of the forcing term itself | section `// 3) VGTA forcing` in `ADVECT`, `shaders.js` | — |

## Using pre-computed armilla fields instead (Approach B)

The whole celestial computation reduces to one shader function, `vgtaBody()`, plus the uniforms
`u_moon0/1`, `u_sun0/1` and `u_k3`. To drive the model from Python/armilla output:

1. Export a_z on the equirectangular grid as a float texture or an RG-encoded PNG.
2. Upload two time slices as `sampler2D`s and replace `az0` / `az1` in `ADVECT` (and `az` in
   `DISPLAY`) with texture reads.

Steps (a) and (b) need no other changes.
