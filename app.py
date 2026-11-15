import os
import tempfile
import time
from pathlib import Path
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymannkendall as mk
import requests
import streamlit as st
import streamlit.components.v1 as components
from packaging.version import Version
from scipy import stats
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Climate Signal | NASA Earth Trend Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

NEW_WIDTH_API = Version(st.__version__) >= Version("1.51.0")
STRETCH = {"width": "stretch"} if NEW_WIDTH_API else {"use_container_width": True}

APP_DIR = Path(__file__).resolve().parent
GIOVANNI_CSV = APP_DIR / "giovanni_svalbard_validation.csv"

SPIN_DEG_PER_SEC = 18.0

GLOBE_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body { margin: 0; padding: 0; background: #050914; overflow: hidden; }
#globe { width: 100%; height: 390px; color: #cbd5e1; font-family: sans-serif; font-size: 14px; user-select: none; -webkit-user-select: none; }
</style>
</head>
<body>
<div id="globe"></div>
<script>
(function () {
  var HEIGHT = 390;
  var DEFAULT_SPIN = 18;
  var PLOTLY_PRIMARY = "https://cdn.plot.ly/plotly-2.35.2.min.js";
  var PLOTLY_BACKUP = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js";
  var CONFIG = { displayModeBar: false, scrollZoom: false, doubleClick: false, responsive: true };

  var el = document.getElementById("globe");
  var meshLat = [];
  var meshLon = [];
  for (var la = -84; la <= 84; la += 6) {
    for (var lo = -180; lo < 180; lo += 6) {
      meshLat.push(la);
      meshLon.push(lo);
    }
  }

  var rot = { lon: 0, lat: 20 };
  var focusId = null;
  var spinning = true;
  var spinSpeed = DEFAULT_SPIN;
  var hovering = false;
  var dragging = false;
  var wasDragged = false;
  var downX = 0;
  var downY = 0;
  var busy = false;
  var created = false;
  var ready = false;
  var pending = null;
  var latestArgs = null;
  var plotlyState = "loading";
  var lastTs = 0;
  var pauseUntil = 0;
  var clickCounter = 0;

  function post(type, data) {
    var msg = { isStreamlitMessage: true, type: type };
    for (var k in data) {
      if (Object.prototype.hasOwnProperty.call(data, k)) {
        msg[k] = data[k];
      }
    }
    window.parent.postMessage(msg, "*");
  }

  function wrapLon(v) {
    return ((((v + 180) % 360) + 360) % 360) - 180;
  }

  function syncRot() {
    try {
      var r = el._fullLayout.geo.projection.rotation;
      if (r && typeof r.lon === "number" && typeof r.lat === "number") {
        rot.lon = r.lon;
        rot.lat = r.lat;
      }
    } catch (err) {}
  }

  function buildData(a) {
    var data = [{
      type: "scattergeo",
      mode: "markers",
      lat: meshLat,
      lon: meshLon,
      marker: { size: 26, opacity: 0.01, color: "#ffffff" },
      hovertemplate: "Click to select: %{lat:.0f}\u00b0, %{lon:.0f}\u00b0<extra></extra>",
      showlegend: false
    }];
    if (a.selected && a.cell_lat && a.cell_lat.length) {
      data.push({
        type: "scattergeo",
        mode: "lines",
        lat: a.cell_lat,
        lon: a.cell_lon,
        line: { color: "#EF553B", width: 2.2 },
        hoverinfo: "text",
        text: a.cell_label,
        showlegend: false
      });
      data.push({
        type: "scattergeo",
        mode: "markers",
        lat: [a.focus_lat],
        lon: [a.focus_lon],
        marker: { size: 13, color: "#EF553B", symbol: "circle-cross", line: { color: "white", width: 1.8 } },
        hovertemplate: "Target: " + a.coord_text + "<extra></extra>",
        showlegend: false
      });
    }
    return data;
  }

  function buildLayout() {
    return {
      geo: {
        projection: { type: "orthographic", rotation: { lon: rot.lon, lat: rot.lat, roll: 0 } },
        showland: true, landcolor: "rgb(40, 74, 54)",
        showocean: true, oceancolor: "rgb(10, 30, 58)",
        showlakes: true, lakecolor: "rgb(16, 46, 86)",
        showcountries: true, countrycolor: "rgba(226, 232, 240, 0.38)", countrywidth: 0.7,
        showcoastlines: true, coastlinecolor: "rgba(248, 250, 252, 0.7)", coastlinewidth: 0.85,
        bgcolor: "#050914"
      },
      margin: { l: 0, r: 0, t: 0, b: 0 },
      height: HEIGHT,
      paper_bgcolor: "#050914",
      dragmode: "pan",
      hovermode: "closest",
      hoverdistance: 25,
      uirevision: "globe-" + String(focusId)
    };
  }

  function bind() {
    el.on("plotly_click", function (ev) {
      if (wasDragged) { return; }
      if (!ev || !ev.points || !ev.points.length) { return; }
      var p = ev.points[0];
      if (p.curveNumber !== 0) { return; }
      var idx = (typeof p.pointNumber === "number") ? p.pointNumber : p.pointIndex;
      if (typeof idx !== "number" || idx < 0 || idx >= meshLat.length) { return; }
      clickCounter += 1;
      post("streamlit:setComponentValue", {
        value: { lat: meshLat[idx], lon: meshLon[idx], seq: String(Date.now()) + "-" + String(clickCounter) },
        dataType: "json"
      });
    });
    el.on("plotly_relayout", function () { syncRot(); });
    el.addEventListener("pointerenter", function () { hovering = true; });
    el.addEventListener("pointerdown", function (e) {
      hovering = true;
      dragging = true;
      wasDragged = false;
      downX = e.clientX;
      downY = e.clientY;
      pauseUntil = performance.now() + 2500;
    }, true);
    window.addEventListener("pointermove", function (e) {
      if (!dragging) { return; }
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 6) {
        wasDragged = true;
      }
    }, true);
    window.addEventListener("pointerup", function () {
      if (!dragging) { return; }
      dragging = false;
      syncRot();
      lastTs = performance.now();
    }, true);
    el.addEventListener("pointerleave", function () {
      syncRot();
      hovering = false;
      dragging = false;
      lastTs = performance.now();
    });
    el.addEventListener("pointercancel", function () {
      hovering = false;
      dragging = false;
      lastTs = performance.now();
    });
  }

  function applyArgs(a) {
    var sel = !!a.selected;
    spinSpeed = (typeof a.spin_speed === "number") ? a.spin_speed : DEFAULT_SPIN;
    if (!created || a.focus_id !== focusId) {
      focusId = a.focus_id;
      rot.lon = a.focus_lon;
      rot.lat = sel ? a.focus_lat * 0.65 : 20;
    }
    spinning = true;
    if (!created) {
      created = true;
      window.Plotly.newPlot(el, buildData(a), buildLayout(), CONFIG).then(function () {
        ready = true;
        bind();
        if (pending) {
          var next = pending;
          pending = null;
          applyArgs(next);
        }
      });
    } else if (!ready) {
      pending = a;
    } else {
      window.Plotly.react(el, buildData(a), buildLayout(), CONFIG);
    }
  }

  function tick(ts) {
    window.requestAnimationFrame(tick);
    if (!ready || !spinning || hovering || dragging || busy || ts < pauseUntil) {
      lastTs = ts;
      return;
    }
    if (ts - lastTs < 33) { return; }
    var dt = Math.min((ts - lastTs) / 1000, 0.1);
    lastTs = ts;
    rot.lon = wrapLon(rot.lon + spinSpeed * dt);
    busy = true;
    try {
      window.Plotly.relayout(el, { "geo.projection.rotation.lon": rot.lon }).then(
        function () { busy = false; },
        function () { busy = false; }
      );
    } catch (err) {
      busy = false;
    }
  }

  function loadScript(src, onOk, onFail) {
    var s = document.createElement("script");
    s.src = src;
    s.onload = onOk;
    s.onerror = onFail;
    document.head.appendChild(s);
  }

  function ensurePlotly(done) {
    if (typeof window.Plotly !== "undefined") {
      done(true);
      return;
    }
    loadScript(PLOTLY_PRIMARY, function () { done(typeof window.Plotly !== "undefined"); }, function () {
      loadScript(PLOTLY_BACKUP, function () { done(typeof window.Plotly !== "undefined"); }, function () { done(false); });
    });
  }

  window.addEventListener("message", function (event) {
    var d = event.data;
    if (!d || d.type !== "streamlit:render") { return; }
    latestArgs = d.args || {};
    if (plotlyState === "ok") { applyArgs(latestArgs); }
    post("streamlit:setFrameHeight", { height: HEIGHT });
  });

  post("streamlit:componentReady", { apiVersion: 1 });
  post("streamlit:setFrameHeight", { height: HEIGHT });

  ensurePlotly(function (ok) {
    if (!ok) {
      plotlyState = "failed";
      el.textContent = "The 3D globe could not be loaded. Use the 2D map or the Quick-Jump menu instead.";
      return;
    }
    plotlyState = "ok";
    window.requestAnimationFrame(tick);
    if (latestArgs) { applyArgs(latestArgs); }
  });
})();
</script>
</body>
</html>
"""


def prepare_globe_dir():
    for base in (Path(tempfile.gettempdir()), APP_DIR):
        try:
            folder = base / "climate_signal_globe"
            folder.mkdir(parents=True, exist_ok=True)
            index = folder / "index.html"
            if not index.exists() or index.read_text(encoding="utf-8") != GLOBE_HTML:
                with tempfile.NamedTemporaryFile("w", dir=str(folder), delete=False, encoding="utf-8") as tmp:
                    tmp.write(GLOBE_HTML)
                    tmp_path = tmp.name
                os.replace(tmp_path, str(index))
            return folder
        except OSError:
            continue
    return None


GLOBE_DIR = prepare_globe_dir()
globe_component = (
    components.declare_component("climate_signal_globe", path=str(GLOBE_DIR))
    if GLOBE_DIR is not None
    else None
)

st.markdown(
    """
<style>
.block-container { padding-top: 3.6rem; padding-bottom: 2.5rem; }
.hero-box {
    background: radial-gradient(circle at 85% 20%, rgba(56, 189, 248, 0.16), transparent 45%),
                linear-gradient(135deg, #070d1f 0%, #111c3a 55%, #0c1428 100%);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 16px;
    padding: 1.35rem 1.75rem;
    margin-top: 0.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
}
.hero-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.32);
    padding: 0.22rem 0.7rem;
    border-radius: 999px;
    margin-bottom: 0.5rem;
}
.hero-title { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: #f8fafc; margin: 0; line-height: 1.15; }
.hero-sub { color: #cbd5e1; font-size: 0.96rem; margin-top: 0.4rem; margin-bottom: 0; max-width: 780px; }
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(17, 25, 40, 0.85) 0%, rgba(11, 17, 30, 0.9) 100%);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 14px;
    padding: 0.95rem 1.1rem;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.22);
    transition: transform 0.18s ease, border-color 0.18s ease;
}
div[data-testid="stMetric"]:hover { transform: translateY(-2px); border-color: rgba(56, 189, 248, 0.45); }
div[data-testid="stMetricLabel"] { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: #94a3b8; }
div[data-testid="stMetricValue"] { font-size: 1.48rem; font-weight: 800; color: #f8fafc; }
.target-card {
    background: linear-gradient(180deg, rgba(17, 25, 40, 0.82) 0%, rgba(11, 17, 30, 0.88) 100%);
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 14px;
    padding: 1.15rem 1.25rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
}
.target-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 0.15rem; }
.target-val { font-size: 1.04rem; font-weight: 600; color: #f1f5f9; margin-bottom: 0.75rem; }
button[data-baseweb="tab"] { font-weight: 600; font-size: 0.92rem; padding-top: 0.6rem; padding-bottom: 0.6rem; }
</style>
""",
    unsafe_allow_html=True,
)

VARIABLES = {
    "T2M": {
        "name": "Air Temperature (2m above surface)",
        "unit": "°C",
        "source": "NASA MERRA-2",
        "desc": "Average air temperature 2 meters above the ground. Tracks regional warming or cooling.",
    },
    "PRECTOTCORR": {
        "name": "Precipitation (Rainfall & Snow)",
        "unit": "mm/day",
        "source": "NASA MERRA-2 (bias-corrected with satellite rainfall data)",
        "desc": "Bias-corrected total precipitation. Reveals long-term drying (drought) or wetting trends.",
    },
    "GWETTOP": {
        "name": "Surface Soil Wetness",
        "unit": "index (0-1)",
        "source": "NASA MERRA-2",
        "desc": "Moisture in the top ~5 cm of soil (0 = completely dry, 1 = fully saturated). Only meaningful over land.",
    },
    "ALLSKY_SFC_SW_DWN": {
        "name": "Surface Solar Radiation",
        "unit": "kWh/m²/day",
        "source": "NASA CERES",
        "desc": "Sunlight reaching Earth's surface. Shifts with cloud cover and atmospheric aerosols.",
    },
    "RH2M": {
        "name": "Relative Humidity",
        "unit": "%",
        "source": "NASA MERRA-2",
        "desc": "Moisture in the air relative to the maximum it can hold at that temperature.",
    },
    "WS10M": {
        "name": "Wind Speed (10m)",
        "unit": "m/s",
        "source": "NASA MERRA-2",
        "desc": "Average wind speed 10 meters above the surface, reflecting atmospheric circulation changes.",
    },
}

NO_PRESET = "Select or Click Map/Globe"
SVALBARD_NAME = "Arctic (Svalbard, Norway)"
PRESETS = {
    NO_PRESET: None,
    "Amazon Rainforest (Manaus)": (-3.12, -60.02),
    "Sahara Desert (Tamanrasset)": (22.79, 5.52),
    "Mumbai Coast, India": (19.07, 72.87),
    SVALBARD_NAME: (78.22, 15.65),
    "Coastal Peru (El Niño Zone)": (-5.19, -80.63),
    "Borneo Rainforest (Indonesia)": (-1.25, 116.83),
}

MIN_MONTHS = 36
ALPHA = 0.05


def wrap_lon(lon):
    return ((lon + 180) % 360) - 180


def fmt_coords(lat, lon):
    return (f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, "
            f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}")


def grid_cell(lat, lon, var_code):
    if var_code == "ALLSKY_SFC_SW_DWN":
        dlat, dlon = 1.0, 1.0
        label = "Approximate CERES-based solar grid cell (~1° × 1°)"
        c_lat = min(float(np.floor(lat)) + 0.5, 89.5)
        c_lon = min(float(np.floor(lon)) + 0.5, 179.5)
    else:
        dlat, dlon = 0.5, 0.625
        label = "Approximate NASA MERRA-2 grid cell (~0.5° × 0.625°)"
        c_lat = round(lat / dlat) * dlat
        c_lon = round((lon + 180) / dlon) * dlon - 180
    south = max(c_lat - dlat / 2, -90)
    north = min(c_lat + dlat / 2, 90)
    return [[south, c_lon - dlon / 2], [north, c_lon + dlon / 2]], label


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_nasa_data(lat, lon, param, start_year, end_year):
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": param,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_year,
        "end": end_year,
        "format": "JSON",
    }

    payload, last_err = None, None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                last_err = "rate limit reached"
                time.sleep(2.0 * (attempt + 1))
                continue
            if 400 <= r.status_code < 500:
                raise RuntimeError(f"NASA POWER rejected the request (HTTP {r.status_code}). "
                                   "Try a different location, variable or period.")
            r.raise_for_status()
            payload = r.json()
            break
        except (requests.RequestException, ValueError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))

    if payload is None:
        raise RuntimeError(f"The NASA POWER service could not be reached ({last_err}). "
                           "Please try again in a moment.")

    msgs = [str(m) for m in (payload.get("messages") or [])]
    if any("error" in m.lower() or "invalid" in m.lower() for m in msgs):
        raise RuntimeError(f"NASA POWER API message: {' '.join(msgs)}")

    try:
        raw = payload["properties"]["parameter"][param]
    except (KeyError, TypeError):
        raise RuntimeError("NASA POWER returned an unexpected response structure for this location/variable.")

    records = []
    for ym, val in raw.items():
        if ym.endswith("13") or val is None or float(val) <= -990:
            continue
        date = pd.to_datetime(ym, format="%Y%m")
        records.append({"Date": date, "Year": date.year, "Month": date.month, "Value": float(val)})

    if not records:
        raise ValueError("NASA has no valid data for this variable at this location "
                         "(for example, soil wetness over open ocean). Try a land point.")

    df = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
    df["t"] = (df["Year"] - df["Year"].min()) * 12 + (df["Month"] - 1)
    df["Anomaly"] = df["Value"] - df.groupby("Month")["Value"].transform("mean")
    return df


@st.cache_data(show_spinner=False)
def load_giovanni_validation():
    if not GIOVANNI_CSV.exists():
        return None
    try:
        lines = GIOVANNI_CSV.read_text(encoding="utf-8", errors="ignore").splitlines()
        header_idx = 0
        for i, line in enumerate(lines):
            low = line.strip().lower()
            if low.startswith("time,") or low.startswith("date,"):
                header_idx = i
                break
        return pd.read_csv(GIOVANNI_CSV, skiprows=header_idx)
    except Exception:
        return None


def compare_with_giovanni(val_df, power_df):
    tcol = val_df.columns[0]
    num_cols = [c for c in val_df.columns[1:] if pd.api.types.is_numeric_dtype(val_df[c])]
    if not num_cols:
        return None, "No numeric data column was found in the Giovanni file."
    vcol = num_cols[0]
    if "t2m" not in str(vcol).lower():
        return None, (f"The Giovanni file's data column (`{vcol}`) does not look like air temperature "
                      "(T2M), so no automatic comparison was run.")

    g = pd.DataFrame({
        "Date": (pd.to_datetime(val_df[tcol], errors="coerce", utc=True)
                 .dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()),
        "Giovanni": pd.to_numeric(val_df[vcol], errors="coerce"),
    }).dropna()
    if g.empty:
        return None, "The Giovanni file's time column could not be read."
    if g["Giovanni"].mean() > 150:
        g["Giovanni"] = g["Giovanni"] - 273.15

    merged = (power_df[["Date", "Value"]].rename(columns={"Value": "POWER"})
              .merge(g, on="Date", how="inner"))
    if len(merged) < 24:
        return None, (f"Only {len(merged)} overlapping months between the two sources for the selected "
                      "period; at least 24 are needed. Widen the analysis period.")

    months = merged["Date"].dt.month
    anom_p = merged["POWER"] - merged.groupby(months)["POWER"].transform("mean")
    anom_g = merged["Giovanni"] - merged.groupby(months)["Giovanni"].transform("mean")
    diff = merged["POWER"] - merged["Giovanni"]
    metrics = {
        "n": len(merged),
        "bias": float(diff.mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "r_anom": float(np.corrcoef(anom_p, anom_g)[0, 1]),
    }
    return merged, metrics


@st.cache_data(show_spinner=False)
def analyze_trend(df, n_boot=300, block=12, seed=42):
    y = df["Anomaly"].to_numpy(dtype=float)
    t = df["t"].to_numpy(dtype=float)
    n = len(y)

    slope, intercept, _, _ = stats.theilslopes(y, t)
    fitted = intercept + slope * t
    resid = y - fitted

    mk_res = mk.hamed_rao_modification_test(y)

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        r = np.concatenate([resid[s:s + block] for s in starts])[:n]
        boots[i] = stats.theilslopes(fitted + r, t)[0]
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])

    return {
        "slope_decade": slope * 120,
        "ci_low": ci_low * 120,
        "ci_high": ci_high * 120,
        "total_change": slope * (t.max() - t.min()),
        "mk_p": float(mk_res.p),
        "trend_line": fitted,
    }


def annual_means(df):
    g = df.groupby("Year")["Value"].agg(["mean", "count"]).reset_index()
    g = g[g["count"] == 12]
    return g[["Year", "mean"]].rename(columns={"mean": "Annual Mean"})


def verdict(res):
    sig = res["mk_p"] < ALPHA
    ci_excludes_zero = res["ci_low"] > 0 or res["ci_high"] < 0
    if sig and ci_excludes_zero:
        return "significant"
    if sig or ci_excludes_zero:
        return "borderline"
    return "none"


def style_plotly_chart(fig, height=410):
    fig.update_layout(
        height=height,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#111b30",
        font=dict(color="#e2e8f0"),
        title_font=dict(size=16, color="#f8fafc"),
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.14)", zerolinecolor="rgba(148, 163, 184, 0.25)"),
        yaxis=dict(gridcolor="rgba(148, 163, 184, 0.14)", zerolinecolor="rgba(148, 163, 184, 0.25)"),
        hovermode="x unified",
    )
    return fig


st.session_state.setdefault("lat", 19.07)
st.session_state.setdefault("lon", 72.87)
st.session_state.setdefault("zoom", 3)
st.session_state.setdefault("has_selected", False)
st.session_state.setdefault("last_click", None)
st.session_state.setdefault("last_globe_seq", None)
st.session_state.setdefault("globe_focus_nonce", 0)
st.session_state.setdefault("preset", NO_PRESET)
st.session_state.setdefault("in_lat", 19.07)
st.session_state.setdefault("in_lon", 72.87)

if "pending_preset" in st.session_state:
    st.session_state["preset"] = st.session_state.pop("pending_preset")

if st.session_state.pop("sync_inputs", False):
    st.session_state["in_lat"] = float(st.session_state["lat"])
    st.session_state["in_lon"] = float(st.session_state["lon"])


def set_location(lat, lon, zoom=5):
    st.session_state["lat"] = round(float(lat), 4)
    st.session_state["lon"] = round(wrap_lon(float(lon)), 4)
    st.session_state["zoom"] = max(st.session_state["zoom"], zoom)
    st.session_state["has_selected"] = True
    st.session_state["sync_inputs"] = True


def on_preset_change():
    coords = PRESETS[st.session_state["preset"]]
    if coords is not None:
        set_location(*coords)
        st.session_state["zoom"] = 5
        st.session_state["globe_focus_nonce"] += 1


def on_manual_coords():
    set_location(st.session_state["in_lat"], st.session_state["in_lon"])
    st.session_state["preset"] = NO_PRESET
    st.session_state["globe_focus_nonce"] += 1


def render_globe(var_code):
    if globe_component is None:
        st.warning("The 3D globe is unavailable in this environment. Please use the 2D map or the Quick-Jump menu.")
        return
    lat_c, lon_c = st.session_state.lat, st.session_state.lon
    has_sel = st.session_state.has_selected

    cell_lat, cell_lon, cell_label = [], [], ""
    if has_sel:
        bounds, cell_label = grid_cell(lat_c, lon_c, var_code)
        (south, west), (north, east) = bounds
        cell_lat = [float(south), float(north), float(north), float(south), float(south)]
        cell_lon = [float(west), float(west), float(east), float(east), float(west)]

    event = globe_component(
        focus_id=int(st.session_state.globe_focus_nonce),
        focus_lat=float(lat_c),
        focus_lon=float(lon_c),
        selected=bool(has_sel),
        spin_speed=float(SPIN_DEG_PER_SEC),
        cell_lat=cell_lat,
        cell_lon=cell_lon,
        cell_label=cell_label,
        coord_text=fmt_coords(lat_c, lon_c),
        key="globe_component",
        default=None,
    )
    st.caption("💡 Click anywhere on the globe to select a location, or drag to rotate it. "
               "The globe pauses while your cursor is on it.")

    if isinstance(event, dict):
        seq = event.get("seq")
        new_lat, new_lon = event.get("lat"), event.get("lon")
        if seq is not None and new_lat is not None and new_lon is not None \
                and seq != st.session_state.last_globe_seq:
            st.session_state.last_globe_seq = seq
            set_location(new_lat, new_lon)
            st.session_state["pending_preset"] = NO_PRESET
            st.rerun()


def render_2d_map(var_code):
    lat_c, lon_c = st.session_state.lat, st.session_state.lon
    has_sel = st.session_state.has_selected

    m = folium.Map(
        location=[lat_c, lon_c],
        zoom_start=st.session_state.zoom,
        tiles=None,
        control_scale=True,
    )
    m.get_root().html.add_child(folium.Element(
        """
        <style>
        .leaflet-control-scale-line {
            box-sizing: content-box !important;
            white-space: nowrap !important;
            font-size: 11px !important;
            line-height: 1.2 !important;
            padding: 2px 6px !important;
        }
        </style>
        """
    ))

    satellite_group = folium.FeatureGroup(name="Satellite Imagery", overlay=False, show=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
    ).add_to(satellite_group)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
    ).add_to(satellite_group)
    satellite_group.add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Street & Terrain Map",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    if has_sel:
        bounds, cell_label = grid_cell(lat_c, lon_c, var_code)
        folium.Rectangle(
            bounds=bounds,
            color="#EF553B",
            weight=2,
            fill=True,
            fill_color="#EF553B",
            fill_opacity=0.15,
            tooltip=cell_label,
        ).add_to(m)
        folium.Marker(
            [lat_c, lon_c],
            tooltip=f"Target: {fmt_coords(lat_c, lon_c)}",
            icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
        ).add_to(m)

    folium.LayerControl(position="topright", collapsed=True).add_to(m)

    map_data = st_folium(
        m,
        height=394,
        width=None,
        returned_objects=["last_clicked"],
        key="interactive_map",
    )

    if map_data and map_data.get("last_clicked"):
        click = (round(map_data["last_clicked"]["lat"], 4), round(map_data["last_clicked"]["lng"], 4))
        if click != st.session_state.last_click:
            st.session_state.last_click = click
            set_location(*click)
            st.session_state["pending_preset"] = NO_PRESET
            st.session_state["globe_focus_nonce"] += 1
            st.rerun()


st.markdown(
    """
<div class="hero-box">
<div class="hero-badge">NASA POWER · MERRA-2 &amp; CERES</div>
<h1 class="hero-title">🌍 Climate Signal</h1>
<p class="hero-sub">Click anywhere on the interactive 3D globe or the 2D satellite map to separate
statistically significant environmental trends from natural seasonal variability.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("1. Variable & Timeframe")
    var_code = st.selectbox(
        "Earth Science Variable",
        options=list(VARIABLES.keys()),
        format_func=lambda k: f"{VARIABLES[k]['name']} ({k})",
    )
    year_range = st.slider("Analysis Period", 1985, 2025, (2001, 2025))

    st.header("2. Quick-Jump Regions")
    st.selectbox("Jump to a Region", list(PRESETS.keys()), key="preset", on_change=on_preset_change)

    with st.expander("📌 Or enter coordinates"):
        st.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.5, key="in_lat")
        st.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.5, key="in_lon")
        st.button("Use these coordinates", on_click=on_manual_coords, **STRETCH)

    st.divider()
    if st.button("🔄 Clear cached data", **STRETCH):
        st.cache_data.clear()
        st.toast("Cache cleared. Fetching fresh data.")

    with st.expander("ℹ️ How the analysis works"):
        st.markdown(
            "1. Monthly data are downloaded from the **NASA POWER** API.\n"
            "2. The **seasonal cycle is removed** (each month minus its long-term average) so we "
            "see change, not summer vs. winter.\n"
            "3. **Trend size** = Theil-Sen slope (robust to outliers).\n"
            "4. **Significance** = Hamed-Rao modified Mann-Kendall test, which accounts for "
            "month-to-month correlation.\n"
            "5. **Uncertainty** = 95% interval from a block bootstrap."
        )

v_meta = VARIABLES[var_code]
unit = v_meta["unit"]

col_map, col_info = st.columns([2.15, 1], gap="large")
with col_map:
    view_mode = st.radio(
        "Spatial View",
        options=["🌍 3D Globe", "🗺️ 2D Satellite & Street Map"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view_mode == "🌍 3D Globe":
        render_globe(var_code)
    else:
        render_2d_map(var_code)

with col_info:
    coord_text = fmt_coords(st.session_state.lat, st.session_state.lon)
    coord_display = f"📍 {coord_text}" if st.session_state.has_selected else f"📍 {coord_text} (Default)"
    st.markdown(
        f"""
<div class="target-card">
<div class="target-label">Selected Coordinates</div>
<div class="target-val">{coord_display}</div>
<div class="target-label">Earth Science Variable</div>
<div class="target-val">{v_meta['name']}</div>
<div class="target-label">NASA Source Dataset</div>
<div class="target-val" style="margin-bottom: 0;">🛰️ {v_meta['source']}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.info(v_meta["desc"])
    st.caption("Click any location on the globe or 2D map to drop a target marker. "
               "Results describe the whole grid cell, not the exact point.")

st.divider()

if year_range[1] - year_range[0] < 9:
    st.warning("Short period selected: trends over fewer than ~10 years are dominated by natural variability.")

try:
    with st.spinner("Fetching satellite & model data from NASA POWER API..."):
        df = fetch_nasa_data(round(st.session_state.lat, 2), round(st.session_state.lon, 2),
                             var_code, year_range[0], year_range[1])
except (RuntimeError, ValueError) as e:
    st.error(str(e))
    st.stop()

if len(df) < MIN_MONTHS:
    st.error(f"Only {len(df)} valid months were returned; at least {MIN_MONTHS} are needed for a "
             "reliable trend test. Choose a longer period or a different location.")
    st.stop()

n_expected = (year_range[1] - year_range[0] + 1) * 12
if len(df) < n_expected:
    st.caption(f"ℹ️ {n_expected - len(df)} month(s) had no data and were skipped.")

with st.spinner("Running trend statistics..."):
    try:
        res = analyze_trend(df)
    except Exception as e:
        st.error(f"The trend test could not be computed for this series ({e}).")
        st.stop()

state = verdict(res)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Rate of Change (Sen's slope)", f"{res['slope_decade']:+.3f} {unit}/decade")
m2.metric("95% Interval (block bootstrap)", f"{res['ci_low']:+.3f} to {res['ci_high']:+.3f}",
          help=f"Units: {unit}/decade")
m3.metric("Total Change over Period", f"{res['total_change']:+.2f} {unit}")
m4.metric("Mann-Kendall p-value (Hamed-Rao)", f"{res['mk_p']:.4f}")

direction = "an increasing" if res["slope_decade"] > 0 else "a decreasing"
if state == "significant":
    st.success(
        f"✅ **Statistically significant trend.** The data show {direction} trend of "
        f"{res['slope_decade']:+.3f} {unit} per decade (p = {res['mk_p']:.4f}); the 95% interval "
        "excludes zero."
    )
elif state == "borderline":
    st.warning(
        "⚠️ **Mixed evidence.** The significance test and the uncertainty interval disagree "
        f"(p = {res['mk_p']:.4f}; interval {res['ci_low']:+.3f} to {res['ci_high']:+.3f} {unit}/decade). "
        "Treat this trend as possible but not confirmed."
    )
else:
    st.info(
        f"**No statistically significant trend** (p = {res['mk_p']:.4f}). Changes at this location "
        "are within natural variability for the selected period."
    )

is_svalbard = (abs(st.session_state.lat - PRESETS[SVALBARD_NAME][0]) < 0.01
               and abs(st.session_state.lon - PRESETS[SVALBARD_NAME][1]) < 0.01)

tab_names = ["📈 Deseasonalized Trend", "🌊 Raw Monthly Observations",
             "🕵️ Detective: Early vs. Recent", "📋 Data Table"]
if is_svalbard:
    tab_names.append("🛰️ Giovanni Spot-Check")
tabs = st.tabs(tab_names)

with tabs[0]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Anomaly"], mode="lines",
                             name="Monthly anomaly (season removed)", line=dict(color="#636EFA", width=1.1)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Anomaly"].rolling(12, center=True).mean(),
                             mode="lines", name="12-month average", line=dict(color="#00CC96", width=2.2)))
    fig.add_trace(go.Scatter(x=df["Date"], y=res["trend_line"], mode="lines",
                             name="Sen's slope trend", line=dict(color="#EF553B", width=3)))
    fig.update_layout(title=f"Long-term trend (seasonal cycle removed): {v_meta['name']}",
                      yaxis_title=f"Anomaly ({unit})", legend=dict(orientation="h", y=-0.18))
    style_plotly_chart(fig, height=420)
    st.plotly_chart(fig, **STRETCH)

with tabs[1]:
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=df["Date"], y=df["Value"], mode="lines",
                                 name="Raw monthly value", line=dict(color="#00CC96", width=1.5)))
    fig_raw.update_layout(title=f"Raw monthly measurements: {v_meta['name']}", yaxis_title=unit)
    style_plotly_chart(fig_raw, height=420)
    st.plotly_chart(fig_raw, **STRETCH)

with tabs[2]:
    ann = annual_means(df)
    if len(ann) < 6:
        st.info("Need at least 6 complete years to compare early vs. recent periods.")
    else:
        half = len(ann) // 2
        early, recent = ann.iloc[:half], ann.iloc[half:]
        e_mean, r_mean = early["Annual Mean"].mean(), recent["Annual Mean"].mean()
        diff = r_mean - e_mean
        _, p_welch = stats.ttest_ind(recent["Annual Mean"], early["Annual Mean"], equal_var=False)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Early mean ({int(early['Year'].min())}-{int(early['Year'].max())})", f"{e_mean:.3f} {unit}")
        c2.metric(f"Recent mean ({int(recent['Year'].min())}-{int(recent['Year'].max())})", f"{r_mean:.3f} {unit}")
        c3.metric("Difference (recent - early)", f"{diff:+.3f} {unit}", help=f"Welch t-test p = {p_welch:.4f}")

        fig_a = go.Figure()
        for part, color, name, mean_v in ((early, "#636EFA", "Early period", e_mean),
                                          (recent, "#EF553B", "Recent period", r_mean)):
            fig_a.add_trace(go.Scatter(x=part["Year"], y=part["Annual Mean"], mode="lines+markers",
                                       name=name, line=dict(color=color, width=2), marker=dict(size=8)))
            fig_a.add_shape(type="line", x0=part["Year"].min(), x1=part["Year"].max(),
                            y0=mean_v, y1=mean_v, line=dict(color=color, dash="dash", width=2))
        fig_a.update_layout(title="Annual means with period averages (dashed)",
                            yaxis_title=unit, xaxis_title="Year")
        style_plotly_chart(fig_a, height=400)
        st.plotly_chart(fig_a, **STRETCH)
        st.caption(f"Welch t-test on annual means: p = {p_welch:.4f}. A simple complementary check on the "
                   "same data; it ignores year-to-year correlation, so rely on the main trend test for "
                   "the final verdict.")

with tabs[3]:
    st.dataframe(df.drop(columns=["t"]), **STRETCH)
    csv = df.drop(columns=["t"]).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download NASA Data (CSV)", csv,
                       f"nasa_{var_code}_{st.session_state.lat}_{st.session_state.lon}.csv", "text/csv")

if is_svalbard:
    with tabs[4]:
        st.subheader("🛰️ NASA Giovanni spot-check (Svalbard preset)")
        st.markdown(
            "A single-location check of the monthly values used in this app against an export from "
            "**NASA Giovanni**, an independent NASA data-access tool. It is a sanity check, not a full "
            "validation. Giovanni exports are typically area-averaged, so small differences are expected."
        )
        val_df = load_giovanni_validation()
        if val_df is None:
            st.info("`giovanni_svalbard_validation.csv` was not found in the project repository.")
        else:
            if var_code != "T2M":
                st.info("Select **Air Temperature (T2M)** in the sidebar to run the automatic comparison.")
            else:
                merged, out = compare_with_giovanni(val_df, df)
                if merged is None:
                    st.info(out)
                else:
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Overlapping months", f"{out['n']}")
                    k2.metric("Mean difference (POWER - Giovanni)", f"{out['bias']:+.2f} °C")
                    k3.metric("RMSE", f"{out['rmse']:.2f} °C")
                    k4.metric("Correlation (seasonal cycle removed)", f"{out['r_anom']:.3f}")
                    fig_g = go.Figure()
                    fig_g.add_trace(go.Scatter(x=merged["Date"], y=merged["POWER"], mode="lines",
                                               name="NASA POWER (this app)", line=dict(color="#636EFA", width=2)))
                    fig_g.add_trace(go.Scatter(x=merged["Date"], y=merged["Giovanni"], mode="lines",
                                               name="NASA Giovanni export", line=dict(color="#EF553B", dash="dot", width=2)))
                    fig_g.update_layout(title="Monthly air temperature: POWER vs. Giovanni",
                                        yaxis_title="°C", legend=dict(orientation="h", y=-0.18))
                    style_plotly_chart(fig_g, height=380)
                    st.plotly_chart(fig_g, **STRETCH)
            with st.expander("Show raw Giovanni file"):
                st.dataframe(val_df, **STRETCH)

with st.expander("⚠️ How to interpret these results (limitations)"):
    st.markdown(
        "- **Grid cell, not a point:** NASA POWER draws on MERRA-2 (~0.5° × 0.625°, roughly 55-70 km) "
        "and CERES-based products, so each result represents an area, not a single spot.\n"
        "- **Model-assimilated data:** reanalysis trends can be affected by changes in the "
        "observations assimilated over time.\n"
        "- **Short records vs. natural cycles:** 10-25 years can be dominated by El Niño/La Niña and "
        "decadal variability, so a trend is not automatically caused by climate change.\n"
        "- **Statistical significance ≠ importance:** always check the size of the change "
        "(total change and interval), not just the p-value.\n"
        "- **Limited external check:** the Svalbard preset was spot-checked against NASA Giovanni "
        "(`giovanni_svalbard_validation.csv`); other locations and variables were not independently "
        "validated."
    )
