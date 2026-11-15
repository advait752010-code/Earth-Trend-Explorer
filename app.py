import time

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymannkendall as mk
import requests
import streamlit as st
from packaging.version import Version
from scipy import stats
from streamlit_folium import st_folium

st.set_page_config(page_title="NASA Earth Trend Explorer", layout="wide")

# Newer Streamlit versions replace use_container_width with width="stretch"
NEW_WIDTH_API = Version(st.__version__) >= Version("1.50.0")
STRETCH = {"width": "stretch"} if NEW_WIDTH_API else {"use_container_width": True}

# ----------------------------------------------------------------------------
# 1. NASA POWER variable catalog
# ----------------------------------------------------------------------------
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
        "desc": "Moisture in the top ~5 cm of soil (0 = completely dry, 1 = fully saturated). "
                "Only meaningful over land.",
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

PRESETS = {
    "Custom (Click Map)": None,
    "Amazon Rainforest (Manaus)": (-3.12, -60.02),
    "Sahara Desert (Tamanrasset)": (22.79, 5.52),
    "Mumbai Coast, India": (19.07, 72.87),
    "Arctic (Svalbard, Norway)": (78.22, 15.65),
    "Coastal Peru (El Niño Zone)": (-5.19, -80.63),
    "Borneo Rainforest (Indonesia)": (-1.25, 116.83),
}

MIN_MONTHS = 36          # minimum record length we are willing to analyse
ALPHA = 0.05


# ----------------------------------------------------------------------------
# 2. Helpers & data access
# ----------------------------------------------------------------------------
def wrap_lon(lon):
    """Map any longitude (folium can return values beyond ±180) into [-180, 180)."""
    return ((lon + 180) % 360) - 180


def fmt_coords(lat, lon):
    return (f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}, "
            f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}")


def grid_cell(lat, lon, var_code):
    """Approximate footprint of the source grid cell containing the selected point.

    MERRA-2 cells are 0.5° lat x 0.625° lon on a fixed grid; the CERES-based solar
    product is coarser (about 1° x 1°). Both are approximate.
    """
    if var_code == "ALLSKY_SFC_SW_DWN":
        dlat, dlon = 1.0, 1.0
        label = "Approximate CERES-based solar grid cell (~1° × 1°)"
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
    """Download monthly NASA POWER data and compute deseasonalized anomalies.

    Raises RuntimeError / ValueError with a human-readable message on failure
    (exceptions are never cached by Streamlit).
    """
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
            if 400 <= r.status_code < 500:
                # Client error: retrying will not help
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

    try:
        raw = payload["properties"]["parameter"][param]
    except (KeyError, TypeError):
        raise RuntimeError("NASA POWER returned an unexpected response for this location/variable.")

    records = []
    for ym, val in raw.items():
        # '13' = annual summary; values <= -990 are NASA's missing-data fill (-999)
        if ym.endswith("13") or val is None or float(val) <= -990:
            continue
        date = pd.to_datetime(ym, format="%Y%m")
        records.append({"Date": date, "Year": date.year, "Month": date.month, "Value": float(val)})

    if not records:
        raise ValueError("NASA has no valid data for this variable at this location "
                         "(for example, soil wetness over open ocean). Try a land point.")

    df = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
    # Months elapsed since the first record: correct even if some months are missing
    df["t"] = (df["Year"] - df["Year"].min()) * 12 + (df["Month"] - 1)
    # Remove the seasonal cycle: subtract each calendar month's long-term mean
    df["Anomaly"] = df["Value"] - df.groupby("Month")["Value"].transform("mean")
    return df


# ----------------------------------------------------------------------------
# 3. Statistics
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def analyze_trend(df, n_boot=300, block=12, seed=42):
    """Robust trend analysis for autocorrelated monthly anomalies.

    - Trend size: Theil-Sen slope (robust to outliers).
    - Significance: Hamed-Rao modified Mann-Kendall test, which corrects the
      p-value for serial correlation (the plain Mann-Kendall test does not).
    - Uncertainty: moving-block bootstrap of the Theil-Sen slope (12-month
      blocks preserve autocorrelation), giving a 95% interval.
    """
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
        "slope_decade": slope * 120,                        # per month -> per decade
        "ci_low": ci_low * 120,
        "ci_high": ci_high * 120,
        "total_change": slope * (t.max() - t.min()),
        "mk_p": float(mk_res.p),
        "trend_line": fitted,
    }


def annual_means(df):
    g = df.groupby("Year")["Value"].agg(["mean", "count"]).reset_index()
    g = g[g["count"] == 12]  # complete years only so the seasonal cycle cancels out
    return g[["Year", "mean"]].rename(columns={"mean": "Annual Mean"})


def verdict(res):
    """Combine the significance test and the bootstrap interval into one honest verdict."""
    sig = res["mk_p"] < ALPHA
    ci_excludes_zero = res["ci_low"] > 0 or res["ci_high"] < 0
    if sig and ci_excludes_zero:
        return "significant"
    if sig or ci_excludes_zero:
        return "borderline"
    return "none"


# ----------------------------------------------------------------------------
# 4. Session state & callbacks
# ----------------------------------------------------------------------------
st.session_state.setdefault("lat", 19.07)
st.session_state.setdefault("lon", 72.87)
st.session_state.setdefault("zoom", 3)
st.session_state.setdefault("last_click", None)


def on_preset_change():
    coords = PRESETS[st.session_state["preset"]]
    if coords is not None:
        st.session_state["lat"], st.session_state["lon"] = coords
        st.session_state["zoom"] = 5


# ----------------------------------------------------------------------------
# 5. UI
# ----------------------------------------------------------------------------
st.title("🌍 NASA Earth Science Trend Explorer")
st.markdown(
    "Click anywhere on the map (or pick a preset region), choose a NASA variable, and test whether "
    "environmental change is **statistically significant** or just natural variability."
)

with st.sidebar:
    st.header("1. Select Variable & Timeframe")
    var_code = st.selectbox(
        "Earth Science Variable",
        options=list(VARIABLES.keys()),
        format_func=lambda k: f"{VARIABLES[k]['name']} ({k})",
    )
    year_range = st.slider("Analysis Period", 1985, 2025, (2001, 2025))

    st.header("2. Quick-Jump Regions")
    st.selectbox("Jump to a Region", list(PRESETS.keys()), key="preset", on_change=on_preset_change)

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

# --- Map -----------------------------------------------------------------
col_map, col_info = st.columns([2, 1])
with col_map:
    lat_c, lon_c = st.session_state.lat, st.session_state.lon

    m = folium.Map(
        location=[lat_c, lon_c],
        zoom_start=st.session_state.zoom,
        tiles=None,
        control_scale=True,
    )

    # Satellite imagery is the default base layer (show=True); street map is selectable
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
        name="Satellite Imagery",
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Street & Terrain Map",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # Borders & place names overlay (switch off when using the street map)
    folium.TileLayer(
        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Place Labels (turn off for Street Map)",
        overlay=True,
        control=True,
        show=True,
    ).add_to(m)

    # Approximate footprint of the grid cell that the NASA value represents
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

    # Only return clicks, so panning/zooming doesn't trigger reruns or recentre the map
    try:
        map_data = st_folium(m, height=380, use_container_width=True, returned_objects=["last_clicked"])
    except TypeError:
        map_data = st_folium(m, height=380, width=700, returned_objects=["last_clicked"])

    if map_data and map_data.get("last_clicked"):
        click = (round(map_data["last_clicked"]["lat"], 4), round(map_data["last_clicked"]["lng"], 4))
        # Act only on NEW clicks: a stale click must never override a preset choice
        if click != st.session_state.last_click:
            st.session_state.last_click = click
            st.session_state.lat = click[0]
            st.session_state.lon = round(wrap_lon(click[1]), 4)
            st.session_state.zoom = max(st.session_state.zoom, 5)
            st.rerun()

with col_info:
    st.subheader("Selected Target")
    st.write(f"**Coordinates:** `{fmt_coords(st.session_state.lat, st.session_state.lon)}`")
    st.write(f"**Variable:** {v_meta['name']}")
    st.write(f"**Dataset:** {v_meta['source']}")
    st.info(v_meta["desc"])
    st.caption("The red box shows the approximate grid cell the NASA value represents. "
               "Results describe the whole cell, not the exact point.")

# --- Data & analysis -------------------------------------------------------
st.divider()

if year_range[1] - year_range[0] < 9:
    st.warning("Short period selected: trends over fewer than ~10 years are dominated by natural "
               "variability. Consider a longer window.")

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
    except Exception as e:  # e.g. constant series
        st.error(f"The trend test could not be computed for this series ({e}).")
        st.stop()

state = verdict(res)

# --- Verdict metrics -------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rate of Change (Sen's slope)", f"{res['slope_decade']:+.3f} {unit}/decade")
m2.metric("95% Interval (block bootstrap)",
          f"{res['ci_low']:+.3f} to {res['ci_high']:+.3f}", help=f"Units: {unit}/decade")
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
        f"➖ **No statistically significant trend** (p = {res['mk_p']:.4f}). Changes at this location "
        "are within natural variability for the selected period."
    )

# --- Charts ----------------------------------------------------------------
tab_trend, tab_raw, tab_annual, tab_table = st.tabs([
    "📈 Deseasonalized Trend",
    "🌊 Raw Monthly Observations",
    "🕵️ Detective: Early vs. Recent",
    "📋 Data Table",
])

with tab_trend:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Anomaly"], mode="lines",
                             name="Monthly anomaly (season removed)",
                             line=dict(color="#636EFA", width=1)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Anomaly"].rolling(12, center=True).mean(),
                             mode="lines", name="12-month average",
                             line=dict(color="#00CC96", width=2)))
    fig.add_trace(go.Scatter(x=df["Date"], y=res["trend_line"], mode="lines",
                             name="Sen's slope trend", line=dict(color="#EF553B", width=3)))
    fig.update_layout(title=f"Long-term trend (seasonal cycle removed): {v_meta['name']}",
                      yaxis_title=f"Anomaly ({unit})", height=420,
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, **STRETCH)

with tab_raw:
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=df["Date"], y=df["Value"], mode="lines",
                                 name="Raw monthly value", line=dict(color="#00CC96")))
    fig_raw.update_layout(title=f"Raw monthly measurements: {v_meta['name']}",
                          yaxis_title=unit, height=420)
    st.plotly_chart(fig_raw, **STRETCH)

with tab_annual:
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
        c1.metric(f"Early mean ({int(early['Year'].min())}-{int(early['Year'].max())})",
                  f"{e_mean:.3f} {unit}")
        c2.metric(f"Recent mean ({int(recent['Year'].min())}-{int(recent['Year'].max())})",
                  f"{r_mean:.3f} {unit}")
        c3.metric("Difference (recent - early)", f"{diff:+.3f} {unit}", help=f"Welch t-test p = {p_welch:.4f}")

        # Markers + period-mean lines (a bar chart with a truncated axis would exaggerate differences)
        fig_a = go.Figure()
        for part, color, name, mean_v in ((early, "#636EFA", "Early period", e_mean),
                                          (recent, "#EF553B", "Recent period", r_mean)):
            fig_a.add_trace(go.Scatter(x=part["Year"], y=part["Annual Mean"], mode="lines+markers",
                                       name=name, line=dict(color=color), marker=dict(size=8)))
            fig_a.add_shape(type="line", x0=part["Year"].min(), x1=part["Year"].max(),
                            y0=mean_v, y1=mean_v, line=dict(color=color, dash="dash", width=2))
        fig_a.update_layout(title="Annual means with period averages (dashed)",
                            yaxis_title=unit, xaxis_title="Year", height=400)
        st.plotly_chart(fig_a, **STRETCH)
        st.caption(f"Welch t-test on annual means: p = {p_welch:.4f}. This is a simple complementary "
                   "check on the same data; it does not account for year-to-year correlation, so "
                   "rely on the main trend test above for the final verdict.")

with tab_table:
    st.dataframe(df.drop(columns=["t"]), **STRETCH)
    csv = df.drop(columns=["t"]).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download NASA Data (CSV)", csv,
                       f"nasa_{var_code}_{st.session_state.lat}_{st.session_state.lon}.csv", "text/csv")

# --- Limitations -----------------------------------------------------------
with st.expander("⚠️ How to interpret these results (limitations)"):
    st.markdown(
        "- **Grid cell, not a point:** NASA POWER draws on MERRA-2 (about 0.5° × 0.625°, roughly "
        "55-70 km) and CERES-based satellite products, so each result represents an area, not "
        "a single spot.\n"
        "- **Model-assimilated data:** reanalysis trends can be affected by changes in the "
        "observations assimilated over time; confirm important findings with independent datasets.\n"
        "- **Short records vs. natural cycles:** 10-25 years can be dominated by El Niño/La Niña and "
        "decadal variability, so a trend is not automatically caused by climate change.\n"
        "- **Statistical significance ≠ importance:** always check the size of the change "
        "(total change and interval), not just the p-value.\n"
        "- **Spot-check against NASA Giovanni:** MERRA-2 values for the Svalbard preset were "
        "compared with NASA Giovanni's area-averaged time series "
        "(see `giovanni_svalbard_validation.csv` in the project repository). This is a single-location "
        "check, not a full validation."
    )
