import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd
import numpy as np
from scipy import stats
import pymannkendall as mk
import plotly.graph_objects as go

st.set_page_config(page_title="NASA Earth Trend Explorer", layout="wide")

# 1. NASA POWER Variable Catalog (Plain-English definitions for students)
VARIABLES = {
    "T2M": {
        "name": "Air Temperature (2m above surface)",
        "unit": "°C",
        "source": "NASA MERRA-2",
        "desc": "Average air temperature 2 meters above the ground. Tracks regional warming or cooling."
    },
    "PRECTOTCORR": {
        "name": "Precipitation (Rainfall & Snow)",
        "unit": "mm/day",
        "source": "NASA MERRA-2 / GPM",
        "desc": "Bias-corrected total precipitation. Reveals long-term drying (drought) or wetting trends."
    },
    "GWETTOP": {
        "name": "Surface Soil Wetness",
        "unit": "0 to 1 index",
        "source": "NASA MERRA-2",
        "desc": "Moisture content in the top 5 cm of soil (0 = completely dry, 1 = fully saturated)."
    },
    "ALLSKY_SFC_SW_DWN": {
        "name": "Surface Solar Radiation",
        "unit": "kWh/m²/day",
        "source": "NASA CERES",
        "desc": "Sunlight reaching Earth's surface. Shifts with cloud cover and atmospheric aerosols."
    },
    "RH2M": {
        "name": "Relative Humidity",
        "unit": "%",
        "source": "NASA MERRA-2",
        "desc": "Percentage of moisture in the air relative to the maximum it can hold at that temperature."
    },
    "WS10M": {
        "name": "Wind Speed (10m)",
        "unit": "m/s",
        "source": "NASA MERRA-2",
        "desc": "Average wind speed 10 meters above the surface, reflecting atmospheric circulation changes."
    }
}

@st.cache_data(show_spinner=False)
def fetch_nasa_data(lat, lon, param, start_year, end_year):
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": param,
        "community": "AG",
        "longitude": round(lon, 4),
        "latitude": round(lat, 4),
        "start": start_year,
        "end": end_year,
        "format": "JSON"
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    raw = r.json()["properties"]["parameter"][param]
    
    records = []
    for ym, val in raw.items():
        # Skip annual summary month '13' and missing fill values (-999)
        if ym.endswith("13") or val == -999:
            continue
        date = pd.to_datetime(ym, format="%Y%m")
        records.append({"Date": date, "Year": date.year, "Month": date.month, "Value": val})
    
    df = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
    # Remove seasonal cycle by subtracting each calendar month's long-term average
    monthly_means = df.groupby("Month")["Value"].transform("mean")
    df["Anomaly"] = df["Value"] - monthly_means
    return df

def analyze_trend(df):
    x = np.arange(len(df))
    y = df["Anomaly"].values
    slope, intercept, r_val, p_val_ols, std_err = stats.linregress(x, y)
    mk_result = mk.original_test(y)
    
    slope_per_decade = slope * 120  # 120 months in a decade
    ci_95 = 1.96 * std_err * 120
    trend_line = intercept + slope * x
    return {
        "slope_decade": slope_per_decade,
        "ci_95": ci_95,
        "mk_trend": mk_result.trend,
        "mk_p": mk_result.p,
        "significant": mk_result.p < 0.05,
        "trend_line": trend_line
    }

# --- UI LAYOUT ---
st.title("🌍 NASA Earth Science Trend Explorer")
st.markdown("Click anywhere on the map (or pick a preset region), select a NASA variable, and test whether environmental changes are **statistically significant**.")

# Sidebar Controls
with st.sidebar:
    st.header("1. Select Variable & Timeframe")
    var_code = st.selectbox(
        "Earth Science Variable",
        options=list(VARIABLES.keys()),
        format_func=lambda k: f"{VARIABLES[k]['name']} ({k})"
    )
    year_range = st.slider("Analysis Period", 1985, 2025, (2001, 2025))
    
    st.header("2. Quick-Jump Regions")
    presets = {
        "Custom (Click Map)": None,
        "Amazon Rainforest (Manaus)": (-3.12, -60.02),
        "Sahara Desert (Tamanrasset)": (22.79, 5.52),
        "Mumbai Coast, India": (19.07, 72.87),
        "Arctic (Svalbard, Norway)": (78.22, 15.65),
        "Coastal Peru (El Niño Zone)": (-5.19, -80.63),
        "Borneo Rainforest (Indonesia)": (-1.25, 116.83)
    }
    preset_choice = st.selectbox("Jump to a Region", list(presets.keys()))

# Session state for coordinates
if "lat" not in st.session_state:
    st.session_state.lat, st.session_state.lon = 19.07, 72.87
if presets[preset_choice] is not None:
    st.session_state.lat, st.session_state.lon = presets[preset_choice]

# Interactive Map
col_map, col_info = st.columns([2, 1])
with col_map:
    m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=3)
    folium.Marker(
        [st.session_state.lat, st.session_state.lon],
        tooltip=f"Selected: {st.session_state.lat:.2f}, {st.session_state.lon:.2f}"
    ).add_to(m)
    map_data = st_folium(m, height=350, width="100%")
    
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]
        if (round(clicked_lat, 4), round(clicked_lon, 4)) != (st.session_state.lat, st.session_state.lon):
            st.session_state.lat = round(clicked_lat, 4)
            st.session_state.lon = round(clicked_lon, 4)
            st.rerun()

with col_info:
    v_meta = VARIABLES[var_code]
    st.subheader("Selected Target")
    st.write(f"**Coordinates:** `{st.session_state.lat}°N, {st.session_state.lon}°E`")
    st.write(f"**Variable:** {v_meta['name']}")
    st.write(f"**Dataset:** {v_meta['source']}")
    st.info(v_meta["desc"])

# Fetch & Display All Data
st.divider()
with st.spinner("Fetching satellite & model data from NASA POWER API..."):
    df = fetch_nasa_data(st.session_state.lat, st.session_state.lon, var_code, year_range[0], year_range[1])
    res = analyze_trend(df)

# Verdict Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rate of Change", f"{res['slope_decade']:+.3f} {v_meta['unit']}/decade")
m2.metric("95% Confidence Margin", f"±{res['ci_95']:.3f} {v_meta['unit']}/decade")
m3.metric("Mann-Kendall p-value", f"{res['mk_p']:.4f}")
if res["significant"]:
    m4.success(f"SIGNIFICANT ({res['mk_trend'].upper()})")
else:
    m4.warning("NOT SIGNIFICANT (p ≥ 0.05)")

# Charts
tab_trend, tab_raw, tab_table = st.tabs(["📈 Deseasonalized Trend Analysis", "🌊 Raw Monthly Observations", "📋 Complete Data Table"])

with tab_trend:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Anomaly"], mode="lines", name="Monthly Anomaly (Season Removed)", line=dict(color="#636EFA", width=1.5)))
    fig.add_trace(go.Scatter(x=df["Date"], y=res["trend_line"], mode="lines", name="Linear Trend", line=dict(color="#EF553B", width=3)))
    fig.update_layout(title=f"Long-Term Trend (Seasonal Cycle Removed) — {v_meta['name']}", yaxis_title=f"Anomaly ({v_meta['unit']})", height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab_raw:
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=df["Date"], y=df["Value"], mode="lines", name="Raw Monthly Value", line=dict(color="#00CC96")))
    fig_raw.update_layout(title=f"Raw Monthly Measurements — {v_meta['name']}", yaxis_title=v_meta["unit"], height=400)
    st.plotly_chart(fig_raw, use_container_width=True)

with tab_table:
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download NASA Data (CSV)", csv, f"nasa_{var_code}_{st.session_state.lat}_{st.session_state.lon}.csv", "text/csv")
