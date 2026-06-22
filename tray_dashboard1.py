import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from datetime import datetime
from textwrap import dedent
import html
import json
from pathlib import Path

DASHBOARD_TITLE = "Dashboard Pemantauan NDVI dan Indikasi Risiko Kekeringan Jawa Timur"

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# STYLE
# =========================
st.markdown(
    """
    <style>
        .main {
            background-color: #f8fafc;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 0.8rem;
            padding-right: 1rem;
            max-width: 100% !important;
            width: 100% !important;
        }

        .dashboard-hero {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 55%, #ecfdf5 100%);
            border: 1px solid #e5e7eb;
            border-radius: 22px;
            padding: 24px 28px;
            margin-bottom: 24px;
            box-shadow: 0 6px 22px rgba(15, 23, 42, 0.06);
        }

        .hero-badge {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background-color: #dcfce7;
            color: #166534;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 12px;
        }

        .dashboard-title {
            font-size: 34px;
            font-weight: 850;
            color: #0f172a;
            margin: 0;
            line-height: 1.18;
            letter-spacing: -0.4px;
        }

        .dashboard-subtitle {
            font-size: 15px;
            color: #64748b;
            margin-top: 10px;
            margin-bottom: 16px;
            line-height: 1.6;
        }

        .hero-meta {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 10px;
        }

        .hero-chip {
            background-color: #ffffff;
            border: 1px solid #dbe4ee;
            color: #334155;
            padding: 7px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 600;
        }

        .metric-card {
            background: #ffffff;
            border: 1px solid #dbe2ea;
            border-radius: 18px;
            padding: 20px 22px;
            height: 170px;              /* paksa semua card sama tinggi */
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        .metric-title {
            font-size: 25px;
            color:#0f172a;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .metric-value {
            font-size: 30px;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.1;
        }

        .metric-help {
            font-size: 17px;
            color: #0f172a;
            line-height: 1.4;
            margin-top: 6px;
        }

        .metric-desc {
            font-size: 14px;
            color: #0f172a;
            line-height: 1.4;
            margin-top: 6px;
        }

        .section-card {
            background: #ffffff;
            padding: 18px;
            border-radius: 18px;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
            border: 1px solid #e5e7eb;
            margin-bottom: 18px;
        }
        .risk-pill {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            color: white;
            font-size: 13px;
            font-weight: 700;
            margin-right: 8px;
            margin-bottom: 8px;
        }
        .chart-card-title {
            font-size: 22px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 4px;
        }

        .chart-card-caption {
            font-size: 15px;
            color: #64748b;
            margin-bottom: 12px;
        }
        
        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e5e7eb;
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.2rem;
        }

        /* BRAND ATAS */
        .sidebar-brand {
            padding: 8px 2px 10px 2px;
            margin-bottom: 8px;
        }

        .sidebar-brand-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .sidebar-logo {
            width: 38px;
            height: 38px;
            border-radius: 12px;
            background: #dcfce7;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
        }

        .sidebar-title {
            font-size: 18px;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.15;
        }

        .sidebar-subtitle {
            font-size: 12px;
            color: #64748b;
            line-height: 1.35;
            margin-top: 6px;
        }

        /* JUDUL SECTION */
        .sidebar-section-title {
            font-size: 12px;
            font-weight: 800;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 10px;
            margin-bottom: 6px;
        }

        /* GARIS PEMISAH */
        section[data-testid="stSidebar"] hr {
            margin: 12px 0;
            border: none;
            border-top: 1px solid #e5e7eb;
        }

        /* NAVIGATION BUTTON STYLE */
        section[data-testid="stSidebar"] div[role="radiogroup"] {
            width: 100%;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            width: 100% !important;
            min-width: 100% !important;
            display: flex !important;
            align-items: center !important;
            background-color: transparent;
            border-radius: 12px;
            padding: 8px 12px !important;
            margin-bottom: 2px;
            min-height: 38px;
            transition: all 0.2s ease;
            border: 1px solid transparent;
        }

        /* Hilangkan bulatan radio */
        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {
            display: none !important;
        }

        /* Hover */
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background-color: #e8f7ef;
            border: 1px solid #bbf7d0;
        }

        /* Menu aktif */
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background-color: #16a34a;
            border: 1px solid #16a34a;
            box-shadow: 0 4px 12px rgba(22, 163, 74, 0.18);
        }

        /* Teks menu aktif */
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * {
            color: white !important;
            font-weight: 700 !important;
        }

        /* Ukuran teks menu */
        section[data-testid="stSidebar"] div[role="radiogroup"] p {
            font-size: 15px;
            margin: 0 !important;
        }

        /* RAPATKAN SELECTBOX FILTER */
        section[data-testid="stSidebar"] div[data-testid="stSelectbox"] {
            margin-bottom: 8px;
        }

        .stApp {
        background-color: #f8fafc;
        }

        [data-testid="stAppViewContainer"] {
            background-color: #f8fafc;
        }

        section.main > div {
            max-width: 100% !important;
        }

        div[data-testid="stExpander"] summary p {
        font-size: 17px !important;
        font-weight: 500 !important;
        color: #0f172a !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <style>
        header[data-testid="stHeader"] {{
            background-color: #ffffff;
            border-bottom: 1px solid #e5e7eb;
            height: 4.2rem;
        }}

        header[data-testid="stHeader"]::before {{
            content: "{DASHBOARD_TITLE}";
            position: fixed;
            top: 1.05rem;
            left: 18rem;
            z-index: 999999;
            font-size: 35px;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.2;
            white-space: nowrap;
        }}

        .block-container {{
            padding-top: 3rem !important;
        }}

        @media (max-width: 768px) {{

            header[data-testid="stHeader"] {{
                height: 3.6rem !important;
            }}

            header[data-testid="stHeader"]::before {{
                display: none !important;
                left: 3.5rem !important;
                top: 0.75rem !important;
                font-size: 13px !important;
                max-width: 72vw !important;
                white-space: normal !important;
                line-height: 1.2 !important;
            }}

            .block-container {{
                padding-top: 1rem !important;
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
                max-width: 100% !important;
                overflow-x: hidden !important;
            }}

            div[data-testid="stHorizontalBlock"] {{
                flex-wrap: wrap !important;
            }}

            div[data-testid="column"] {{
                width: 100% !important;
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }}

            .metric-card {{
                height: auto !important;
                min-height: 120px !important;
                padding: 14px 16px !important;
                margin-bottom: 12px !important;
            }}

            .metric-title {{
                font-size: 17px !important;
                line-height: 1.25 !important;
            }}

            .metric-value {{
                font-size: 24px !important;
            }}

            .metric-help,
            .metric-desc {{
                font-size: 13px !important;
            }}

            .chart-card-title {{
                font-size: 18px !important;
                line-height: 1.3 !important;
            }}

            .chart-card-caption {{
                font-size: 13px !important;
                line-height: 1.4 !important;
            }}

            div[data-testid="stPlotlyChart"] {{
                width: 100% !important;
                overflow-x: hidden !important;
            }}

            table {{
                font-size: 13px !important;
            }}
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.html("""
<style>
.shift-table-wrap {
    width: 100%;
    overflow-x: auto;
    border: 1px solid #dbe2ea;
    border-radius: 12px;
    background: #ffffff;
}

.shift-table-wrap table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    color: #0f172a;
}

.shift-table-wrap th {
    background: #f1f5f9;
    color: #0f172a;
    font-weight: 800;
    padding: 10px 12px;
    border-bottom: 1px solid #cbd5e1;
    text-align: left;
    white-space: nowrap;
}

.shift-table-wrap td {
    padding: 10px 12px;
    border-bottom: 1px solid #e5e7eb;
    white-space: nowrap;
}

@media (max-width: 768px) {
    .shift-table-wrap table {
        font-size: 12px;
    }

    .shift-table-wrap th,
    .shift-table-wrap td {
        padding: 8px 10px;
    }
}
</style>
""")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_forecast_data():
    df = pd.read_csv("historis_plus_forecast_2025.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    df["tahun"] = df.index.year
    df["bulan"] = df.index.month
    return df

@st.cache_data
def load_forecast_file(file_name):
    data = pd.read_csv(file_name, index_col=0)
    data.index = pd.to_datetime(data.index)
    data["tahun"] = data.index.year
    data["bulan"] = data.index.month
    return data

@st.cache_data
def load_classification_data():
    return pd.read_csv("Data_Lengkap_Kekeringan_Jatim.csv")

@st.cache_data
def load_eval_data():
    return pd.read_csv("evaluasi_gstar_jatim.csv")

@st.cache_data
def load_eval_file(file_name):
    return pd.read_csv(file_name)


@st.cache_data
def load_lstm_folder(folder_name="Prediksi_LSTM"):
    folder = Path(folder_name)
    files = sorted(folder.glob("prediksi_lstm_*.csv"))

    if len(files) == 0:
        return None, None

    forecast_series = []
    eval_rows = []

    for file in files:
        city_name = (
            file.stem
            .replace("prediksi_lstm_", "")
            .replace("_", " ")
        )

        temp = pd.read_csv(file)

        temp["TANGGAL"] = pd.to_datetime(temp["TANGGAL"])
        temp = temp.sort_values("TANGGAL")

        aktual = pd.to_numeric(temp["Aktual"], errors="coerce")
        prediksi = pd.to_numeric(temp["Prediksi"], errors="coerce")

        s = pd.Series(
            prediksi.values,
            index=temp["TANGGAL"],
            name=city_name
        )

        forecast_series.append(s)

        valid = pd.DataFrame({
            "Aktual": aktual,
            "Prediksi": prediksi
        }).dropna()

        if not valid.empty:
            error = valid["Aktual"] - valid["Prediksi"]

            mse = np.mean(error ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(error))

            nonzero = valid["Aktual"] != 0
            if nonzero.sum() > 0:
                mape = np.mean(
                    np.abs(error[nonzero] / valid.loc[nonzero, "Aktual"])
                ) * 100
            else:
                mape = np.nan

            denom = np.abs(valid["Aktual"]) + np.abs(valid["Prediksi"])
            smape_valid = denom != 0
            if smape_valid.sum() > 0:
                smape = np.mean(
                    2 * np.abs(error[smape_valid]) / denom[smape_valid]
                ) * 100
            else:
                smape = np.nan

            eval_rows.append({
                "Kabupaten/Kota": city_name,
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE (%)": mape,
                "SMAPE (%)": smape
            })

    lstm_forecast = pd.concat(forecast_series, axis=1)
    lstm_forecast.index = pd.to_datetime(lstm_forecast.index)
    lstm_forecast["tahun"] = lstm_forecast.index.year
    lstm_forecast["bulan"] = lstm_forecast.index.month

    lstm_eval = pd.DataFrame(eval_rows)

    return lstm_forecast, lstm_eval

@st.cache_data
def load_historical_ndvi():
    possible_files = [
        Path("ndvi17-24.xlsx"),
        Path("ndvi17-24.xls"),
        Path("ndvi17-24.csv"),
        Path("NDVI17-24.xlsx"),
        Path("NDVI17-24.xls"),
        Path("NDVI17-24.csv"),
    ]

    file_path = None

    for file in possible_files:
        if file.exists():
            file_path = file
            break

    if file_path is None:
        st.error(
            "File data historis NDVI tidak ditemukan. "
            "Pastikan file bernama ndvi17-24.xlsx / ndvi17-24.xls / ndvi17-24.csv "
            "berada di folder yang sama dengan file dashboard."
        )

        with st.expander("Cek file yang terbaca di folder dashboard"):
            st.write([str(p) for p in Path(".").glob("*")])

        st.stop()

    if file_path.suffix.lower() in [".xlsx", ".xls"]:
        hist = pd.read_excel(file_path)
    else:
        hist = pd.read_csv(file_path)

    date_col = None
    for col in hist.columns:
        if str(col).lower().strip() in ["date", "tanggal", "time", "periode"]:
            date_col = col
            break

    if date_col is None:
        date_col = hist.columns[0]

    hist[date_col] = pd.to_datetime(hist[date_col])
    hist = hist.set_index(date_col)
    hist = hist.sort_index()

    hist["tahun"] = hist.index.year
    hist["bulan"] = hist.index.month

    return hist

df = load_forecast_data()
df_cls = load_classification_data()
df_eval = load_eval_data()
df_hist = load_historical_ndvi()

# =========================
# DATA MODEL PREDIKSI
# =========================
forecast_data_map = {
    "GSTAR": df
}

forecast_eval_map = {
    "GSTAR": df_eval
}

# GSTAR-ARIMA
if Path("GSTAR_ARIMA_Forecast_NDVI.csv").exists() and Path("GSTAR_ARIMA_Evaluation_Metrics.csv").exists():
    df_gstar_arima = load_forecast_file("GSTAR_ARIMA_Forecast_NDVI.csv")
    df_eval_gstar_arima = load_eval_file("GSTAR_ARIMA_Evaluation_Metrics.csv")

    forecast_data_map["GSTAR-ARIMA"] = df_gstar_arima
    forecast_eval_map["GSTAR-ARIMA"] = df_eval_gstar_arima

# LSTM dari folder Prediksi_LSTM
df_lstm, df_eval_lstm = load_lstm_folder("Prediksi_LSTM")

if df_lstm is not None and df_eval_lstm is not None:
    forecast_data_map["LSTM"] = df_lstm
    forecast_eval_map["LSTM"] = df_eval_lstm

# df_gstar_arima = load_forecast_file("historis_plus_forecast_gstar_arima.csv")
# df_lstm = load_forecast_file("historis_plus_forecast_lstm.csv")
# df_eval_gstar_arima = pd.read_csv("evaluasi_gstar_arima_jatim.csv")
# df_eval_lstm = pd.read_csv("evaluasi_lstm_jatim.csv")

# =========================
# HELPER FUNCTIONS
# =========================
def classify_ndvi(ndvi):
    if pd.isna(ndvi):
        return "Tidak Diketahui"
    if ndvi <= 0.21:
        return "Kehijauan Sangat Rendah"
    elif ndvi <= 0.42:
        return "Kehijauan Rendah"
    elif ndvi <= 0.63:
        return "Kehijauan Sedang"
    else:
        return "Kehijauan Tinggi"

def risk_label(x):
    mapping = {
        0: "Kehijauan Sangat Rendah",
        1: "Kehijauan Rendah",
        2: "Kehijauan Sedang",
        3: "Kehijauan Tinggi"
    }
    try:
        return mapping.get(int(x), "Tidak Diketahui")
    except Exception:
        return "Tidak Diketahui"

def risk_level(label):
    mapping = {
        "Kehijauan Sangat Rendah": "Indikasi Risiko Sangat Tinggi",
        "Kehijauan Rendah": "Indikasi Risiko Tinggi",
        "Kehijauan Sedang": "Indikasi Risiko Sedang",
        "Kehijauan Tinggi": "Indikasi Risiko Rendah",
        "Tidak Diketahui": "Tidak Diketahui"
    }
    return mapping.get(label, "Tidak Diketahui")

def class_color(label):
    color_map = {
        "Kehijauan Sangat Rendah": "#6B3F16",  # coklat tua
        "Kehijauan Rendah": "#B7791F",         # coklat muda
        "Kehijauan Sedang": "#D9C75F",         # kuning kehijauan
        "Kehijauan Tinggi": "#2E7D32",         # hijau
        "Tidak Diketahui": "#94A3B8"           # abu-abu
    }
    return color_map.get(label, "#94A3B8")

def metric_card(title, value, help_text=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def small_metric_card(title, value, help_text=""):
    st.markdown(
        f"""
        <div style="
            background:#ffffff;
            padding:14px 16px;
            border-radius:16px;
            box-shadow:0 3px 12px rgba(15,23,42,0.06);
            border:1px solid #e5e7eb;
            min-height:92px;
            margin-bottom:14px;
        ">
            <div style="font-size:13px;color:#64748b;margin-bottom:6px;">{title}</div>
            <div style="font-size:24px;font-weight:800;color:#0f172a;">{value}</div>
            <div style="font-size:11px;color:#94a3b8;margin-top:5px;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def recommendation_by_class(kelas):
    if kelas == "Kehijauan Sangat Rendah":
        return "Prioritas pemantauan lapangan, antisipasi kekurangan air, dan koordinasi mitigasi wilayah."
    elif kelas == "Kehijauan Rendah":
        return "Perlu peningkatan monitoring vegetasi dan kesiapan intervensi apabila kondisi memburuk."
    elif kelas == "Kehijauan Sedang":
        return "Lakukan pemantauan rutin untuk menjaga kestabilan kondisi vegetasi."
    elif kelas == "Kehijauan Tinggi":
        return "Kondisi vegetasi relatif baik, monitoring berkala tetap diperlukan."
    else:
        return "Data belum cukup untuk diberikan rekomendasi."

def render_city_card(kabupaten, ndvi, kelas):
    warna = class_color(kelas)
    risiko = risk_level(kelas)

    st.markdown(
        f"""
        <div style="
            background: {warna};
            color: white;
            padding: 15px 16px;
            border-radius: 16px;
            min-height: 155px;
            margin-bottom: 12px;
            box-shadow: 0 4px 12px rgba(15,23,42,0.12);
        ">
            <div style="font-size:16px;font-weight:800;line-height:1.25;margin-bottom:10px;">
                {html.escape(str(kabupaten))}
            </div>
            <div style="font-size:13px;margin-bottom:8px;">{kelas}</div>
            <div style="font-size:13px;margin-bottom:8px;">{risiko}</div>
            <div style="font-size:18px;font-weight:800;">NDVI: {ndvi:.3f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-brand-row">
            <div class="sidebar-logo">🌾</div>
            <div>
                <div class="sidebar-title">Pemantauan NDVI</div>
                <div class="sidebar-title">Jawa Timur</div>
            </div>
        </div>
        <div class="sidebar-subtitle">
            Dashboard pemantauan kondisi vegetasi
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="sidebar-section-title">Navigasi</div>',
    unsafe_allow_html=True
)

nav_options = {
    "🏠 Beranda": "Beranda",
    "📈 Prediksi NDVI": "Prediksi NDVI",
    "🧩 Klasifikasi Risiko": "Klasifikasi Indikasi Risiko"
}

st.markdown(
    """
    <div style="
        background:#ffffff;
        border:1px solid #e5e7eb;
        border-radius:16px;
        padding:12px 14px;
        margin-bottom:16px;
        box-shadow:0 2px 8px rgba(15,23,42,0.04);
    ">
        <div style="font-size:13px;color:#64748b;font-weight:700;margin-bottom:6px;">
            Pilih Halaman Dashboard
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

selected_nav = st.radio(
    "Pilih Halaman Dashboard",
    list(nav_options.keys()),
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation"
)

menu = nav_options[selected_nav]

st.sidebar.markdown("---")

st.sidebar.markdown(
    '<div class="sidebar-section-title">Filter Data</div>',
    unsafe_allow_html=True
)

kabupaten_cols = [
    col for col in df.columns
    if col not in ["tahun", "bulan"]
]

if menu == "Prediksi NDVI":
    model_options_prediksi = list(forecast_data_map.keys())

    selected_forecast_model = st.sidebar.selectbox(
        "Model Prediksi",
        model_options_prediksi,
        index=0,
        key="prediksi_model_sidebar"
    )

    df_year_source = forecast_data_map[selected_forecast_model]

    year_options = sorted(
        [int(y) for y in df_year_source["tahun"].dropna().unique()],
        reverse=True
    )

    # Tahun prediksi otomatis mengikuti data model yang dipilih
    selected_year = year_options[0]

    st.sidebar.markdown(
        f"""
        <div style="
            background:#ffffff;
            border:1px solid #e5e7eb;
            border-radius:10px;
            padding:10px 12px;
            margin-bottom:10px;
            font-size:13px;
            color:#334155;
        ">
            <div style="font-size:11px;color:#64748b;font-weight:700;margin-bottom:4px;">
                Tahun Hasil Prediksi
            </div>
            <div style="font-size:18px;font-weight:800;color:#0f172a;">
                {selected_year}
            </div>
            <div style="font-size:11px;color:#64748b;margin-top:3px;">
                Otomatis mengikuti model {selected_forecast_model}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    city_options = [
        col for col in df_year_source.columns
        if col not in ["tahun", "bulan"]
    ]

else:
    selected_forecast_model = None

    year_options = sorted(
        [int(y) for y in df["tahun"].dropna().unique()],
        reverse=True
    )

    selected_year = st.sidebar.selectbox(
        "Tahun",
        year_options
    )

    city_options = kabupaten_cols

selected_city = st.sidebar.selectbox(
    "Kabupaten/Kota",
    ["Semua"] + sorted(city_options)
)

filtered = df[df["tahun"] == selected_year].copy()

# =========================
# HEADER
# =========================

# =========================
# PAGE 1: BERANDA
# =========================

st.markdown(
    f"""
    <div class="dashboard-hero">
        <div class="hero-badge">🌾 Pemantauan NDVI Jawa Timur</div>
        <h1 class="dashboard-title">{DASHBOARD_TITLE}</h1>
        <div class="dashboard-subtitle">
            Dashboard pemantauan kondisi vegetasi dan indikasi risiko kekeringan berbasis NDVI.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if menu == "Beranda":

    if selected_city == "Semua":
        display_cols = kabupaten_cols
    else:
        display_cols = [selected_city]

    eda_df = pd.DataFrame({
        "Kabupaten/Kota": display_cols,
        "NDVI": [
            pd.to_numeric(filtered[col], errors="coerce").mean()
            for col in display_cols
        ]
    })

    eda_df["Kelas"] = eda_df["NDVI"].apply(classify_ndvi)
    eda_df["Risiko"] = eda_df["Kelas"].apply(risk_level)
    eda_df["Warna"] = eda_df["Kelas"].apply(class_color)

    total_wilayah = eda_df["Kabupaten/Kota"].nunique()
    rata_ndvi = eda_df["NDVI"].mean()
    ndvi_min = eda_df["NDVI"].min()
    ndvi_max = eda_df["NDVI"].max()
    wilayah_prioritas = (eda_df["Kelas"] == "Kehijauan Sangat Rendah").sum()

    jumlah_risiko_sangat_tinggi = (
        eda_df["Risiko"] == "Indikasi Risiko Sangat Tinggi"
    ).sum()

    kondisi_umum_kelas = classify_ndvi(rata_ndvi)
    kondisi_umum_risiko = risk_level(kondisi_umum_kelas).replace("Indikasi ", "")

    risk_card_color_map = {
        "Risiko Sangat Tinggi": ("#6B3F16", "#FFFFFF", "rgba(255,255,255,0.78)"),
        "Risiko Tinggi": ("#B7791F", "#FFFFFF", "rgba(255,255,255,0.82)"),
        "Risiko Sedang": ("#D9C75F", "#1F2937", "#475569"),
        "Risiko Rendah": ("#2E7D32", "#FFFFFF", "rgba(255,255,255,0.82)")
    }

    card_bg, card_text, card_desc = risk_card_color_map.get(
        kondisi_umum_risiko,
        ("#ffffff", "#0f172a", "#0f172a")
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Indikasi Risiko Kekeringan Sangat Tinggi",
            f"{jumlah_risiko_sangat_tinggi}",
            "Wilayah prioritas utama pemantauan"
        )

    with c2:
        metric_card(
            "Rata-rata NDVI",
            f"{rata_ndvi:.3f}",
            "Indikator kondisi vegetasi wilayah"
        )

    with c3:
        metric_card(
            "Rentang NDVI",
            f"{ndvi_min:.3f}–{ndvi_max:.3f}",
            "Nilai minimum–maksimum wilayah"
        )

    with c4:
        st.html(
            f"""
            <div style="
                background:#ffffff;
                border:1px solid #dbe2ea;
                border-radius:18px;
                padding:18px 22px;
                height:auto;
                box-sizing:border-box;
                display:flex;
                flex-direction:column;
                justify-content:space-between;
                box-shadow:0 2px 8px rgba(15,23,42,0.04);
            ">
                <div style="
                    font-size:22px;
                    color:#0f172a;
                    font-weight:700;
                    margin-bottom:6px;
                    line-height:1.15;
                ">
                    Kondisi Umum Tahun {selected_year}
                </div>

                <div style="
                    display:inline-flex;
                    align-items:center;
                    justify-content:center;
                    width:fit-content;
                    min-width:150px;
                    background:{card_bg};
                    color:{card_text};
                    padding:8px 16px;
                    border-radius:999px;
                    font-size:30px;
                    font-weight:800;
                    line-height:1.1;
                    box-sizing:border-box;
                ">
                    {kondisi_umum_risiko}
                </div>

                <div style="
                    font-size:17px;
                    color:#0f172a;
                    line-height:1.4;
                    margin-top:6px;
                ">
                    Berdasarkan rata-rata NDVI wilayah
                </div>
            </div>
            """
        )

    st.markdown(
        """
        <div style="height:16px;"></div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Lihat Dasar Klasifikasi Risiko Kekeringan Berbasis NDVI"):
        klasifikasi_df = pd.DataFrame({
            "Nilai NDVI": [
                "NDVI ≤ 0,21",
                "0,21 < NDVI ≤ 0,42",
                "0,42 < NDVI ≤ 0,63",
                "NDVI > 0,63"
            ],
            "Interpretasi Kondisi Vegetasi": [
                "Sinyal vegetasi sangat rendah / non-vegetasi",
                "Vegetasi tidak rapat",
                "Vegetasi cukup rapat",
                "Vegetasi rapat"
            ],
            "Indikasi Risiko Kekeringan": [
                "Sangat Tinggi",
                "Tinggi",
                "Sedang",
                "Rendah"
            ]
        })

        def style_risiko(row):
            risiko = row["Indikasi Risiko Kekeringan"]

            color_map = {
                "Sangat Tinggi": ("#6B3F16", "#FFFFFF"),
                "Tinggi": ("#B7791F", "#FFFFFF"),
                "Sedang": ("#D9C75F", "#1F2937"),
                "Rendah": ("#2E7D32", "#FFFFFF")
            }

            bg_color, text_color = color_map.get(risiko, ("#FFFFFF", "#0F172A"))

            return [
                f"background-color:{bg_color}; color:{text_color}; font-weight:800;"
                if col == "Indikasi Risiko Kekeringan"
                else ""
                for col in row.index
            ]

        risk_color_map = {
            "Sangat Tinggi": ("#6B3F16", "#FFFFFF"),
            "Tinggi": ("#B7791F", "#FFFFFF"),
            "Sedang": ("#D9C75F", "#1F2937"),
            "Rendah": ("#2E7D32", "#FFFFFF")
        }

        rows_html = ""

        for _, row in klasifikasi_df.iterrows():
            risiko = row["Indikasi Risiko Kekeringan"]
            bg_color, text_color = risk_color_map.get(risiko, ("#FFFFFF", "#0F172A"))

            rows_html += f"""
            <tr>
                <td style="
                    padding:10px 12px;
                    border-bottom:1px solid #e5e7eb;
                    line-height:1.45;
                ">
                    {row["Nilai NDVI"]}
                </td>

                <td style="
                    padding:10px 12px;
                    border-bottom:1px solid #e5e7eb;
                    line-height:1.45;
                ">
                    {row["Interpretasi Kondisi Vegetasi"]}
                </td>

                <td style="
                    padding:10px 12px;
                    border-bottom:1px solid #e5e7eb;
                    line-height:1.45;
                    background:{bg_color};
                    color:{text_color};
                    font-weight:500;
                ">
                    {risiko}
                </td>
            </tr>
            """

        st.html(
            f"""
            <div style="
                border:1px solid #dbe2ea;
                border-radius:10px;
                overflow:hidden;
                background:#ffffff;
            ">
                <table style="
                    width:100%;
                    border-collapse:collapse;
                    font-size:14px;
                    color:#0f172a;
                ">
                    <thead>
                        <tr style="
                            background:#f1f5f9;
                            color:#0f172a;
                            font-weight:800;
                        ">
                            <th style="text-align:left; padding:12px; font-size:16px; font-weight:600; border-bottom:1px solid #cbd5e1;">
                                Nilai NDVI
                            </th>
                            <th style="text-align:left; padding:12px; font-size:16px; font-weight:600; border-bottom:1px solid #cbd5e1;">
                                Interpretasi Kondisi Vegetasi
                            </th>
                            <th style="text-align:left; padding:12px; font-size:16px; font-weight:600; border-bottom:1px solid #cbd5e1;">
                                Indikasi Risiko Kekeringan
                            </th>
                        </tr>
                    </thead>

                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            """
        )

    eda_df["Prioritas"] = (1 - eda_df["NDVI"]).clip(lower=0.001)

    wilayah_prioritas = eda_df.sort_values("Prioritas", ascending=False).iloc[0]
    wilayah_terbaik = eda_df.sort_values("NDVI", ascending=False).iloc[0]
    kelas_dominan = eda_df["Kelas"].value_counts().idxmax()

    st.markdown("")

    CARD_HEIGHT = 530
    CHART_HEIGHT = 410

    left, middle, factor_col = st.columns(3, gap="medium")

    with left:
        with st.container(border=True):
            st.markdown(
                """
                <div class="chart-card-title">Distribusi Indikasi Risiko Kekeringan</div>
                <div class="chart-card-caption">
                    Menunjukkan komposisi wilayah berdasarkan indikasi risiko kekeringan yang diturunkan dari kelas NDVI
                </div>
                """,
                unsafe_allow_html=True
            )

            eda_df["Risiko_Label"] = (
                eda_df["Risiko"]
                .str.replace("Indikasi Risiko ", "", regex=False)
            )

            dist = (
                eda_df["Risiko_Label"]
                .value_counts()
                .rename_axis("Risiko")
                .reset_index(name="Jumlah")
            )

            risk_order = [
                "Sangat Tinggi",
                "Tinggi",
                "Sedang",
                "Rendah"
            ]

            risk_color_map = {
                "Sangat Tinggi": "#6B3F16",
                "Tinggi": "#B7791F",
                "Sedang": "#D9C75F",
                "Rendah": "#2E7D32"
            }

            fig_pie = px.pie(
                dist,
                names="Risiko",
                values="Jumlah",
                hole=0.45,
                color="Risiko",
                category_orders={"Risiko": risk_order},
                color_discrete_map=risk_color_map
            )

            pie_text_color_map = {
                "Sangat Tinggi": "#ffffff",
                "Tinggi": "#ffffff",
                "Sedang": "#0f172a",
                "Rendah": "#ffffff"
            }

            pie_text_colors = [
                pie_text_color_map.get(label, "#0f172a")
                for label in fig_pie.data[0].labels
            ]

            fig_pie.update_traces(
                textinfo="percent",
                textfont=dict(
                    size=14,
                    color=pie_text_colors
                )
            )

            fig_pie.update_layout(
                height=410,
                margin=dict(l=5, r=5, t=5, b=5),
                legend_title_text="Indikasi Risiko Kekeringan",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True,
                theme=None,
                config={"displayModeBar": False},
                key="pie_distribusi_kehijauan"
            )

    with middle:
        with st.container(border=True):
            st.markdown(
                """
                <div class="chart-card-title">Wilayah dengan NDVI Terendah</div>
                <div class="chart-card-caption">
                    Menampilkan wilayah dengan nilai NDVI paling rendah pada tahun terpilih
                </div>
                """,
                unsafe_allow_html=True
            )

            top_low = eda_df.sort_values("NDVI", ascending=True).head(8)

            fig_bar = go.Figure()
            fig_bar.add_trace(
                go.Bar(
                    x=top_low["NDVI"],
                    y=top_low["Kabupaten/Kota"],
                    orientation="h",
                    marker=dict(color=top_low["Warna"]),
                    text=[f"{v:.3f}" for v in top_low["NDVI"]],
                    textposition="outside"
                )
            )

            fig_bar.update_layout(
                height=410,
                xaxis_title="Nilai NDVI",
                yaxis_title="",
                showlegend=False,
                margin=dict(l=115, r=20, t=5, b=45),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    range=[-0.02, top_low["NDVI"].max() * 1.15],
                    zeroline=True,
                    zerolinecolor="#cbd5e1",
                    gridcolor="#e5e7eb"
                ),
                yaxis=dict(
                    automargin=True,
                    tickfont=dict(size=12)
                )
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True,
                theme=None,
                config={"displayModeBar": False},
                key="bar_ndvi_terendah"
            )

    with factor_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="chart-card-title">Faktor yang Berkaitan dengan NDVI</div>
                <div class="chart-card-caption">
                    Ringkasan hasil analisis faktor untuk mengetahui kelompok variabel yang berkaitan dengan variasi nilai NDVI
                </div>
                """,
                unsafe_allow_html=True
            )

            st.html(
                f"""
                <div style="
                    height:{CHART_HEIGHT}px;
                    box-sizing:border-box;
                    overflow:auto;
                    padding-right:6px;
                    color:#1e293b;
                    font-family:Arial, sans-serif;
                ">

                    <div style="
                        display:grid;
                        grid-template-columns:1fr 1fr;
                        gap:10px;
                        margin-bottom:12px;
                    ">
                        <div style="
                            background:#ecfdf5;
                            border:1px solid #bbf7d0;
                            border-radius:14px;
                            padding:12px 14px;
                        ">
                            <div style="font-size:12px;color:#166534;font-weight:700;margin-bottom:6px;">
                                Faktor Utama
                            </div>
                            <div style="font-size:26px;font-weight:850;color:#14532d;">
                                5
                            </div>
                            <div style="font-size:12px;color:#166534;margin-top:4px;">
                                Hasil EFA
                            </div>
                        </div>

                        <div style="
                            background:#eff6ff;
                            border:1px solid #bfdbfe;
                            border-radius:14px;
                            padding:12px 14px;
                        ">
                            <div style="font-size:12px;color:#1d4ed8;font-weight:700;margin-bottom:6px;">
                                Model Terbaik
                            </div>
                            <div style="font-size:20px;font-weight:850;color:#1e3a8a;">
                                Random Forest Regression
                            </div>
                            <div style="font-size:12px;color:#1e40af;margin-top:4px;">
                                R² = 0.7078
                            </div>
                        </div>
                    </div>

                    <div style="
                        background:#ffffff;
                        border:1px solid #e5e7eb;
                        border-left:6px solid #2E7D32;
                        border-radius:14px;
                        padding:13px 15px;
                        margin-bottom:10px;
                    ">
                        <div style="font-size:13px;color:#64748b;font-weight:700;margin-bottom:5px;">
                            Faktor 1
                        </div>
                        <div style="font-size:16px;color:#0f172a;font-weight:800;margin-bottom:4px;">
                            Kondisi Iklim Kering-Basah
                        </div>
                        <div style="font-size:13px;color:#475569;line-height:1.55;">
                            Variabel dominan: <b>Kelembapan Udara Rata-rata, Lama Penyinaran Matahari, Curah Hujan</b>
                        </div>
                    </div>

                    <div style="
                        background:#ffffff;
                        border:1px solid #e5e7eb;
                        border-left:6px solid #B7791F;
                        border-radius:14px;
                        padding:13px 15px;
                        margin-bottom:10px;
                    ">
                        <div style="font-size:13px;color:#64748b;font-weight:700;margin-bottom:5px;">
                            Faktor 2
                        </div>
                        <div style="font-size:16px;color:#0f172a;font-weight:800;margin-bottom:4px;">
                            Kecepatan Angin
                        </div>
                        <div style="font-size:13px;color:#475569;line-height:1.55;">
                            Variabel dominan: <b>Kecepatan Angin Maksimum, Kecepatan Angin Rata-rata</b>
                        </div>
                    </div>

                    <div style="
                        background:#ffffff;
                        border:1px solid #e5e7eb;
                        border-left:6px solid #D9C75F;
                        border-radius:14px;
                        padding:13px 15px;
                        margin-bottom:10px;
                    ">
                        <div style="font-size:13px;color:#64748b;font-weight:700;margin-bottom:5px;">
                            Faktor 3
                        </div>
                        <div style="font-size:16px;color:#0f172a;font-weight:800;margin-bottom:4px;">
                            Temperatur Udara
                        </div>
                        <div style="font-size:13px;color:#475569;line-height:1.55;">
                            Variabel dominan: <b>Suhu Minimum, Suhu Maksimum</b>
                        </div>
                    </div>

                    <div style="
                        background:#ffffff;
                        border:1px solid #e5e7eb;
                        border-left:6px solid #2563eb;
                        border-radius:14px;
                        padding:13px 15px;
                        margin-bottom:10px;
                    ">
                        <div style="font-size:13px;color:#64748b;font-weight:700;margin-bottom:5px;">
                            Faktor 4
                        </div>
                        <div style="font-size:16px;color:#0f172a;font-weight:800;margin-bottom:4px;">
                            Penggunaan Lahan dan Kependudukan
                        </div>
                        <div style="font-size:13px;color:#475569;line-height:1.55;">
                            Variabel dominan: <b>Luas Lahan Irigasi, Luas Hutan, Jumlah Penduduk Laki-laki</b>
                        </div>
                    </div>

                    <div style="
                        background:#ffffff;
                        border:1px solid #e5e7eb;
                        border-left:6px solid #64748b;
                        border-radius:14px;
                        padding:13px 15px;
                    ">
                        <div style="font-size:13px;color:#64748b;font-weight:700;margin-bottom:5px;">
                            Faktor 5
                        </div>
                        <div style="font-size:16px;color:#0f172a;font-weight:800;margin-bottom:4px;">
                            Arah Angin dan Temporal
                        </div>
                        <div style="font-size:13px;color:#475569;line-height:1.55;">
                            Variabel dominan: <b>Komponen Arah Angin, Kondisi Angin Tenang, Tahun</b>
                        </div>
                    </div>

                </div>
                """
            )

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="chart-card-title">Catatan Interpretasi</div>
            <div class="chart-card-caption">
                Panduan membaca dashboard indikasi risiko kekeringan berbasis NDVI tahun {selected_year}
            </div>
            """,
            unsafe_allow_html=True
        )

        note1, note2, note3, note4 = st.columns(4, gap="medium")

        with note1:
            st.html(
                """
                <div style="
                    background:#f8fafc;
                    border:1px solid #dbe2ea;
                    border-radius:14px;
                    padding:14px 15px;
                    min-height:135px;
                ">
                    <div style="font-size:12px;color:#334155;font-weight:800;margin-bottom:6px;">
                        About Dashboard?
                    </div>
                    <div style="font-size:14px;line-height:1.6;color:#334155;">
                        Dashboard ini menyajikan pemantauan kondisi vegetasi wilayah Jawa Timur
                        berdasarkan nilai NDVI. Nilai NDVI digunakan sebagai indikator tidak langsung
                        untuk mengidentifikasi wilayah yang berpotensi memerlukan perhatian lebih
                        terhadap indikasi risiko kekeringan
                    </div>
                </div>
                """
            )

        with note2:
            st.html(
                """
                <div style="
                    background:#eff6ff;
                    border:1px solid #bfdbfe;
                    border-radius:14px;
                    padding:14px 15px;
                    min-height:135px;
                ">
                    <div style="font-size:12px;color:#1d4ed8;font-weight:800;margin-bottom:6px;">
                        Home Page
                    </div>
                    <div style="font-size:14px;line-height:1.6;color:#1e3a8a;">
                        Donut chart menunjukkan komposisi kelas risiko. Bar chart menunjukkan wilayah
                        dengan NDVI terendah. Treemap memperlihatkan skala prioritas pemantauan antarwilayah.
                        Semakin rendah NDVI, semakin tinggi prioritas pemantauan
                    </div>
                </div>
                """
            )

        with note3:
            st.html(
                """
                <div style="
                    background:#fff7ed;
                    border:1px solid #fed7aa;
                    border-radius:14px;
                    padding:14px 15px;
                    min-height:135px;
                ">
                    <div style="font-size:12px;color:#9a3412;font-weight:800;margin-bottom:6px;">
                        Hal yang Perlu Dipahami
                    </div>
                    <div style="font-size:14px;line-height:1.6;color:#7c2d12;">
                        Kategori risiko pada dashboard ini bersifat indikatif. Artinya, wilayah dengan
                        risiko tinggi tidak langsung berarti pasti mengalami kekeringan, tetapi menunjukkan
                        kondisi vegetasi yang perlu diperhatikan lebih lanjut
                    </div>
                </div>
                """
            )

        with note4:
            st.html(
                """
                <div style="
                    background:#ecfdf5;
                    border:1px solid #bbf7d0;
                    border-radius:14px;
                    padding:14px 15px;
                    min-height:135px;
                ">
                    <div style="font-size:12px;color:#166534;font-weight:800;margin-bottom:6px;">
                        Rekomendasi Pemanfaatan
                    </div>
                    <div style="font-size:14px;line-height:1.6;color:#14532d;">
                        Hasil dashboard dapat digunakan sebagai dasar awal untuk menentukan wilayah
                        prioritas pemantauan. Interpretasi sebaiknya tetap dibandingkan dengan data
                        pendukung seperti curah hujan, kelembapan, irigasi, dan kondisi lapangan
                    </div>
                </div>
                """
            )

    with st.container(border=True):
        st.markdown(
            """
            <div class="chart-card-title">Treemap Prioritas Pemantauan per Kabupaten/Kota</div>
            <div class="chart-card-caption">
                Ukuran kotak menunjukkan skor prioritas pemantauan, dihitung dari 1 - NDVI.
                Semakin rendah NDVI, semakin besar ukuran kotak. Warna menunjukkan kelas kehijauan wilayah.
            </div>
            """,
            unsafe_allow_html=True
        )

        treemap_df = eda_df.copy()

        treemap_df["NDVI"] = pd.to_numeric(treemap_df["NDVI"], errors="coerce")
        treemap_df = treemap_df.dropna(subset=["NDVI"])

        treemap_df["Prioritas"] = (1 - treemap_df["NDVI"]).clip(lower=0.001)

        fig_treemap = px.treemap(
            treemap_df,
            path=[px.Constant("Jawa Timur"), "Kabupaten/Kota"],
            values="Prioritas",
            color="Kelas",
            color_discrete_map={
                "Kehijauan Sangat Rendah": "#6B3F16",
                "Kehijauan Rendah": "#B7791F",
                "Kehijauan Sedang": "#D9C75F",
                "Kehijauan Tinggi": "#2E7D32",
                "Tidak Diketahui": "#94A3B8"
            },
            hover_data={
                "NDVI": ":.3f",
                "Risiko": True,
                "Prioritas": ":.3f"
            }
        )

        fig_treemap.update_traces(
            texttemplate="<b>%{label}</b><br>Prioritas: %{value:.3f}",
            textposition="middle center",
            textfont_size=16
        )

        label_to_class = dict(
            zip(treemap_df["Kabupaten/Kota"], treemap_df["Kelas"])
        )

        text_colors = []

        for label in fig_treemap.data[0].labels:
            kelas = label_to_class.get(label, "")

            if kelas in [
                "Kehijauan Sangat Rendah",
                "Kehijauan Rendah",
                "Kehijauan Tinggi"
            ]:
                text_colors.append("#ffffff")
            else:
                text_colors.append("#0f172a")

        fig_treemap.update_traces(
            textfont=dict(
                size=15,
                color=text_colors
            )
        )
        fig_treemap.update_layout(
            height=480,
            margin=dict(l=5, r=5, t=5, b=5),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig_treemap,
            use_container_width=True,
            theme=None,
            config={"displayModeBar": False},
            key="treemap_kondisi_wilayah"
        )

    with st.expander("Lihat Urutan Prioritas Pemantauan Wilayah"):
        detail_prioritas = eda_df.copy()

        detail_prioritas["Skor Prioritas"] = (
            1 - detail_prioritas["NDVI"]
        ).clip(lower=0.001)

        detail_prioritas = detail_prioritas.sort_values(
            "NDVI",
            ascending=True
        ).reset_index(drop=True)

        detail_prioritas["Peringkat"] = detail_prioritas.index + 1

        detail_prioritas["Indikasi Risiko Kekeringan"] = (
            detail_prioritas["Risiko"]
            .str.replace("Indikasi Risiko ", "", regex=False)
        )

        detail_prioritas["NDVI"] = detail_prioritas["NDVI"].round(3)
        detail_prioritas["Skor Prioritas"] = detail_prioritas["Skor Prioritas"].round(3)

        tabel_prioritas = detail_prioritas[
            [
                "Peringkat",
                "Kabupaten/Kota",
                "NDVI",
                "Indikasi Risiko Kekeringan",
                "Skor Prioritas"
            ]
        ].copy()

        styled_tabel_prioritas = (
            tabel_prioritas
            .style
            .set_properties(**{
                "font-size": "15px",
                "color": "#0f172a"
            })
            .set_table_styles([
                {
                    "selector": "thead th",
                    "props": [
                        ("font-size", "15px"),
                        ("font-weight", "800"),
                        ("color", "#0f172a"),
                        ("background-color", "#f8fafc")
                    ]
                }
            ])
        )

        rows_prioritas_html = ""

        for _, row in detail_prioritas.iterrows():
            rows_prioritas_html += f"""
            <tr>
                <td>{row["Peringkat"]}</td>
                <td>{html.escape(str(row["Kabupaten/Kota"]))}</td>
                <td>{row["NDVI"]:.3f}</td>
                <td>{row["Indikasi Risiko Kekeringan"]}</td>
                <td>{row["Skor Prioritas"]:.3f}</td>
            </tr>
            """

        st.html(
            f"""
            <div style="
                border:1px solid #dbe2ea;
                border-radius:10px;
                overflow:hidden;
                background:#ffffff;
            ">
                <table style="
                    width:100%;
                    border-collapse:collapse;
                    color:#0f172a;
                ">
                    <thead>
                        <tr style="background:#f1f5f9;">
                            <th style="text-align:left; padding:12px; border-bottom:1px solid #cbd5e1; font-size:16px; font-weight:700;">
                                Peringkat
                            </th>
                            <th style="text-align:left; padding:12px; border-bottom:1px solid #cbd5e1; font-size:16px; font-weight:700;">
                                Kabupaten/Kota
                            </th>
                            <th style="text-align:left; padding:12px; border-bottom:1px solid #cbd5e1; font-size:16px; font-weight:700;">
                                NDVI
                            </th>
                            <th style="text-align:left; padding:12px; border-bottom:1px solid #cbd5e1; font-size:16px; font-weight:700;">
                                Indikasi Risiko Kekeringan
                            </th>
                            <th style="text-align:left; padding:12px; border-bottom:1px solid #cbd5e1; font-size:16px; font-weight:700;">
                                Skor Prioritas
                            </th>
                        </tr>
                    </thead>

                    <tbody>
                        {rows_prioritas_html}
                    </tbody>
                </table>
            </div>

            <style>
                tbody td {{
                    padding:10px 12px;
                    border-bottom:1px solid #e5e7eb;
                    font-size:16px;
                    font-weight:400;
                    line-height:1.45;
                }}
            </style>
            """
        )

# =========================
# PAGE 2: PREDIKSI NDVI
# =========================
elif menu == "Prediksi NDVI":

    # =========================
    # DROPDOWN MODEL PREDIKSI
    # =========================

    df_pred = forecast_data_map[selected_forecast_model]
    df_eval_model = forecast_eval_map[selected_forecast_model]

    pred_kabupaten_cols = [
        col for col in df_pred.columns
        if col not in ["tahun", "bulan"]
    ]

    city_for_pred = selected_city

    if selected_city != "Semua" and selected_city not in pred_kabupaten_cols:
        st.warning(
            f"Wilayah {selected_city} tidak tersedia pada data model {selected_forecast_model}. "
            "Dashboard menampilkan rata-rata seluruh wilayah."
        )
        city_for_pred = "Semua"
        
    filtered_pred = df_pred[df_pred["tahun"] == selected_year].copy()

    if filtered_pred.empty:
        st.warning("Tidak ada data prediksi untuk filter tahun yang dipilih.")
    else:

        # =========================
        # METRIK EVALUASI MODEL
        # =========================
        if city_for_pred == "Semua":
            rmse_val = df_eval_model["RMSE"].mean()
            mse_val = df_eval_model["MSE"].mean()
            mae_val = df_eval_model["MAE"].mean()
        else:
            row_eval = df_eval_model[df_eval_model["Kabupaten/Kota"] == city_for_pred]

            if not row_eval.empty:
                rmse_val = row_eval["RMSE"].iloc[0]
                mse_val = row_eval["MSE"].iloc[0]
                mae_val = row_eval["MAE"].iloc[0]
            else:
                rmse_val, mse_val, mae_val = np.nan, np.nan, np.nan

        # =========================
        # DATA UNTUK LINE CHART
        # =========================

        # Data prediksi tahun terpilih, misalnya 2025
        if city_for_pred == "Semua":
            y_data = filtered_pred[pred_kabupaten_cols].apply(
                pd.to_numeric,
                errors="coerce"
            ).mean(axis=1)
            chart_title = "Rata-rata NDVI Jawa Timur"
        else:
            y_data = pd.to_numeric(
                filtered_pred[city_for_pred],
                errors="coerce"
            )
            chart_title = city_for_pred

        y_data = y_data.dropna()
        y_data = y_data.groupby(y_data.index).mean().sort_index()


        # =========================
        # DATA HISTORIS 2017–SEBELUM TAHUN PREDIKSI
        # =========================
        hist_df = df_hist[
            df_hist["tahun"] < selected_year
        ].copy()

        hist_value_cols = [
            col for col in hist_df.columns
            if col not in ["tahun", "bulan"]
        ]

        if city_for_pred == "Semua":
            y_hist = hist_df[hist_value_cols].apply(
                pd.to_numeric,
                errors="coerce"
            ).mean(axis=1)

        else:
            # Cocokkan nama wilayah dengan lebih fleksibel
            hist_col_map = {
                " ".join(str(col).lower().strip().split()): col
                for col in hist_value_cols
            }

            city_key = " ".join(str(city_for_pred).lower().strip().split())

            if city_key in hist_col_map:
                y_hist = pd.to_numeric(
                    hist_df[hist_col_map[city_key]],
                    errors="coerce"
                )
            else:
                y_hist = pd.Series(dtype=float)

        y_hist = y_hist.dropna()
        y_hist = y_hist.groupby(y_hist.index).mean().sort_index()

        if y_data.empty:
            st.warning("Tidak ada data NDVI untuk wilayah yang dipilih.")
        else:

            # =========================
            # HITUNG ARAH TREN
            # =========================
            first_ndvi = y_data.iloc[0]
            last_ndvi = y_data.iloc[-1]
            delta_ndvi = last_ndvi - first_ndvi

            if abs(delta_ndvi) < 0.01:
                trend_status = "Stabil"
                trend_note = "Perubahan NDVI relatif kecil"
                trend_color = "#64748b"
            elif delta_ndvi < 0:
                trend_status = "Menurun"
                trend_note = "Vegetasi cenderung melemah"
                trend_color = "#B7791F"
            else:
                trend_status = "Meningkat"
                trend_note = "Vegetasi cenderung membaik"
                trend_color = "#2E7D32"

            # =========================
            # CARD ATAS
            # =========================
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="medium")

            with c1:
                metric_card(
                    "Rata-rata Error",
                    f"{mae_val:.4f}" if not pd.isna(mae_val) else "-",
                    "Rata-rata selisih prediksi terhadap nilai aktual"
                )

            with c2:
                metric_card(
                    "Error Model",
                    f"{rmse_val:.4f}" if not pd.isna(rmse_val) else "-",
                    "Semakin kecil nilai error, semakin baik prediksi"
                )

            with c3:
                metric_card(
                    "Rata-rata NDVI Prediksi",
                    f"{y_data.mean():.3f}",
                    f"Berdasarkan model {selected_forecast_model}"
                )

            with c4:
                first_ndvi = y_data.iloc[0]
                last_ndvi = y_data.iloc[-1]
                delta_ndvi = last_ndvi - first_ndvi

                if abs(delta_ndvi) < 0.01:
                    trend_status = "Stabil"
                    trend_note = "Perubahan NDVI relatif kecil"
                    trend_color = "#64748b"
                elif delta_ndvi < 0:
                    trend_status = "Menurun"
                    trend_note = "Vegetasi cenderung melemah"
                    trend_color = "#B7791F"
                else:
                    trend_status = "Meningkat"
                    trend_note = "Vegetasi cenderung membaik"
                    trend_color = "#2E7D32"

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Arah Tren NDVI</div>
                    <div class="metric-value" style="color:{trend_color};">{trend_status}</div>
                    <div style="font-size:13px; color:#64748b;">Perubahan: {delta_ndvi:+.3f}</div>
                    <div class="metric-desc">{trend_note}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            # =========================
            # DATA BAR CHART
            # =========================
            pred_rank = pd.DataFrame({
                "Kabupaten/Kota": pred_kabupaten_cols,
                "Rata-rata NDVI Prediksi": [
                    pd.to_numeric(filtered_pred[col], errors="coerce").mean()
                    for col in pred_kabupaten_cols
                ]
            })

            pred_rank["Kelas"] = pred_rank["Rata-rata NDVI Prediksi"].apply(classify_ndvi)
            pred_rank["Risiko"] = pred_rank["Kelas"].apply(risk_level)
            pred_rank["Warna"] = pred_rank["Kelas"].apply(class_color)

            top_pred_low = pred_rank.sort_values(
                "Rata-rata NDVI Prediksi",
                ascending=True
            ).head(10)

            # =========================
            # ANALISIS PERGESERAN HISTORIS KE PREDIKSI
            # =========================
            hist_compare_start = selected_year - 3

            hist_compare_df = df_hist[
                (df_hist["tahun"] >= hist_compare_start) &
                (df_hist["tahun"] < selected_year)
            ].copy()


            def normalize_wilayah_name(name):
                name = str(name).lower().strip()
                name = name.replace("kabupaten ", "")
                name = name.replace("kab. ", "")
                name = name.replace("kab ", "")
                name = name.replace("kota ", "")
                name = name.replace(".", "")
                name = name.replace("-", " ")
                name = name.replace("_", " ")
                name = " ".join(name.split())
                return name


            def kondisi_kelompok(kelas):
                if kelas in ["Kehijauan Sangat Rendah", "Kehijauan Rendah"]:
                    return "Rendah"
                elif kelas in ["Kehijauan Sedang", "Kehijauan Tinggi"]:
                    return "Baik"
                else:
                    return "Tidak Diketahui"


            # Deteksi apakah data historis bentuknya long format:
            # kolom wilayah + kolom NDVI
            area_col = None
            for col in hist_compare_df.columns:
                if str(col).lower().strip() in [
                    "kabupaten/kota",
                    "kabupaten kota",
                    "wilayah",
                    "kabupaten",
                    "kota"
                ]:
                    area_col = col
                    break

            ndvi_col = None
            for col in hist_compare_df.columns:
                if str(col).upper().strip() == "NDVI":
                    ndvi_col = col
                    break

            hist_is_long = area_col is not None and ndvi_col is not None


            # Kalau bukan long format, berarti dianggap wide format:
            # tiap wilayah menjadi kolom sendiri
            hist_value_cols = [
                col for col in hist_compare_df.columns
                if col not in ["tahun", "bulan"]
            ]

            hist_col_map = {
                normalize_wilayah_name(col): col
                for col in hist_value_cols
            }


            shift_rows = []

            if city_for_pred == "Semua":
                shift_cols = pred_kabupaten_cols
            else:
                shift_cols = [city_for_pred]

            for kota in shift_cols:
                if kota not in filtered_pred.columns:
                    continue

                # Ambil data historis
                if hist_is_long:
                    kota_key = normalize_wilayah_name(kota)

                    hist_temp = hist_compare_df[
                        hist_compare_df[area_col].apply(normalize_wilayah_name) == kota_key
                    ].copy()

                    hist_series = pd.to_numeric(
                        hist_temp[ndvi_col],
                        errors="coerce"
                    ).dropna()

                else:
                    kota_key = normalize_wilayah_name(kota)
                    hist_col = hist_col_map.get(kota_key, None)

                    if hist_col is None:
                        continue

                    hist_series = pd.to_numeric(
                        hist_compare_df[hist_col],
                        errors="coerce"
                    ).dropna()

                # Ambil data prediksi
                pred_series = pd.to_numeric(
                    filtered_pred[kota],
                    errors="coerce"
                ).dropna()

                if hist_series.empty or pred_series.empty:
                    continue

                ndvi_hist = hist_series.mean()
                ndvi_pred = pred_series.mean()
                perubahan = ndvi_pred - ndvi_hist

                kelas_hist = classify_ndvi(ndvi_hist)
                kelas_pred = classify_ndvi(ndvi_pred)

                kelompok_hist = kondisi_kelompok(kelas_hist)
                kelompok_pred = kondisi_kelompok(kelas_pred)

                if kelompok_hist == "Baik" and kelompok_pred == "Rendah":
                    status_pergeseran = "Waspada Baru"
                elif kelompok_hist == "Rendah" and kelompok_pred == "Rendah":
                    status_pergeseran = "Konsisten Prioritas"
                elif kelompok_hist == "Rendah" and kelompok_pred == "Baik":
                    status_pergeseran = "Mulai Membaik"
                elif kelompok_hist == "Baik" and kelompok_pred == "Baik":
                    status_pergeseran = "Relatif Stabil"
                else:
                    status_pergeseran = "Tidak Diketahui"

                shift_rows.append({
                    "Kabupaten/Kota": kota,
                    "NDVI Historis": ndvi_hist,
                    "Kelas Historis": kelas_hist,
                    "NDVI Prediksi": ndvi_pred,
                    "Kelas Prediksi": kelas_pred,
                    "Perubahan NDVI": perubahan,
                    "Status Pergeseran": status_pergeseran
                })

            shift_df = pd.DataFrame(shift_rows)

            if not shift_df.empty:
                status_order = {
                    "Waspada Baru": 1,
                    "Konsisten Prioritas": 2,
                    "Mulai Membaik": 3,
                    "Relatif Stabil": 4,
                    "Tidak Diketahui": 5
                }

                shift_df["Urutan Status"] = shift_df["Status Pergeseran"].map(status_order)

                shift_df = shift_df.sort_values(
                    ["Urutan Status", "NDVI Prediksi"],
                    ascending=[True, True]
                ).reset_index(drop=True)

            # =========================
            # LINE CHART + BAR CHART
            # =========================
            left_chart, right_chart = st.columns([1.45, 1], gap="medium")

            with left_chart:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="chart-card-title">Pergerakan NDVI {selected_forecast_model}: {chart_title}</div>
                        <div class="chart-card-caption">
                            Menampilkan data NDVI historis 2017–{selected_year - 1} dan hasil prediksi model {selected_forecast_model} pada tahun {selected_year}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    combined_line = pd.concat([y_hist, y_data])

                    y_min = combined_line.min()
                    y_max = combined_line.max()
                    buffer = max((y_max - y_min) * 0.25, 0.03)

                    y_axis_min = max(0, y_min - buffer)
                    y_axis_max = min(1, y_max + buffer)

                    fig_line = go.Figure()

                    # Garis historis
                    if not y_hist.empty:
                        fig_line.add_trace(
                            go.Scatter(
                                x=y_hist.index,
                                y=y_hist,
                                mode="lines",
                                line=dict(
                                    width=3,
                                    color="#64748b"
                                ),
                                name="NDVI Historis"
                            )
                        )

                    # Garis prediksi
                    fig_line.add_trace(
                        go.Scatter(
                            x=y_data.index,
                            y=y_data,
                            mode="lines+markers",
                            line=dict(
                                width=3,
                                color="#16a34a",
                                dash="dash"
                            ),
                            marker=dict(
                                size=7,
                                color="#16a34a"
                            ),
                            name=f"Prediksi {selected_forecast_model}"
                        )
                    )

                    # Garis penanda awal prediksi
                    if not y_data.empty:
                        fig_line.add_vline(
                            x=y_data.index.min(),
                            line_width=2,
                            line_dash="dot",
                            line_color="#94a3b8"
                        )

                        fig_line.add_annotation(
                            x=y_data.index.min(),
                            y=y_axis_max,
                            text="Awal prediksi",
                            showarrow=False,
                            yshift=12,
                            font=dict(
                                size=12,
                                color="#64748b"
                            )
                        )

                    x_start = y_hist.index.min() if not y_hist.empty else y_data.index.min()
                    x_end = y_data.index.max()

                    fig_line.update_layout(
                        height=430,
                        xaxis_title="Periode",
                        yaxis_title="Nilai NDVI",
                        xaxis=dict(
                            range=[x_start, x_end],
                            tickformat="%Y",
                            dtick="M12",
                            gridcolor="#e5e7eb"
                        ),
                        yaxis=dict(
                            range=[y_axis_min, y_axis_max],
                            tickformat=".3f",
                            gridcolor="#e5e7eb",
                            title_standoff=18
                        ),
                        hovermode="x unified",
                        margin=dict(l=75, r=20, t=10, b=45),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )

                    st.plotly_chart(
                        fig_line,
                        use_container_width=True,
                        theme=None,
                        config={"displayModeBar": False}
                    )

            with right_chart:
                with st.container(border=True):
                    st.markdown(
                        """
                        <div class="chart-card-title">Wilayah dengan Prediksi NDVI Terendah</div>
                        <div class="chart-card-caption">
                            Menampilkan 10 wilayah dengan rata-rata prediksi NDVI terendah
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    fig_pred_rank = go.Figure()
                    fig_pred_rank.add_trace(
                        go.Bar(
                            x=top_pred_low["Rata-rata NDVI Prediksi"],
                            y=top_pred_low["Kabupaten/Kota"],
                            orientation="h",
                            marker=dict(color=top_pred_low["Warna"]),
                            text=[f"{v:.3f}" for v in top_pred_low["Rata-rata NDVI Prediksi"]],
                            textposition="outside"
                        )
                    )

                    max_x = top_pred_low["Rata-rata NDVI Prediksi"].max()
                    left_pad = max_x * 0.035

                    fig_pred_rank.update_traces(
                        cliponaxis=False
                    )

                    fig_pred_rank.update_layout(
                        height=430,
                        xaxis_title="Rata-rata NDVI Prediksi",
                        yaxis_title="",
                        showlegend=False,
                        margin=dict(l=115, r=25, t=10, b=55),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff",
                        xaxis=dict(
                            range=[-left_pad, max_x * 1.22],
                            tickvals=[0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                            tickformat=".2f",
                            zeroline=True,
                            zerolinecolor="#cbd5e1",
                            gridcolor="#e5e7eb"
                        ),
                        yaxis=dict(
                            automargin=True,
                            tickfont=dict(size=11)
                        )
                    )

                    st.plotly_chart(
                        fig_pred_rank,
                        use_container_width=True,
                        theme=None,
                        config={"displayModeBar": False}
                    )

            # =========================
            # ANALISIS PERGESERAN HISTORIS KE PREDIKSI
            # =========================
            def shift_metric_card(title, value, subtitle, bg_color, border_color, text_color, badge_text):
                st.html(
                    f"""
                    <div style="
                        background:{bg_color};
                        border:1px solid {border_color};
                        border-left:7px solid {border_color};
                        border-radius:16px;
                        padding:18px 20px;
                        min-height:125px;
                        box-shadow:0 2px 8px rgba(15,23,42,0.05);
                    ">
                        <div style="
                            font-size:13px;
                            font-weight:850;
                            color:{text_color};
                            margin-bottom:10px;
                        ">
                            {title}
                        </div>

                        <div style="
                            font-size:34px;
                            font-weight:900;
                            color:{text_color};
                            line-height:1.1;
                            margin-bottom:6px;
                        ">
                            {value}
                            <span style="
                                font-size:14px;
                                font-weight:700;
                                color:{text_color};
                                opacity:0.75;
                            ">
                                wilayah
                            </span>
                        </div>

                        <div style="
                            font-size:13px;
                            color:{text_color};
                            opacity:0.9;
                            line-height:1.45;
                        ">
                            {subtitle}
                        </div>
                    </div>
                    """
                )

            if not shift_df.empty:
                jumlah_waspada_baru = (shift_df["Status Pergeseran"] == "Waspada Baru").sum()
                jumlah_konsisten = (shift_df["Status Pergeseran"] == "Konsisten Prioritas").sum()
                jumlah_membaik = (shift_df["Status Pergeseran"] == "Mulai Membaik").sum()
                jumlah_stabil = (shift_df["Status Pergeseran"] == "Relatif Stabil").sum()

                shift_show = shift_df.copy()

                for col in ["NDVI Historis", "NDVI Prediksi", "Perubahan NDVI"]:
                    shift_show[col] = shift_show[col].round(3)

                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="chart-card-title">Analisis Pergeseran Kondisi Historis Terkini ke Prediksi</div>
                        <div class="chart-card-caption">
                            Membandingkan rata-rata NDVI historis {hist_compare_start}–{selected_year - 1}
                            dengan hasil prediksi model {selected_forecast_model} tahun {selected_year}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    s1, s2, s3, s4 = st.columns(4, gap="medium")

                    with s1:
                        shift_metric_card(
                            title="Waspada Baru",
                            value=jumlah_waspada_baru,
                            subtitle="Historis baik → prediksi rendah",
                            bg_color="#FEF2F2",
                            border_color="#EF4444",
                            text_color="#7F1D1D",
                            badge_text="⚠ Warning"
                        )

                    with s2:
                        shift_metric_card(
                            title="Konsisten Prioritas",
                            value=jumlah_konsisten,
                            subtitle="Historis rendah → prediksi rendah",
                            bg_color="#FFF7ED",
                            border_color="#F97316",
                            text_color="#7C2D12",
                            badge_text="🔥 Prioritas"
                        )

                    with s3:
                        shift_metric_card(
                            title="Mulai Membaik",
                            value=jumlah_membaik,
                            subtitle="Historis rendah → prediksi baik",
                            bg_color="#F0FDF4",
                            border_color="#22C55E",
                            text_color="#14532D",
                            badge_text="⬆ Membaik"
                        )

                    with s4:
                        shift_metric_card(
                            title="Relatif Stabil",
                            value=jumlah_stabil,
                            subtitle="Historis baik → prediksi baik",
                            bg_color="#EFF6FF",
                            border_color="#3B82F6",
                            text_color="#1E3A8A",
                            badge_text="✓ Stabil"
                        )

                    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

                    st.markdown(
                        """
                        <div style="
                            margin-top:8px;
                            margin-bottom:12px;
                            color:#334155;
                            font-size:16px;
                            line-height:1.65;
                            font-weight:500;
                        ">
                            Tabel berikut menampilkan wilayah dengan perubahan NDVI yang perlu diperhatikan
                            berdasarkan perbandingan kondisi historis terkini dan hasil prediksi.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    shift_table = shift_show.copy()

                    shift_table["NDVI Historis"] = shift_table["NDVI Historis"].apply(
                        lambda x: f"{x:.3f}"
                    )

                    shift_table["NDVI Prediksi"] = shift_table["NDVI Prediksi"].apply(
                        lambda x: f"{x:.3f}"
                    )

                    shift_table["Perubahan NDVI"] = shift_table["Perubahan NDVI"].apply(
                        lambda x: f"{x:+.3f}"
                    )

                    shift_table_final = shift_table[
                        [
                            "Kabupaten/Kota",
                            "NDVI Historis",
                            "NDVI Prediksi",
                            "Perubahan NDVI",
                            "Status Pergeseran"
                        ]
                    ].copy()

                    table_html = shift_table_final.to_html(
                        index=False,
                        escape=False,
                        classes="shift-table"
                    )

                    st.html(
                        f"""
                        <div class="shift-table-wrap">
                            {table_html}
                        </div>
                        """
                    )

            else:
                st.info(
                    "Data historis belum dapat dicocokkan dengan data prediksi untuk analisis pergeseran."
                )

# =========================
# PAGE 3: KLASIFIKASI RISIKO
# =========================
elif menu == "Klasifikasi Indikasi Risiko":

    # =========================
    # DATA EVALUASI MODEL KLASIFIKASI
    # =========================
    model_eval_df = pd.DataFrame({
        "Algoritma": [
            "XGBoost",
            "Random Forest",
            "XGBoost",
            "Random Forest",
            "Random Forest",
            "XGBoost"
        ],
        "Pendekatan": [
            "Oversampling",
            "Oversampling",
            "SMOTE",
            "Base Model",
            "SMOTE",
            "Base Model"
        ],
        "Model": [
            "XGBoost dengan Oversampling",
            "Random Forest dengan Oversampling",
            "XGBoost dengan SMOTE",
            "Random Forest Base Model",
            "Random Forest dengan SMOTE",
            "XGBoost Base Model"
        ],
        "Accuracy": [0.7810, 0.7920, 0.7755, 0.7974, 0.7810, 0.7682],
        "Precision Weighted": [0.7795, 0.7954, 0.7735, 0.8017, 0.7822, 0.7639],
        "Recall Weighted": [0.7810, 0.7920, 0.7755, 0.7974, 0.7810, 0.7682],
        "F1-Macro": [0.7505, 0.7486, 0.7479, 0.7445, 0.7401, 0.7223],
        "F1-Weighted": [0.7793, 0.7870, 0.7734, 0.7917, 0.7802, 0.7597]
    })

    best_model_name = "XGBoost dengan Oversampling"

    # =========================
    # DROPDOWN MODEL
    # =========================
    d1, d2 = st.columns(2)

    with d1:
        selected_algorithm = st.selectbox(
            "Pilih Algoritma",
            ["XGBoost", "Random Forest"],
            key="klasifikasi_algoritma"
        )

    available_approaches = model_eval_df[
        model_eval_df["Algoritma"] == selected_algorithm
    ]["Pendekatan"].tolist()

    approach_order = ["Base Model", "SMOTE", "Oversampling"]
    available_approaches = [
        app for app in approach_order if app in available_approaches
    ]

    with d2:
        selected_approach = st.selectbox(
            "Pilih Pendekatan",
            available_approaches,
            index=available_approaches.index("Oversampling") if "Oversampling" in available_approaches else 0,
            key="klasifikasi_pendekatan"
        )

    selected_model_row = model_eval_df[
        (model_eval_df["Algoritma"] == selected_algorithm) &
        (model_eval_df["Pendekatan"] == selected_approach)
    ].iloc[0]

    selected_model_name = selected_model_row["Model"]
    accuracy_selected = selected_model_row["Accuracy"]
    precision_selected = selected_model_row["Precision Weighted"]
    recall_selected = selected_model_row["Recall Weighted"]
    f1_macro_selected = selected_model_row["F1-Macro"]
    f1_weighted_selected = selected_model_row["F1-Weighted"]

    if selected_model_name == best_model_name:
        status_model = "Model Terbaik"
        status_help = "F1-Macro tertinggi"
    else:
        status_model = "Model Pembanding"
        status_help = f"Model terbaik: {best_model_name}"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Model Digunakan",
            selected_algorithm,
            "Algoritma klasifikasi yang dipilih"
        )

    with c2:
        metric_card(
            "Ketepatan Prediksi",
            f"{accuracy_selected:.2%}",
            "Gambaran umum prediksi yang benar"
        )

    with c3:
        metric_card(
            "Keseimbangan Klasifikasi",
            f"{f1_macro_selected:.2%}",
            "Acuan utama pemilihan model"
        )

    with c4:
        metric_card(
            "Skenario Model",
            selected_approach,
            "Pendekatan yang digunakan"
        )
    
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    with st.expander("Lihat Perbandingan Performa Model"):
        st.markdown(
            f"""
            <div class="section-card">
                Model terbaik pada penelitian ini adalah <b>{best_model_name}</b> karena memiliki nilai
                <b>F1-Macro tertinggi</b>, sehingga lebih seimbang dalam mengenali seluruh kelas risiko.
                Akurasi tetap ditampilkan sebagai gambaran umum ketepatan prediksi model.
            </div>
            """,
            unsafe_allow_html=True
        )

        detail_all_df = model_eval_df.copy()

        for col in ["Accuracy", "Precision Weighted", "Recall Weighted", "F1-Macro", "F1-Weighted"]:
            detail_all_df[col] = detail_all_df[col].apply(lambda x: f"{x:.2%}")

        model_rows_html = ""

        for _, row in detail_all_df.iterrows():
            is_best = row["Model"] == best_model_name

            row_bg = "#f8fafc" if is_best else "#ffffff"
            row_weight = "700" if is_best else "600"

            model_rows_html += f"""
            <tr style="background:{row_bg};">
                <td>{html.escape(str(row["Model"]))}</td>
                <td>{row["Accuracy"]}</td>
                <td>{row["Precision Weighted"]}</td>
                <td>{row["Recall Weighted"]}</td>
                <td>{row["F1-Macro"]}</td>
                <td>{row["F1-Weighted"]}</td>
            </tr>
            """

        st.html(
            f"""
            <div style="
                border:1px solid #dbe2ea;
                border-radius:10px;
                overflow:hidden;
                background:#ffffff;
                margin-top:14px;
            ">
                <table style="
                    width:100%;
                    border-collapse:collapse;
                    color:#0f172a;
                ">
                    <thead>
                        <tr style="background:#f1f5f9;">
                            <th>Model</th>
                            <th>Ketepatan Prediksi</th>
                            <th>Precision Weighted</th>
                            <th>Recall Weighted</th>
                            <th>F1-Macro</th>
                            <th>F1-Weighted</th>
                        </tr>
                    </thead>

                    <tbody>
                        {model_rows_html}
                    </tbody>
                </table>
            </div>

            <style>
                thead th {{
                    text-align:left;
                    padding:12px;
                    border-bottom:1px solid #cbd5e1;
                    font-size:16px;
                    font-weight:700;
                    color:#0f172a;
                    white-space:nowrap;
                }}

                tbody td {{
                    padding:10px 12px;
                    border-bottom:1px solid #e5e7eb;
                    font-size:16px;
                    font-weight:500;
                    line-height:1.45;
                    color:#0f172a;
                }}
            </style>
            """
        )

    # =========================
    # FILTER DATA KLASIFIKASI
    # =========================
    filtered_cls = df_cls.copy()

    if "tahun" in filtered_cls.columns:
        filtered_cls = filtered_cls[filtered_cls["tahun"] == selected_year]

    if selected_city != "Semua" and "Kabupaten/Kota" in filtered_cls.columns:
        filtered_cls = filtered_cls[filtered_cls["Kabupaten/Kota"] == selected_city]

    if filtered_cls.empty:
        st.warning("Tidak ada data klasifikasi untuk filter yang dipilih.")
    else:
        filtered_cls["kelas_ndvi_num"] = pd.to_numeric(
            filtered_cls["kelas_ndvi"],
            errors="coerce"
        )

        filtered_cls["pred_num"] = pd.to_numeric(
            filtered_cls["Prediksi_Risiko_Model"],
            errors="coerce"
        )

        eval_cls = filtered_cls.dropna(
            subset=["kelas_ndvi_num", "pred_num"]
        ).copy()

        eval_cls["kelas_ndvi_num"] = eval_cls["kelas_ndvi_num"].astype(int)
        eval_cls["pred_num"] = eval_cls["pred_num"].astype(int)

        valid_labels = [0, 1, 2, 3]

        eval_cls = eval_cls[
            eval_cls["kelas_ndvi_num"].isin(valid_labels) &
            eval_cls["pred_num"].isin(valid_labels)
        ].copy()

        if eval_cls.empty:
            st.warning("Data tidak cukup untuk menampilkan hasil klasifikasi")
        else:

            # =========================
            # DATA PETA DAN RINGKASAN HASIL KLASIFIKASI
            # =========================

            # Tahun 2022 menggunakan hasil klasifikasi model.
            # Tahun selain 2022 menggunakan kelas NDVI yang sudah dibentuk dari nilai NDVI.
            if selected_year == 2022:
                sumber_kelas_col = "pred_num"
                judul_peta = "Peta Sebaran Hasil Klasifikasi Model Tahun 2022"
                caption_peta = (
                    "Menampilkan sebaran indikasi risiko kekeringan berdasarkan hasil klasifikasi model pada tahun 2022"
                )
                label_kelas_peta = "Hasil Klasifikasi Model"
            else:
                sumber_kelas_col = "kelas_ndvi_num"
                judul_peta = "Peta Sebaran Kelas NDVI sebagai Indikasi Risiko Kekeringan"
                caption_peta = (
                    "Menampilkan sebaran kelas NDVI tiap kabupaten/kota yang digunakan sebagai indikator tidak langsung terhadap indikasi risiko kekeringan"
                )
                label_kelas_peta = "Kelas NDVI"

            map_df = eval_cls.copy()
            map_df["kelas_peta"] = map_df[sumber_kelas_col].apply(risk_label)
            map_df["risiko_peta"] = map_df["kelas_peta"].apply(risk_level)

            def mode_value(series):
                mode_result = series.dropna().mode()
                if len(mode_result) > 0:
                    return mode_result.iloc[0]
                return "Tidak Diketahui"

            map_summary = (
                map_df
                .groupby("Kabupaten/Kota", as_index=False)
                .agg({
                    "NDVI": "mean",
                    "kelas_peta": mode_value,
                    "risiko_peta": mode_value
                })
            )

            map_summary["risiko_label"] = (
                map_summary["risiko_peta"]
                .str.replace("Indikasi Risiko ", "", regex=False)
            )

            # =========================
            # PETA SEBARAN RISIKO
            # =========================
            st.markdown(
                f"""
                <div class="chart-card-title">{judul_peta}</div>
                <div class="chart-card-caption">
                    {caption_peta}
                </div>
                """,
                unsafe_allow_html=True
            )

            def normalize_area_name(x):
                x = str(x).strip().lower()
                x = x.replace("kab. ", "kabupaten ")
                x = x.replace("kab ", "kabupaten ")
                x = " ".join(x.split())
                return x

            map_summary["wilayah_key"] = map_summary["Kabupaten/Kota"].apply(
                normalize_area_name
            )

            try:
                with open("jatim_kabkota.geojson", "r", encoding="utf-8") as f:
                    jatim_geojson = json.load(f)

                def normalize_area_name(x):
                    x = str(x).strip().lower()
                    x = x.replace("kab. ", "kabupaten ")
                    x = x.replace("kab ", "kabupaten ")
                    x = x.replace(".", "")
                    x = x.replace("-", " ")
                    x = " ".join(x.split())
                    return x


                def build_geo_wilayah_name(properties):
                    tipe = str(properties.get("TYPE_2", "")).strip()
                    nama = str(properties.get("NAME_2", "")).strip()

                    # Beberapa nama kota di GeoJSON bentuknya KotaBlitar, KotaKediri, dll.
                    if tipe.lower() == "kota":
                        if nama.lower().startswith("kota"):
                            nama = nama[4:].strip()
                        return f"Kota {nama}"

                    if tipe.lower() == "kabupaten":
                        return f"Kabupaten {nama}"

                    return nama

                map_summary["wilayah_key"] = map_summary["Kabupaten/Kota"].apply(normalize_area_name)

                for feature in jatim_geojson["features"]:
                    geo_full_name = build_geo_wilayah_name(feature["properties"])
                    feature["properties"]["wilayah_key"] = normalize_area_name(geo_full_name)

                data_keys = set(map_summary["wilayah_key"])
                geo_keys = {
                    feature["properties"]["wilayah_key"]
                    for feature in jatim_geojson["features"]
                }

                unmatched_data = sorted(data_keys - geo_keys)

                if len(unmatched_data) > 0:
                    with st.expander("Cek wilayah yang belum cocok dengan GeoJSON"):
                        st.write(unmatched_data)

                fig_map = px.choropleth_mapbox(
                    map_summary,
                    geojson=jatim_geojson,
                    locations="wilayah_key",
                    featureidkey="properties.wilayah_key",
                    color="risiko_label",
                    category_orders={
                        "risiko_label": ["Sangat Tinggi", "Tinggi", "Sedang", "Rendah"]
                    },
                    color_discrete_map={
                        "Rendah": "#2E7D32",
                        "Sedang": "#D9C75F",
                        "Tinggi": "#B7791F",
                        "Sangat Tinggi": "#6B3F16",
                        "Tidak Diketahui": "#94A3B8"
                    },
                    hover_name="Kabupaten/Kota",
                    hover_data={
                        "NDVI": ":.3f",
                        "kelas_peta": True,
                        "risiko_label": True,
                        "wilayah_key": False
                    },
                    labels={
                        "NDVI": "Rata-rata NDVI",
                        "kelas_peta": label_kelas_peta,
                        "risiko_label": "Indikasi Risiko Kekeringan"
                    },
                    mapbox_style="carto-positron",
                    center={"lat": -7.54, "lon": 112.23},
                    zoom=6.4,
                    opacity=0.78
                )

                fig_map.update_layout(
                    height=560,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    legend_title_text="Indikasi Risiko Kekeringan"
                )

                st.plotly_chart(
                    fig_map,
                    use_container_width=True,
                    theme=None,
                    config={"displayModeBar": False}
                )

            except FileNotFoundError:
                st.error(
                    "File GeoJSON tidak ditemukan. Pastikan file `jatim_kabkota.geojson` "
                    "ada dalam folder yang sama dengan file Streamlit."
                )

            # =========================
            # DETAIL DISTRIBUSI BULANAN HASIL KLASIFIKASI
            # =========================
            st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

            with st.container(border=True):
                if selected_year == 2022:
                    judul_distribusi = "Detail Distribusi Bulanan Hasil Klasifikasi Model"
                    caption_distribusi = (
                        "Menampilkan ringkasan dan jumlah kabupaten/kota pada setiap kategori indikasi risiko kekeringan "
                        "berdasarkan hasil klasifikasi model pada bulan yang dipilih"
                    )
                else:
                    judul_distribusi = "Detail Distribusi Bulanan Kelas NDVI sebagai Indikasi Risiko"
                    caption_distribusi = (
                        "Menampilkan ringkasan dan jumlah kabupaten/kota pada setiap kategori kelas NDVI "
                        "sebagai indikator tidak langsung terhadap indikasi risiko kekeringan pada bulan yang dipilih"
                    )

                st.markdown(
                    f"""
                    <div class="chart-card-title">{judul_distribusi}</div>
                    """,
                    unsafe_allow_html=True
                )

                # =========================
                # DATA DISTRIBUSI TAHUNAN PER BULAN
                # =========================
                month_name_map = {
                    1: "Jan",
                    2: "Feb",
                    3: "Mar",
                    4: "Apr",
                    5: "Mei",
                    6: "Jun",
                    7: "Jul",
                    8: "Agu",
                    9: "Sep",
                    10: "Okt",
                    11: "Nov",
                    12: "Des"
                }

                dist_df = eval_cls.copy()

                dist_df["bulan"] = pd.to_numeric(
                    dist_df["bulan"],
                    errors="coerce"
                ).astype(int)

                # Tahun 2022 memakai prediksi model, tahun selain 2022 memakai kelas NDVI
                if selected_year == 2022:
                    dist_df["kelas_distribusi"] = dist_df["pred_num"].apply(risk_label)
                else:
                    dist_df["kelas_distribusi"] = dist_df["kelas_ndvi_num"].apply(risk_label)

                dist_df["risiko_distribusi"] = dist_df["kelas_distribusi"].apply(risk_level)

                dist_df["risiko_label"] = (
                    dist_df["risiko_distribusi"]
                    .str.replace("Indikasi Risiko ", "", regex=False)
                )

                dist_df["Bulan"] = dist_df["bulan"].map(month_name_map)

                risk_order = ["Sangat Tinggi", "Tinggi", "Sedang", "Rendah"]

                risk_color_map = {
                    "Sangat Tinggi": "#6B3F16",
                    "Tinggi": "#B7791F",
                    "Sedang": "#D9C75F",
                    "Rendah": "#2E7D32"
                }

                # Rekap jumlah wilayah per bulan dan per kategori risiko
                distribusi_bulanan = (
                    dist_df
                    .groupby(["bulan", "risiko_label"])["Kabupaten/Kota"]
                    .nunique()
                    .reset_index(name="Jumlah Wilayah")
                )

                bulan_tersedia = sorted(
                    dist_df["bulan"].dropna().astype(int).unique()
                )

                # Buat kombinasi lengkap bulan x kategori risiko
                template_bulanan = pd.DataFrame(
                    [
                        {"bulan": bulan, "risiko_label": risiko}
                        for bulan in bulan_tersedia
                        for risiko in risk_order
                    ]
                )

                # Gabungkan hasil rekap dengan template
                distribusi_bulanan = template_bulanan.merge(
                    distribusi_bulanan,
                    on=["bulan", "risiko_label"],
                    how="left"
                )

                distribusi_bulanan["Jumlah Wilayah"] = (
                    distribusi_bulanan["Jumlah Wilayah"]
                    .fillna(0)
                    .astype(int)
                )

                distribusi_bulanan["Bulan"] = distribusi_bulanan["bulan"].map(month_name_map)

                # =========================
                # RINGKASAN TAHUNAN
                # =========================
                total_wilayah_tahun = dist_df["Kabupaten/Kota"].nunique()

                risiko_dominan_tahun = (
                    dist_df["risiko_label"]
                    .value_counts()
                    .idxmax()
                )

                wilayah_prioritas_tahun = (
                    dist_df
                    .groupby("Kabupaten/Kota", as_index=False)["NDVI"]
                    .mean()
                    .sort_values("NDVI", ascending=True)
                    .iloc[0]["Kabupaten/Kota"]
                )

                ndvi_prioritas_tahun = (
                    dist_df
                    .groupby("Kabupaten/Kota", as_index=False)["NDVI"]
                    .mean()
                    .sort_values("NDVI", ascending=True)
                    .iloc[0]["NDVI"]
                )

                jumlah_sangat_tinggi_tahun = (
                    dist_df["risiko_label"] == "Sangat Tinggi"
                ).sum()

                jumlah_tinggi_tahun = (
                    dist_df["risiko_label"] == "Tinggi"
                ).sum()

                ringkasan_col, chart_col = st.columns([0.95, 1.55], gap="large")

                with ringkasan_col:
                    st.html(
                        f"""
                        <div style="
                            background:#ffffff;
                            border-radius:18px;
                            border-left:8px solid #2E7D32;
                            padding:20px 24px;
                            min-height:390px;
                            box-shadow:0 4px 18px rgba(15,23,42,0.05);
                            border-top:1px solid #e5e7eb;
                            border-right:1px solid #e5e7eb;
                            border-bottom:1px solid #e5e7eb;
                        ">
                            <div style="
                                font-size:22px;
                                font-weight:800;
                                color:#0f172a;
                                margin-bottom:6px;
                            ">
                                Ringkasan Tahun {selected_year}
                            </div>

                            <div style="
                                font-size:16px;
                                color:#64748b;
                                font-weight:500;
                                margin-bottom:14px;
                            ">
                                Hasil distribusi indikasi risiko kekeringan per bulan
                            </div>

                            <div style="
                                height:1px;
                                background:#e5e7eb;
                                margin:12px 0 14px 0;
                            "></div>

                            <div style="
                                font-size:20px;
                                line-height:1.85;
                                color:#334155;
                                font-weight:400;
                            ">
                                Pada tahun
                                <span style="color:#2563eb; font-weight:800;">{selected_year}</span>,
                                visualisasi menampilkan distribusi risiko pada
                                <span style="color:#2563eb; font-weight:800;">{total_wilayah_tahun}</span>
                                kabupaten/kota untuk setiap bulan.

                                Kategori indikasi risiko yang paling sering muncul adalah
                                <span style="color:#ca8a04; font-weight:800;">{risiko_dominan_tahun}</span>.

                                Wilayah dengan rata-rata NDVI terendah sepanjang tahun adalah
                                <span style="color:#dc2626; font-weight:800;">{html.escape(str(wilayah_prioritas_tahun))}</span>
                                dengan nilai NDVI
                                <span style="color:#2563eb; font-weight:800;">{ndvi_prioritas_tahun:.3f}</span>.

                                Secara akumulatif, terdapat
                                <span style="color:#7c2d12; font-weight:800;">{jumlah_sangat_tinggi_tahun}</span>
                                kemunculan wilayah-bulan pada kategori risiko sangat tinggi dan
                                <span style="color:#b45309; font-weight:800;">{jumlah_tinggi_tahun}</span>
                                pada kategori risiko tinggi.
                            </div>
                        </div>
                        """
                    )

                with chart_col:
                    fig_dist_cls = go.Figure()

                    for risiko in risk_order:
                        data_risiko = distribusi_bulanan[
                            distribusi_bulanan["risiko_label"] == risiko
                        ]

                        fig_dist_cls.add_trace(
                            go.Bar(
                                x=data_risiko["Bulan"],
                                y=data_risiko["Jumlah Wilayah"],
                                name=risiko,
                                marker=dict(
                                    color=risk_color_map.get(risiko, "#94A3B8")
                                ),
                                text=data_risiko["Jumlah Wilayah"],
                                textposition="outside"
                            )
                        )

                    max_jumlah = distribusi_bulanan["Jumlah Wilayah"].max()

                    fig_dist_cls.update_layout(
                        height=390,
                        barmode="group",
                        xaxis_title="Bulan",
                        yaxis_title="Jumlah Wilayah",
                        legend_title="Indikasi Risiko",
                        margin=dict(l=50, r=30, t=20, b=60),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(
                            range=[0, max_jumlah * 1.18],
                            gridcolor="#e5e7eb",
                            zeroline=True,
                            zerolinecolor="#cbd5e1"
                        ),
                        xaxis=dict(
                            categoryorder="array",
                            categoryarray=[month_name_map[m] for m in bulan_tersedia],
                            tickfont=dict(size=12)
                        ),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )

                    st.plotly_chart(
                        fig_dist_cls,
                        use_container_width=True,
                        theme=None,
                        config={"displayModeBar": False}
                    )

            with st.expander("Lihat Rekap Bulanan Kelas Risiko"):
                rekap_bulanan_df = eval_cls.copy()

                if selected_year == 2022:
                    rekap_bulanan_df["kelas_rekap"] = rekap_bulanan_df["pred_num"].apply(risk_label)
                else:
                    rekap_bulanan_df["kelas_rekap"] = rekap_bulanan_df["kelas_ndvi_num"].apply(risk_label)

                rekap_bulanan_df["risiko_label"] = (
                    rekap_bulanan_df["kelas_rekap"]
                    .apply(risk_level)
                    .str.replace("Indikasi Risiko ", "", regex=False)
                )

                rekap_bulanan_df["bulan"] = pd.to_numeric(
                    rekap_bulanan_df["bulan"],
                    errors="coerce"
                ).astype(int)

                rekap_bulanan = (
                    rekap_bulanan_df
                    .groupby(["bulan", "risiko_label"])["Kabupaten/Kota"]
                    .nunique()
                    .reset_index(name="Jumlah Wilayah")
                )

                tabel_rekap_bulanan = (
                    rekap_bulanan
                    .pivot_table(
                        index="bulan",
                        columns="risiko_label",
                        values="Jumlah Wilayah",
                        fill_value=0
                    )
                    .reset_index()
                )

                risk_cols = ["Sangat Tinggi", "Tinggi", "Sedang", "Rendah"]

                for col in risk_cols:
                    if col not in tabel_rekap_bulanan.columns:
                        tabel_rekap_bulanan[col] = 0

                tabel_rekap_bulanan["Bulan"] = tabel_rekap_bulanan["bulan"].map(month_name_map)

                tabel_rekap_bulanan = tabel_rekap_bulanan[
                    ["Bulan", "Sangat Tinggi", "Tinggi", "Sedang", "Rendah"]
                ]

                st.dataframe(
                    tabel_rekap_bulanan,
                    use_container_width=True,
                    hide_index=True
                )

st.markdown("---")
st.caption(
    f"Terakhir diperbarui: {datetime.now().strftime('%d %B %Y %H:%M')} | "
    f"{DASHBOARD_TITLE}"
)