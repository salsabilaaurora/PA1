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

DASHBOARD_TITLE = "Dashboard Pemantauan NDVI dan Indikasi Risiko Kekeringan Jawa Timur"

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
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
            font-size: 14px;
            color: #64748b;
            font-weight: 500;
            margin-bottom: 6px;
        }

        .metric-value {
            font-size: 28px;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.1;
        }

        .metric-desc {
            font-size: 12px;
            color: #94a3b8;
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
            font-size: 20px;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 4px;
        }

        .chart-card-caption {
            font-size: 13px;
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
    </style>
    """,
    unsafe_allow_html=True
)

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

df = load_forecast_data()
df_cls = load_classification_data()
df_eval = load_eval_data()

# =========================
# DATA MODEL PREDIKSI
# =========================
forecast_data_map = {
    "GSTAR": df,
    # "GSTAR-ARIMA": df_gstar_arima,
    # "LSTM": df_lstm,
}

forecast_eval_map = {
    "GSTAR": df_eval,
    # "GSTAR-ARIMA": df_eval_gstar_arima,
    # "LSTM": df_eval_lstm,
}

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
    "🏠  Beranda": "Beranda",
    "📈  Prediksi NDVI": "Prediksi NDVI",
    "🧩  Klasifikasi Indikasi Risiko": "Klasifikasi Indikasi Risiko"
}

selected_nav = st.sidebar.radio(
    "Navigasi",
    list(nav_options.keys()),
    label_visibility="collapsed"
)

menu = nav_options[selected_nav]

st.sidebar.markdown("---")

st.sidebar.markdown(
    '<div class="sidebar-section-title">Filter Data</div>',
    unsafe_allow_html=True
)

if menu == "Prediksi NDVI":
    year_options = sorted(
        [y for y in df["tahun"].unique() if y >= 2025],
        reverse=True
    )

    selected_year = st.sidebar.selectbox(
        "Tahun Prediksi",
        year_options
    )
else:
    year_options = sorted(df["tahun"].unique(), reverse=True)

    selected_year = st.sidebar.selectbox(
        "Tahun",
        year_options
    )

kabupaten_cols = [col for col in df.columns if col not in ["tahun", "bulan"]]
selected_city = st.sidebar.selectbox(
    "Kabupaten/Kota",
    ["Semua"] + sorted(kabupaten_cols)
)

filtered = df[df["tahun"] == selected_year].copy()

# =========================
# HEADER
# =========================

# =========================
# PAGE 1: BERANDA
# =========================
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

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Prioritas Utama Pemantauan", f"{wilayah_prioritas}", "Wilayah dengan NDVI sangat rendah")
    with c2:
        metric_card("Total Wilayah", f"{total_wilayah}", "Kabupaten/kota Jawa Timur")
    with c3:
        metric_card("Rata-rata NDVI", f"{rata_ndvi:.3f}", "Rata-rata seluruh wilayah")
    with c4:
        metric_card("Rentang NDVI", f"{ndvi_min:.3f}–{ndvi_max:.3f}", "Minimum–maksimum")

    eda_df["Prioritas"] = (1 - eda_df["NDVI"]).clip(lower=0.001)

    wilayah_prioritas = eda_df.sort_values("Prioritas", ascending=False).iloc[0]
    wilayah_terbaik = eda_df.sort_values("NDVI", ascending=False).iloc[0]
    kelas_dominan = eda_df["Kelas"].value_counts().idxmax()

    st.markdown("")

    CARD_HEIGHT = 530
    CHART_HEIGHT = 410

    left, middle, insight_col = st.columns(3, gap="medium")

    with left:
        with st.container(border=True, height=CARD_HEIGHT):
            st.markdown(
                """
                <div class="chart-card-title">Distribusi Tingkat Kehijauan</div>
                <div class="chart-card-caption">
                    Menunjukkan komposisi jumlah wilayah berdasarkan kelas kehijauan NDVI
                </div>
                """,
                unsafe_allow_html=True
            )

            dist = (
                eda_df["Kelas"]
                .value_counts()
                .rename_axis("Kelas")
                .reset_index(name="Jumlah")
            )

            class_order = [
                "Kehijauan Sangat Rendah",
                "Kehijauan Rendah",
                "Kehijauan Sedang",
                "Kehijauan Tinggi"
            ]

            fig_pie = px.pie(
                dist,
                names="Kelas",
                values="Jumlah",
                hole=0.45,
                color="Kelas",
                category_orders={"Kelas": class_order},
                color_discrete_map={k: class_color(k) for k in class_order}
            )

            fig_pie.update_traces(
                textinfo="percent",
                textfont_size=14
            )

            fig_pie.update_layout(
                height=410,
                margin=dict(l=5, r=5, t=5, b=5),
                legend_title_text="Kategori",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff"
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True,
                theme=None,
                config={"displayModeBar": False},
                key="pie_distribusi_kehijauan"
            )

    with middle:
        with st.container(border=True, height=CARD_HEIGHT):
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
                margin=dict(l=170, r=50, t=5, b=45),
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
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

    with insight_col:
        jumlah_sangat_rendah = (eda_df["Kelas"] == "Kehijauan Sangat Rendah").sum()
        jumlah_rendah = (eda_df["Kelas"] == "Kehijauan Rendah").sum()
        jumlah_sedang = (eda_df["Kelas"] == "Kehijauan Sedang").sum()
        jumlah_tinggi = (eda_df["Kelas"] == "Kehijauan Tinggi").sum()

        kelas_dominan = eda_df["Kelas"].value_counts().idxmax()
        persen_dominan = eda_df["Kelas"].value_counts(normalize=True).max() * 100

        top_prioritas = eda_df.sort_values("NDVI", ascending=True).head(3)

        wilayah_utama = top_prioritas.iloc[0]["Kabupaten/Kota"]

        top_prioritas_html = ""

        for nomor, (_, row) in enumerate(top_prioritas.iterrows(), start=1):
            top_prioritas_html += dedent(f"""
            <div style="
                background:#ffffff;
                border:1px solid #dbe2ea;
                border-radius:12px;
                padding:11px 12px;
                margin-bottom:9px;
            ">
                <div style="
                    font-size:11px;
                    color:#64748b;
                    font-weight:800;
                    margin-bottom:4px;
                ">
                    #{nomor}
                </div>
                <div style="
                    font-size:15px;
                    color:#0f172a;
                    font-weight:800;
                    margin-bottom:4px;
                    line-height:1.35;
                ">
                    {html.escape(str(row["Kabupaten/Kota"]))}
                </div>
                <div style="
                    font-size:12px;
                    color:#475569;
                    line-height:1.5;
                    font-weight:500;
                ">
                    NDVI <b style="color:#0f172a;">{row["NDVI"]:.3f}</b> · {row["Kelas"]}
                </div>
            </div>
            """).strip()

        with st.container(border=True, height=CARD_HEIGHT):
        # MARKDOWN 1: judul, biarin
            st.markdown(
                f"""
                <div class="chart-card-title">Ringkasan Insight</div>
                <div class="chart-card-caption">
                    Gambaran cepat kondisi kehijauan wilayah berdasarkan NDVI tahun {selected_year}
                </div>
                """,
                unsafe_allow_html=True
            )

            # MARKDOWN 2: isi HTML panjang, ini yang pakai dedent
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
                        background:linear-gradient(135deg,#f8fafc 0%,#ecfdf5 100%);
                        border:1px solid #bfdbfe;
                        border-radius:14px;
                        padding:14px 15px;
                        margin-bottom:14px;
                    ">
                        <div style="
                            font-size:11px;
                            color:#475569;
                            font-weight:800;
                            letter-spacing:0.04em;
                            margin-bottom:6px;
                        ">
                            KONDISI UMUM
                        </div>

                        <div style="
                            font-size:15px;
                            line-height:1.7;
                            color:#334155;
                            font-weight:500;
                        ">
                            Rata-rata NDVI wilayah berada pada nilai 
                            <b style="color:#0f172a; font-size:16px;">{rata_ndvi:.3f}</b>. 
                            Kategori dominan adalah 
                            <b style="color:{class_color(kelas_dominan)}; font-size:15px;">{kelas_dominan}</b> 
                            dengan proporsi <b style="color:#0f172a;">{persen_dominan:.1f}%</b> dari wilayah yang ditampilkan.
                        </div>
                    </div>

                    <div style="
                        display:grid;
                        grid-template-columns:1fr 1fr;
                        gap:10px;
                        margin-bottom:14px;
                    ">

                        <div style="
                            background:#ffffff;
                            border:1px solid #dbe2ea;
                            border-radius:12px;
                            padding:12px;
                        ">
                            <div style="
                                font-size:11px;
                                color:#64748b;
                                font-weight:800;
                                margin-bottom:6px;
                            ">
                                RATA-RATA NDVI
                            </div>
                            <div style="
                                font-size:30px;
                                font-weight:850;
                                color:#0f172a;
                                line-height:1.1;
                            ">
                                {rata_ndvi:.3f}
                            </div>
                        </div>

                        <div style="
                            background:#ffffff;
                            border:1px solid #dbe2ea;
                            border-radius:12px;
                            padding:12px;
                        ">
                            <div style="
                                font-size:11px;
                                color:#64748b;
                                font-weight:800;
                                margin-bottom:6px;
                            ">
                                PRIORITAS
                            </div>
                            <div style="
                                font-size:30px;
                                font-weight:850;
                                color:#B7791F;
                                line-height:1.1;
                            ">
                                {jumlah_sangat_rendah}
                            </div>
                            <div style="
                                font-size:12px;
                                color:#64748b;
                                font-weight:600;
                                margin-top:3px;
                            ">
                                wilayah
                            </div>
                        </div>

                        <div style="
                            background:#ffffff;
                            border:1px solid #dbe2ea;
                            border-radius:12px;
                            padding:12px;
                        ">
                            <div style="
                                font-size:11px;
                                color:#64748b;
                                font-weight:800;
                                margin-bottom:6px;
                            ">
                                DOMINAN
                            </div>
                            <div style="
                                font-size:15px;
                                font-weight:800;
                                color:{class_color(kelas_dominan)};
                                line-height:1.35;
                            ">
                                {kelas_dominan}
                            </div>
                        </div>

                        <div style="
                            background:#ffffff;
                            border:1px solid #dbe2ea;
                            border-radius:12px;
                            padding:12px;
                        ">
                            <div style="
                                font-size:11px;
                                color:#64748b;
                                font-weight:800;
                                margin-bottom:6px;
                            ">
                                NDVI TERENDAH
                            </div>
                            <div style="
                                font-size:15px;
                                font-weight:800;
                                color:#0f172a;
                                line-height:1.35;
                            ">
                                {html.escape(str(wilayah_utama))}
                            </div>
                        </div>

                    </div>

                    <div style="
                        font-size:14px;
                        font-weight:800;
                        color:#0f172a;
                        margin-bottom:10px;
                    ">
                        Wilayah Prioritas Pemantauan
                    </div>

                    {top_prioritas_html}

                    <div style="
                        background:#fffbeb;
                        border:1px solid #fcd34d;
                        border-radius:14px;
                        padding:12px 13px;
                        margin-top:12px;
                        color:#92400e;
                        font-size:13px;
                        line-height:1.65;
                        font-weight:500;
                    ">
                        <b>Catatan:</b> Wilayah dengan nilai NDVI rendah dapat dijadikan prioritas awal
                        untuk pemantauan kondisi vegetasi dan indikasi risiko kekeringan.
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
            textfont_size=13
        )

        fig_treemap.update_layout(
            height=680,
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

    with st.expander("Lihat Detail Data Wilayah"):
        st.dataframe(
            eda_df[["Kabupaten/Kota", "NDVI", "Kelas", "Risiko"]],
            use_container_width=True,
            hide_index=True
        )

# =========================
# PAGE 2: PREDIKSI NDVI
# =========================
elif menu == "Prediksi NDVI":

    # =========================
    # DROPDOWN MODEL PREDIKSI
    # =========================
    selected_forecast_model = st.selectbox(
        "Pilih Model Prediksi",
        list(forecast_data_map.keys()),
        index=0,
        key="prediksi_model"
    )

    df_pred = forecast_data_map[selected_forecast_model]
    df_eval_model = forecast_eval_map[selected_forecast_model]

    filtered_pred = df_pred[df_pred["tahun"] == selected_year].copy()

    if filtered_pred.empty:
        st.warning("Tidak ada data prediksi untuk filter tahun yang dipilih.")
    else:

        # =========================
        # METRIK EVALUASI MODEL
        # =========================
        if selected_city == "Semua":
            rmse_val = df_eval_model["RMSE"].mean()
            mse_val = df_eval_model["MSE"].mean()
            mae_val = df_eval_model["MAE"].mean()
        else:
            row_eval = df_eval_model[df_eval_model["Kabupaten/Kota"] == selected_city]

            if not row_eval.empty:
                rmse_val = row_eval["RMSE"].iloc[0]
                mse_val = row_eval["MSE"].iloc[0]
                mae_val = row_eval["MAE"].iloc[0]
            else:
                rmse_val, mse_val, mae_val = np.nan, np.nan, np.nan

        if pd.isna(rmse_val):
            status_prediksi = "Tidak Tersedia"
        elif rmse_val <= 0.05:
            status_prediksi = "Sangat Baik"
        elif rmse_val <= 0.10:
            status_prediksi = "Baik"
        elif rmse_val <= 0.15:
            status_prediksi = "Cukup Baik"
        else:
            status_prediksi = "Perlu Evaluasi"

        # =========================
        # DATA UNTUK LINE CHART
        # =========================
        if selected_city == "Semua":
            y_data = filtered_pred[kabupaten_cols].apply(
                pd.to_numeric,
                errors="coerce"
            ).mean(axis=1)
            chart_title = "Rata-rata NDVI Jawa Timur"
        else:
            y_data = pd.to_numeric(
                filtered_pred[selected_city],
                errors="coerce"
            )
            chart_title = selected_city

        y_data = y_data.dropna()

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
                    "Status Prediksi",
                    status_prediksi,
                    "Berdasarkan nilai RMSE model"
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
                "Kabupaten/Kota": kabupaten_cols,
                "Rata-rata NDVI Prediksi": [
                    pd.to_numeric(filtered_pred[col], errors="coerce").mean()
                    for col in kabupaten_cols
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
            # LINE CHART + BAR CHART
            # =========================
            left_chart, right_chart = st.columns([1.45, 1], gap="medium")

            with left_chart:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="chart-card-title">Pergerakan NDVI: {chart_title}</div>
                        <div class="chart-card-caption">
                            Menampilkan perubahan nilai NDVI bulanan pada tahun {selected_year}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    y_min = y_data.min()
                    y_max = y_data.max()
                    buffer = max((y_max - y_min) * 0.25, 0.03)

                    y_axis_min = max(0, y_min - buffer)
                    y_axis_max = min(1, y_max + buffer)

                    fig_line = go.Figure()
                    fig_line.add_trace(
                        go.Scatter(
                            x=y_data.index,
                            y=y_data,
                            mode="lines+markers",
                            line=dict(width=3),
                            marker=dict(size=7),
                            name="NDVI"
                        )
                    )

                    fig_line.update_layout(
                        height=430,
                        xaxis_title="Bulan",
                        yaxis_title="Nilai NDVI",
                        yaxis=dict(
                            range=[y_axis_min, y_axis_max],
                            tickformat=".3f",
                            gridcolor="#e5e7eb",
                            title_standoff=18
                        ),
                        hovermode="x unified",
                        margin=dict(l=75, r=20, t=10, b=45),
                        paper_bgcolor="#ffffff",
                        plot_bgcolor="#ffffff"
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
                        margin=dict(l=170, r=75, t=10, b=55),
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
            # INTERPRETASI
            # =========================
            kondisi = classify_ndvi(y_data.mean())

            st.markdown(
                f"""
                <div class="section-card">
                    Rata-rata NDVI prediksi pada tahun {selected_year} berada pada kategori 
                    <span style="color:{class_color(kondisi)};font-weight:800;">{kondisi}</span>. 
                    Kondisi ini menunjukkan perlunya pemantauan lebih lanjut terhadap wilayah dengan nilai NDVI rendah. 
                    Arah tren NDVI menunjukkan kondisi <b>{trend_status.lower()}</b> 
                    dengan perubahan sebesar <b>{delta_ndvi:+.3f}</b>.
                </div>
                """,
                unsafe_allow_html=True
            )

            # =========================
            # DETAIL EVALUASI MODEL
            # =========================
            with st.expander("Lihat Detail Evaluasi Model"):
                eval_detail_df = pd.DataFrame({
                    "Metrik": ["RMSE", "MSE", "MAE"],
                    "Nilai": [
                        f"{rmse_val:.4f}" if not pd.isna(rmse_val) else "-",
                        f"{mse_val:.4f}" if not pd.isna(mse_val) else "-",
                        f"{mae_val:.4f}" if not pd.isna(mae_val) else "-"
                    ],
                    "Keterangan": [
                        "Root Mean Squared Error",
                        "Mean Squared Error",
                        "Mean Absolute Error"
                    ]
                })

                st.dataframe(
                    eval_detail_df,
                    use_container_width=True,
                    hide_index=True
                )

            # =========================
            # DETAIL FORECAST BULANAN
            # =========================
            if selected_city == "Semua":
                detail_df = filtered_pred[kabupaten_cols].copy()
            else:
                detail_df = filtered_pred[[selected_city]].copy()

            detail_df = detail_df.reset_index()
            detail_df = detail_df.rename(columns={detail_df.columns[0]: "Tanggal"})
            detail_df["Tanggal"] = pd.to_datetime(detail_df["Tanggal"]).dt.strftime("%B %Y")

            with st.expander("Lihat Detail Forecast Bulanan"):
                st.dataframe(
                    detail_df,
                    use_container_width=True,
                    hide_index=True
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

    c1, c2, c3 = st.columns(3)

    with c1:
        small_metric_card(
            "Accuracy",
            f"{accuracy_selected:.2%}",
            "Proporsi klasifikasi benar"
        )

    with c2:
        small_metric_card(
            "F1-Macro",
            f"{f1_macro_selected:.2%}",
            "Metrik utama pemilihan model"
        )

    with c3:
        small_metric_card(
            "Status Model",
            status_model,
            f"{status_help} | F1-Macro: {f1_macro_selected:.2%}"
        )

    with st.expander("Lihat Perbandingan Seluruh Model"):
        detail_all_df = model_eval_df.copy()

        for col in ["Accuracy", "Precision Weighted", "Recall Weighted", "F1-Macro", "F1-Weighted"]:
            detail_all_df[col] = detail_all_df[col].apply(lambda x: f"{x:.2%}")

        st.dataframe(
            detail_all_df[
                ["Model", "Accuracy", "Precision Weighted", "Recall Weighted", "F1-Macro", "F1-Weighted"]
            ],
            use_container_width=True,
            hide_index=True
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
            # PETA SEBARAN RISIKO
            # =========================
            st.markdown("#### Peta Sebaran Klasifikasi Indikasi Risiko Kekeringan Berbasis NDVI")

            map_df = eval_cls.copy()
            map_df["kelas_prediksi"] = map_df["pred_num"].apply(risk_label)
            map_df["risiko_prediksi"] = map_df["kelas_prediksi"].apply(risk_level)

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
                    "kelas_prediksi": mode_value,
                    "risiko_prediksi": mode_value
                })
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
                    color="kelas_prediksi",
                    color_discrete_map={
                        "Kehijauan Sangat Rendah": "#6B3F16",
                        "Kehijauan Rendah": "#B7791F",
                        "Kehijauan Sedang": "#D9C75F",
                        "Kehijauan Tinggi": "#2E7D32",
                        "Tidak Diketahui": "#94A3B8"
                    },
                    hover_name="Kabupaten/Kota",
                    hover_data={
                        "NDVI": ":.3f",
                        "kelas_prediksi": True,
                        "risiko_prediksi": True,
                        "wilayah_key": False
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
                    legend_title_text="Kelas Prediksi"
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
            # CONFUSION MATRIX
            # =========================
            with st.expander("Lihat Confusion Matrix"):
                y_true = eval_cls["kelas_ndvi_num"]
                y_pred = eval_cls["pred_num"]

                label_num = [0, 1, 2, 3]
                label_text = [risk_label(x) for x in label_num]

                cm = confusion_matrix(
                    y_true,
                    y_pred,
                    labels=label_num
                )

                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    x=label_text,
                    y=label_text,
                    color_continuous_scale="YlOrBr",
                    aspect="auto"
                )

                fig_cm.update_layout(
                    height=420,
                    xaxis_title="Prediksi",
                    yaxis_title="Aktual",
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff"
                )

                st.plotly_chart(
                    fig_cm,
                    use_container_width=True,
                    theme=None,
                    config={"displayModeBar": False}
                )

            # =========================
            # DETAIL KLASIFIKASI
            # =========================
            show_cols = [
                "Kabupaten/Kota",
                "bulan",
                "tahun",
                "NDVI",
                "RR",
                "RH_AVG",
                "TAVG",
                "kelas_ndvi",
                "Prediksi_Risiko_Model"
            ]

            available_cols = [
                col for col in show_cols
                if col in filtered_cls.columns
            ]

            detail_cls = filtered_cls[available_cols].copy()

            if "kelas_ndvi" in detail_cls.columns:
                detail_cls["kelas_ndvi"] = pd.to_numeric(
                    detail_cls["kelas_ndvi"],
                    errors="coerce"
                )

                detail_cls["kelas_aktual"] = detail_cls["kelas_ndvi"].apply(
                    lambda x: risk_label(x) if pd.notna(x) else "Tidak Diketahui"
                )

            if "Prediksi_Risiko_Model" in detail_cls.columns:
                detail_cls["Prediksi_Risiko_Model"] = pd.to_numeric(
                    detail_cls["Prediksi_Risiko_Model"],
                    errors="coerce"
                )

                detail_cls["kelas_prediksi"] = detail_cls["Prediksi_Risiko_Model"].apply(
                    lambda x: risk_label(x) if pd.notna(x) else "Tidak Diketahui"
                )

            detail_cols_final = [
                col for col in [
                    "Kabupaten/Kota",
                    "bulan",
                    "tahun",
                    "NDVI",
                    "RR",
                    "RH_AVG",
                    "TAVG",
                    "kelas_aktual",
                    "kelas_prediksi"
                ]
                if col in detail_cls.columns
            ]

            with st.expander("Lihat Detail Klasifikasi"):
                st.dataframe(
                    detail_cls[detail_cols_final],
                    use_container_width=True,
                    hide_index=True
                )

st.markdown("---")
st.caption(
    f"Terakhir diperbarui: {datetime.now().strftime('%d %B %Y %H:%M')} | "
    f"{DASHBOARD_TITLE}"
)