import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import html
import json
import re

st.set_page_config(
    page_title="Dashboard Prediksi dan Klasifikasi Kekeringan Jawa Timur",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_data():
    df = pd.read_csv(r"historis_plus_forecast_2025.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    df['tahun'] = df.index.year
    return df

df = load_data()

@st.cache_data
def load_classification_data():
    df_cls = pd.read_csv(r"Data_Lengkap_Kekeringan_Jatim.csv")
    return df_cls

df_cls = load_classification_data()

df_eval = pd.read_csv("evaluasi_gstar_jatim.csv")

# load model
with open(r"C:\Users\salsa\Downloads\model_gstar.pkl", "rb") as f:
    model = pickle.load(f)

# Palet warna berbasis kondisi vegetasi: hijau -> kuning -> coklat
# Semakin coklat, semakin rendah kehijauan vegetasi dan semakin tinggi risiko kekeringan.
COLOR_MAP = {
    "Kehijauan Sangat Rendah": "#6B3F16",  # coklat tua
    "Kehijauan Rendah": "#B7791F",         # coklat muda
    "Kehijauan Sedang": "#D9C75F",         # kuning kehijauan
    "Kehijauan Tinggi": "#2E7D32",         # hijau
    "Tidak Diketahui": "#94A3B8",          # abu
}

RISK_MAP = {
    "Kehijauan Sangat Rendah": "Risiko Sangat Tinggi",
    "Kehijauan Rendah": "Risiko Tinggi",
    "Kehijauan Sedang": "Risiko Sedang",
    "Kehijauan Tinggi": "Risiko Rendah",
    "Tidak Diketahui": "Tidak Diketahui",
}

CLASS_ORDER = [
    "Kehijauan Sangat Rendah",
    "Kehijauan Rendah",
    "Kehijauan Sedang",
    "Kehijauan Tinggi",
]

def class_color_cls(label: str):
    return COLOR_MAP.get(str(label).strip(), "#94A3B8")

def risk_level(label: str):
    return RISK_MAP.get(str(label).strip(), "Tidak Diketahui")

def risk_label(x):
    mapping = {
        0: "Kehijauan Sangat Rendah",
        1: "Kehijauan Rendah",
        2: "Kehijauan Sedang",
        3: "Kehijauan Tinggi",
    }
    try:
        return mapping.get(int(x), "Tidak Diketahui")
    except Exception:
        return "Tidak Diketahui"

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
    
def metric_card(title, value, help_text=None):
    st.markdown(
        f"""
        <div style="background:#ffffff;padding:18px 20px;border-radius:16px;box-shadow:0 4px 14px rgba(0,0,0,0.06);border:1px solid #e5e7eb;">
            <div style="font-size:16px;color:#475569;margin-bottom:8px;">{title}</div>
            <div style="font-size:32px;font-weight:700;color:#0f172a;">{value}</div>
            <div style="font-size:12px;color:#64748b;">{help_text or ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def recommendation_rule(row):
    if row["prob_kering"] >= 0.75:
        return "Prioritas intervensi air bersih, pompanisasi, dan monitoring mingguan."
    if row["prob_kering"] >= 0.60:
        return "Siapkan distribusi air, efisiensi irigasi, dan peringatan dini wilayah."
    if row["prob_kering"] >= 0.40:
        return "Lakukan pemantauan rutin dan antisipasi penurunan kelembaban lahan."
    return "Kondisi relatif aman, tetap lanjutkan monitoring berkala."

@st.cache_data
def load_jatim_geojson(path="jatim.geojson"):
    """Memuat GeoJSON Jawa Timur. File harus berada pada folder yang sama dengan app.py."""
    with open(path, "r", encoding="utf-8") as f:
        geo = json.load(f)

    # GeoJSON ini memakai NAME_2 dan TYPE_2.
    # Beberapa kota memakai NAME_2 seperti KotaBlitar, sehingga perlu dirapikan.
    for feature in geo.get("features", []):
        props = feature.get("properties", {})
        tipe = str(props.get("TYPE_2", "")).strip()
        nama = str(props.get("NAME_2", "")).strip()

        if tipe.lower() == "kota":
            nama_clean = re.sub(r"^Kota", "", nama, flags=re.IGNORECASE).strip()
            nama_lengkap = f"Kota {nama_clean}"
        else:
            nama_lengkap = f"Kabupaten {nama}"

        props["nama_lengkap"] = nama_lengkap
        feature["id"] = normalize_area_name(nama_lengkap)

    return geo

def normalize_area_name(name):
    """Menyamakan format nama wilayah agar data CSV bisa cocok dengan GeoJSON."""
    name = str(name).lower().strip()
    name = name.replace("kota ", "kota ")
    name = name.replace("kab. ", "kabupaten ")
    name = name.replace("kab ", "kabupaten ")
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def mode_value(series):
    mode = series.dropna().mode()
    if len(mode) == 0:
        return np.nan
    return mode.iloc[0]

st.markdown(
    """
    <style>
        .main {background-color: #F7F3EA;}
        .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
        section[data-testid="stSidebar"] {background: #3B3326;}
        section[data-testid="stSidebar"] * {color: #ffffff !important;}
        div[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
        .small-note {font-size: 0.9rem; color: #64748b;}
        .info-box {
            background:#FFFDF7;
            padding:16px 18px;
            border-radius:16px;
            border:1px solid #E7DEC8;
            box-shadow:0 4px 14px rgba(76,64,35,0.08);
            color:#3B3326;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
df = load_data()

st.sidebar.title("Navigasi Dashboard")
menu = st.sidebar.radio(
    "Pilih halaman",
    ["Beranda / EDA", "Analisis Prediksi", "Analisis Klasifikasi", "Insight & Rekomendasi"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Data")

selected_year = st.sidebar.selectbox(
    "Tahun",
    sorted(df["tahun"].unique(), reverse=True)
)

list_kabupaten = [col for col in df.columns if col != "tahun"]
selected_city = st.sidebar.selectbox(
    "Kabupaten/Kota",
    ["Semua"] + sorted(list_kabupaten)
)

filtered = df[df["tahun"] == selected_year].copy()

eda_source = df.copy()

st.title("Dashboard Prediksi dan Klasifikasi Kekeringan di Jawa Timur")
st.markdown(
    """
    <div style="display:flex;gap:16px;align-items:center;font-size:14px;color:#64748b;margin-top:-10px;">
        <span>🌾 Berbasis NDVI</span>
        <span>•</span>
        <span style="color:#6B3F16;font-weight:600;">Data historis dan hasil pemodelan</span>
        <span>•</span>
        <span>38 Kabupaten/Kota</span>
    </div>
    """,
    unsafe_allow_html=True
)

# EDA
def safe_classify_ndvi(x):
    """Klasifikasi NDVI yang aman untuk nilai kosong."""
    if pd.isna(x):
        return "Tidak Diketahui"
    return classify_ndvi(float(x))


def safe_color_class(kelas):
    """Mapping warna yang aman, selalu return warna valid."""
    return class_color_cls(kelas)


def render_eda_card(kabupaten, kelas, ndvi, warna):
    """Card kabupaten tanpa multiline HTML yang rawan kebaca sebagai code block."""
    kabupaten = html.escape(str(kabupaten))
    kelas = html.escape(str(kelas))

    try:
        ndvi_text = f"{float(ndvi):.2f}"
    except Exception:
        ndvi_text = "-"

    card_html = f"""
    <div style="background:{warna};padding:14px 16px;border-radius:14px;color:white;margin-bottom:12px;height:150px;box-sizing:border-box;overflow:hidden;">
        <div style="font-size:17px;font-weight:700;line-height:1.25;min-height:44px;word-break:break-word;">
            {kabupaten}
        </div>
        <div style="font-size:13px;line-height:1.35;margin-top:10px;">
            {kelas}
        </div>
        <div style="font-size:14px;line-height:1.35;margin-top:14px;">
            NDVI: <b>{ndvi_text}</b>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

if menu == "Beranda / EDA":
    st.subheader("Ringkasan Kondisi Jawa Timur")
    st.caption(
        "Halaman ini menampilkan ringkasan seluruh kabupaten/kota di Jawa Timur "
        "dan tidak mengikuti filter sidebar."
    )

    # ambil semua kolom kabupaten/kota kecuali tahun
    kabupaten_cols = [col for col in eda_source.columns if col != "tahun"]

    if not kabupaten_cols:
        st.warning("Kolom kabupaten/kota tidak ditemukan pada data EDA.")
        st.stop()

    # hitung rerata NDVI per kabupaten/kota
    eda_df = pd.DataFrame({
        "kabupaten": kabupaten_cols,
        "ndvi": [pd.to_numeric(eda_source[col], errors="coerce").mean() for col in kabupaten_cols]
    })

    # bersihkan data
    eda_df["kabupaten"] = eda_df["kabupaten"].astype(str).str.strip()
    eda_df["ndvi"] = pd.to_numeric(eda_df["ndvi"], errors="coerce")
    eda_df["kelas"] = eda_df["ndvi"].apply(safe_classify_ndvi).astype(str).str.strip()
    eda_df["warna"] = eda_df["kelas"].apply(safe_color_class)

    class_order = [
        "Kehijauan Sangat Rendah",
        "Kehijauan Rendah",
        "Kehijauan Sedang",
        "Kehijauan Tinggi",
        "Tidak Diketahui",
    ]

    # metric summary
    total_wilayah = int(eda_df["kabupaten"].nunique())
    rata_ndvi = float(eda_df["ndvi"].mean()) if eda_df["ndvi"].notna().any() else 0.0
    ndvi_min = float(eda_df["ndvi"].min()) if eda_df["ndvi"].notna().any() else 0.0
    ndvi_max = float(eda_df["ndvi"].max()) if eda_df["ndvi"].notna().any() else 0.0
    kritis_count = int((eda_df["kelas"] == "Kehijauan Sangat Rendah").sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Wilayah Kritis", f"{kritis_count}", "Kehijauan sangat rendah")
    with c2:
        metric_card("Total Kabupaten/Kota", f"{total_wilayah}", "Seluruh Jawa Timur")
    with c3:
        metric_card("Rata-rata NDVI", f"{rata_ndvi:.2f}", "Rata-rata seluruh wilayah")
    with c4:
        metric_card("Rentang NDVI", f"{ndvi_min:.2f} - {ndvi_max:.2f}", "Minimum hingga maksimum")

    #pie chart dkk
    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Distribusi Tingkat Kehijauan")

        dist = (
            eda_df["kelas"]
            .value_counts(dropna=False)
            .rename_axis("kelas")
            .reset_index(name="jumlah")
        )

        dist["kelas"] = dist["kelas"].astype(str).str.strip()
        existing_classes = [k for k in class_order if k in dist["kelas"].tolist()]

        if not dist.empty:
            fig_pie = px.pie(
                dist,
                names="kelas",
                values="jumlah",
                hole=0.35,
                color="kelas",
                category_orders={"kelas": existing_classes},
                color_discrete_map={k: safe_color_class(k) for k in existing_classes},
            )
            fig_pie.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=380
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Distribusi kelas belum tersedia.")

    with right:
        st.markdown("#### Top Wilayah NDVI Terendah")

        top_area = (
            eda_df.dropna(subset=["ndvi"])
            .sort_values("ndvi", ascending=True)
            .head(6)
            .copy()
        )

        if not top_area.empty:
            fig_bar = go.Figure()
            fig_bar.add_trace(
                go.Bar(
                    x=top_area["ndvi"],
                    y=top_area["kabupaten"],
                    orientation="h",
                    marker=dict(color=top_area["warna"].tolist()),
                    text=[f"{v:.2f}" for v in top_area["ndvi"]],
                    textposition="outside",
                )
            )
            fig_bar.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=380,
                xaxis_title="Nilai NDVI",
                yaxis_title="",
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Data NDVI terendah belum tersedia.")

    legend_items = []
    for kelas in class_order:
        jumlah = int((eda_df["kelas"] == kelas).sum())
        if jumlah > 0:
            legend_items.append((kelas, jumlah))

    if legend_items:
        pills_html = '<div style="display:flex;flex-wrap:wrap;gap:12px;justify-content:flex-start;align-items:center;margin-top:8px;">'

        for kelas, jumlah in legend_items:
            pills_html += (
                f'<div style="display:inline-block;'
                f'background:{safe_color_class(kelas)};'
                f'color:white;padding:8px 14px;border-radius:999px;'
                f'font-size:14px;font-weight:600;white-space:nowrap;'
                f'box-shadow:0 2px 6px rgba(0,0,0,0.1);">'
                f'{html.escape(kelas)}: {jumlah}'
                f'</div>'
            )

        pills_html += '</div>'

        st.markdown(pills_html, unsafe_allow_html=True)

    #card kabupaten
    st.markdown("#### Ringkasan Kabupaten/Kota")

    eda_cards = (
        eda_df.sort_values(["ndvi", "kabupaten"], ascending=[True, True], na_position="last")
        .reset_index(drop=True)
        .copy()
    )

    if not eda_cards.empty:
        n_cols = 6
        for i in range(0, len(eda_cards), n_cols):
            row_slice = eda_cards.iloc[i:i+n_cols]
            cols = st.columns(n_cols)

            for j, (_, row) in enumerate(row_slice.iterrows()):
                with cols[j]:
                    render_eda_card(
                        kabupaten=row["kabupaten"],
                        kelas=row["kelas"],
                        ndvi=row["ndvi"],
                        warna=row["warna"],
                    )
    else:
        st.info("Ringkasan kabupaten/kota belum tersedia.")

    #tabel
    st.markdown("#### Detail Kabupaten/Kota Jawa Timur")

    detail_eda = eda_df.sort_values(["ndvi", "kabupaten"], ascending=[True, True], na_position="last").copy()
    detail_eda = detail_eda.rename(columns={
        "kabupaten": "Kabupaten/Kota",
        "kelas": "Status",
        "ndvi": "NDVI"
    })

    st.dataframe(
        detail_eda[["Kabupaten/Kota", "Status", "NDVI"]],
        use_container_width=True,
        hide_index=True
    )

#prediksi
elif menu == "Analisis Prediksi":
    st.subheader(f"Tren Prediksi NDVI Tahun {selected_year}")

    if selected_city == "Semua":
        rmse_val = df_eval["RMSE"].mean()
        mae_val = df_eval["MAE"].mean()
        mse_val = df_eval["MSE"].mean()
    else:
        row = df_eval[df_eval["Kabupaten/Kota"] == selected_city]

        if not row.empty:
            rmse_val = row["RMSE"].values[0]
            mae_val = row["MAE"].values[0]
            mse_val = row["MSE"].values[0]
        else:
            rmse_val, mae_val, mse_val = 0, 0, 0

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("RMSE", f"{rmse_val:.4f}")

    with c2:
        metric_card("MSE", f"{mse_val:.4f}")

    with c3:
        metric_card("MAE", f"{mae_val:.4f}")

    df_filtered = df[df["tahun"] == selected_year].copy()
    list_kabupaten = [col for col in df.columns if col != "tahun"]

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown(f"#### Pergerakan Nilai NDVI: {selected_city}")

        fig_line = go.Figure()

        if selected_city == "Semua":
            avg_jatim = df_filtered[list_kabupaten].mean(axis=1)
            fig_line.add_trace(go.Scatter(
                x=df_filtered.index,
                y=avg_jatim,
                mode="lines+markers",
                name="Rata-rata Jawa Timur",
                fill="tozeroy"
            ))
        else:
            fig_line.add_trace(go.Scatter(
                x=df_filtered.index,
                y=df_filtered[selected_city],
                mode="lines+markers",
                name=selected_city,
                fill="tozeroy"
            ))

        fig_line.update_layout(
            height=420,
            xaxis_title="Bulan",
            yaxis_title="Nilai NDVI",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with right:
        st.markdown("#### Ringkasan Forecast")
        if selected_city == "Semua":
            ringkasan = pd.DataFrame({
                "Metric": ["Rata-rata NDVI", "Nilai Minimum", "Nilai Maksimum", "Jumlah Bulan"],
                "Nilai": [
                    round(df_filtered[list_kabupaten].mean(axis=1).mean(), 4),
                    round(df_filtered[list_kabupaten].mean(axis=1).min(), 4),
                    round(df_filtered[list_kabupaten].mean(axis=1).max(), 4),
                    len(df_filtered)
                ]
            })
        else:
            ringkasan = pd.DataFrame({
                "Metric": ["Rata-rata NDVI", "Nilai Minimum", "Nilai Maksimum", "Jumlah Bulan"],
                "Nilai": [
                    round(df_filtered[selected_city].mean(), 4),
                    round(df_filtered[selected_city].min(), 4),
                    round(df_filtered[selected_city].max(), 4),
                    len(df_filtered)
                ]
            })

        st.dataframe(ringkasan, use_container_width=True, hide_index=True)

    st.markdown("#### Detail Angka Forecast (Bulanan)")
    if selected_city == "Semua":
        detail_df = df_filtered[list_kabupaten].copy()
    else:
        detail_df = df_filtered[[selected_city]].copy()

    detail_df = detail_df.reset_index()
    detail_df = detail_df.rename(columns={detail_df.columns[0]: "tanggal"})
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    st.info("💡 Gunakan filter tahun dan kabupaten/kota di sidebar untuk melihat tren NDVI sesuai pilihan.")

#klasifikasi
elif menu == "Analisis Klasifikasi":
    st.subheader("Peta Klasifikasi Risiko Kekeringan")
    st.caption(
        "Halaman ini menampilkan hasil klasifikasi risiko kekeringan berbasis NDVI "
        "menggunakan model terbaik, yaitu XGBoost dengan oversampling."
    )

    # Metrik final sesuai hasil Bab 4: XGBoost dengan Oversampling
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Accuracy", f"{0.7810:.2%}", "Data uji")
    with c2:
        metric_card("Precision", f"{0.7795:.2%}", "Weighted")
    with c3:
        metric_card("Recall", f"{0.7810:.2%}", "Weighted")
    with c4:
        metric_card("F1-Macro", f"{0.7505:.2%}", "Metrik utama")
    with c5:
        metric_card("F1-Weighted", f"{0.7793:.2%}", "Weighted")

    st.markdown(
        """
        <div class="info-box">
            <b>Model terbaik:</b> XGBoost dengan Oversampling.<br>
            Model dipilih berdasarkan nilai F1-Macro tertinggi karena distribusi kelas risiko tidak seimbang.
            Palet warna dibuat dari hijau, kuning, hingga coklat. Semakin coklat warna wilayah,
            semakin rendah kehijauan vegetasi dan semakin tinggi risiko kekeringan berbasis NDVI.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Jika tahun filter dari data prediksi tidak tersedia pada data klasifikasi,
    # gunakan tahun terbaru pada data klasifikasi agar halaman tidak kosong.
    cls_years = sorted(pd.to_numeric(df_cls["tahun"], errors="coerce").dropna().astype(int).unique())
    map_year = selected_year
    if cls_years and selected_year not in cls_years:
        map_year = max(cls_years)
        st.info(f"Data klasifikasi untuk tahun {selected_year} tidak tersedia. Menampilkan tahun {map_year}.")

    filtered_cls = df_cls[df_cls["tahun"] == map_year].copy()

    if selected_city != "Semua":
        filtered_cls = filtered_cls[filtered_cls["Kabupaten/Kota"] == selected_city].copy()

    if filtered_cls.empty:
        st.warning("Tidak ada data klasifikasi untuk filter yang dipilih.")
    else:
        filtered_cls["kelas_ndvi_num"] = pd.to_numeric(filtered_cls["kelas_ndvi"], errors="coerce")
        filtered_cls["pred_num"] = pd.to_numeric(filtered_cls["Prediksi_Risiko_Model"], errors="coerce")

        eval_cls = filtered_cls.dropna(subset=["kelas_ndvi_num", "pred_num"]).copy()
        eval_cls["kelas_ndvi_num"] = eval_cls["kelas_ndvi_num"].astype(int)
        eval_cls["pred_num"] = eval_cls["pred_num"].astype(int)

        valid_labels = [0, 1, 2, 3]
        eval_cls = eval_cls[
            eval_cls["kelas_ndvi_num"].isin(valid_labels) &
            eval_cls["pred_num"].isin(valid_labels)
        ].copy()

        if eval_cls.empty:
            st.warning("Data klasifikasi tidak valid untuk divisualisasikan.")
        else:
            map_basis = st.radio(
                "Warna peta berdasarkan:",
                ["Kelas Prediksi", "Kelas Aktual"],
                horizontal=True
            )
            target_col = "pred_num" if map_basis == "Kelas Prediksi" else "kelas_ndvi_num"

            # Agregasi per wilayah agar peta menampilkan satu warna untuk setiap kabupaten/kota.
            map_df = (
                eval_cls
                .groupby("Kabupaten/Kota")
                .agg(
                    NDVI=("NDVI", "mean"),
                    kelas_kode=(target_col, mode_value),
                    jumlah_data=(target_col, "count")
                )
                .reset_index()
            )
            map_df["kelas_kode"] = map_df["kelas_kode"].astype(int)
            map_df["Kelas"] = map_df["kelas_kode"].apply(risk_label)
            map_df["Risiko"] = map_df["Kelas"].apply(risk_level)
            map_df["geo_id"] = map_df["Kabupaten/Kota"].apply(normalize_area_name)
            map_df["Kelas"] = pd.Categorical(map_df["Kelas"], categories=CLASS_ORDER, ordered=True)

            left, right = st.columns([1.55, 1])

            with left:
                st.markdown("#### Peta Risiko Kekeringan Jawa Timur")
                try:
                    jatim_geojson = load_jatim_geojson("jatim.geojson")

                    fig_map = px.choropleth(
                        map_df,
                        geojson=jatim_geojson,
                        locations="geo_id",
                        featureidkey="id",
                        color="Kelas",
                        color_discrete_map=COLOR_MAP,
                        hover_name="Kabupaten/Kota",
                        hover_data={
                            "NDVI": ":.3f",
                            "Risiko": True,
                            "jumlah_data": True,
                            "geo_id": False,
                            "kelas_kode": False,
                        },
                    )
                    fig_map.update_geos(fitbounds="locations", visible=False)
                    fig_map.update_layout(
                        height=520,
                        margin=dict(l=0, r=0, t=0, b=0),
                        legend_title_text="Kelas NDVI",
                    )
                    st.plotly_chart(fig_map, use_container_width=True)

                except FileNotFoundError:
                    st.warning(
                        "File `jatim.geojson` belum ditemukan. Letakkan file tersebut "
                        "di folder yang sama dengan file dashboard Streamlit."
                    )
                except Exception as e:
                    st.error(f"Peta belum dapat ditampilkan: {e}")

            with right:
                st.markdown("#### Ringkasan Risiko Wilayah")
                jumlah_wilayah = map_df["Kabupaten/Kota"].nunique()
                sangat_rendah = int((map_df["Kelas"] == "Kehijauan Sangat Rendah").sum())
                rendah = int((map_df["Kelas"] == "Kehijauan Rendah").sum())
                sedang = int((map_df["Kelas"] == "Kehijauan Sedang").sum())
                tinggi = int((map_df["Kelas"] == "Kehijauan Tinggi").sum())

                metric_card("Wilayah Terpetakan", f"{jumlah_wilayah}", f"Tahun {map_year}")
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="info-box">
                        <b>Distribusi Kelas</b><br><br>
                        <span style="color:#6B3F16;font-weight:700;">● Kehijauan sangat rendah:</span> {sangat_rendah} wilayah<br>
                        <span style="color:#B7791F;font-weight:700;">● Kehijauan rendah:</span> {rendah} wilayah<br>
                        <span style="color:#D9C75F;font-weight:700;">● Kehijauan sedang:</span> {sedang} wilayah<br>
                        <span style="color:#2E7D32;font-weight:700;">● Kehijauan tinggi:</span> {tinggi} wilayah
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("#### Wilayah Prioritas")
                prioritas = map_df.sort_values("NDVI", ascending=True).head(6)
                st.dataframe(
                    prioritas[["Kabupaten/Kota", "NDVI", "Kelas", "Risiko"]],
                    use_container_width=True,
                    hide_index=True
                )

            bottom_left, bottom_right = st.columns([1, 1])

            with bottom_left:
                st.markdown("#### Distribusi Kelas")
                dist_cls = (
                    map_df["Kelas"].astype(str)
                    .value_counts()
                    .rename_axis("Kelas")
                    .reset_index(name="Jumlah")
                )
                dist_cls["Kelas"] = pd.Categorical(dist_cls["Kelas"], categories=CLASS_ORDER, ordered=True)
                dist_cls = dist_cls.sort_values("Kelas")

                fig_cls = px.bar(
                    dist_cls,
                    x="Kelas",
                    y="Jumlah",
                    color="Kelas",
                    text="Jumlah",
                    color_discrete_map=COLOR_MAP,
                    category_orders={"Kelas": CLASS_ORDER},
                )
                fig_cls.update_traces(textposition="outside")
                fig_cls.update_layout(
                    height=390,
                    showlegend=False,
                    xaxis_title="Kelas NDVI",
                    yaxis_title="Jumlah Wilayah",
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_cls, use_container_width=True)

            with bottom_right:
                st.markdown("#### Confusion Matrix")
                y_true = eval_cls["kelas_ndvi_num"]
                y_pred = eval_cls["pred_num"]
                label_order_num = [0, 1, 2, 3]
                label_order_text = [risk_label(x) for x in label_order_num]
                cm = confusion_matrix(y_true, y_pred, labels=label_order_num)

                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    x=label_order_text,
                    y=label_order_text,
                    aspect="auto",
                    color_continuous_scale="YlOrBr",
                )
                fig_cm.update_layout(
                    height=390,
                    xaxis_title="Prediksi",
                    yaxis_title="Aktual",
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown("#### Detail Klasifikasi per Wilayah")
            detail_table = map_df[["Kabupaten/Kota", "NDVI", "Kelas", "Risiko", "jumlah_data"]].copy()
            detail_table = detail_table.rename(columns={"jumlah_data": "Jumlah Data"})
            st.dataframe(
                detail_table.sort_values("NDVI", ascending=True),
                use_container_width=True,
                hide_index=True
            )

#rekomendasi
else:
    st.subheader("Insight dan Rekomendasi")
    st.markdown(
        """
        **Insight utama:**
        1. Nilai NDVI yang lebih rendah menunjukkan tingkat kehijauan vegetasi yang lebih rendah.
        2. Wilayah dengan kehijauan sangat rendah perlu dipantau lebih intensif.
        3. Hasil prediksi NDVI dapat digunakan sebagai dasar pemantauan kondisi vegetasi antar wilayah.
        """
    )

    kabupaten_cols = [col for col in filtered.columns if col != "tahun"]

    rank = pd.DataFrame({
        "kabupaten": kabupaten_cols,
        "ndvi": [filtered[col].mean() for col in kabupaten_cols]
    })

    rank["kelas"] = rank["ndvi"].apply(classify_ndvi)

    def recommendation_green(row):
        if row["kelas"] == "Kehijauan Sangat Rendah":
            return "Prioritaskan pemantauan lapangan dan intervensi cepat."
        elif row["kelas"] == "Kehijauan Rendah":
            return "Tingkatkan monitoring vegetasi dan antisipasi penurunan kondisi lahan."
        elif row["kelas"] == "Kehijauan Sedang":
            return "Lakukan pemantauan rutin untuk menjaga kestabilan kondisi vegetasi."
        return "Kondisi vegetasi relatif baik, lanjutkan monitoring berkala."

    rank["rekomendasi"] = rank.apply(recommendation_green, axis=1)
    rank = rank.sort_values("ndvi", ascending=True)

    st.markdown("#### Prioritas Wilayah Berdasarkan Tingkat Kehijauan")
    st.dataframe(
        rank[["kabupaten", "ndvi", "kelas", "rekomendasi"]],
        use_container_width=True,
        hide_index=True,
    )

    top_rank = rank.head(10).copy()

    class_order = [
        "Kehijauan Sangat Rendah",
        "Kehijauan Rendah",
        "Kehijauan Sedang",
        "Kehijauan Tinggi",
    ]

    existing_rank_classes = [k for k in class_order if k in top_rank["kelas"].values]

    if top_rank.empty or not existing_rank_classes:
        st.warning("Tidak ada data insight untuk ditampilkan.")
    else:
        top_rank["kelas"] = pd.Categorical(
            top_rank["kelas"],
            categories=existing_rank_classes,
            ordered=True
        )
        top_rank = top_rank.sort_values("kelas")

        fig_rank = px.bar(
            top_rank,
            x="kabupaten",
            y="ndvi",
            color="kelas",
            category_orders={"kelas": existing_rank_classes},
            color_discrete_map={k: class_color_cls(k) for k in existing_rank_classes},
        )
        fig_rank.update_layout(
            height=420,
            xaxis_title="Kabupaten/Kota",
            yaxis_title="Nilai NDVI"
        )
        st.plotly_chart(fig_rank, use_container_width=True)

st.markdown("---")
st.caption(f"Terakhir diperbarui: {datetime.now().strftime('%d %B %Y %H:%M')} | Prototype dashboard Streamlit")
