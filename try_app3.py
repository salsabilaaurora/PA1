import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pickle

st.set_page_config(
    page_title="Dashboard Prediksi dan Klasifikasi Kekeringan Jawa Timur",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_data
def load_sample_data():
    data = [
        ["Sampang", 2025, 12, 0.11, 5, 34, 42, 1045, 178000, 12, "Ekstrim", 0.83],
        ["Bangkalan", 2025, 12, 0.14, 8, 33, 45, 892, 125000, 18, "Ekstrim", 0.81],
        ["Pamekasan", 2025, 12, 0.19, 12, 33, 46, 567, 98000, 22, "Parah", 0.76],
        ["Situbondo", 2025, 12, 0.21, 15, 32, 48, 678, 89000, 25, "Parah", 0.74],
        ["Jember", 2025, 12, 0.22, 18, 31, 50, 734, 156000, 28, "Parah", 0.73],
        ["Bojonegoro", 2025, 12, 0.24, 17, 32, 47, 610, 77000, 24, "Parah", 0.72],
        ["Probolinggo", 2025, 12, 0.25, 20, 31, 52, 590, 86000, 30, "Parah", 0.71],
        ["Tuban", 2025, 12, 0.27, 22, 31, 54, 480, 70000, 31, "Parah", 0.69],
        ["Bondowoso", 2025, 12, 0.28, 24, 30, 56, 530, 65000, 35, "Parah", 0.68],
        ["Banyuwangi", 2025, 12, 0.35, 38, 29, 63, 310, 42000, 44, "Sedang", 0.55],
        ["Kediri", 2025, 12, 0.38, 42, 29, 66, 280, 35000, 48, "Sedang", 0.51],
        ["Magetan", 2025, 12, 0.39, 43, 28, 67, 260, 30000, 49, "Sedang", 0.50],
        ["Probolinggo Kota", 2025, 12, 0.40, 44, 29, 68, 230, 29000, 50, "Sedang", 0.49],
        ["Ponorogo", 2025, 12, 0.41, 45, 28, 69, 220, 28000, 51, "Sedang", 0.48],
        ["Mojokerto", 2025, 12, 0.42, 44, 29, 68, 240, 31000, 50, "Sedang", 0.50],
        ["Nganjuk", 2025, 12, 0.42, 43, 29, 67, 250, 32000, 49, "Sedang", 0.50],
        ["Tulungagung", 2025, 12, 0.43, 41, 28, 66, 270, 33000, 48, "Sedang", 0.52],
        ["Lamongan", 2025, 12, 0.44, 40, 29, 65, 275, 34000, 47, "Sedang", 0.53],
        ["Kota Kediri", 2025, 12, 0.44, 40, 29, 65, 190, 26000, 47, "Sedang", 0.54],
        ["Madiun", 2025, 12, 0.47, 48, 28, 70, 210, 27000, 53, "Ringan", 0.40],
        ["Pacitan", 2025, 12, 0.56, 57, 27, 75, 150, 22000, 60, "Ringan", 0.28],
        ["Malang", 2025, 12, 0.58, 60, 26, 77, 145, 21000, 62, "Ringan", 0.25],
        ["Kota Batu", 2025, 12, 0.68, 75, 24, 82, 90, 12000, 73, "Normal", 0.12],
    ]

    cols = [
        "kabupaten", "tahun", "bulan", "ndvi", "curah_hujan", "suhu", "kelembaban",
        "luas_terdampak", "populasi", "reservoir", "kelas", "prob_kering"
    ]
    return pd.DataFrame(data, columns=cols)

def class_color(label: str):
    mapping = {
        "Normal": "#16a34a",
        "Ringan": "#eab308",
        "Sedang": "#f97316",
        "Parah": "#ef4444",
        "Ekstrim": "#7f1d1d",
    }
    return mapping.get(label, "#64748b")


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

st.markdown(
    """
    <style>
        .main {background-color: #f8fafc;}
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        section[data-testid="stSidebar"] {background-color: #ffffff;}
        div[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
        .small-note {font-size: 0.9rem; color: #64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)
df = load_sample_data()

st.sidebar.title("Navigasi Dashboard")
menu = st.sidebar.radio(
    "Pilih halaman",
    ["Beranda / EDA", "Analisis Prediksi", "Analisis Klasifikasi", "Insight & Rekomendasi"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Data")
selected_year = st.sidebar.selectbox("Tahun", sorted(df["tahun"].unique(), reverse=True))
selected_city = st.sidebar.selectbox("Kabupaten/Kota", ["Semua"] + sorted(df["kabupaten"].unique().tolist()))

filtered = df[df["tahun"] == selected_year].copy()
if selected_city != "Semua":
    filtered = filtered[filtered["kabupaten"] == selected_city]

st.title("Dashboard Prediksi dan Klasifikasi Kekeringan di Jawa Timur")
st.markdown(
    """
    <div style="display:flex;gap:16px;align-items:center;font-size:14px;color:#64748b;margin-top:-10px;">
        <span>📅 Senin, 22 Desember 2025</span>
        <span>•</span>
        <span style="color:#16a34a;font-weight:600;">Data Real-time</span>
        <span>•</span>
        <span>38 Kabupaten/Kota</span>
    </div>
    """,
    unsafe_allow_html=True
)

#EDA
if menu == "Beranda / EDA":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Wilayah", f"{filtered['kabupaten'].nunique()}", "Kabupaten/kota terfilter")
    with c2:
        metric_card("Rata-rata NDVI", f"{filtered['ndvi'].mean():.2f}", "Semakin rendah, risiko makin tinggi")
    with c3:
        metric_card("Total Luas Terdampak", f"{filtered['luas_terdampak'].sum():,.0f} km²", "Akumulasi wilayah terdampak")
    with c4:
        metric_card("Total Populasi", f"{filtered['populasi'].sum():,}", "Estimasi populasi terdampak")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Distribusi Tingkat Kekeringan")
        dist = filtered["kelas"].value_counts().reset_index()
        dist.columns = ["kelas", "jumlah"]
        fig_pie = px.pie(
            dist,
            names="kelas",
            values="jumlah",
            hole=0.35,
            color="kelas",
            color_discrete_map={k: class_color(k) for k in dist["kelas"].unique()},
        )
        fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with right:
        st.subheader("Top Wilayah Terdampak")
        top_area = filtered.sort_values("luas_terdampak", ascending=True).tail(10)
        fig_bar = px.bar(
            top_area,
            x="luas_terdampak",
            y="kabupaten",
            orientation="h",
            color="kelas",
            color_discrete_map={
                "Normal": class_color("Normal"),
                "Ringan": class_color("Ringan"),
                "Sedang": class_color("Sedang"),
                "Parah": class_color("Parah"),
                "Ekstrim": class_color("Ekstrim"),
            },
        )
        fig_bar.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=380, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Kartu Ringkas Kabupaten/Kota")
    cards = st.columns(4)
    for idx, (_, row) in enumerate(filtered.reset_index(drop=True).iterrows()):
        col = cards[idx % 4]
        with col:
            st.markdown(
                f"""
                <div style="background:{class_color(row['kelas'])};padding:16px;border-radius:14px;color:white;margin-bottom:12px;min-height:124px;">
                    <div style="font-size:20px;font-weight:700;">{row['kabupaten']}</div>
                    <div style="font-size:15px;margin-top:6px;">{row['kelas']}</div>
                    <div style="font-size:14px;margin-top:10px;">NDVI: {row['ndvi']:.2f}</div>
                    <div style="font-size:14px;">Curah hujan: {row['curah_hujan']} mm</div>
                    <div style="font-size:14px;">Reservoir: {row['reservoir']}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Detail Data")
    show_cols = [
        "kabupaten", "kelas", "ndvi", "curah_hujan", "suhu", "kelembaban",
        "luas_terdampak", "populasi", "reservoir", "prob_kering"
    ]
    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)

#prediksi
elif menu == "Analisis Prediksi":
    st.subheader("Analisis Prediksi Kekeringan")

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("RMSE", "0.098", "Contoh hasil model GSTAR")
    with c2:
        metric_card("MSE", "0.010", "Contoh hasil evaluasi")
    with c3:
        metric_card("MAE", "0.0795", "Contoh hasil evaluasi")

    pred_df = filtered[["kabupaten", "ndvi", "prob_kering", "curah_hujan", "luas_terdampak"]].copy()
    pred_df["prediksi_status"] = pd.cut(
        pred_df["prob_kering"],
        bins=[0, 0.4, 0.6, 0.75, 1.0],
        labels=["Aman", "Waspada", "Siaga", "Prioritas"],
        include_lowest=True,
    )

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Probabilitas Risiko Kekeringan")
        fig_pred = px.bar(
            pred_df.sort_values("prob_kering", ascending=False),
            x="kabupaten",
            y="prob_kering",
            color="prediksi_status",
        )
        fig_pred.update_layout(height=420, xaxis_title="Kabupaten/Kota", yaxis_title="Probabilitas")
        st.plotly_chart(fig_pred, use_container_width=True)

    with right:
        st.markdown("#### Hubungan NDVI dan Curah Hujan")
        fig_scatter = px.scatter(
            pred_df,
            x="curah_hujan",
            y="ndvi",
            size="luas_terdampak",
            color="prob_kering",
            hover_name="kabupaten",
        )
        fig_scatter.update_layout(height=420)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("#### Tabel Hasil Prediksi")
    pred_view = pred_df.copy()
    pred_view["rekomendasi"] = filtered.apply(recommendation_rule, axis=1).values
    st.dataframe(pred_view, use_container_width=True, hide_index=True)

#klasifikasi
elif menu == "Analisis Klasifikasi":
    st.subheader("Analisis Klasifikasi Kekeringan")
    st.caption("Halaman ini menampilkan performa klasifikasi dan distribusi kelas, tanpa rekomendasi kebijakan.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Accuracy", "0.82")
    with c2:
        metric_card("Precision", "0.80")
    with c3:
        metric_card("Recall", "0.78")
    with c4:
        metric_card("F1 Score", "0.79")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Distribusi Kelas Aktual")
        fig_cls = px.histogram(
            filtered,
            x="kelas",
            color="kelas",
            category_orders={"kelas": ["Normal", "Ringan", "Sedang", "Parah", "Ekstrim"]},
            color_discrete_map={
                "Normal": class_color("Normal"),
                "Ringan": class_color("Ringan"),
                "Sedang": class_color("Sedang"),
                "Parah": class_color("Parah"),
                "Ekstrim": class_color("Ekstrim"),
            },
        )
        fig_cls.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_cls, use_container_width=True)

    with right:
        st.markdown("#### Simulasi Confusion Matrix")
        cm = np.array([
            [12, 1, 0, 0, 0],
            [1, 10, 2, 0, 0],
            [0, 1, 9, 2, 0],
            [0, 0, 1, 8, 1],
            [0, 0, 0, 1, 6],
        ])
        labels = ["Normal", "Ringan", "Sedang", "Parah", "Ekstrim"]
        fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels, aspect="auto")
        fig_cm.update_layout(height=400, xaxis_title="Prediksi", yaxis_title="Aktual")
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("#### Detail Kelas per Wilayah")
    st.dataframe(filtered[["kabupaten", "kelas", "ndvi", "curah_hujan", "suhu", "kelembaban"]], use_container_width=True, hide_index=True)

#rekomendasi
else:
    st.subheader("Insight dan Rekomendasi")
    st.markdown(
        """
        **Insight utama:**
        1. Wilayah dengan NDVI rendah, curah hujan rendah, dan reservoir kecil cenderung memiliki probabilitas kekeringan lebih tinggi.
        2. Konsentrasi risiko tertinggi terlihat pada wilayah Madura dan beberapa kabupaten di pantura.
        3. Klasifikasi berguna untuk pemetaan status, tetapi rekomendasi tindakan difokuskan pada hasil prediksi.
        """
    )

    rank = filtered.sort_values("prob_kering", ascending=False).copy()
    rank["rekomendasi"] = rank.apply(recommendation_rule, axis=1)

    st.markdown("#### Prioritas Rekomendasi Berbasis Prediksi")
    st.dataframe(
        rank[["kabupaten", "prob_kering", "kelas", "luas_terdampak", "populasi", "rekomendasi"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Ringkasan untuk Bab Sistem")
    st.info(
        "Pada desain sistem, sistem rekomendasi ditempatkan sebagai keluaran dari modul prediksi, bukan modul klasifikasi."
    )

    fig_rank = px.bar(
        rank.head(10),
        x="kabupaten",
        y="prob_kering",
        color="kelas",
        color_discrete_map={
            "Normal": class_color("Normal"),
            "Ringan": class_color("Ringan"),
            "Sedang": class_color("Sedang"),
            "Parah": class_color("Parah"),
            "Ekstrim": class_color("Ekstrim"),
        },
    )
    fig_rank.update_layout(height=420, xaxis_title="Kabupaten/Kota", yaxis_title="Probabilitas Kekeringan")
    st.plotly_chart(fig_rank, use_container_width=True)

st.markdown("---")
st.caption(f"Terakhir diperbarui: {datetime.now().strftime('%d %B %Y %H:%M')} | Prototype dashboard Streamlit")
