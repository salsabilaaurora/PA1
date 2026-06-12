import streamlit as st 
import pandas as pd
import numpy as np

st.set_page_config(
    page_title='Streamlit Dashboard', 
    layout='wide', 
    page_icon='💹'
)
st.markdown(""" /*padding atas banget*/
    <style>
    .block-container {
        padding-top: 2rem;    
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; margin-bottom: 0px;'>DASHBOARD PREDIKSI DAN KLASIFIKASI KEKERINGAN DI JAWA TIMUR</h3>", unsafe_allow_html=True)
#st.write("---"))

first_kpi, second_kpi, third_kpi, fourth_kpi = st.columns(4, gap="small")

def kpi_card(title, value):
    st.markdown(f"""
    <div style="
        background-color: #1f2937;
        padding: 10px 12px; 
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        align-items: center;    /* Angka otomatis di tengah */
        gap: 2px;
        line-height: 1.1;
    ">
        <span style="
            color: white; 
            font-size: 1.2rem; 
            align-self: flex-start; /* Judul ke kiri */
        ">{title}</span>
        <span style="
            color: #4ade80; 
            font-size: 1.8rem; 
            font-weight: 700;
        ">{value}</span>
    </div>
    """, unsafe_allow_html=True)

with first_kpi:
    kpi_card("Accuracy", "0.82")
with second_kpi:
    kpi_card("F1 Score", "0.79")
with third_kpi:
    kpi_card("Data Points", "1200")
with fourth_kpi:
    kpi_card("Loss", "0.15")

# ... (kode bagian atas tetap sama hingga bagian penutup kpi_card)

# Baris baru untuk layout konten di bawah KPI
# Filter ada di kolom 1 (kiri), Chart ada di kolom 2 (kanan)
col_filter, col_chart = st.columns([1, 3], gap="medium")

with col_filter:
    st.write("") # Spacer
    with st.container(border=True):
        st.markdown("**Choose Filter**")
        city = st.selectbox("City:", ["Bojonegoro", "Lamongan", "Surabaya"], key="f_city")
        year = st.selectbox("Year:", [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024], key="f_year")
        tipe_analisis = st.radio("Analysis Type:", ["Classification", "Prediction"], key="f_type")

with col_chart:
    st.write("") # Agar sejajar dengan bagian filter
    st.subheader(f"📈 Tren Data {tipe_analisis} - {city} ({year})")
    
    # --- SIMULASI DATA (Ganti dengan data asli Anda) ---
    chart_data = pd.DataFrame(
        np.random.randn(20, 1),
        columns=['Nilai']
    )
    
    # Menampilkan Line Chart
    # Gunakan use_container_width=True agar grafik memenuhi kolom
    st.line_chart(chart_data, use_container_width=True)

# Bagian Sebaran Risiko (Baris Baru)
st.markdown("---")
col_map, col_trend_extra = st.columns(2)

with col_map:
    st.markdown("### 🌍 Sebaran Risiko Wilayah")
    # st.image(...) atau map logic di sini

with col_trend_extra:
    st.markdown("### 📊 Tren Data Lainnya")
    # visualisasi tambahan

























































### second row 
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("## Secondary KPIs")

first_kpi, second_kpi, third_kpi, fourth_kpi, fifth_kpi, sixth_kpi = st.columns(6)

with first_kpi:
    st.markdown("**First KPI**")
    number1 = 111 
    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)

with second_kpi:
    st.markdown("**Second KPI**")
    number2 = 222 
    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

with third_kpi:
    st.markdown("**Third KPI**")
    number3 = 333 
    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

with fourth_kpi:
    st.markdown("**First KPI**")
    number1 = 111 
    st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)

with fifth_kpi:
    st.markdown("**Second KPI**")
    number2 = 222 
    st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

with sixth_kpi:
    st.markdown("**Third KPI**")
    number3 = 333 
    st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)


st.markdown("## Chart Section: 1")

first_chart, second_chart = st.columns(2)


with first_chart:
    chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
    st.line_chart(chart_data)

with second_chart:
    chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
    st.line_chart(chart_data)


st.markdown("## Chart Section: 2")

first_chart, second_chart = st.columns(2)


with first_chart:
    chart_data = pd.DataFrame(np.random.randn(100, 3),columns=['a', 'b', 'c'])
    st.line_chart(chart_data)

with second_chart:
    chart_data = pd.DataFrame(np.random.randn(2000, 3),columns=['a', 'b', 'c'])
    st.line_chart(chart_data)