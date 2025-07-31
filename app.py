
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import base64
import plotly.graph_objects as go
import folium
import json
import plotly.express as px
from folium import Choropleth
from streamlit_folium import folium_static
from streamlit_folium import st_folium
from statsmodels.tsa.arima.model import ARIMA
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Dashboard SPKLU",
    page_icon="logo spklu.png",  # Path ke file
    layout="wide"
)
    
# Optional: use wide layout
st.set_page_config(layout="wide")
warnings.filterwarnings("ignore")

# Sidebar navigation
from streamlit_option_menu import option_menu
import base64

# Fungsi konversi gambar ke base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Ambil gambar logo dan ubah ke base64
logo_base64 = get_base64_image("logo spklu.png")



# Sidebar
with st.sidebar:
    # Tampilkan logo dan teks berdampingan
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <img src="data:image/png;base64,{logo_base64}" width="500" style="margin-right:20px;" />
            <h4 style="margin: 0;">SPKLU Dashboard</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Menu Navigasi
    selected = option_menu(
        menu_title="",
        options=["Menu Utama", "Analisis", "Prediksi", "Tentang"],
        icons=["bar-chart", "activity", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles = {
    "container": {
        "padding": "5!important",
        "background-color": "#D6EBFF",  # Soft blue sidebar background
    },
    "icon": {
        "color": "#007ACC",  # PLN blue for icons
        "font-size": "18px"
    },
    "nav-link": {
        "font-size": "16px",
        "text-align": "left",
        "color": "#444444",  # Neutral gray text
        "margin": "4px",
        "border-radius": "8px"
    },
    "nav-link-selected": {
        "background-color": "#B3D9FF",  # Lighter blue for selected
        "color": "#003366",  # Darker navy text
        "font-weight": "bold"
    }
}

    )


# Load file
try:
    df = pd.read_excel('Coba kp.xlsx')
    df['TGL BAYAR'] = pd.to_datetime(df['TGL BAYAR'], format='%d/%m/%Y', errors='coerce')
    df['Efisiensi'] = df['RPKWH'] / df['PEMKWH']
except Exception as e:
    st.error(f"Gagal memuat file: {e}")
    st.stop()

if selected == "Menu Utama":
    st.title('Dashboard Ringkasan SPKLU')
    st.write("Ringkasan data transaksi SPKLU.")

    # 1. Format kolom tanggal dan buat kolom bulan
    df['TANGGAL'] = pd.to_datetime(df['TGL BAYAR'])
    df['BULAN'] = df['TANGGAL'].dt.to_period('M').dt.to_timestamp()

    # 2. Ambil daftar bulan unik & urutkan dari terbaru ke terlama
    daftar_bulan = df[['BULAN']].drop_duplicates().sort_values(by='BULAN', ascending=False)
    daftar_bulan_display = daftar_bulan['BULAN'].dt.strftime('%B %Y').tolist()
    pilihan_display = ["Semua Bulan"] + daftar_bulan_display

    # 3. Dropdown filter bulan
    pilihan_bulan_display = st.selectbox("Pilih Bulan", pilihan_display)

    if pilihan_bulan_display != "Semua Bulan":
        bulan_terpilih = pd.to_datetime(pilihan_bulan_display)
        df_filter = df[df['BULAN'] == bulan_terpilih]
    else:
        df_filter = df.copy()

    # -------------------------
    # 4. Pratinjau Data
    st.subheader("Pratinjau Data")
    st.dataframe(df_filter.head())

    # -------------------------
    # 5. Ringkasan Statistik
    total_kwh = df_filter['PEMKWH'].sum()
    total_income = df_filter['RPKWH'].sum()
    total_transaksi = df_filter['No'].nunique()

    st.subheader("Ringkasan Statistik")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total KWH Terjual", f"{total_kwh:,.2f}")
    with col2:
        st.metric("Total Pendapatan", f"Rp{total_income:,.2f}")
    with col3:
        st.metric("Jumlah Transaksi", total_transaksi)

    # -------------------------
    st.subheader("Ranking SPKLU Berdasarkan KWH Terjual")

    # Ambil data SPKLU dengan total KWH berdasarkan bulan yang dipilih (df_filter)
    spklu_ranking_kwh = df_filter.groupby('NAMA_SPKLU')['PEMKWH'].sum().reset_index()
    spklu_ranking_kwh = spklu_ranking_kwh.sort_values(by='PEMKWH', ascending=False)

    top_n = 11
    top_spklu = spklu_ranking_kwh.head(top_n)

    fig = go.Figure(go.Bar(
        x=top_spklu['PEMKWH'],
        y=top_spklu['NAMA_SPKLU'],
        orientation='h',
        marker=dict(color='#007ACC'),
        text=top_spklu['PEMKWH'].round(0),
        textposition='outside',
        insidetextanchor='start'
    ))

    fig.update_layout(
        title=f'Urutan SPKLU berdasarkan KWH Terjual',
        xaxis_title='Total KWH',
        yaxis=dict(autorange="reversed"),
        height=500,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff'
    )

    st.plotly_chart(fig, use_container_width=True)

 # -------------------------
    st.subheader("Ranking SPKLU Berdasarkan Pendapatan Terjual")

    # Ambil data SPKLU dengan total KWH berdasarkan bulan yang dipilih (df_filter)
    spklu_ranking_kwh = df_filter.groupby('NAMA_SPKLU')['RPKWH'].sum().reset_index()
    spklu_ranking_kwh = spklu_ranking_kwh.sort_values(by='RPKWH', ascending=False)

    top_n = 11
    top_spklu = spklu_ranking_kwh.head(top_n)

    fig = go.Figure(go.Bar(
        x=top_spklu['RPKWH'],
        y=top_spklu['NAMA_SPKLU'],
        orientation='h',
        marker=dict(color='#007ACC'),
        text=top_spklu['RPKWH'].round(0),
        textposition='outside',
        insidetextanchor='start'
    ))

    fig.update_layout(
        title=f'Urutan SPKLU berdasarkan Pendapatan',
        xaxis_title='Total Pendapatan',
        yaxis=dict(autorange="reversed"),
        height=500,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff'
    )

    st.plotly_chart(fig, use_container_width=True)


   
    st.title("🗺️ Peta Kota Bandung Berdasarkan Unit Layanan Pelanggan (ULP)")
    
    # Load GeoJSON
    with open("kecamatan_bandung_ulp.geojson", "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
        
    
    for feature in geojson_data["features"]:
        if "nama_kecamatan" not in feature["properties"]:
            feature["properties"]["nama_kecamatan"] = "Tidak Diketahui"
        if "ULP" not in feature["properties"]:
            feature["properties"]["ULP"] = "Tidak Diketahui"

    st.subheader(geojson_data["features"][0]["properties"])
    # Inisialisasi peta
    m = folium.Map(location=[-6.9, 107.6], zoom_start=11)
    
    # Buat tooltip aman
    tooltip = folium.GeoJsonTooltip(
        fields=["nama_kecamatan", "ULP"],
        aliases=["Kecamatan:", "Unit Layanan Pelanggan (ULP):"],
        localize=True,
        sticky=True,
        labels=True,
        toLocaleString=True
    )
    
    # Tambahkan ke peta
    folium.GeoJson(
        geojson_data,
        name="Kecamatan dan ULP",
        tooltip=tooltip,
        style_function=lambda feature: {
            'fillColor': '#007ACC',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6
        }
    ).add_to(m)
    
    folium.LayerControl().add_to(m)
    folium_static(m, width=1200, height=800)



elif selected == "Analisis":
    st.title('🔍 Analisis Detail Data SPKLU')
    st.write("Analisis mendalam berdasarkan unit dan SPKLU.")

    # Analisis per UNITUP yang spesifik
    st.subheader('Analisis Transaksi per SPKLU dalam Unit Tertentu')
    selected_unit_analysis = st.selectbox('Pilih UNITUP untuk Analisis Detail', df['UNITUP'].unique())

    if selected_unit_analysis:
        df_unit_analysis = df[df['UNITUP'] == selected_unit_analysis]
        spklu_unit_analysis = df_unit_analysis.groupby('NAMA_SPKLU')['No'].nunique().reset_index()
        spklu_unit_analysis = spklu_unit_analysis.sort_values(by='No', ascending=False).rename(columns={'No': 'jumlah_transaksi'})
        st.write(f'Jumlah Transaksi per SPKLU di UNITUP {selected_unit_analysis}:')
        st.dataframe(spklu_unit_analysis)

        # Visualisasi transaksi per SPKLU di unit yang dipilih
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(x='NAMA_SPKLU', y='jumlah_transaksi', data=spklu_unit_analysis, ax=ax, palette='viridis')
        ax.set_title(f'Jumlah Transaksi per SPKLU di UNITUP {selected_unit_analysis}')
        ax.set_xlabel('NAMA SPKLU')
        ax.set_ylabel('Jumlah Transaksi')
        plt.xticks(rotation=90)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader('Ranking SPKLU Berdasarkan Total KWH dan Pendapatan')
    spklu_summary = df.groupby('NAMA_SPKLU').agg({
        'PEMKWH': 'sum',
        'RPKWH': 'sum'
    }).reset_index()

    ranking_spklu_kwh = spklu_summary.sort_values(by='PEMKWH', ascending=False).rename(columns={'PEMKWH': 'total_kwh'})
    ranking_spklu_pendapatan = spklu_summary.sort_values(by='RPKWH', ascending=False).rename(columns={'RPKWH': 'total_pendapatan'})

    col1_rank, col2_rank = st.columns(2)
    with col1_rank:
        st.write("Ranking NAMA_SPKLU Berdasarkan Total KWH Terjual:")
        st.dataframe(ranking_spklu_kwh)
    with col2_rank:
        st.write("Ranking NAMA_SPKLU Berdasarkan Total Pendapatan:")
        st.dataframe(ranking_spklu_pendapatan)

    st.subheader('Visualisasi Total KWH dan Pendapatan per SPKLU')
    ranking_unit = df.groupby('UNITUP').agg({'PEMKWH': 'sum', 'RPKWH': 'sum'}).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    sns.barplot(ax=axes[0], x='NAMA_SPKLU', y='total_kwh', data=ranking_spklu_kwh, palette='viridis')
    axes[0].set_title('Total KWH Terjual per NAMA_SPKLU')
    axes[0].set_xlabel('NAMA SPKLU')
    axes[0].set_ylabel('Total KWH')
    axes[0].tick_params(axis='x', rotation=90)

    sns.barplot(ax=axes[1], x='NAMA_SPKLU', y='total_pendapatan', data=ranking_spklu_pendapatan, palette='viridis')
    axes[1].set_title('Total Pendapatan per NAMA_SPKLU')
    axes[1].set_xlabel('NAMA SPKLU')
    axes[1].set_ylabel('Total Pendapatan (IDR)')
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader('Proporsi KWH dan Pendapatan per Unit')
    ranking_unit = df.groupby('UNITUP').agg({'PEMKWH': 'sum', 'RPKWH': 'sum'}).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].pie(ranking_unit['PEMKWH'], labels=ranking_unit['UNITUP'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(ranking_unit)))
    axes[0].set_title('Proporsi Total KWH per Unit')
    axes[0].axis('equal')

    axes[1].pie(ranking_unit['RPKWH'], labels=ranking_unit['UNITUP'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(ranking_unit)))
    axes[1].set_title('Proporsi Total Pendapatan per Unit')
    axes[1].axis('equal')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)





elif selected == "Prediksi":
    st.title('Prediksi Jumlah Transaksi Harian SPKLU')
    st.write("Memprediksi jumlah transaksi SPKLU untuk beberapa hari ke depan menggunakan model ARIMA.")

    df_valid_dates = df.dropna(subset=['TGL BAYAR'])

    if not df_valid_dates.empty:
        df_valid_dates['Tanggal'] = df_valid_dates['TGL BAYAR'].dt.date
        transaksi_per_hari = df_valid_dates.groupby('Tanggal')['No'].nunique().reset_index()
        transaksi_per_hari = transaksi_per_hari.rename(columns={'No': 'jumlah_transaksi'})
        transaksi_per_hari['Tanggal'] = pd.to_datetime(transaksi_per_hari['Tanggal'])
        transaksi_per_hari = transaksi_per_hari.sort_values(by='Tanggal')

        st.subheader('Tren Harian Total Transaksi SPKLU (Data Historis)')
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.lineplot(x='Tanggal', y='jumlah_transaksi', data=transaksi_per_hari, ax=ax)
        ax.set_title('Tren Harian Total Transaksi SPKLU')
        ax.set_xlabel('Tanggal Pembayaran')
        ax.set_ylabel('Jumlah Transaksi')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


        st.subheader('Prediksi Jumlah Transaksi Harian')

        if len(transaksi_per_hari) >= 2:
            time_series_data = transaksi_per_hari.set_index('Tanggal')['jumlah_transaksi']

            p, d, q = 5, 1, 0
            days_to_predict = st.slider('Pilih jumlah hari ke depan untuk diprediksi', 1, 30, 7)

            try:
                model = ARIMA(time_series_data, order=(p, d, q))
                model_fit = model.fit()

                last_date = time_series_data.index[-1]
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict, freq='D')
                forecast = model_fit.predict(start=len(time_series_data), end=len(time_series_data) + days_to_predict - 1)
                forecast.index = forecast_index

                st.write(f"Prediksi Jumlah Transaksi untuk {days_to_predict} Hari ke Depan:")
                st.dataframe(forecast.reset_index().rename(columns={'index': 'Tanggal', 'predicted_mean': 'Jumlah Prediksi'}))

                fig, ax = plt.subplots(figsize=(15, 6))
                ax.plot(time_series_data.index, time_series_data.values, label='Data Historis')
                ax.plot(forecast.index, forecast.values, label='Prediksi', color='red')
                ax.set_title(f'Tren Harian Total Transaksi SPKLU dengan Prediksi ({days_to_predict} Hari ke Depan)')
                ax.set_xlabel('Tanggal')
                ax.set_ylabel('Jumlah Transaksi')
                plt.xticks(rotation=45)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat fitting atau forecasting model ARIMA: {e}")
                st.warning("Kemungkinan data tidak cocok untuk order ARIMA ini atau jumlah data terlalu sedikit.")

        else:
            st.warning("Data transaksi harian yang valid tidak cukup untuk melakukan prediksi (dibutuhkan minimal 2 data point).")

    else:
        st.warning("Tidak ada data tanggal pembayaran yang valid untuk analisis tren.")

elif selected == "Tentang":
    st.title('Tentang Dashboard Ini')
    st.write("""
    Dashboard ini dibuat untuk menganalisis data transaksi SPKLU secara interaktif dan informatif.

    ### 🔌 Apa itu SPKLU?
    SPKLU (Stasiun Pengisian Kendaraan Listrik Umum) adalah fasilitas yang disediakan untuk mengisi daya baterai kendaraan listrik. SPKLU menjadi bagian penting dalam ekosistem kendaraan listrik di Indonesia untuk mendukung transisi menuju energi bersih dan berkelanjutan.

    ### Fitur dalam Dashboard:
    """)

    # Fitur 1: Data Awal
    with st.expander("Data Awal"):
        st.write("Menampilkan cuplikan awal data transaksi SPKLU yang digunakan dalam analisis.")

    # Fitur 2: Ringkasan Data
    with st.expander("Ringkasan Data"):
        st.write("Memberikan informasi jumlah total transaksi dalam dataset.")

    # Fitur 3: Transaksi per Unit
    with st.expander("Transaksi per Unit"):
        st.write("Menampilkan jumlah transaksi yang terjadi di setiap unit pelayanan PLN (UNITUP).")

    # Fitur 4: Analisis per Unit
    with st.expander("Analisis per Unit"):
        st.write("Menampilkan transaksi per SPKLU pada unit tertentu yang dipilih, lengkap dengan grafik batang (bar chart).")

    # Fitur 5: KWH & Pendapatan
    with st.expander("KWH & Pendapatan"):
        st.write("Menunjukkan total energi (kWh) yang terjual serta pendapatan yang dihasilkan dari seluruh transaksi.")

    # Fitur 6: Ranking Unit
    with st.expander("Ranking Unit"):
        st.write("Mengurutkan UNITUP berdasarkan total energi terjual dan pendapatan terbesar.")

    # Fitur 7: Ranking SPKLU
    with st.expander("Ranking SPKLU"):
        st.write("Menyediakan daftar SPKLU berdasarkan performa (total kWh terjual dan total pendapatan).")

    # Fitur 8: Proporsi Pie Chart
    with st.expander("Proporsi Pie Chart"):
        st.write("Visualisasi pie chart untuk menunjukkan proporsi energi terjual dan pendapatan di masing-masing UNITUP.")

    # Fitur 9: Tren Transaksi & Prediksi
    with st.expander("Tren Transaksi & Prediksi"):
        st.write("Menampilkan tren harian jumlah transaksi dalam bentuk grafik garis. Fitur ini juga menggunakan **model ARIMA** untuk memprediksi jumlah transaksi di masa depan berdasarkan pola historis.")

    st.write("""
    ### 📁 Sumber Data:
    Data berasal dari file Excel: Rincian SPKLU Bulan Juni 2025
    """)

