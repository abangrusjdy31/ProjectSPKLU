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
from streamlit_folium import folium_static
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from statsmodels.tsa.arima.model import ARIMA
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler


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


# Koordinat bounding box tiap UNITUP (lat_min, lon_min, lat_max, lon_max)
unitup_bounds = {
    "5351": {
        "nama": "Bandung Barat",
        "bounds": [[-6.95, 107.55], [-6.90, 107.60]]
    },
    "53567": {
        "nama": "Bandung Selatan",
        "bounds": [[-7.00, 107.60], [-6.95, 107.65]]
    },
    "53563": {
        "nama": "Bandung Timur",
        "bounds": [[-6.95, 107.70], [-6.90, 107.75]]
    },
    "53575": {
        "nama": "Bandung Utara",
        "bounds": [[-6.88, 107.60], [-6.83, 107.65]]
    },
    "53559": {
        "nama": "Kopo",
        "bounds": [[-6.94, 107.58], [-6.89, 107.63]]
    },
    "53751": {
        "nama": "Cijaura",
        "bounds": [[-6.92, 107.65], [-6.87, 107.70]]
    },
    "53555": {
        "nama": "Ujungberung",
        "bounds": [[-6.91, 107.73], [-6.86, 107.78]]
    }
}




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
    st.write("Ringkasan data transaksi SPKLU di Kota Bandung Raya")

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

    # Pilih dan ubah nama kolom
    selected_columns = ['UNITUP', 'NAMA_SPKLU', 'PEMKWH', 'RP PERKWH', 'RPKWH', 'RPTOTAL']

    # Alias kolom
    column_alias = {
        'UNITUP': 'Unit',
        'NAMA_SPKLU': 'Nama SPKLU',
        'PEMKWH': 'Jumlah kWh',
        'RP PERKWH': 'Harga per kWh',
        'RPKWH': 'Total Biaya',
        'RPTOTAL': 'Total Biaya + PPN'
    }

    # Terapkan alias dan tampilkan tanpa index
    df_display = df_filter[selected_columns].rename(columns=column_alias)
    st.dataframe(df_display.reset_index(drop=True), height=200, hide_index=True)

    
    
    # -------------------------
    # 5. Ringkasan Statistik
        total_kwh = df_filter['PEMKWH'].sum()
        total_income = df_filter['RPKWH'].sum()
        total_transaksi = df_filter['No'].nunique()
        
        # Spasi atas
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.subheader("Ringkasan Statistik")
        
        # CSS Responsive
        st.markdown("""
            <style>
            .metric-container {
                display: flex;
                justify-content: space-between;
                gap: 10px;
                flex-wrap: wrap; /* supaya bisa turun kalau layar kecil */
            }
            .metric-box {
                flex: 1;
                min-width: 200px; /* batas minimum ukuran kotak */
                border: 2px solid #e6e6e6;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                background-color: #f9f9f9;
                box-sizing: border-box;
            }
            .metric-box.wide {
                flex: 1.5;
            }
            .metric-label {
                font-weight: bold;
                font-size: 22px;
                margin-bottom: -5px;
                word-wrap: break-word;
            }
            .metric-value {
                font-size: 22px;
                color: #333;
            }
        
            /* Responsif untuk layar kecil */
            @media (max-width: 768px) {
                .metric-container {
                    flex-direction: column;
                    align-items: stretch;
                }
                .metric-box {
                    width: 100%;
                }
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Display metrics
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-box">
                    <div class="metric-label">Total KWH Terjual</div>
                    <div class="metric-value">{total_kwh:,.0f}</div>
                </div>
                <div class="metric-box wide">
                    <div class="metric-label">Total Pendapatan</div>
                    <div class="metric-value">Rp{total_income:,.0f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Jumlah Transaksi</div>
                    <div class="metric-value">{total_transaksi}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)



    # -------------------------
    
    
    # Tambahkan <br> sebagai spasi atas
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Ranking SPKLU Berdasarkan KWH Terjual")

    # Ambil data SPKLU dengan total KWH berdasarkan bulan yang dipilih (df_filter)
    spklu_ranking_kwh = df_filter.groupby('NAMA_SPKLU')['PEMKWH'].sum().reset_index()
    spklu_ranking_kwh = spklu_ranking_kwh.sort_values(by='PEMKWH', ascending=False)

    top_n = 10
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
        title=f'Urutan SPKLU berdasarkan Total KWH Terjual',
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

    top_n = 10
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
        title=f'Urutan SPKLU berdasarkan Total Pendapatan',
        xaxis_title='Total Pendapatan',
        yaxis=dict(autorange="reversed"),
        height=500,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff'
    )

    st.plotly_chart(fig, use_container_width=True)

#-----------------------------------

     # -------------------------
    st.subheader("Ranking SPKLU Berdasarkan Jumlah Transaksi")

    # Hitung jumlah transaksi untuk setiap SPKLU
    spklurank_transaksi = df_filter.groupby('NAMA_SPKLU').size().reset_index(name='JUMLAH_TRANSAKSI')

    # Urutkan dari yang terbesar
    spklurank_transaksi = spklurank_transaksi.sort_values(by='JUMLAH_TRANSAKSI', ascending=False)

    # Ambil 10 SPKLU teratas
    top_n = 10
    top_spklu = spklurank_transaksi.head(top_n)

    # Plot horizontal bar chart
    fig = go.Figure(go.Bar(
        x=top_spklu['JUMLAH_TRANSAKSI'],
        y=top_spklu['NAMA_SPKLU'],
        orientation='h',
        marker=dict(color='#007ACC'),
        text=top_spklu['JUMLAH_TRANSAKSI'],
        textposition='outside',
        insidetextanchor='start'
    ))

    fig.update_layout(
        title='Urutan SPKLU berdasarkan Jumlah Transaksi',
        xaxis_title='Jumlah Transaksi',
        yaxis=dict(autorange="reversed"),
        height=500,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff'
    )

    st.plotly_chart(fig, use_container_width=True)

#--------------------------

    # Judul
    st.title("Peta Lokasi SPKLU di Bandung")

    # Data lokasi SPKLU
    spklu_locations = [
        ["SPKLU PLN UP3 BANDUNG", -6.948691482456584, 107.61219619688617],
        ["SPKLU PLN ULP BANDUNG UTARA", -6.920962, 107.608129],
        ["SPKLU PLN ULP BANDUNG BARAT", -6.933869, 107.57143],
        ["SPKLU PLN ULP BANDUNG TIMUR", -6.899030, 107.641179],
        ["SPKLU PLN ULP CIJAWURA", -6.898929, 107.641115],
        ["SPKLU PLN ULP UJUNGBERUNG", -6.2528476, 107.0104015],
        ["SPKLU PLN ULP KOPO", -6.954014, 107.640576],
        ["SPKLU PLN TRANS STUDIO MALL BANDUNG", -6.9254, 107.6365],
        ["SPKLU PLN UID JAWA BARAT", -6.919962, 107.60901],
        ["SPKLU POLDA JABAR", -6.936625, 107.7033697],
        ["SPKLU PLN ICON HUB (BRAGA HERITAGE)", -6.9199358706829255, 107.6098682273523],
        ["SPKLU PLN UIP JBT", -6.938285, 107.627942],
        ["SPKLU (ARISTA POWER) BYD BANDUNG", -6.93866, 107.6759],
        ["SPKLU PLN GEOWISATA INN", -6.91758, 107.57838],
    ]

    # Buat DataFrame dari lokasi SPKLU
    df_lokasi = pd.DataFrame(spklu_locations, columns=["NAMA_SPKLU", "LAT", "LON"])

    # Buat ringkasan data per SPKLU
    summary = df_filter.groupby('NAMA_SPKLU').agg({
        'No': 'count',
        'PEMKWH': 'sum',
        'RPKWH': 'sum'
    }).reset_index().rename(columns={
        'No': 'Jumlah Transaksi',
        'PEMKWH': 'Total kWh',
        'RPKWH': 'Total Pendapatan'
    })

    # Gabungkan lokasi dan data summary
    df_map = pd.merge(df_lokasi, summary, on="NAMA_SPKLU", how="left")

    # Isi NaN dengan 0 agar tetap ditampilkan
    df_map[['Jumlah Transaksi', 'Total kWh', 'Total Pendapatan']] = df_map[[
        'Jumlah Transaksi', 'Total kWh', 'Total Pendapatan'
    ]].fillna(0)

    # Inisialisasi peta
    m = folium.Map(location=[-6.92, 107.62], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m)

    # Tambahkan marker
    for _, row in df_map.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; font-size: 13px; line-height: 1.5">
            <strong>{row['NAMA_SPKLU']}</strong><br>
            Bulan : {pilihan_bulan_display}<br><br>
            <table style="width: 250px">
                <tr><td>🔁 Jumlah Transaksi:</td><td><strong>{int(row['Jumlah Transaksi']):,}</strong></td></tr>
                <tr><td>⚡ Total kWh:</td><td><strong>{row['Total kWh']:.0f}</strong></td></tr>
                <tr><td>💰 Total Pendapatan:</td><td><strong>Rp {row['Total Pendapatan']:,.0f}</strong></td></tr>
            </table>
        </div>
        """

        folium.Marker(
            location=[row['LAT'], row['LON']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=row['NAMA_SPKLU'],
            icon=folium.Icon(color="green", icon="bolt", prefix="fa")
        ).add_to(marker_cluster)

    # Tampilkan peta di Streamlit
    folium_static(m, width=1100, height=700)






elif selected == "Analisis":
    st.title('Analisis Detail Data SPKLU')
    st.write("Analisis mendalam SPKLU")

    # Analisis per UNITUP yang spesifik
    st.subheader('Analisis Transaksi per SPKLU dalam Unit Tertentu')
    selected_unit_analysis = st.selectbox('Pilih UNIT', df['UNITUP'].unique())

    if selected_unit_analysis:
        df_unit_analysis = df[df['UNITUP'] == selected_unit_analysis]
        spklu_unit_analysis = df_unit_analysis.groupby('NAMA_SPKLU')['No'].nunique().reset_index()
        spklu_unit_analysis = spklu_unit_analysis.sort_values(by='No', ascending=False).rename(columns={'No': 'Jumlah Transaksi', 'NAMA_SPKLU': 'SPKLU'})
        st.write(f'Jumlah Transaksi SPKLU di UNIT {selected_unit_analysis}:')
        st.dataframe(spklu_unit_analysis, use_container_width=True, hide_index=True)

        # Donut Chart
    fig = go.Figure(data=[go.Pie(
        labels=spklu_unit_analysis['SPKLU'],
        values=spklu_unit_analysis['Jumlah Transaksi'],
        hole=0.4,  # Donut style
        marker=dict(colors=px.colors.qualitative.Vivid),
        textinfo='percent',
        hoverinfo='label+value'
    )])

    st.plotly_chart(fig, use_container_width=True)



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
    Data berasal dari Penjualan SPKLU di kota Bandung selama Bulan Juni 2024 - Juni 2025
    """)
