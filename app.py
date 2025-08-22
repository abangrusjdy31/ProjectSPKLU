import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings
import base64
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from streamlit_option_menu import option_menu

# Optional: use wide layout
st.set_page_config(layout="wide")
warnings.filterwarnings("ignore")

# ==== Fungsi Konversi Gambar ke Base64 ====
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Logo
logo_base64 = get_base64_image("logo spklu.png")

# ==== Sidebar ====
with st.sidebar:
    # Logo + judul
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <img src="data:image/png;base64,{logo_base64}" width="220" style="margin-right:20px;" />            
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
        styles={
            "container": {
                "padding": "5!important",
                "background-color": "#D6EBFF",
            },
            "icon": {
                "color": "#007ACC",
                "font-size": "18px"
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "color": "#444444",
                "margin": "4px",
                "border-radius": "8px"
            },
            "nav-link-selected": {
                "background-color": "#B3D9FF",
                "color": "#003366",
                "font-weight": "bold"
            }
        }
    )

# ==== Load Dataset ====
url_data2 = "https://docs.google.com/spreadsheets/d/16cyvXwvucVb7EM1qiikZpbK8J8isbktuiw-MR1EJDEY/export?format=csv&gid=829004516"
df2 = pd.read_csv(url_data2)
df2.columns = df2.columns.str.strip()

# ==== Halaman Berdasarkan Menu ====
if selected == "Menu Utama":
    st.title("Dashboard Ringkasan SPKLU")

    if 'Bulan & Tahun' in df2.columns:
        bulan_tahun_list = df2['Bulan & Tahun'].unique().tolist()
        bulan_tahun_list.insert(0, "Semua")

        selected_bulan_tahun = st.selectbox("Pilih Periode", bulan_tahun_list)

        # Filter data
        if selected_bulan_tahun == "Semua":
            filtered_df = df2
        else:
            filtered_df = df2[df2['Bulan & Tahun'] == selected_bulan_tahun]

        # Ringkasan Statistik
        total_pendapatan = filtered_df['Total Pendapatan'].sum()
        total_kwh = filtered_df['Jumlah KWH'].sum()
        total_transaksi = filtered_df['Jumlah Transaksi'].sum()

        # Tabel data
        st.dataframe(filtered_df)

        # Custom CSS untuk KPI box
        st.markdown("""
            <style>
            .metric-container {
                display: flex;
                justify-content: space-between;
                gap: 10px;
            }
            .metric-box {
                flex: 1;
                border: 2px solid #e6e6e6;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                background-color: #f9f9f9;
                box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
            }
            .metric-box.wide {
                flex: 1.5;
            }
            .metric-label {
                font-weight: bold;
                font-size: 18px;
                margin-bottom: 8px;
            }
            .metric-value {
                font-size: 22px;
                color: #333;
            }
            </style>
        """, unsafe_allow_html=True)

        # KPI Box
        st.markdown(f"""
            <div class="metric-container">
                <div class="metric-box">
                    <div class="metric-label">Total KWH Terjual</div>
                    <div class="metric-value">{total_kwh:,.0f}</div>
                </div>
                <div class="metric-box wide">
                    <div class="metric-label">Total Pendapatan</div>
                    <div class="metric-value">Rp{total_pendapatan:,.0f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Jumlah Transaksi</div>
                    <div class="metric-value">{total_transaksi:,.0f}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Summary agregat per SPKLU
    summary = df2.groupby("Nama SPKLU").agg({
        "Jumlah Transaksi": "sum",
        "Jumlah KWH": "sum",
        "Total Pendapatan": "sum"
    }).reset_index().rename(columns={
        "Jumlah KWH": "Total kWh"
    })

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
        ["SPKLU ONE STOP CHARGING STATION SURAPATI", -6.898765265153307, 107.62126970807917],
        ["SPKLU REST AREA KM 147 A RUAS PADALARANG - CILEUNYI",-6.967293465888548, 107.68142490895632],
        ["SPKLU PLN MALAGA RESTO",-6.885339711686332, 107.61274315460332],
        ["SPKLU PLN ICON PLUS BANDUNG", -6.908399449687401, 107.63129068871457],
        ["SPKLU PLN TENTH AVENUE BANDUNG", -6.9463929250433205, 107.64099958463902],
        ["SPKLU PLN HOTEL NEWTON",-6.914804225417974, 107.629904095603],
        ["SPKLU PLN RS ADVENT BANDUNG", -6.892129849618786, 107.6030436604091],
        ["SPKLU PLN TRANSMART CIPADUNG", -6.925710208848551, 107.71158241839437],
        ["SPKLU PLN HOTEL CEMERLANG", -6.9122875525627645, 107.59752849325291],
        ["SPKLU PLN Best Western Hotel Setiabudhi Bandung", -6.8611386080112595, 107.59520463558238]
    ]

    df_lokasi = pd.DataFrame(spklu_locations, columns=["NAMA_SPKLU", "LAT", "LON"])
    df_map = pd.merge(df_lokasi, summary, left_on="NAMA_SPKLU", right_on="Nama SPKLU", how="left")
    df_map[['Jumlah Transaksi', 'Total kWh', 'Total Pendapatan']] = df_map[
        ['Jumlah Transaksi', 'Total kWh', 'Total Pendapatan']
    ].fillna(0)

    # Inisialisasi peta
    m = folium.Map(location=[-6.92, 107.62], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m)

    # Tambahkan marker
    for _, row in df_map.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; font-size: 13px; line-height: 1.5">
            <strong>{row['NAMA_SPKLU']}</strong><br><br>
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
    st.title("📊 Analisis Data SPKLU")

    def plot_top10(df, kolom, judul, warna):
        top10 = df.groupby("Nama SPKLU")[kolom].sum().nlargest(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        top10.sort_values().plot(kind="barh", color=warna, ax=ax)
        ax.set_title(judul)
        ax.set_xlabel(kolom)
        ax.set_ylabel("SPKLU")
        st.pyplot(fig)

    st.subheader("Ranking SPKLU")
    plot_top10(df2, "Total Pendapatan", "Top 10 SPKLU - Total Pendapatan", "skyblue")
    plot_top10(df2, "Jumlah KWH", "Top 10 SPKLU - Jumlah KWH", "lightgreen")
    plot_top10(df2, "Jumlah Transaksi", "Top 10 SPKLU - Jumlah Transaksi", "orange")
    



elif selected == "Prediksi":
    st.title("🔮 Prediksi Model")
    st.info("Halaman ini untuk integrasi model prediksi (nanti load .pkl dari .ipynb)")

elif selected == "Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    st.write("Dashboard ini dibuat untuk monitoring SPKLU. Dikembangkan dengan Streamlit.")
