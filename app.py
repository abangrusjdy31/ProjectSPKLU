import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings
import base64
import folium
import joblib
import altair as alt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from plotly.subplots import make_subplots
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from streamlit_option_menu import option_menu
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor




st.set_page_config(
    page_title="Dashboard SPKLU",   # tulisan di bar
    page_icon="logo spklu2.png",           # logo custom (bisa file gambar)
    layout="wide"
)
    
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
                "background-color": "#D8F9FF",
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
                "background-color": "#FFFFFF",
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
    st.write("Ringkasan Data Penjualan SPKLU se-Kota Bandung Raya")

    # ==== Filter Bulan & Tahun ====
    if 'Bulan & Tahun' in df2.columns:
        # Mapping nama bulan Indonesia ke angka
        bulan_map = {
            "Januari": 1, "Februari": 2, "Maret": 3, "April": 4,
            "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8,
            "September": 9, "Oktober": 10, "November": 11, "Desember": 12
        }

        # Pisahkan kolom Bulan & Tahun
        df2[['Bulan', 'Tahun']] = df2['Bulan & Tahun'].str.split(" ", expand=True)
        df2['Tahun'] = df2['Tahun'].astype(int)
        df2['BulanNum'] = df2['Bulan'].map(bulan_map)

        # Urutkan berdasarkan Tahun dan Bulan
        df2 = df2.sort_values(by=['Tahun', 'BulanNum'])

        # Buat list opsi (urut sesuai waktu)
        bulan_tahun_list = df2['Bulan & Tahun'].unique().tolist()
        bulan_tahun_list.insert(0, "Semua")  # tambahkan opsi "Semua" di paling depan

        # Selectbox dengan default "Semua"
        selected_bulan_tahun = st.selectbox("Pilih Periode", bulan_tahun_list, index=0)

        # Filter DataFrame sesuai pilihan
        filtered_df = df2 if selected_bulan_tahun == "Semua" else df2[df2['Bulan & Tahun'] == selected_bulan_tahun]

        # ==== Ringkasan Statistik ====
        total_pendapatan = filtered_df['Total Pendapatan'].sum()
        total_kwh = filtered_df['Jumlah KWH'].sum()
        total_transaksi = filtered_df['Jumlah Transaksi'].sum()

        # KPI Box
        st.markdown("""
            <style>
            .metric-container {display: flex; justify-content: space-between; gap: 10px;}
            .metric-box {flex: 1; border: 2px solid #F3FCFA; border-radius: 10px;
                         padding: 20px; text-align: center; background-color: #FFFFFF;
                         box-shadow: 2px 2px 8px rgba(0,0,0,0.05);}
            .metric-box.wide {flex: 1.5;}
            .metric-label {font-weight: bold; font-size: 18px; margin-bottom: 8px;}
            .metric-value {font-size: 22px; color: #333;}
            </style>
        """, unsafe_allow_html=True)

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

    # ==== Ranking SPKLU ====
    st.subheader("Ranking SPKLU")

    def plot_top5(df, kolom, judul, warna):
        top5 = df.groupby("Nama SPKLU")[kolom].sum().nlargest(5).sort_values()
    
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="none")  # transparan
        ax.set_facecolor("none")  # mengikuti background Streamlit
    
        # Buat bar horizontal dengan rounded edge
        bars = ax.barh(
            top5.index,
            top5.values,
            color=warna,
            edgecolor="black",
            height=0.6
        )
    
        # Percantik style
        ax.set_title(judul, fontsize=14, weight="bold", pad=15)
        ax.set_xlabel(kolom, fontsize=12)
        ax.set_ylabel("")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
    
        # Hilangkan spines (garis tepi)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
    
        # Tambahkan label nilai di ujung bar
        for bar in bars:
            ax.text(
                bar.get_width() + (0.01 * max(top5.values)),
                bar.get_y() + bar.get_height()/2,
                f"{int(bar.get_width()):,}",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black"
            )

        st.pyplot(fig)
        plt.close(fig)


    tab1, tab2, tab3 = st.tabs(["Total KWH Terjual", "Total Pendapatan", "Jumlah Transaksi"])
    with tab2:
        plot_top5(filtered_df, "Total Pendapatan", "Top 5 SPKLU Bedasarkan Total Pendapatan", "#FA8072")
    with tab1:
        plot_top5(filtered_df, "Jumlah KWH", "Top 5 SPKLU Bedasarkan Jumlah KWH Terjual", "lightgreen")
    with tab3:
        plot_top5(filtered_df, "Jumlah Transaksi", "Top 5 SPKLU Bedasarkan Jumlah Transaksi Terbanyak", "#FFBD31")






    # ==== Dropdown Pilihan SPKLU ====
    st.markdown("<br>", unsafe_allow_html=True)
    spklu_list = ["Silahkan pilih SPKLU"] + sorted(df2['Nama SPKLU'].unique().tolist())
    pilihan_spklu = st.selectbox("Pilih SPKLU yang ingin ditampilkan", spklu_list)

    # ==== Summary semua SPKLU untuk map ====
    summary_all = filtered_df.groupby('Nama SPKLU', as_index=False).agg({
        'Jumlah Transaksi': 'sum',
        'Jumlah KWH': 'sum',
        'Total Pendapatan': 'sum'
    }).rename(columns={'Jumlah KWH': 'Total kWh'})

    # ==== Jika pilih salah satu SPKLU -> tampilkan ringkasannya ====
    if pilihan_spklu != "Silahkan pilih SPKLU":
        df_filter_spklu = summary_all[summary_all['Nama SPKLU'] == pilihan_spklu]
        if not df_filter_spklu.empty:
            row = df_filter_spklu.iloc[0]




            st.markdown(f"""
                <div style="font-size:24px; font-weight:bold; text-align:center; margin-top:10px;">
                    {row['Nama SPKLU']}<br>Bulan : {selected_bulan_tahun}
                </div>
                <div class="metric-container">
                    <div class="metric-box">
                        <div class="metric-label">Total KWH Terjual</div>
                        <div class="metric-value">{row['Total kWh']:,.0f}</div>
                    </div>
                    <div class="metric-box wide">
                        <div class="metric-label">Total Pendapatan</div>
                        <div class="metric-value">Rp{row['Total Pendapatan']:,.0f}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Jumlah Transaksi</div>
                        <div class="metric-value">{int(row['Jumlah Transaksi']):,}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # ==== Peta Lokasi ====
    st.title("Peta Lokasi SPKLU di Bandung")

    spklu_locations = [
        ["SPKLU PLN UP3 BANDUNG", -6.948691, 107.612196],
        ["SPKLU PLN ULP BANDUNG UTARA", -6.920962, 107.608129],
        ["SPKLU PLN ULP BANDUNG BARAT", -6.933869, 107.57143],
        ["SPKLU PLN ULP BANDUNG TIMUR", -6.899030, 107.641179],
        ["SPKLU PLN ULP CIJAWURA", -6.898929, 107.641115],
        ["SPKLU PLN ULP UJUNGBERUNG", -6.9038, 107.6657],
        ["SPKLU PLN ULP KOPO", -6.954014, 107.640576],
        ["SPKLU PLN TRANS STUDIO MALL BANDUNG", -6.9254, 107.6365],
        ["SPKLU PLN UID JAWA BARAT", -6.919962, 107.60901],
        ["SPKLU POLDA JABAR", -6.936625, 107.7033697],
        ["SPKLU PLN ICON HUB (BRAGA HERITAGE)", -6.919935, 107.609868],
        ["SPKLU PLN UIP JBT", -6.938285, 107.627942],
        ["SPKLU (ARISTA POWER) BYD BANDUNG", -6.93866, 107.6759],
        ["SPKLU PLN GEOWISATA INN", -6.91758, 107.57838],
        ["SPKLU ONE STOP CHARGING STATION SURAPATI", -6.898765, 107.62127],
        ["SPKLU REST AREA KM 147 A RUAS PADALARANG - CILEUNYI",-6.967293, 107.681425],
        ["SPKLU PLN MALAGA RESTO",-6.88534, 107.61274],
        ["SPKLU PLN ICON PLUS BANDUNG", -6.908399, 107.631291],
        ["SPKLU PLN TENTH AVENUE BANDUNG", -6.946393, 107.640999],
        ["SPKLU PLN HOTEL NEWTON",-6.914804, 107.629904],
        ["SPKLU PLN RS ADVENT BANDUNG", -6.89213, 107.603044],
        ["SPKLU PLN TRANSMART CIPADUNG", -6.92571, 107.711582],
        ["SPKLU PLN HOTEL CEMERLANG", -6.912288, 107.597528],
        ["SPKLU PLN Best Western Hotel Setiabudhi Bandung", -6.861139, 107.595205]
    ]

    df_lokasi = pd.DataFrame(spklu_locations, columns=["NAMA_SPKLU", "LAT", "LON"])
    df_map = pd.merge(df_lokasi, summary_all, left_on="NAMA_SPKLU", right_on="Nama SPKLU", how="left")
    df_map[['Jumlah Transaksi','Total kWh','Total Pendapatan']] = df_map[['Jumlah Transaksi','Total kWh','Total Pendapatan']].fillna(0)

    m = folium.Map(location=[-6.92, 107.62], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df_map.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; font-size: 13px; line-height: 1.5">
            <strong>{row['NAMA_SPKLU']}</strong><br>
            <em>Periode : {selected_bulan_tahun}</em><br><br>
            <table style="width: 250px">
                <tr><td>🔁 Jumlah Transaksi:</td><td><strong>{int(row['Jumlah Transaksi']):,}</strong></td></tr>
                <tr><td>⚡ Total kWh:</td><td><strong>{row['Total kWh']:,.0f}</strong></td></tr>
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

    folium_static(m, width=1100, height=700)






elif selected == "Analisis":   
    st.title("Analisis Data SPKLU")

    # Load Data4 (kapasitas & kategori)
    url_data4 = "https://docs.google.com/spreadsheets/d/16cyvXwvucVb7EM1qiikZpbK8J8isbktuiw-MR1EJDEY/export?format=csv&gid=1731077450"    
    df4 = pd.read_csv(url_data4)
    df4.columns = df4.columns.str.strip()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Ranking SPKLU", 
        "Compare SPKLU", 
        "Compare ULP", 
        "Kapasitas & Kategori",
        "Level Spklu",
        "Tren Bulanan Unit SPKLU",
        "Test3",
        "Test4"
    ])

    # ============================
    # Tab 1 - Ranking
    # ============================
    with tab1:
        st.subheader("Ranking SPKLU")

        # Pilihan ranking: 5 teratas / 5 terbawah
        pilihan_ranking = st.radio("Pilih Ranking", ["5 Teratas", "5 Terbawah"], horizontal=True)

        def plot_ranking(df, kolom, judul, warna, ranking, bg_color="#D9F9FF"):
            grouped = df.groupby("Nama SPKLU")[kolom].sum()
            if ranking == "5 Teratas":
                data = grouped.nlargest(5).sort_values(ascending=True)  # kecil di bawah, besar di atas
            else:
                data = grouped.nsmallest(5).sort_values(ascending=False)  # kecil di atas, besar di bawah
        
            fig, ax = plt.subplots(figsize=(8, 5))
        
            # Atur warna background agar sesuai background utama
            fig.patch.set_facecolor(bg_color)
            ax.set_facecolor(bg_color)
        
            # Plot barh
            data.plot(kind="barh", color=warna, ax=ax)
        
            # Tambahan styling
            ax.set_xlabel("")   # Hapus keterangan bawah (xlabel)
            ax.set_ylabel("")
        
            # Judul ditaruh tengah atas
            ax.set_title(kolom, fontsize=14, weight="bold", loc="center", pad=15)
        
            ax.tick_params(axis="x", labelsize=10)
            ax.tick_params(axis="y", labelsize=10)
            ax.grid(axis="x", linestyle="--", alpha=0.5)
        
            # Hapus spines biar lebih clean
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
        
            st.pyplot(fig)


        # Tampilkan grafik berdasarkan pilihan        
        plot_ranking(df2, "Jumlah Transaksi", "SPKLU", "#FFBD31", pilihan_ranking) 
        st.divider()  # garis pemisah
        plot_ranking(df2, "Jumlah KWH", "SPKLU", "lightgreen", pilihan_ranking)
        st.divider()  # garis pemisah
        plot_ranking(df2, "Total Pendapatan", "SPKLU", "#FA8072", pilihan_ranking)
        st.divider()  # garis pemisah
        



    # ============================
    # Tab 2 - Membandingkan SPKLU
    # ============================
    with tab2:
      st.subheader("Perbandingan Antar SPKLU")

      # Urutan bulan
      bulan_urut = {
          "Januari": 1, "Februari": 2, "Maret": 3, "April": 4,
          "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8,
          "September": 9, "Oktober": 10, "November": 11, "Desember": 12
      }

      # Pisahkan kolom "Bulan & Tahun"
      df2["Bulan"] = df2["Bulan & Tahun"].str.split(" ").str[0]
      df2["Tahun"] = df2["Bulan & Tahun"].str.split(" ").str[1].astype(int)

      # Tambahkan kolom "key" untuk urutan (Tahun * 12 + Bulan)
      df2["key"] = df2["Tahun"] * 12 + df2["Bulan"].map(bulan_urut)

      # Ambil daftar unik & urut
      opsi_bulan_tahun = (
          df2.sort_values("key")[["Bulan", "Tahun"]]
          .drop_duplicates()
          .assign(label=lambda x: x["Bulan"] + " " + x["Tahun"].astype(str))
          ["label"].tolist()
      )

      # Dropdown dari - ke
      col1, col2 = st.columns(2)
      with col1:
          start_option = st.selectbox("Dari Bulan", opsi_bulan_tahun, index=0)
      with col2:
          end_option = st.selectbox("Sampai Bulan", opsi_bulan_tahun, index=len(opsi_bulan_tahun)-1)

      # Ambil key start & end
      key_start = df2.loc[df2["Bulan & Tahun"] == start_option, "key"].iloc[0]
      key_end = df2.loc[df2["Bulan & Tahun"] == end_option, "key"].iloc[0]

      # Filter rentang
      if key_start <= key_end:
          df_bulan = df2[(df2["key"] >= key_start) & (df2["key"] <= key_end)]
      else:
          st.warning("Bulan awal harus sebelum atau sama dengan bulan akhir!")
          df_bulan = df2.copy()


      # Pilih SPKLU untuk perbandingan (pakai data yang sudah difilter bulan)
      spklu_list = df_bulan["Nama SPKLU"].dropna().unique()
      col1, col2 = st.columns(2)
      spklu_a = col1.selectbox("Pilih SPKLU A", spklu_list, index=0, key="spklu_a")
      spklu_b = col2.selectbox(
          "Pilih SPKLU B",
          spklu_list,
          index=min(1, len(spklu_list) - 1) if len(spklu_list) > 1 else 0,
          key="spklu_b"
      )

      # Filter data per SPKLU (pakai data bulan)
      data_a = df_bulan[df_bulan["Nama SPKLU"] == spklu_a]
      data_b = df_bulan[df_bulan["Nama SPKLU"] == spklu_b]

      # Ringkasan
      summary_a = {
          "Jumlah Transaksi": int(data_a["Jumlah Transaksi"].sum()),
          "Total kWh": float(data_a["Jumlah KWH"].sum()),
          "Pendapatan": float(data_a["Total Pendapatan"].sum())
      }
      summary_b = {
          "Jumlah Transaksi": int(data_b["Jumlah Transaksi"].sum()),
          "Total kWh": float(data_b["Jumlah KWH"].sum()),
          "Pendapatan": float(data_b["Total Pendapatan"].sum())
      }

      # Buat dataframe ringkasan
      df_summary = pd.DataFrame([summary_a, summary_b], index=[spklu_a, spklu_b])

      # Styling dataframe
      styled_df = (
          df_summary.style
          .set_table_styles(
              [
                  {"selector": "th", "props": [("background-color", "#FFFFFF"), ("color", "black"), ("font-weight", "bold")]},
                  {"selector": "td", "props": [("padding", "8px")]},
              ]
          )
          .highlight_max(axis=0, color="#FFCDD2")  # highlight nilai terbesar per kolom
          .highlight_min(axis=0, color="#C8E6C9")  # highlight nilai terkecil per kolom
          .format("{:,.0f}")  # format angka dengan pemisah ribuan
      )
      st.subheader("Ringkasan Perbandingan")
      st.dataframe(styled_df, use_container_width=True)


      # Buat subplot 1 baris 3 kolom
      fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])

      # Donut Transaksi
      fig.add_trace(go.Pie(
          labels=[spklu_a, spklu_b],
          values=[summary_a["Jumlah Transaksi"], summary_b["Jumlah Transaksi"]],
          hole=0.6,
          marker=dict(colors=px.colors.qualitative.Set2),
          textinfo='percent',
          showlegend=True  # legend hanya muncul sekali
      ), 1, 1)

      # Donut kWh
      fig.add_trace(go.Pie(
          labels=[spklu_a, spklu_b],
          values=[summary_a["Total kWh"], summary_b["Total kWh"]],
          hole=0.6,
          marker=dict(colors=px.colors.qualitative.Set2),
          textinfo='percent',
          showlegend=False
      ), 1, 2)

      # Donut Pendapatan
      fig.add_trace(go.Pie(
          labels=[spklu_a, spklu_b],
          values=[summary_a["Pendapatan"], summary_b["Pendapatan"]],
          hole=0.6,
          marker=dict(colors=px.colors.qualitative.Set2),
          textinfo='percent',
          showlegend=False
      ), 1, 3)

      # Tambahkan judul per chart
      fig.update_layout(
          annotations=[
              dict(text="Transaksi", x=0.11, y=0.5, font_size=14, showarrow=False),
              dict(text="Total kWh", x=0.50, y=0.5, font_size=14, showarrow=False),
              dict(text="Pendapatan", x=0.90, y=0.5, font_size=14, showarrow=False)
          ],
          legend=dict(
              orientation="h",
              yanchor="top",
              y=-0.2,
              xanchor="center",
              x=0.5
          ),
      )

      st.plotly_chart(fig, use_container_width=True)




    # ============================
    # Tab 3 - Perbandingan ULP
    # ============================
    with tab3:
        st.subheader("Perbandingan Antar ULP di Kota Bandung")

        # --- Merge Data2 dan Data4 berdasarkan Nama SPKLU ---
        df_wilayah = df2.merge(df4, on="Nama SPKLU", how="left")

        # Urutan bulan
        bulan_urut = {
            "Januari": 1, "Februari": 2, "Maret": 3, "April": 4,
            "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8,
            "September": 9, "Oktober": 10, "November": 11, "Desember": 12
        }

        # Pisahkan Bulan & Tahun
        df_wilayah["Bulan"] = df_wilayah["Bulan & Tahun"].str.split(" ").str[0]
        df_wilayah["Tahun"] = df_wilayah["Bulan & Tahun"].str.split(" ").str[1].astype(int)
        df_wilayah["key"] = df_wilayah["Tahun"] * 12 + df_wilayah["Bulan"].map(bulan_urut)

        # Dropdown rentang bulan
        opsi_bulan_tahun = (
            df_wilayah.sort_values("key")[["Bulan", "Tahun"]]
            .drop_duplicates()
            .assign(label=lambda x: x["Bulan"] + " " + x["Tahun"].astype(str))
            ["label"].tolist()
        )

        col1, col2 = st.columns(2)
        start_option = col1.selectbox("Dari Bulan", opsi_bulan_tahun, index=0, key="wil_start")
        end_option = col2.selectbox("Sampai Bulan", opsi_bulan_tahun, index=len(opsi_bulan_tahun)-1, key="wil_end")

        key_start = df_wilayah.loc[df_wilayah["Bulan & Tahun"] == start_option, "key"].iloc[0]
        key_end = df_wilayah.loc[df_wilayah["Bulan & Tahun"] == end_option, "key"].iloc[0]

        if key_start <= key_end:
            df_bulan = df_wilayah[(df_wilayah["key"] >= key_start) & (df_wilayah["key"] <= key_end)]
        else:
            st.warning("Bulan awal harus <= bulan akhir!")
            df_bulan = df_wilayah.copy()

        # Dropdown pilih wilayah (ULP)
        wilayah_list = df_bulan["Wilayah"].dropna().unique()
        col1, col2 = st.columns(2)
        wilayah_a = col1.selectbox("Pilih Wilayah A", wilayah_list, index=0, key="wilayah_a")
        wilayah_b = col2.selectbox(
            "Pilih Wilayah B",
            wilayah_list,
            index=min(1, len(wilayah_list)-1) if len(wilayah_list) > 1 else 0,
            key="wilayah_b"
        )

        # Filter data per wilayah
        data_a = df_bulan[df_bulan["Wilayah"] == wilayah_a]
        data_b = df_bulan[df_bulan["Wilayah"] == wilayah_b]

        # Ringkasan
        summary_a = {
            "Jumlah Transaksi": int(data_a["Jumlah Transaksi"].sum()),
            "Total kWh": float(data_a["Jumlah KWH"].sum()),
            "Pendapatan": float(data_a["Total Pendapatan"].sum())
        }
        summary_b = {
            "Jumlah Transaksi": int(data_b["Jumlah Transaksi"].sum()),
            "Total kWh": float(data_b["Jumlah KWH"].sum()),
            "Pendapatan": float(data_b["Total Pendapatan"].sum())
        }

        # Tabel ringkasan
        st.subheader("Ringkasan Perbandingan ULP di Kota Bandung")
        st.dataframe(pd.DataFrame([summary_a, summary_b], index=[wilayah_a, wilayah_b]))

        # Donut chart (mirip perbandingan SPKLU)
        fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])

        fig.add_trace(go.Pie(labels=[wilayah_a, wilayah_b], values=[summary_a["Jumlah Transaksi"], summary_b["Jumlah Transaksi"]],
                            hole=0.6, marker=dict(colors=px.colors.qualitative.Set2), textinfo='percent', showlegend=True), 1, 1)

        fig.add_trace(go.Pie(labels=[wilayah_a, wilayah_b], values=[summary_a["Total kWh"], summary_b["Total kWh"]],
                            hole=0.6, marker=dict(colors=px.colors.qualitative.Set2), textinfo='percent', showlegend=False), 1, 2)

        fig.add_trace(go.Pie(labels=[wilayah_a, wilayah_b], values=[summary_a["Pendapatan"], summary_b["Pendapatan"]],
                            hole=0.6, marker=dict(colors=px.colors.qualitative.Set2), textinfo='percent', showlegend=False), 1, 3)

        fig.update_layout(
            annotations=[
                dict(text="Transaksi", x=0.11, y=0.5, font_size=14, showarrow=False),
                dict(text="Total kWh", x=0.50, y=0.5, font_size=14, showarrow=False),
                dict(text="Pendapatan", x=0.90, y=0.5, font_size=14, showarrow=False)
            ],
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        )

        st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Tab 4 - Kapasitas & Kategori
    # ============================
    with tab4:
        st.subheader("Kapasitas & Kategori SPKLU")
        # --- Agregasi df2: transaksi per SPKLU ---
        df2_agg = df2.groupby("Nama SPKLU")[["Jumlah Transaksi", "Jumlah KWH", "Total Pendapatan"]].sum().reset_index()

        # --- Agregasi df4: kapasitas max, rata-rata, jumlah tipe, dan kategori tertinggi ---
        # --- Cleaning kolom Kapasitas jadi numerik ---
        df4["Kapasitas"] = pd.to_numeric(
            df4["Kapasitas"]
                .astype(str)
                .str.replace("kW", "", regex=False)
                .str.replace("KW", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.extract(r"(\d+)")[0],  # ambil hanya angka
            errors="coerce"
        )
        # Mapping ranking kategori
        kategori_rank = {
            "Ultra Fast Charging": 3,
            "Fast Charging": 2,
            "Medium Charging": 1,
            "Slow Charging": 0
        }
        def pilih_kategori(x):
            # Buang NaN
            x = x.dropna()
            if x.empty:
                return "Unknown"

            # Map rank hanya yg valid
            ranks = x.map(kategori_rank).fillna(0)

            # Ambil kategori dengan rank tertinggi
            best_idx = ranks.idxmax()
            return x.loc[best_idx]

        df4_agg = df4.groupby("Nama SPKLU").agg({
            # kapasitas: total + variasi
            "Kapasitas": ["max", "mean", lambda x: x.nunique()],
            # kategori: ambil tertinggi
            "Kategori": pilih_kategori
        }).reset_index()

        # rename kolom multiindex
        df4_agg.columns = ["Nama SPKLU", "Kapasitas_Max", "Kapasitas_Mean", "Jumlah_Tipe", "Kategori"]
        df4_agg = df4_agg.dropna(subset=["Kapasitas_Max", "Kapasitas_Mean"])

        # --- Gabungkan df2 + df4 ---
        df_merged = pd.merge(df2_agg, df4_agg, on="Nama SPKLU", how="left")

        # --- One-hot encoding kategori ---
        df_encoded = pd.get_dummies(df_merged, columns=["Kategori"])

        # --- Pilih fitur untuk clustering ---
        features = [
            "Jumlah Transaksi",
            "Jumlah KWH",
            "Total Pendapatan",
            "Kapasitas_Max",
            "Kapasitas_Mean",
            "Jumlah_Tipe"
        ] + [col for col in df_encoded.columns if "Kategori_" in col]

        # --- Standardisasi ---
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_encoded[features].fillna(0))

        # --- K-Means ---
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_encoded["Cluster"] = kmeans.fit_predict(X_scaled)

        # Mapping cluster berdasarkan rata-rata pendapatan
        cluster_order = df_encoded.groupby("Cluster")["Total Pendapatan"].mean().sort_values().index
        mapping = {cluster_order[0]: "Rendah", cluster_order[1]: "Sedang", cluster_order[2]: "Tinggi"}
        df_encoded["Level"] = df_encoded["Cluster"].map(mapping)

        # --- Output ---
        # === Bagian Rekomendasi (kotak kotak per level) ===
        rekomendasi = {
            "Tinggi": "SPKLU ini ramai digunakan. Tambah unit charger, upgrade kapasitas, atau buka cabang di lokasi serupa.",
            "Sedang": "SPKLU memiliki potensi. Dorong dengan promosi, kerjasama merchant, atau peningkatan fasilitas.",
            "Rendah": "SPKLU relatif sepi. Evaluasi lokasi, cek teknis, atau strategi marketing. Jika tetap rendah, pertimbangkan relokasi."
        }

        df_tinggi = df_encoded.query("Level == 'Tinggi'").sort_values("Total Pendapatan", ascending=False)
        df_sedang = df_encoded.query("Level == 'Sedang'").sort_values("Total Pendapatan", ascending=False)
        df_rendah = df_encoded.query("Level == 'Rendah'").sort_values("Total Pendapatan", ascending=False)

        tinggi = df_tinggi["Nama SPKLU"].tolist()
        sedang = df_sedang["Nama SPKLU"].tolist()
        rendah = df_rendah["Nama SPKLU"].tolist()

        col1, col2, col3 = st.columns(3)

        def buat_kotak(judul, data):
            with st.container():
                st.markdown(f"### {judul}")
                st.markdown(
                    f"""
                    <div style="border:1px solid #ddd; border-radius:10px; padding:10px;
                                max-height:200px; overflow-y:auto; background-color:#FFFFFF; margin-bottom:15px;">
                        {"".join([f"<p style='margin:5px 0;'>{item}</p>" for item in data])}
                    </div>
                    <p style='font-weight:bold; margin-top:5px;'>
                        Jumlah Unit: {len(data)}
                    </p>
                    """,
                    unsafe_allow_html=True
                )

        with col1:
            buat_kotak("Tinggi", tinggi)

        with col2:
            buat_kotak("Sedang", sedang)

        with col3:
            buat_kotak("Rendah", rendah)

        import plotly.express as px
        # --- Pilih SPKLU ---
        spklu_list = df4["Nama SPKLU"].unique()
        selected_spklu = st.selectbox("Pilih SPKLU", spklu_list)

        # --- Filter data sesuai pilihan ---
        df_selected = df4[df4["Nama SPKLU"] == selected_spklu]

        col1, col2 = st.columns(2)


        with col1:
            import plotly.express as px
            kategori_order = ["Standar", "Medium", "Fast", "Ultra Fast"]

            fig_bar = px.bar(
                df_selected,
                x="Kategori",
                y="Kapasitas",
                color="Kategori",
                text="Kapasitas",
                title=f"Kapasitas per Kategori - {selected_spklu}",
                color_discrete_sequence=px.colors.qualitative.Set2,
                category_orders={"Kategori": kategori_order}
            )

            # Tambahkan ruang di atas batang agar label tidak kepotong
            max_val = df_selected["Kapasitas"].max()
            fig_bar.update_yaxes(range=[0, max_val * 1.25])  # 25% padding atas

            # Atur margin frame
            fig_bar.update_layout(margin=dict(t=60, b=40, l=40, r=40))

            # Teks di atas batang
            fig_bar.update_traces(textposition="outside", textfont_size=12)

            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            import plotly.graph_objects as go

            # Hitung jumlah unit per kategori
            unit_per_kategori = df_selected["Kategori"].value_counts()

            # Hitung total kapasitas (untuk teks tengah)
            total_kapasitas = df_selected["Kapasitas"].sum()

            # Buat Donut Chart
            fig = go.Figure(data=[go.Pie(
                labels=unit_per_kategori.index,
                values=unit_per_kategori.values,
                hole=0.4,
                textinfo='percent',
                texttemplate='<br>%{percent:.1%}',  # tidak menampilkan jumlah unit
                insidetextfont=dict(color="black")
            )])

            # Tambahkan total kapasitas di tengah
            fig.add_annotation(
                text=f"<b>Total<br>{int(total_kapasitas)} kW</b>",  # tetap kapasitas
                x=0.5, y=0.5,
                font=dict(size=14, color="black"),
                showarrow=False
            )

            # Layout
            fig.update_layout(
                title_text=f"Proporsi Kategori Charger - {selected_spklu}",
                title_x=0.5,
                margin=dict(l=10, r=10, t=40, b=10)
            )

            st.plotly_chart(fig, use_container_width=True)
    # ============================
    # Tab 5 - Level Spklu (Klustering berdasarkan level banyak jumlah transaksi)
    # ============================
    with tab5:
        # --- Klustering SPKLU ---
        st.subheader("Klustering SPKLU")

        # --- Persiapan data ---
        fitur_group = df2.groupby("Nama SPKLU")[["Jumlah Transaksi", "Jumlah KWH", "Total Pendapatan"]].sum().reset_index()

        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(fitur_group[["Jumlah Transaksi", "Jumlah KWH", "Total Pendapatan"]])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        fitur_group["Cluster"] = kmeans.fit_predict(X_scaled)

        # Mapping cluster
        cluster_order = fitur_group.groupby("Cluster")["Total Pendapatan"].mean().sort_values().index
        mapping = {cluster_order[0]: "Rendah", cluster_order[1]: "Sedang", cluster_order[2]: "Tinggi"}
        fitur_group["Level"] = fitur_group["Cluster"].map(mapping)

        # === Bagian Rekomendasi (kotak kotak per level) ===
        rekomendasi = {
            "Tinggi": "SPKLU ini ramai digunakan. Tambah unit charger, upgrade kapasitas, atau buka cabang di lokasi serupa.",
            "Sedang": "SPKLU memiliki potensi. Dorong dengan promosi, kerjasama merchant, atau peningkatan fasilitas.",
            "Rendah": "SPKLU relatif sepi. Evaluasi lokasi, cek teknis, atau strategi marketing. Jika tetap rendah, pertimbangkan relokasi."
        }

        df_tinggi = fitur_group.query("Level == 'Tinggi'").sort_values("Total Pendapatan", ascending=False)
        df_sedang = fitur_group.query("Level == 'Sedang'").sort_values("Total Pendapatan", ascending=False)
        df_rendah = fitur_group.query("Level == 'Rendah'").sort_values("Total Pendapatan", ascending=False)
        tinggi = df_tinggi["Nama SPKLU"].tolist()
        sedang = df_sedang["Nama SPKLU"].tolist()
        rendah = df_rendah["Nama SPKLU"].tolist()

        col1, col2, col3 = st.columns(3)

        def buat_kotak(judul, data):
            with st.container():
                st.markdown(f"### {judul}")
                st.markdown(
                    f"""
                    <div style="border:1px solid #ddd; border-radius:10px; padding:10px;
                                max-height:200px; overflow-y:auto; background-color:#FFFFFF; margin-bottom:15px;">
                        {"".join([f"<p style='margin:5px 0;'>{item}</p>" for item in data])}
                    </div>
                    <p style='font-weight:bold; margin-top:5px;'>
                        Jumlah Unit: {len(data)}
                    </p>
                    """,
                    unsafe_allow_html=True
                )

        with col1:
            buat_kotak("Tinggi", tinggi)

        with col2:
            buat_kotak("Sedang", sedang)

        with col3:
            buat_kotak("Rendah", rendah)

        st.markdown("---")

        # === Layout Bawah: Scatter kiri & Pie kanan ===
        col_left, col_right = st.columns([2, 1])  # scatter lebih lebar

        with col_left:
            import plotly.express as px
            label_map = {
                "Tinggi": "Intensitas Tinggi",
                "Sedang": "Intensitas Sedang",
                "Rendah": "Intensitas Rendah"
            }
            fitur_group["Level Deskriptif"] = fitur_group["Level"].map(label_map)

            fig1 = px.scatter(
                fitur_group,
                x="Jumlah Transaksi",
                y="Jumlah KWH",
                size="Total Pendapatan",
                color="Level Deskriptif",
                hover_data=["Nama SPKLU"],
                color_discrete_map={
                    "Intensitas Rendah (sepi penggunaan)": "lightcoral",
                    "Intensitas Sedang (potensial berkembang)": "gold",
                    "Intensitas Tinggi (ramai digunakan)": "seagreen"
                },
                category_orders={"Level Deskriptif": [
                    "Intensitas Tinggi (ramai digunakan)",
                    "Intensitas Sedang (potensial berkembang)",
                    "Intensitas Rendah (sepi penggunaan)"]
                }
            )
            st.plotly_chart(fig1, use_container_width=True, key="scatter_plot")

        with col_right:
            import plotly.graph_objects as go

            # Hitung value counts
            level_count = fitur_group["Level"].value_counts()

            # Buat pie chart dengan donut style
            fig2 = go.Figure(data=[go.Pie(
                labels=level_count.index,
                values=level_count.values,
                hole=0.4,  # bikin jadi donut
                marker=dict(colors=["lightcoral", "gold", "seagreen"]),
                textinfo='percent'  # tampilkan label + persentase
            )])

            # Tambahkan total di tengah
            fig2.add_annotation(
                text=f"<b>Total<br>{level_count.sum()}</b>",
                x=0.5, y=0.5,
                font=dict(size=14, color="black"),
                showarrow=False
            )

            # Atur judul
            fig2.update_layout(
                title_text="Proporsi SPKLU per Level",
                title_x=0.5
            )

            st.plotly_chart(fig2, use_container_width=True)

        # Fitur 1: Informasi Lebih Lanjut
        with st.expander("Data Awal"):
            st.write("Menampilkan cuplikan awal data transaksi SPKLU yang digunakan dalam analisis.")
    
    
    # ============================
    # Tab 6 - Tren Bulanan Unit SPKLU
    # ============================
    with tab6:
        st.subheader("Tren Bulanan Unit SPKLU")

        # --- Pastikan kolom waktu tersedia tanpa duplikasi ---
        if not {"Bulan", "Tahun", "BulanNum"}.issubset(df2.columns):
            bulan_map = {
                "Januari": 1, "Februari": 2, "Maret": 3, "April": 4,
                "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8,
                "September": 9, "Oktober": 10, "November": 11, "Desember": 12
            }
            df2[["Bulan", "Tahun"]] = df2["Bulan & Tahun"].str.split(" ", n=1, expand=True)
            df2["Tahun"] = df2["Tahun"].astype(int)
            df2["BulanNum"] = df2["Bulan"].map(bulan_map)

        # --- Index bulan terurut + key numerik YYYYMM ---
        bulan_index = (
            df2[["Bulan & Tahun", "Bulan", "Tahun", "BulanNum"]]
            .drop_duplicates()
            .sort_values(["Tahun", "BulanNum"])
            .assign(key=lambda d: d["Tahun"] * 100 + d["BulanNum"])
        )
        opsi_bulan = bulan_index["Bulan & Tahun"].tolist()
        bulan2key = dict(zip(bulan_index["Bulan & Tahun"], bulan_index["key"]))

        # --- Pilihan Unit & Rentang Bulan ---
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            selected_spklu = st.selectbox("Pilih Unit SPKLU", sorted(df2["Nama SPKLU"].unique()))
        with c2:
            start_bulan = st.selectbox("Bulan Awal", opsi_bulan, index=0)
        with c3:
            end_bulan = st.selectbox("Bulan Akhir", opsi_bulan, index=len(opsi_bulan) - 1)

        start_key, end_key = bulan2key[start_bulan], bulan2key[end_bulan]
        lo, hi = (start_key, end_key) if start_key <= end_key else (end_key, start_key)

        # --- Kalender bulan pada rentang (agar bulan kosong jadi 0) ---
        kalender = bulan_index.loc[(bulan_index["key"] >= lo) & (bulan_index["key"] <= hi), ["key", "Bulan & Tahun"]]

        # --- Agregasi per bulan untuk unit terpilih ---
        df_unit = df2[df2["Nama SPKLU"] == selected_spklu].copy()
        df_unit["key"] = df_unit["Tahun"] * 100 + df_unit["BulanNum"]

        agg = (
            df_unit.groupby("key", as_index=False)
            .agg({
                "Total Pendapatan": "sum",
                "Jumlah KWH": "sum",
                "Jumlah Transaksi": "sum"
            })
        )

        # --- Gabungkan dengan kalender & isi nol untuk bulan tanpa data ---
        df_tren = kalender.merge(agg, on="key", how="left").fillna(0.0)
        df_tren["BulanLabel"] = df_tren["Bulan & Tahun"]  # pakai label Indonesia asli
        order_x = df_tren["BulanLabel"].tolist()

        import plotly.express as px

        col_a, col_b, col_c = st.columns(3)

        # Grafik 1: Total Pendapatan
        with col_a:
            fig_pendapatan = px.line(
                df_tren, x="BulanLabel", y="Total Pendapatan", markers=True,
                title="Total Pendapatan",
                category_orders={"BulanLabel": order_x},
                color_discrete_sequence=["#FA8072"]
            )
            fig_pendapatan.update_traces(
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate="Periode=%{x}<br>Pendapatan=Rp%{y:,.0f}<extra></extra>"
            )
            fig_pendapatan.update_yaxes(tickformat=",")
            fig_pendapatan.update_layout(margin=dict(t=50, b=10, l=10, r=10),
                xaxis_title=None,   # Hilangkan label X
                yaxis_title=None    # Hilangkan label Y
            )
            st.plotly_chart(fig_pendapatan, use_container_width=True)

        # Grafik 2: Jumlah KWH
        with col_b:
            fig_kwh = px.line(
                df_tren, x="BulanLabel", y="Jumlah KWH", markers=True,
                title="Jumlah KWH",
                category_orders={"BulanLabel": order_x},
                color_discrete_sequence=["lightgreen"]
            )
            fig_kwh.update_traces(
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate="Periode=%{x}<br>KWH=%{y:,.0f}<extra></extra>"
            )
            fig_kwh.update_yaxes(tickformat=",")
            fig_kwh.update_layout(margin=dict(t=50, b=10, l=10, r=10),
                xaxis_title=None,   # Hilangkan label X
                yaxis_title=None    # Hilangkan label Y
            )
            st.plotly_chart(fig_kwh, use_container_width=True)

        # Grafik 3: Jumlah Transaksi
        with col_c:
            fig_transaksi = px.line(
                df_tren, x="BulanLabel", y="Jumlah Transaksi", markers=True,
                title="Jumlah Transaksi",
                category_orders={"BulanLabel": order_x},
                color_discrete_sequence=["#FFBD31"]
            )
            fig_transaksi.update_traces(
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate="Periode=%{x}<br>Transaksi=%{y:,.0f}<extra></extra>"
            )
            fig_transaksi.update_yaxes(tickformat=",")
            fig_transaksi.update_layout(margin=dict(t=50, b=10, l=10, r=10),
                xaxis_title=None,   # Hilangkan label X
                yaxis_title=None    # Hilangkan label Y
            )
            st.plotly_chart(fig_transaksi, use_container_width=True)

        # Opsional: lihat data
        with st.expander("Lihat Data Tren Bulanan"):
            st.dataframe(
            df_tren[["Bulan & Tahun", "Jumlah Transaksi", "Jumlah KWH", "Total Pendapatan"]]
            .reset_index(drop=True),  # hilangkan index
            use_container_width=True
        )





# ======================
# PREDIKSI
# ======================
elif selected == "Prediksi": 
    st.title("Prediksi Jumlah Transaksi SPKLU")
    st.write("Prediksi menggunakan model Machine Learning (XGBoost) dan Prophet (opsional).")

    # ============ Load Model ============
    model_daily = joblib.load("model_daily.pkl")
    model_monthly = joblib.load("model_monthly.pkl")
    
    # ==== Load Dataset ====
    url_data5 = "https://docs.google.com/spreadsheets/d/16cyvXwvucVb7EM1qiikZpbK8J8isbktuiw-MR1EJDEY/export?format=csv&gid=2075790964"
    df5 = pd.read_csv(url_data5)
    df5.columns = df5.columns.str.strip()

    if "TGL BAYAR" not in df5.columns:
        st.error("Kolom 'TGL BAYAR' tidak ditemukan pada dataset.")
        st.stop()

    df5 = df5.dropna(subset=["TGL BAYAR"]).copy()
    df5["TGL BAYAR"] = pd.to_datetime(df5["TGL BAYAR"], errors="coerce")
    df5 = df5.dropna(subset=["TGL BAYAR"])
    df5["Tanggal"] = df5["TGL BAYAR"].dt.normalize()

    # Styling global
    plt.style.use("default")
    plt.rcParams["axes.facecolor"] = "#f9f9f9"
    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["axes.edgecolor"] = "#888888"
    plt.rcParams["axes.labelcolor"] = "#444444"
    plt.rcParams["xtick.color"] = "#444444"
    plt.rcParams["ytick.color"] = "#444444"
    plt.rcParams["grid.color"] = "#cccccc"
    
    # Tabs Harian & Bulanan
    tab1, tab2 = st.tabs(["Harian", "Bulanan"])

    # ========================
    # TAB 1: Prediksi Harian
    # ========================
    with tab1:
        st.subheader("Prediksi Harian (XGBoost)")
        daily = df5.groupby("Tanggal")["No"].nunique().rename("y").reset_index()

        if len(daily) < 14:
            st.warning("Data harian terlalu sedikit (minimal 14 hari).")
            st.stop()

        s = daily.set_index("Tanggal")["y"].asfreq("D").fillna(0)

        # Visualisasi historis
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(s.index, s.values, marker="o", label="Historis", color="#e63946")
        ax.set_title("Tren Harian — Jumlah Transaksi")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Pilihan horizon
        horizon = st.slider("Pilih Prediksi Harian", 1, 30, 7)

        # Fitur untuk prediksi
        df_feat = s.to_frame().reset_index().rename(columns={"Tanggal": "ds", "y": "y"})
        df_feat["dayofweek"] = df_feat["ds"].dt.dayofweek
        df_feat["month"] = df_feat["ds"].dt.month
        for L in [1, 2, 3, 7]:
            df_feat[f"lag{L}"] = df_feat["y"].shift(L)
        df_feat = df_feat.dropna().reset_index(drop=True)

        FEATURES = ["dayofweek", "month", "lag1", "lag2", "lag3", "lag7"]

        # Prediksi berulang
        last_ds = df_feat["ds"].iloc[-1]
        series = df_feat.set_index("ds")["y"].copy()
        preds = []
        for h in range(1, horizon + 1):
            nxt = last_ds + timedelta(days=h)
            row = {
                "dayofweek": nxt.dayofweek,
                "month": nxt.month,
                "lag1": series.iloc[-1],
                "lag2": series.iloc[-2] if len(series) >= 2 else series.iloc[-1],
                "lag3": series.iloc[-3] if len(series) >= 3 else series.iloc[-1],
                "lag7": series.iloc[-7] if len(series) >= 7 else series.iloc[-1],
            }
            yhat = float(model_daily.predict(pd.DataFrame([row])[FEATURES])[0])
            preds.append((nxt, max(0, yhat)))
            series.loc[nxt] = yhat

        df_pred = pd.DataFrame(preds, columns=["Tanggal", "Prediksi"])
        st.dataframe(df_pred, hide_index=True)

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(s.index, s.values, label="Historis", color="#e63946")
        ax.plot(df_pred["Tanggal"], df_pred["Prediksi"], "--o", label="Forecast", color="#457b9d")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # ========================
    # TAB 2: Prediksi Bulanan
    # ========================
    with tab2:
        st.subheader("Prediksi Bulanan (XGBoost)")
        df5["Periode"] = df5["TGL BAYAR"].dt.to_period("M").dt.to_timestamp(how="start")
        monthly = df5.groupby("Periode")["No"].nunique().rename("y").reset_index()

        if len(monthly) < 6:
            st.warning("Data bulanan terlalu sedikit (butuh ≥ 6 titik).")
            st.stop()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(monthly["Periode"], monthly["y"], "o-", label="Historis", color="#e63946")
        ax.set_title("Tren Bulanan — Jumlah Transaksi")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # Pilihan horizon
        horizon_m = st.slider("Pilih Prediksi Bulan Kedepan", 1, 24, 6)

        # Siapkan fitur bulanan
        dfm = monthly.rename(columns={"Periode": "ds", "y": "y"})
        dfm["month"] = dfm["ds"].dt.month
        dfm["year"] = dfm["ds"].dt.year
        for L in [1, 2, 3, 6, 12]:
            dfm[f"lag{L}"] = dfm["y"].shift(L)
        dfm = dfm.dropna().reset_index(drop=True)

        FEATURES_M = ["month", "year", "lag1", "lag2", "lag3", "lag6", "lag12"]

        # Prediksi berulang
        last_ds = dfm["ds"].iloc[-1]
        cur = dfm.set_index("ds")["y"].copy()
        preds_m = []
        for h in range(1, horizon_m + 1):
            nxt = (last_ds + pd.offsets.MonthBegin(h))
            row = {
                "month": nxt.month, "year": nxt.year,
                "lag1": cur.iloc[-1],
                "lag2": cur.iloc[-2] if len(cur) >= 2 else cur.iloc[-1],
                "lag3": cur.iloc[-3] if len(cur) >= 3 else cur.iloc[-1],
                "lag6": cur.iloc[-6] if len(cur) >= 6 else cur.iloc[-1],
                "lag12": cur.iloc[-12] if len(cur) >= 12 else cur.iloc[-1],
            }
            yhat = float(model_monthly.predict(pd.DataFrame([row])[FEATURES_M])[0])
            preds_m.append((nxt, max(0, yhat)))
            cur.loc[nxt] = yhat

        df_pred_m = pd.DataFrame(preds_m, columns=["Periode", "Forecast"])
        st.dataframe(df_pred_m, hide_index=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(monthly["Periode"], monthly["y"], label="Historis", color="#e63946")
        ax.plot(df_pred_m["Periode"], df_pred_m["Forecast"], "--o", label="Forecast (XGB)", color="#457b9d")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

elif selected == "Tentang":
    st.title('Tentang Dashboard Ini')
    st.write("""
    Dashboard ini dibuat untuk menganalisis data transaksi SPKLU secara interaktif dan informatif.

    ### Apa itu SPKLU?
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
