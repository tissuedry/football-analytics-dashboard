# ⚽ EuroScout AI

> **Advanced Scouting & Performance Intelligence Platform**  
> Europe's Big 5 Football Leagues (2024–2025 Season)

EuroScout AI adalah platform analitik dan visualisasi data performa pemain sepak bola di 5 liga top Eropa (Premier League, La Liga, Serie A, Bundesliga, dan Ligue 1) musim 2024–2025 berbasis data FBref. Aplikasi ini dibangun menggunakan Streamlit, Plotly, Pandas, dan Scikit-learn dengan pembagian 5 modul terintegrasi.

---

## 👥 Pembagian Tugas & Modul Anggota Kelompok

| Anggota | Tugas / Modul Fokus | Fitur / Metrik Utama | Status |
| :--- | :--- | :--- | :---: |
| **Anggota 1** | **Attacking Analytics** | Analisis $xG$ (*Expected Goals*), *finishing efficiency* ($Gls - xG$), *Shot-Creating Actions* (SCA), dan akurasi tembakan. | ✅ Selesai |
| **Anggota 2** | **Playmaking & Passing** | *Progressive Passes*, *Key Passes*, $xAG$ (*Expected Assisted Goals*), dan *pass completion rate* ($Cmp\%$). | ⏳ In Progress |
| **Anggota 3** | **Defensive & Workrate** | *Tackles*, *Interceptions*, *Ball Recoveries*, dan distribusi aksi bertahan (*defensive 3rd, mid 3rd, att 3rd*). | ✅ Selesai |
| **Anggota 4** | **Scouting / ML Clustering** | Model *Cosine Similarity Search* & *K-Means Clustering* (dengan PCA & evaluasi Silhouette / Elbow) untuk *Player Recommendation Tool* serta *Percentile Radar Chart*. | ✅ Selesai |
| **Anggota 5** | **Dashboard & Integration** | Orkestrasi Streamlit UI 5-tabs, *Executive Overview & League Intelligence*, arsitektur data loader (`data_loader.py`), tema visual (`theme.py`), dan dokumentasi repositori. | ✅ Selesai |

---

## 📁 Struktur Direktori Proyek

```text
euroscout-ai/
├── app.py                  # Anggota 5 (Main Orchestrator & Streamlit Layout)
├── requirements.txt        # Anggota 5 (Daftar dependensi Python)
├── README.md               # Anggota 5 (Dokumentasi lengkap repositori)
├── data/
│   └── players_data.csv    # Dataset statistik pemain FBref Big 5 2024-2025
├── utils/
│   ├── data_loader.py      # Anggota 5 (Pembersihan, normalisasi kolom, & caching)
│   └── theme.py            # Anggota 5 (Design system gelap & template Plotly terpadu)
└── modules/
    ├── overview.py         # Anggota 5 (Tab 1: Executive Overview & League Intelligence)
    ├── attacking.py        # Anggota 1 (Tab 2: Attacking Analytics & Finishing)
    ├── playmaking.py       # Anggota 2 (Tab 3: Passing & Playmaking)
    ├── defensive.py        # Anggota 3 (Tab 4: Defensive Workrate)
    └── recommender.py      # Anggota 4 (Tab 5: ML Scouting & Similarity Search)
```

---

## 🚀 Panduan Instalasi & Menjalankan Aplikasi

### 1. Prasyarat
* Python 3.9 atau lebih baru
* pip / virtualenv

### 2. Clone Repositori & Masuk Direktori
```bash
git clone https://github.com/<username>/football-analytics-dashboard.git
cd football-analytics-dashboard
```

### 3. Buat dan Aktifkan Virtual Environment
* **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 4. Pasang Dependensi
```bash
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di peramban pada alamat `http://localhost:8501`.

---

## 📊 Sumber Data
Dataset dihimpun dari data publik FBref untuk 5 liga top Eropa musim 2024–2025:
* **Kaggle**: [Football Players Stats 2024-2025](https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025)
* Data mencakup statistik gabungan: Standard Stats, Shooting, Passing, Goal and Shot Creation, Defense, Possession, Playing Time, dan Goalkeeping.
