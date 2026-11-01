# ⚽ EuroScout AI

> **Advanced Scouting & Performance Intelligence Platform**
> Europe's Big 5 Football Leagues (2024–2025 Season)

EuroScout AI is an analytics and visualization platform for football player performance across Europe's top 5 leagues (Premier League, La Liga, Serie A, Bundesliga, and Ligue 1) for the 2024–2025 season, built on FBref data. The application is built with Streamlit, Plotly, Pandas, and Scikit-learn, and is organized into 5 integrated modules.

---

## 👥 Team Roles & Module Assignments

| Member | Task / Focus Module | Key Features / Metrics | Status |
| :--- | :--- | :--- | :---: |
| **Member 1** | **Attacking Analytics** | $xG$ (*Expected Goals*) analysis, *finishing efficiency* ($Gls - xG$), *Shot-Creating Actions* (SCA), and shot accuracy. | ✅ Done |
| **Member 2** | **Playmaking & Passing** | *Progressive Passes*, *Key Passes*, $xAG$ (*Expected Assisted Goals*), and *pass completion rate* ($Cmp\%$). | ⏳ In Progress |
| **Member 3** | **Defensive & Workrate** | *Tackles*, *Interceptions*, *Ball Recoveries*, and distribution of defensive actions (*defensive 3rd, mid 3rd, att 3rd*). | ✅ Done |
| **Member 4** | **Scouting / ML Clustering** | *Cosine Similarity Search* & *K-Means Clustering* models (with PCA and Silhouette / Elbow evaluation) for the *Player Recommendation Tool* and *Percentile Radar Chart*. | ✅ Done |
| **Member 5** | **Dashboard & Integration** | Streamlit 5-tab UI orchestration, *Executive Overview & League Intelligence*, data loader architecture (`data_loader.py`), visual theme (`theme.py`), and repository documentation. | ✅ Done |

---

## 📁 Project Directory Structure

```text
euroscout-ai/
├── app.py                  # Member 5 (Main Orchestrator & Streamlit Layout)
├── requirements.txt        # Member 5 (Python dependency list)
├── README.md               # Member 5 (Full repository documentation)
├── data/
│   └── players_data.csv    # FBref Big 5 player stats dataset 2024-2025
├── utils/
│   ├── data_loader.py      # Member 5 (Cleaning, column normalization, & caching)
│   └── theme.py            # Member 5 (Dark design system & unified Plotly template)
└── modules/
    ├── overview.py         # Member 5 (Tab 1: Executive Overview & League Intelligence)
    ├── attacking.py        # Member 1 (Tab 2: Attacking Analytics & Finishing)
    ├── playmaking.py       # Member 2 (Tab 3: Passing & Playmaking)
    ├── defensive.py        # Member 3 (Tab 4: Defensive Workrate)
    └── recommender.py      # Member 4 (Tab 5: ML Scouting & Similarity Search)
```

---

## 🚀 Installation & Running Guide

### 1. Prerequisites
* Python 3.9 or newer
* pip / virtualenv

### 2. Clone the Repository & Enter the Directory
```bash
git clone https://github.com/<username>/football-analytics-dashboard.git
cd football-analytics-dashboard
```

### 3. Create and Activate a Virtual Environment
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

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Application
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Data Source
The dataset is compiled from public FBref data for the top 5 European leagues for the 2024–2025 season:
* **Kaggle**: [Football Players Stats 2024-2025](https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025)
* The data covers combined statistics from: Standard Stats, Shooting, Passing, Goal and Shot Creation, Defense, Possession, Playing Time, and Goalkeeping.
