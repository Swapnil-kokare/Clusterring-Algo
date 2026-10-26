# 🎓 Smart Placement Intelligence & Shortlisting Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, end-to-end talent placement intelligence platform. It clusters student candidate profiles based on 18 technical skill dimensions using unsupervised machine learning, visualizes high-dimensional spaces with 2D Principal Component Analysis (PCA), benchmarks clustering algorithms, and parses corporate **Job Descriptions (via JSON or text files)** to generate explainable candidate shortlists.

---

## 🌟 Key Features

- **🎨 Modern Glassmorphic Dashboard**: Dark-mode UI with fluid navigation, SVG icons, and responsive layouts.
- **🏠 Intelligent Landing & Overview Screen**: Platform metrics, system architecture workflow, and feature showcases.
- **👥 Student Directory & Data Ingestion**:
  - Live search and branch filtering across 1,000+ candidate profiles.
  - CSV dataset upload and sample loader.
  - Interactive modal to add new student profiles dynamically, updating the ML model on the fly.
- **🧩 Unsupervised Skill Clustering**:
  - Compare **K-Means**, **Agglomerative Hierarchical (Ward)**, **Gaussian Mixture Models (GMM)**, and **DBSCAN**.
  - Interactive **2D PCA projection** with Chart.js scatter plots and hover tooltips.
  - Cluster profiling with dominant skills and average CGPA metrics.
- **📊 Algorithmic Benchmarking**:
  - Rigorous evaluation metrics: **Silhouette Score**, **Davies-Bouldin Index**, **Calinski-Harabasz Index**, and **Adjusted Rand Index (ARI)** against ground truth profiles.
  - Dual-axis line chart for Elbow (Inertia) vs Silhouette Score across $k \in [2, 10]$.
- **🎯 Automated Job Description (JD) Ingestion & Matching**:
  - Ingest **JSON Job Descriptions** or upload raw text JD files (no manual slider inputs required).
  - Quick presets: *Full Stack Engineer*, *Data Scientist*, *Cloud DevOps*, *Java Backend*, and *Embedded IoT*.
  - Cluster relevance evaluation against cluster centroids.
  - Transparent 0–100 multi-factor matching score with candidate skill gap badges and AI explainability.
  - One-click export to CSV (`shortlist.csv`).
- **📓 Jupyter Data Science Notebook**: Full end-to-end analytical pipeline in `smart_placement_pipeline.ipynb`.

---

## 📐 Mathematical Formulation

### Composite Match Score (0 - 100)
$$\text{Skill Fit} = \frac{1}{M} \sum_{i=1}^M \min\left(\frac{\text{Candidate Proficiency}_i}{\text{Required Level}_i}, 1.0\right)$$

$$\text{Experience} = 0.50 \times \min\left(\frac{\text{Projects}}{4}, 1\right) + 0.30 \times \min\left(\frac{\text{Internships}}{2}, 1\right) + 0.20 \times \min\left(\frac{\text{Certs}}{3}, 1\right)$$

$$\text{Score} = 100 \times \left(0.65 \times \text{Skill Fit} + 0.15 \times \frac{\text{CGPA}}{10} + 0.10 \times \text{Experience} + 0.10 \times \frac{\text{Aptitude}}{100}\right)$$

> **Eligibility Criteria**: Minimum CGPA cutoff and maximum active backlogs function as non-negotiable hard filters. Candidates in target high-affinity clusters are prioritized.

---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Swapnil-kokare/smart-placement-system.git
cd smart-placement-system/smart-placement
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows PowerShell
# source venv/bin/activate    # On Linux / macOS

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
python app.py
```
Open your browser and navigate to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

*(Windows users can also simply run `.\run.ps1` or double-click `run.bat` for one-click startup).*

---

## 📁 Repository Structure

```text
smart-placement/
├── app.py                      # Flask REST API & ML Clustering Engine
├── requirements.txt            # Python dependencies
├── smart_placement_pipeline.ipynb # Full Jupyter Data Science pipeline
├── run.bat                     # Windows Command Prompt launcher
├── run.ps1                     # PowerShell automated launcher
├── data/
│   ├── students.csv            # 1,000 student candidate profiles
│   ├── shortlist_output.csv    # Exported shortlist output
│   └── job_descriptions/       # Sample Job Description JSON files
│       ├── full_stack_developer.json
│       ├── data_scientist.json
│       ├── cloud_devops_engineer.json
│       ├── java_backend_developer.json
│       └── embedded_iot_engineer.json
└── static/
    └── index.html              # Modern Glassmorphic Frontend (Chart.js + SVG Icons)
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the web application |
| `/api/load` | `POST` | Loads or resets the student CSV dataset |
| `/api/students` | `GET` | Search and filter candidates by query & branch |
| `/api/add_student` | `POST` | Dynamically registers a custom candidate |
| `/api/cluster` | `POST` | Performs clustering (K-Means, Ward, GMM, DBSCAN) |
| `/api/compare` | `POST` | Evaluates all algorithms & computes $k$-scan metrics |
| `/api/match` | `POST` | Matches JD requirements against student clusters |

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
