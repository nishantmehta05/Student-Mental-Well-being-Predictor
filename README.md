# 🧠 MindPulse AI — Student Mental Health & Social Media Analytics Dashboard

A modern, production-grade Streamlit web application and analytics platform built on top of a trained Scikit-Learn **Random Forest Regression Pipeline** (`Mental_Health_Model.pkl`).

---

## 🚀 Features

- **🎯 Interactive Student Assessment**:
  - Predict student mental well-being scores based on social media usage, smartphone unlocks, sleep duration, physical activity, and stress levels.
  - Interactive gauge chart with real-time score categorization (Optimal, Good, Moderate Risk, High Vulnerability).
  - Persona quick-fill presets (*Average Student*, *High Screen Time / Burnout*, *Balanced Lifestyle*).
  - Tailored psychological factor analysis & lifestyle recommendations.
- **📁 High-Throughput Batch Inference**:
  - Upload CSV cohorts with multiple students.
  - Download pre-configured sample CSV template for testing.
  - Instant cohort summary statistics, distribution charts, and exportable evaluated CSV with predicted scores and risk levels.
- **📊 Cohort Analytics & Exploratory Data**:
  - Interactive Plotly visualizations analyzing screen time vs mental health, sleep duration impact, social media platform differences, and country-level comparisons.
- **🔬 Model Diagnostics & Technical Specs**:
  - Authentic model evaluation metrics: Test $R^2 = 87.8\%$, Train $R^2 = 98.1\%$, MAE = 0.347, RMSE = 0.463.
  - Live Random Forest feature importance rankings.
  - Full preprocessing architecture overview.

---

## 📂 Project Structure

```text
ml project/
│
├── .venv/                                       # Python virtual environment
├── Mental_Health_Model.pkl                     # Trained Random Forest Scikit-Learn Pipeline
├── ML_Project.ipynb                            # Original EDA & model training notebook
├── Student Social Media And Mental Health Impact.csv # Original student dataset
├── requirements.txt                            # Application dependencies
├── README.md                                   # Documentation and run instructions
├── app.py                                      # Streamlit UI dashboard
└── src/
    ├── __init__.py
    └── ml_service.py                           # Inference, validation & analytics service
```

---

## 🛠️ Installation & Setup

### 1. Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt (cmd):**
```cmd
.\.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

---

### 2. Install Required Dependencies

```powershell
pip install -r requirements.txt
```

---

### 3. Launch the Streamlit Dashboard

Run the following command from the project root directory:

```powershell
streamlit run app.py
```

The application will start and open automatically in your browser at `http://localhost:8501`.

---

## 🧠 Model & Feature Details

The underlying pipeline processes 12 input features across 4 sub-pipelines:
1. **Skewed Numeric:** `Study_Hours` (Log1p transformation + StandardScaler)
2. **Standard Numeric:** `Age`, `Avg_Daily_Usage_Hours`, `Daily_Unlocks`, `Physical_Activity_Hours`, `Sleep_Hours_Per_Night` (StandardScaler)
3. **Ordinal Categorical:** `Stress_Level` (Low, Medium, High, Very High)
4. **Nominal Categorical:** `Gender`, `Academic_Level`, `Most_Used_Platform`, `Purpose_Of_Use`, `Grouped_country` (OneHotEncoder)
5. **Estimator:** `RandomForestRegressor(random_state=42)`

