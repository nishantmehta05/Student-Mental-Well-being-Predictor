"""
MindPulse AI - Student Social Media & Mental Health Impact Dashboard
Calm Wellness Theme: Full-width single dashboard with sage-green accents.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.ml_service import (
    load_model,
    load_cleaned_dataset,
    predict_single,
    CATEGORICAL_OPTIONS,
    TOP_COUNTRIES,
    ORDERED_STRESS_LEVELS
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="MindPulse AI | Student Mental Health Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Calm Wellness CSS Theme (Sage Green & Clean White)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Styling & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Page Background */
    .stApp, [data-testid="stAppViewContainer"], .main {
        background-color: #F5F8F6 !important;
        color: #24332E !important;
    }

    /* Completely Remove Sidebar & Collapse Controls */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    section[data-testid="stSidebar"],
    .stSidebar,
    button[data-testid="baseButton-headerNoPadding"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        width: 0px !important;
        height: 0px !important;
        margin: 0 !important;
        padding: 0 !important;
        pointer-events: none !important;
    }

    /* Full-Width Main Content Layout without Top Gap */
    .main, [data-testid="stMain"] {
        width: 100% !important;
        max-width: 100% !important;
        padding: 0 !important;
    }

    .main .block-container,
    [data-testid="stMainBlockContainer"],
    .block-container {
        max-width: 100% !important;
        width: 100% !important;
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        box-sizing: border-box !important;
    }

    @media (max-width: 768px) {
        .main .block-container,
        [data-testid="stMainBlockContainer"],
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
        }
    }

    /* Calm Wellness Hero Header */
    .hero-header {
        background: #FFFFFF;
        border: 1px solid #DCE5DF;
        border-left: 5px solid #2F7D68;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 22px;
        box-shadow: 0 2px 10px rgba(36, 51, 46, 0.04);
    }
    .hero-title {
        color: #24332E;
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 6px;
        letter-spacing: -0.01em;
    }
    .hero-subtitle {
        color: #687870;
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0;
        line-height: 1.5;
    }

    /* Typography & Headers */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #24332E !important;
    }
    .stMarkdown h4, .stMarkdown h5 {
        color: #24332E !important;
        font-weight: 600 !important;
    }

    /* Cards & Containers */
    [data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DCE5DF !important;
        border-radius: 14px !important;
        padding: 24px !important;
        box-shadow: 0 2px 10px rgba(36, 51, 46, 0.03) !important;
    }

    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #DCE5DF;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(36, 51, 46, 0.03);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .metric-card:hover {
        border-color: #2F7D68;
        transform: translateY(-1px);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #687870;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 700;
        color: #24332E;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #687870;
        margin-top: 5px;
    }

    /* Inputs, Selectboxes & Dropdowns */
    [data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    input {
        background-color: #F8FAF8 !important;
        border-color: #DCE5DF !important;
        color: #24332E !important;
        border-radius: 8px !important;
    }
    [data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"]:focus-within {
        border-color: #2F7D68 !important;
        box-shadow: 0 0 0 2px rgba(47, 125, 104, 0.15) !important;
    }
    div[data-baseweb="popover"], ul[role="listbox"] {
        background-color: #FFFFFF !important;
        color: #24332E !important;
        border: 1px solid #DCE5DF !important;
        border-radius: 8px !important;
    }
    li[role="option"] {
        color: #24332E !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] {
        background-color: #DDEFE8 !important;
        color: #256653 !important;
    }

    /* Sage Green Sliders */
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
        background-color: #2F7D68 !important;
        border-color: #2F7D68 !important;
        box-shadow: 0 0 0 3px rgba(47, 125, 104, 0.2) !important;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] div > div[style*="background"] {
        background-color: #2F7D68 !important;
    }
    div[data-testid="stSelectSlider"] div[data-baseweb="slider"] div[role="slider"] {
        background-color: #2F7D68 !important;
        border-color: #2F7D68 !important;
        box-shadow: 0 0 0 3px rgba(47, 125, 104, 0.2) !important;
    }
    div[data-testid="stSelectSlider"] div[data-baseweb="slider"] div > div[style*="background"] {
        background-color: #2F7D68 !important;
    }

    /* Primary Accent Buttons */
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    .stButton > button[kind="primary"] {
        background-color: #2F7D68 !important;
        color: #FFFFFF !important;
        border: 1px solid #2F7D68 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.2rem !important;
        box-shadow: 0 2px 6px rgba(47, 125, 104, 0.18) !important;
        transition: all 0.2s ease !important;
    }
    button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    .stButton > button[kind="primary"]:hover {
        background-color: #256653 !important;
        border-color: #256653 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 10px rgba(37, 102, 83, 0.25) !important;
    }

    /* Secondary / Preset Buttons */
    button[kind="secondary"],
    button[data-testid="baseButton-secondary"],
    .stButton > button:not([kind="primary"]) {
        background-color: #FFFFFF !important;
        color: #24332E !important;
        border: 1px solid #DCE5DF !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(36, 51, 46, 0.04) !important;
    }
    button[kind="secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover {
        background-color: #DDEFE8 !important;
        border-color: #2F7D68 !important;
        color: #256653 !important;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* Insight Cards */
    .insight-card {
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        font-size: 0.92rem;
        line-height: 1.45;
    }
    .insight-positive {
        background: #DDEFE8;
        border-left: 4px solid #2F7D68;
        color: #24332E;
    }
    .insight-negative {
        background: #FDF0ED;
        border-left: 4px solid #BC5340;
        color: #24332E;
    }

    /* Divider */
    hr {
        border-color: #DCE5DF !important;
        margin: 28px 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Resource Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained ML model pipeline...")
def get_cached_model():
    return load_model()

@st.cache_data(show_spinner="Loading baseline student cohort data...")
def get_cached_dataset():
    return load_cleaned_dataset()

try:
    model = get_cached_model()
    df_cohort = get_cached_dataset()
    dataset_loaded = True
except Exception as e:
    st.error(f"Error loading system assets: {str(e)}")
    st.stop()


# ---------------------------------------------------------
# MAIN DASHBOARD: Student Mental Well-being Predictor
# (Starts immediately at top of page - zero navigation gap)
# ---------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-title">Student Mental Well-being Predictor</div>
    <p class="hero-subtitle">Evaluate the psychological impact of daily social media habits, sleep schedules, and lifestyle factors using the trained Random Forest model.</p>
</div>
""", unsafe_allow_html=True)

# Persona Quick-Fill
st.markdown("##### ⚡ Quick Persona Presets")
col_p1, col_p2, col_p3 = st.columns(3)

# Session state for inputs
if "preset_applied" not in st.session_state:
    st.session_state.age = 21
    st.session_state.gender = "Female"
    st.session_state.academic_level = "Undergraduate"
    st.session_state.country = "USA"
    st.session_state.platform = "Instagram"
    st.session_state.purpose = "Entertainment"
    st.session_state.daily_usage = 4.2
    st.session_state.unlocks = 75
    st.session_state.sleep = 6.8
    st.session_state.study = 3.2
    st.session_state.exercise = 1.2
    st.session_state.stress = "Medium"

if col_p1.button("🎓 Average Student Persona", use_container_width=True):
    st.session_state.age = 21
    st.session_state.gender = "Female"
    st.session_state.academic_level = "Undergraduate"
    st.session_state.country = "USA"
    st.session_state.platform = "Instagram"
    st.session_state.purpose = "Entertainment"
    st.session_state.daily_usage = 4.3
    st.session_state.unlocks = 78
    st.session_state.sleep = 6.6
    st.session_state.study = 3.0
    st.session_state.exercise = 1.0
    st.session_state.stress = "Medium"
    st.rerun()

if col_p2.button("🚨 High Screen Time / Burnout", use_container_width=True):
    st.session_state.age = 20
    st.session_state.gender = "Male"
    st.session_state.academic_level = "Undergraduate"
    st.session_state.country = "India"
    st.session_state.platform = "TikTok"
    st.session_state.purpose = "Entertainment"
    st.session_state.daily_usage = 7.5
    st.session_state.unlocks = 135
    st.session_state.sleep = 4.5
    st.session_state.study = 1.5
    st.session_state.exercise = 0.2
    st.session_state.stress = "Very High"
    st.rerun()

if col_p3.button("🌱 Balanced & High Resilience", use_container_width=True):
    st.session_state.age = 22
    st.session_state.gender = "Female"
    st.session_state.academic_level = "Graduate"
    st.session_state.country = "Canada"
    st.session_state.platform = "YouTube"
    st.session_state.purpose = "Education"
    st.session_state.daily_usage = 1.8
    st.session_state.unlocks = 35
    st.session_state.sleep = 8.2
    st.session_state.study = 4.5
    st.session_state.exercise = 2.5
    st.session_state.stress = "Low"
    st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Input Form
with st.form("student_assessment_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 👤 Student Profile")
        age = st.slider("Age", min_value=17, max_value=35, value=st.session_state.age, help="Student age in years")
        gender = st.selectbox("Gender", options=CATEGORICAL_OPTIONS['Gender'], index=CATEGORICAL_OPTIONS['Gender'].index(st.session_state.gender))
        academic_level = st.selectbox("Academic Level", options=CATEGORICAL_OPTIONS['Academic_Level'], index=CATEGORICAL_OPTIONS['Academic_Level'].index(st.session_state.academic_level))
        country = st.selectbox("Country of Residence", options=TOP_COUNTRIES, index=TOP_COUNTRIES.index(st.session_state.country) if st.session_state.country in TOP_COUNTRIES else 0)

    with col2:
        st.markdown("#### 📱 Digital Habits")
        platform = st.selectbox("Most Used Platform", options=CATEGORICAL_OPTIONS['Most_Used_Platform'], index=CATEGORICAL_OPTIONS['Most_Used_Platform'].index(st.session_state.platform))
        purpose = st.selectbox("Primary Purpose of Use", options=CATEGORICAL_OPTIONS['Purpose_Of_Use'], index=CATEGORICAL_OPTIONS['Purpose_Of_Use'].index(st.session_state.purpose))
        daily_usage = st.slider("Social Media Usage (Hours/Day)", min_value=0.2, max_value=12.0, value=float(st.session_state.daily_usage), step=0.1, help="Total hours spent on social platforms daily")
        unlocks = st.slider("Daily Phone Unlocks", min_value=10, max_value=200, value=int(st.session_state.unlocks), step=1, help="Total times smartphone screen is unlocked daily")

    with col3:
        st.markdown("#### 🌿 Health & Lifestyle")
        sleep = st.slider("Sleep Duration (Hours/Night)", min_value=3.0, max_value=12.0, value=float(st.session_state.sleep), step=0.1, help="Average nightly sleep duration")
        study = st.slider("Study Hours (Hours/Day)", min_value=0.2, max_value=10.0, value=float(st.session_state.study), step=0.1, help="Daily dedicated academic study time")
        exercise = st.slider("Physical Activity (Hours/Day)", min_value=0.0, max_value=5.0, value=float(st.session_state.exercise), step=0.1, help="Exercise, sports, or gym hours per day")
        stress = st.select_slider("Self-Reported Stress Level", options=ORDERED_STRESS_LEVELS, value=st.session_state.stress)

    submit_btn = st.form_submit_button("🔮 Predict Mental Health Impact", use_container_width=True, type="primary")

if submit_btn:
    input_payload = {
        'Study_Hours': study,
        'Age': age,
        'Avg_Daily_Usage_Hours': daily_usage,
        'Daily_Unlocks': unlocks,
        'Physical_Activity_Hours': exercise,
        'Sleep_Hours_Per_Night': sleep,
        'Stress_Level': stress,
        'Gender': gender,
        'Academic_Level': academic_level,
        'Most_Used_Platform': platform,
        'Purpose_Of_Use': purpose,
        'Country': country
    }

    with st.spinner("Executing ML pipeline transformation and inference..."):
        result = predict_single(model, input_payload)

    st.markdown("---")
    st.markdown("### 📊 Prediction Results & Psychological Profile")

    # Top Metric Cards
    score = result["score"]
    category = result["category"]
    
    # Calm Wellness Badge Colors
    if category == "Optimal Well-being":
        badge_bg = "#DDEFE8"
        badge_text = "#256653"
        badge_border = "#2F7D68"
        accent_color = "#2F7D68"
    elif category == "Good Well-being":
        badge_bg = "#EBF3F0"
        badge_text = "#2F7D68"
        badge_border = "#4A8B77"
        accent_color = "#4A8B77"
    elif category == "Moderate Risk / Strain":
        badge_bg = "#FFF5E8"
        badge_text = "#9A6318"
        badge_border = "#D79A45"
        accent_color = "#C28434"
    else:
        badge_bg = "#FDF0ED"
        badge_text = "#A33827"
        badge_border = "#D96B58"
        accent_color = "#BC5340"
    
    # Calculate cohort percentile
    cohort_scores = df_cohort['Mental_Health_Score']
    percentile = (cohort_scores < score).mean() * 100

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Well-being</div>
            <div class="metric-val" style="color: {accent_color};">{score:.2f} <span style="font-size: 1rem; color: #687870;">/ 10</span></div>
            <div class="metric-sub">Cohort Mean: {cohort_scores.mean():.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Classification Status</div>
            <div style="margin-top: 8px;">
                <span class="badge" style="background-color: {badge_bg}; color: {badge_text}; border: 1px solid {badge_border};">
                    {category}
                </span>
            </div>
            <div class="metric-sub">{result['summary']}</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Cohort Percentile</div>
            <div class="metric-val">{percentile:.0f}<span style="font-size: 1rem; color: #687870;">th</span></div>
            <div class="metric-sub">Higher than {percentile:.1f}% of surveyed students</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        ratio = daily_usage / max(sleep, 0.1)
        ratio_status = "Balanced" if ratio < 0.6 else ("Moderate" if ratio < 1.0 else "High Screen Ratio")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Screen-to-Sleep Ratio</div>
            <div class="metric-val">{ratio:.2f}</div>
            <div class="metric-sub">{ratio_status} ({daily_usage}h screen / {sleep}h sleep)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    r_col1, r_col2 = st.columns([1, 1.2])

    with r_col1:
        st.markdown("##### 🎯 Well-being Score Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Well-being Index", 'font': {'size': 16, 'color': '#24332E'}},
            delta={'reference': cohort_scores.mean(), 'increasing': {'color': "#2F7D68"}, 'decreasing': {'color': "#BC5340"}},
            number={'font': {'color': '#24332E'}},
            gauge={
                'axis': {'range': [1, 10], 'tickwidth': 1, 'tickcolor': "#687870"},
                'bar': {'color': "#2F7D68", 'thickness': 0.25},
                'bgcolor': "#FFFFFF",
                'borderwidth': 1,
                'bordercolor': "#DCE5DF",
                'steps': [
                    {'range': [1.0, 5.0], 'color': '#FDF0ED'},
                    {'range': [5.0, 6.2], 'color': '#FFF5E8'},
                    {'range': [6.2, 7.5], 'color': '#EBF3F0'},
                    {'range': [7.5, 10.0], 'color': '#DDEFE8'}
                ],
                'threshold': {
                    'line': {'color': "#24332E", 'width': 3},
                    'thickness': 0.8,
                    'value': score
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font={'color': "#24332E"},
            height=320,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with r_col2:
        st.markdown("##### ⚖️ Student vs Cohort Benchmark")
        
        metrics_compare = ['Daily Screen Time (h)', 'Sleep Duration (h)', 'Study Hours (h)', 'Daily Unlocks (÷10)', 'Physical Activity (h)']
        student_vals = [daily_usage, sleep, study, unlocks / 10, exercise]
        cohort_avg_vals = [
            df_cohort['Avg_Daily_Usage_Hours'].mean(),
            df_cohort['Sleep_Hours_Per_Night'].mean(),
            df_cohort['Study_Hours'].mean(),
            df_cohort['Daily_Unlocks'].mean() / 10,
            df_cohort['Physical_Activity_Hours'].mean()
        ]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name='Assessed Student',
            x=metrics_compare,
            y=student_vals,
            marker_color='#2F7D68'
        ))
        fig_comp.add_trace(go.Bar(
            name='Cohort Average',
            x=metrics_compare,
            y=cohort_avg_vals,
            marker_color='#A3B8AF'
        ))

        fig_comp.update_layout(
            barmode='group',
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font={'color': "#687870"},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=320,
            margin=dict(l=10, r=10, t=20, b=20),
            yaxis=dict(gridcolor='#EAEFEA'),
            xaxis=dict(gridcolor='#EAEFEA')
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Risk and Protective Factors
    st.markdown("##### 🔍 Identified Factors & Actionable Recommendations")
    if result["insights"]:
        for ins in result["insights"]:
            card_class = "insight-positive" if ins["type"] == "positive" else "insight-negative"
            icon = "🌿" if ins["type"] == "positive" else "⚠️"
            st.markdown(f"""
            <div class="insight-card {card_class}">
                <strong>{icon} {ins['title']}</strong><br>
                <span>{ins['desc']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Input habits are close to the balanced central norms of the surveyed population.")



st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("MindPulse AI • Built with Streamlit, Scikit-Learn, and Plotly • Calm Wellness Edition")
