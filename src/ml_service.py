"""
ML Service Module for Student Social Media & Mental Health Impact Prediction.
Handles model loading, data validation, preprocessing, inference, and analytics.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Mental_Health_Model.pkl"
DATASET_PATH = BASE_DIR / "Student Social Media And Mental Health Impact.csv"

# Preprocessing Constants established in training
TOP_COUNTRIES = ['Other', 'India', 'USA', 'Canada', 'Australia', 'UK', 'Germany', 'Mexico', 'Turkey', 'France']
ORDERED_STRESS_LEVELS = ['Low', 'Medium', 'High', 'Very High']
CATEGORICAL_OPTIONS = {
    'Gender': ['Female', 'Male'],
    'Academic_Level': ['High School', 'Undergraduate', 'Graduate'],
    'Most_Used_Platform': [
        'Instagram', 'YouTube', 'TikTok', 'WhatsApp', 'Facebook', 
        'Snapchat', 'Twitter', 'LinkedIn', 'LINE', 'WeChat', 'KakaoTalk', 'VKontakte'
    ],
    'Purpose_Of_Use': ['Entertainment', 'Education', 'Networking', 'News'],
    'Stress_Level': ORDERED_STRESS_LEVELS,
    'Country': TOP_COUNTRIES
}

FEATURE_COLUMNS = [
    'Study_Hours', 'Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks',
    'Physical_Activity_Hours', 'Sleep_Hours_Per_Night', 'Stress_Level',
    'Gender', 'Academic_Level', 'Most_Used_Platform', 'Purpose_Of_Use', 'Grouped_country'
]

# Real benchmark metrics from the trained Random Forest model
MODEL_METRICS = {
    "model_name": "Random Forest Regressor (Pipeline)",
    "n_estimators": 100,
    "random_state": 42,
    "test_r2": 0.8780,
    "train_r2": 0.9809,
    "test_mae": 0.3465,
    "test_rmse": 0.4629,
    "target_variable": "Mental_Health_Score (Scale: 1.0 - 10.0)"
}


def load_model(path: Path = MODEL_PATH):
    """Load the trained scikit-learn Pipeline from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found at {path}")
    return joblib.load(path)


def load_cleaned_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load and apply the project's standard preprocessing to the dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df['Physical_Activity_Hours'] = df['Physical_Activity_Hours'].clip(lower=0)
    
    top_c = df['Country'].value_counts().index[:10].tolist()
    df['Grouped_country'] = df['Country'].apply(lambda x: x if x in top_c else 'Other')
    return df


def preprocess_single_input(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Validate, sanitize, and format single user input into the exact DataFrame
    structure expected by the trained pipeline.
    """
    country = str(data.get('Country', 'Other')).strip()
    grouped_country = country if country in TOP_COUNTRIES else 'Other'
    
    row = {
        'Study_Hours': float(data.get('Study_Hours', 3.0)),
        'Age': int(data.get('Age', 21)),
        'Avg_Daily_Usage_Hours': float(data.get('Avg_Daily_Usage_Hours', 4.0)),
        'Daily_Unlocks': int(data.get('Daily_Unlocks', 75)),
        'Physical_Activity_Hours': max(0.0, float(data.get('Physical_Activity_Hours', 1.0))),
        'Sleep_Hours_Per_Night': float(data.get('Sleep_Hours_Per_Night', 7.0)),
        'Stress_Level': str(data.get('Stress_Level', 'Medium')),
        'Gender': str(data.get('Gender', 'Female')),
        'Academic_Level': str(data.get('Academic_Level', 'Undergraduate')),
        'Most_Used_Platform': str(data.get('Most_Used_Platform', 'Instagram')),
        'Purpose_Of_Use': str(data.get('Purpose_Of_Use', 'Entertainment')),
        'Grouped_country': grouped_country
    }
    
    df_input = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return df_input


def categorize_score(score: float) -> Dict[str, Any]:
    """
    Interpret the mental health score into risk categories, colors, and recommendations.
    Based on dataset distribution (mean=6.23, min=3.6, max=9.4):
    Higher score represents better mental well-being.
    """
    score = round(float(score), 2)
    if score >= 7.5:
        category = "Optimal Well-being"
        badge_color = "#10B981"  # Emerald
        severity = "success"
        summary = "Strong psychological resilience with healthy digital balance."
    elif score >= 6.2:
        category = "Good Well-being"
        badge_color = "#3B82F6"  # Blue
        severity = "info"
        summary = "Healthy baseline well-being with minor areas for lifestyle refinement."
    elif score >= 5.0:
        category = "Moderate Risk / Strain"
        badge_color = "#F59E0B"  # Amber
        severity = "warning"
        summary = "Signs of digital overload or academic stress are impacting well-being."
    else:
        category = "High Vulnerability"
        badge_color = "#EF4444"  # Red
        severity = "danger"
        summary = "Noticeable distress indicators detected. Proactive balance and support recommended."
        
    return {
        "score": score,
        "category": category,
        "badge_color": badge_color,
        "severity": severity,
        "summary": summary
    }


def predict_single(model, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute end-to-end inference for a single student."""
    df_input = preprocess_single_input(input_data)
    pred_val = float(model.predict(df_input)[0])
    
    # Bound logically within plausible score bounds
    pred_val = max(1.0, min(10.0, pred_val))
    result = categorize_score(pred_val)
    
    # Generate contextual risk & protective factor insights
    insights = []
    daily_usage = float(input_data.get('Avg_Daily_Usage_Hours', 4.0))
    sleep_hours = float(input_data.get('Sleep_Hours_Per_Night', 7.0))
    unlocks = int(input_data.get('Daily_Unlocks', 75))
    exercise = float(input_data.get('Physical_Activity_Hours', 1.0))
    stress = str(input_data.get('Stress_Level', 'Medium'))

    if daily_usage > 5.5:
        insights.append({
            "type": "negative",
            "title": "Elevated Screen Time",
            "desc": f"{daily_usage:.1f} hrs/day on social media is higher than average (4.3 hrs), strongly linked with digital fatigue."
        })
    elif daily_usage <= 2.5:
        insights.append({
            "type": "positive",
            "title": "Disciplined Social Media Usage",
            "desc": f"{daily_usage:.1f} hrs/day protects focus and sleep quality."
        })

    if sleep_hours < 6.0:
        insights.append({
            "type": "negative",
            "title": "Insufficient Sleep Window",
            "desc": f"{sleep_hours:.1f} hrs sleep per night diminishes cognitive recovery and mood stability."
        })
    elif sleep_hours >= 7.5:
        insights.append({
            "type": "positive",
            "title": "Restorative Sleep Pattern",
            "desc": f"{sleep_hours:.1f} hrs sleep per night provides strong resilience against daily stressors."
        })

    if unlocks > 100:
        insights.append({
            "type": "negative",
            "title": "High Device Interruption Rate",
            "desc": f"{unlocks} daily unlocks indicates frequent task fragmentation."
        })

    if exercise >= 2.0:
        insights.append({
            "type": "positive",
            "title": "Active Physical Exercise",
            "desc": f"{exercise:.1f} hrs of daily physical activity supports dopamine regulation and emotional release."
        })

    if stress in ['High', 'Very High']:
        insights.append({
            "type": "negative",
            "title": "Elevated Stress Baseline",
            "desc": f"Reported {stress.lower()} academic and personal stress compounding mental load."
        })

    result["insights"] = insights
    result["input_features"] = df_input.to_dict(orient="records")[0]
    return result


def batch_predict(model, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Process batch CSV predictions with schema validation."""
    required = [
        'Study_Hours', 'Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks',
        'Physical_Activity_Hours', 'Sleep_Hours_Per_Night', 'Stress_Level',
        'Gender', 'Academic_Level', 'Most_Used_Platform', 'Purpose_Of_Use'
    ]
    
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing mandatory columns: {', '.join(missing)}")
    
    df_proc = df_raw.copy()
    df_proc['Physical_Activity_Hours'] = df_proc['Physical_Activity_Hours'].clip(lower=0)
    
    # Handle country column if provided, otherwise default to Other
    if 'Country' in df_proc.columns:
        df_proc['Grouped_country'] = df_proc['Country'].apply(
            lambda x: str(x) if str(x) in TOP_COUNTRIES else 'Other'
        )
    elif 'Grouped_country' in df_proc.columns:
        df_proc['Grouped_country'] = df_proc['Grouped_country'].apply(
            lambda x: str(x) if str(x) in TOP_COUNTRIES else 'Other'
        )
    else:
        df_proc['Grouped_country'] = 'Other'
        
    X_input = df_proc[FEATURE_COLUMNS].copy()
    preds = model.predict(X_input)
    preds = np.clip(preds, 1.0, 10.0)
    
    df_raw_res = df_raw.copy()
    df_raw_res['Predicted_Score'] = np.round(preds, 2)
    df_raw_res['Mental_Health_Category'] = [categorize_score(p)['category'] for p in preds]
    
    summary = {
        "total_records": len(df_raw_res),
        "mean_score": round(float(np.mean(preds)), 2),
        "min_score": round(float(np.min(preds)), 2),
        "max_score": round(float(np.max(preds)), 2),
        "category_counts": df_raw_res['Mental_Health_Category'].value_counts().to_dict()
    }
    
    return df_raw_res, summary


def get_feature_importances(model) -> pd.DataFrame:
    """Extract and format feature importances from the trained Pipeline."""
    prep = model.named_steps['preprocessor']
    rf = model.named_steps['random forest']

    numeric_cols = [
        'Study_Hours', 'Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks',
        'Physical_Activity_Hours', 'Sleep_Hours_Per_Night'
    ]
    ordinal_cols = ['Stress_Level']
    
    ohe = prep.named_transformers_['Normal'].named_steps['encode']
    cat_source_cols = ['Gender', 'Academic_Level', 'Most_Used_Platform', 'Purpose_Of_Use', 'Grouped_country']
    ohe_cols = ohe.get_feature_names_out(cat_source_cols).tolist()
    
    all_feature_names = numeric_cols + ordinal_cols + ohe_cols
    importances = rf.feature_importances_
    
    df_imp = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    df_imp['Percentage'] = (df_imp['Importance'] * 100).round(2)
    return df_imp

