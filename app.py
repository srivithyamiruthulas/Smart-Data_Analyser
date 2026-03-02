import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def analyze_data(file):
    df = pd.read_csv(file.name)

    # keep numeric data only
    num_df = df.select_dtypes(include=np.number).dropna()

    summary = df.describe().to_string()
    missing = df.isnull().sum().to_string()

    # ---------- AUTO TARGET DETECTION ----------
    target_col = num_df.var().idxmax()

    X = num_df.drop(columns=[target_col])
    y = num_df[target_col]

    # ---------- FEATURE IMPORTANCE ----------
    model = RandomForestRegressor()
    model.fit(X, y)

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    top_features = importance.head(5).to_string()

    # ---------- PLOT ----------
    plt.figure()
    importance.head(5).plot(kind="bar")
    plt.title(f"Top Features Influencing {target_col}")
    plt.savefig("plot.png")

    # ---------- AI STYLE INSIGHT ----------
insight = f"""
===== AI DATA SCIENTIST REPORT =====

AI Selected Prediction Target:
➡ {target_col}

Top Influential Features:
{top_features}

Interpretation:
The AI trained a Random Forest model and identified
which variables most strongly influence the target.
These features should be prioritized for prediction tasks.
"""
    return summary, missing, "plot.png", insight


interface = gr.Interface(
    fn=analyze_data,
    inputs=gr.File(label="Upload CSV"),
    outputs=["text", "text", "image", "text"],
    title="Smart Data Analyzer — AI Data Scientist Mode"
)

interface.launch()