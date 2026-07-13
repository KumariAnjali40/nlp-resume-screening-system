# src/visualization.py
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to ensure stable multi-threaded execution
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from src.utils import logger

class Visualizer:
    """Generates and exports diagnostic plots to disk."""
    
    @staticmethod
    def plot_distributions(df: pd.DataFrame, output_dir: str):
        """Plots demographic skill frequencies and structural match distribution."""
        plt.figure(figsize=(10, 4))
        sns.histplot(df['skill_match_pct'], bins=15, kde=True, color='skyblue')
        plt.title('Skill Match Percentage Distribution')
        plt.xlabel('Match %')
        plt.savefig(f"{output_dir}/skill_match_distribution.png", bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 4))
        sns.histplot(df['overall_candidate_score'], bins=15, kde=True, color='salmon')
        plt.title('Overall System Suitability Score Histogram')
        plt.xlabel('Candidate Composite Score')
        plt.savefig(f"{output_dir}/candidate_score_histogram.png", bbox_inches='tight')
        plt.close()
        logger.info("Saved data distribution charts.")

    @staticmethod
    def plot_model_diagnostics(model, X_test, y_test, importance_df: pd.DataFrame, output_dir: str):
        """Generates performance evaluation curves and feature importance plots."""
        # Top 10 Core Features Plot
        if not importance_df.empty:
            plt.figure(figsize=(10, 5))
            sns.barplot(x='importance', y='feature', data=importance_df.head(10), palette='viridis')
            plt.title('Top 10 Feature Importances')
            plt.savefig(f"{output_dir}/feature_importance.png", bbox_inches='tight')
            plt.close()

        # Curve Metrics Calculation
        probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        prec, rec, _ = precision_recall_curve(y_test, probs)

        # Plot ROC Curve
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label='Model Performance')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(f"{output_dir}/roc_curve.png", bbox_inches='tight')
        plt.close()

        # Plot Precision-Recall Curve
        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec, color='purple', label='PR Curve')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig(f"{output_dir}/precision_recall_curve.png", bbox_inches='tight')
        plt.close()
        logger.info("Saved model performance graphs successfully.")