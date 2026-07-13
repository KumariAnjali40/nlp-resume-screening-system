# src/evaluation.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
from src.utils import logger

class Evaluator:
    """Validates real-world performance metrics across production models."""
    
    @staticmethod
    def evaluate(model, X_test, y_test) -> dict:
        """Computes confusion array boundaries and metric configurations."""
        logger.info("Calculating system classification performance metrics...")
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1_score": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist()
        }
        
        logger.info(f"Evaluation Complete | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        return metrics

    @staticmethod
    def get_feature_importance(model, feature_names) -> pd.DataFrame:
        """Extracts relative weights assigned to features by the tree architecture."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values(by='importance', ascending=False)
            return df
        return pd.DataFrame()