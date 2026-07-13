# src/model_training.py
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from src.utils import logger

class ModelTrainer:
    """Optimizes, trains, and isolates best-performing classifiers."""
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(random_state=42)
        self.xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        self.best_model = None
        self.best_model_name = ""

    def train_and_tune(self, X, y):
        """Executes Parallel Hyperparameter Grid Searches via stratified datasets."""
        logger.info("Initiating model hyperparameter tuning pipelines...")
        
        # Setup tuning grids
        rf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [8, 15, None],
            'min_samples_split': [2, 5]
        }
        
        xgb_param_grid = {
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 150]
        }
        
        # Optimize Random Forest
        logger.info("Optimizing Random Forest parameters via GridSearchCV...")
        rf_grid = GridSearchCV(self.rf_model, rf_param_grid, cv=3, scoring='f1', n_jobs=-1)
        rf_grid.fit(X, y)
        best_rf_score = rf_grid.best_score_
        logger.info(f"Random Forest Top Cross-Validated F1-Score: {best_rf_score:.4f}")
        
        # Optimize XGBoost
        logger.info("Optimizing XGBoost parameters via GridSearchCV...")
        xgb_grid = GridSearchCV(self.xgb_model, xgb_param_grid, cv=3, scoring='f1', n_jobs=-1)
        xgb_grid.fit(X, y)
        best_xgb_score = xgb_grid.best_score_
        logger.info(f"XGBoost Top Cross-Validated F1-Score: {best_xgb_score:.4f}")
        
        # Model Selection Evaluation
        if best_xgb_score >= best_rf_score:
            self.best_model = xgb_grid.best_estimator_
            self.best_model_name = "XGBoost"
        else:
            self.best_model = rf_grid.best_estimator_
            self.best_model_name = "RandomForest"
            
        logger.info(f"Selected Champion Model Architecture: {self.best_model_name}")
        return self.best_model