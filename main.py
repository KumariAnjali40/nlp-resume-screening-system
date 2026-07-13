# # main.py
# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from src.utils import set_seed, ensure_directories, logger
# from src.data_loader import DataLoader
# from src.preprocessing import DataPreprocessor
# from src.feature_engineering import FeatureEngineer
# from src.model_training import ModelTrainer
# from src.evaluation import Evaluator
# from src.visualization import Visualizer
# from src.report_generator import ReportGenerator
# import joblib

# def run_pipeline(resume_csv: str, jd_csv: str):
#     """Executes the complete end-to-end recruitment screening system pipeline."""
#     set_seed(42)
#     ensure_directories(['data', 'models', 'reports', 'visualizations'])
    
#     # 1. Ingestion
#     try:
#         raw_resumes = DataLoader.load_resumes(resume_csv)
#         raw_jd = DataLoader.load_job_description(jd_csv)
#     except Exception as e:
#         logger.critical(f"Pipeline initialization failed due to missing inputs: {str(e)}")
#         return

#     # 2. Data Cleansing
#     preprocessor = DataPreprocessor()
#     clean_df = preprocessor.fit_transform(raw_resumes, raw_jd)
    
#     if clean_df.empty:
#         logger.error("Dataset empty after pipeline filtering routines. Halting runtime.")
#         return

#     # 3. Extraction & Alignment
#     fe = FeatureEngineer()
#     X, processed_df = fe.transform(clean_df, is_training=True)
#     y = processed_df['shortlisted'].fillna(0).astype(int)

#     # Split datasets for validation testing
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#     # 4. Training Engine Selection
#     trainer = ModelTrainer()
#     champion_model = trainer.train_and_tune(X_train, y_train)

#     # 5. Diagnostic Evaluation
#     metrics = Evaluator.evaluate(champion_model, X_test, y_test)
#     importance_df = Evaluator.get_feature_importance(champion_model, X.columns.tolist())
    
#     # Write metric log file to disk
#     with open('reports/evaluation_metrics.json', 'w') as f:
#         import json
#         json.dump(metrics, f, indent=4)

#     # 6. Build Graphical Dashboards
#     Visualizer.plot_distributions(processed_df, 'visualizations')
#     Visualizer.plot_model_diagnostics(champion_model, X_test, y_test, importance_df, 'visualizations')

#     # 7. Generate Production Candidate Rankings
#     full_pipeline_probabilities = champion_model.predict_proba(X)[:, 1]
#     csv_rep, json_rep = ReportGenerator.generate_ranking_reports(processed_df, full_pipeline_probabilities, 'reports')
    
#     logger.info("System Processing Completed Successfully.")
#     logger.info(f"Review Shortlist Queue: {csv_rep}")

# # if __name__ == "__main__":
# #     # Point paths to relative configurations
# #     run_pipeline('data/sample_resumes.csv', 'data/sample_job_descriptions.csv')


# if __name__ == "__main__":
#     # 1. Point paths to relative configurations
#     # 2. Modify your pipeline function to RETURN the trained model and feature engineering assets
#     champion_model, fe = run_pipeline(
#         'data/sample_resumes.csv', 
#         'data/sample_job_descriptions.csv'
#     )
    
#     # 3. Save the production assets (properly indented)
#     logger.info("Saving trained champion model assets for production inference use...")
#     joblib.dump(champion_model, 'models/champion_model.pkl')
#     joblib.dump(fe.vectorizer, 'models/tfidf_vectorizer.pkl')


import json
import joblib
from sklearn.model_selection import train_test_split

from src.utils import set_seed, ensure_directories, logger
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import Evaluator
from src.visualization import Visualizer
from src.report_generator import ReportGenerator


def run_pipeline(resume_csv: str, jd_csv: str):
    """
    Executes the end-to-end resume screening pipeline.

    Returns:
        champion_model : trained ML model
        fe             : FeatureEngineer object
    """

    set_seed(42)
    ensure_directories([
        "data",
        "models",
        "reports",
        "visualizations"
    ])

    # -------------------------------------------------
    # Step 1 : Load Data
    # -------------------------------------------------
    try:
        raw_resumes = DataLoader.load_resumes(resume_csv)
        raw_jd = DataLoader.load_job_description(jd_csv)

    except Exception as e:
        logger.critical(f"Unable to load input files.\n{e}")
        return None, None

    # -------------------------------------------------
    # Step 2 : Preprocessing
    # -------------------------------------------------
    preprocessor = DataPreprocessor()
    clean_df = preprocessor.fit_transform(raw_resumes, raw_jd)

    if clean_df.empty:
        logger.error("No data left after preprocessing.")
        return None, None

    # -------------------------------------------------
    # Step 3 : Feature Engineering
    # -------------------------------------------------
    fe = FeatureEngineer()

    X, processed_df = fe.transform(
        clean_df,
        is_training=True
    )

    y = processed_df["shortlisted"].fillna(0).astype(int)

    # -------------------------------------------------
    # Step 4 : Train-Test Split
    # -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # -------------------------------------------------
    # Step 5 : Train Model
    # -------------------------------------------------
    trainer = ModelTrainer()

    champion_model = trainer.train_and_tune(
        X_train,
        y_train
    )

    # -------------------------------------------------
    # Step 6 : Evaluate
    # -------------------------------------------------
    metrics = Evaluator.evaluate(
        champion_model,
        X_test,
        y_test
    )

    importance_df = Evaluator.get_feature_importance(
        champion_model,
        X.columns.tolist()
    )

    # -------------------------------------------------
    # Step 7 : Save Metrics
    # -------------------------------------------------
    with open("reports/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # -------------------------------------------------
    # Step 8 : Visualizations
    # -------------------------------------------------
    Visualizer.plot_distributions(
        processed_df,
        "visualizations"
    )

    Visualizer.plot_model_diagnostics(
        champion_model,
        X_test,
        y_test,
        importance_df,
        "visualizations"
    )

    # -------------------------------------------------
    # Step 9 : Ranking Report
    # -------------------------------------------------
    probabilities = champion_model.predict_proba(X)[:, 1]

    csv_report, json_report = ReportGenerator.generate_ranking_reports(
        processed_df,
        probabilities,
        "reports"
    )

    logger.info("Pipeline completed successfully.")
    logger.info(f"CSV Report : {csv_report}")
    logger.info(f"JSON Report: {json_report}")

    return champion_model, fe


if __name__ == "__main__":

    champion_model, fe = run_pipeline(
        "data/sample_resumes.csv",
        "data/sample_job_descriptions.csv"
    )

    if champion_model is None:
        logger.error("Pipeline failed. Model not saved.")

    else:
        logger.info("Saving production assets...")

        joblib.dump(
            champion_model,
            "models/champion_model.pkl"
        )

        if hasattr(fe, "vectorizer"):

            joblib.dump(
                fe.vectorizer,
                "models/tfidf_vectorizer.pkl"
            )

            logger.info("Vectorizer saved.")

        else:
            logger.warning(
                "FeatureEngineer has no 'vectorizer' attribute. Skipping vectorizer save."
            )

        logger.info("Production assets saved successfully.")