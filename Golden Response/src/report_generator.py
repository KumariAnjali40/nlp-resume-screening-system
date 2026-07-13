# src/report_generator.py
import pandas as pd
import json
from src.utils import logger

class ReportGenerator:
    """Ranks candidates and exports structured hiring recommendations."""
    
    @staticmethod
    def generate_ranking_reports(processed_df: pd.DataFrame, probabilities: np.ndarray, output_dir: str):
        """Generates sorted shortlists and exports structured CSV/JSON summaries."""
        logger.info("Sorting applicant pipelines into tiered priority recommendations...")
        
        # Add tracking columns
        processed_df['shortlist_probability'] = probabilities
        # Normalize score tracking standard
        processed_df['suitability_score'] = processed_df['overall_candidate_score'].round(4)
        
        # Sort by classification probability and score tie-breakers
        ranked_df = processed_df.sort_values(by=['shortlist_probability', 'suitability_score'], ascending=False).copy()
        ranked_df['rank'] = range(1, len(ranked_df) + 1)
        
        # Map qualitative status based on confidence levels
        def assign_recommendation(prob):
            if prob >= 0.85: return "Strongly Recommended"
            if prob >= 0.65: return "Recommended"
            if prob >= 0.40: return "Borderline Review"
            return "Not Shortlisted"
            
        ranked_df['recommendation'] = ranked_df['shortlist_probability'].apply(assign_recommendation)
        
        # Filter reporting slice
        reporting_cols = ['rank', 'candidate_id', 'candidate_name', 'suitability_score', 'shortlist_probability', 'recommendation']
        output_report = ranked_df[reporting_cols]
        
        # 1. Export CSV
        csv_path = f"{output_dir}/ranked_candidates.csv"
        output_report.to_csv(csv_path, index=False)
        
        # 2. Export standardized JSON structure
        json_path = f"{output_dir}/ranked_candidates.json"
        json_records = []
        for _, row in output_report.iterrows():
            json_records.append({
                "candidate_id": int(row['candidate_id']),
                "candidate_name": str(row['candidate_name']),
                "ranking": int(row['rank']),
                "probability": float(round(row['shortlist_probability'], 2)),
                "recommendation": str(row['recommendation'])
            })
            
        with open(json_path, 'w') as f:
            json.dump(json_records, f, indent=2)
            
        logger.info(f"Reports successfully written to: {output_dir}")
        return csv_path, json_path