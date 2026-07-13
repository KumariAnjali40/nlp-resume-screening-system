# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import logger

class FeatureEngineer:
    """Generates complex hybrid statistical/semantic features from candidates."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')

    def calculate_skill_match(self, row) -> float:
        """Calculates percentage of required skills satisfied."""
        req = set(row['jd_required_skills'])
        if not req:
            return 1.0
        cand = set(row['parsed_skills'])
        return len(req.intersection(cand)) / len(req)

    def calculate_education_match(self, row) -> int:
        """Binary heuristic validation of baseline education levels."""
        cand_edu = str(row['education']).lower()
        req_edu = row['jd_education_requirement']
        if req_edu in cand_edu:
            return 1
        # Basic hierarchical catch
        if "phd" in cand_edu or "doctorate" in cand_edu:
            return 1
        if "master" in cand_edu and "bachelor" in req_edu:
            return 1
        return 0

    def transform(self, df: pd.DataFrame, is_training: bool = True) -> tuple:
        """Generates matching vectors, interaction terms, and text feature matrices."""
        logger.info("Extracting engineered features and calculating cosine spaces...")
        
        # 1. Base Domain Matching Features
        df['skill_match_pct'] = df.apply(self.calculate_skill_match, axis=1)
        df['exp_gap'] = df['years_experience'] - df['jd_min_experience']
        df['education_match'] = df.apply(self.calculate_education_match, axis=1)
        df['resume_length'] = df['resume_text'].fillna("").apply(len)
        
        # Count configurations
        df['num_projects'] = pd.to_numeric(df['projects'], errors='coerce').fillna(0)
        
        # 2. Text Cosine Vectorization
        # Combine candidate texts and calculate similarity directly matching rows
        corpus = df['cleaned_resume_text'].tolist()
        if is_training:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
        else:
            tfidf_matrix = self.vectorizer.transform(corpus)
            
        jd_tfidf = self.vectorizer.transform(df['jd_description_cleaned'].head(1))
        
        # Row-by-row matrix cosine vectorization
        cos_sims = cosine_similarity(tfidf_matrix, jd_tfidf).flatten()
        df['cosine_similarity'] = cos_sims
        
        # 3. Composite Algorithmic Score Component (Synthesized baseline metric)
        df['overall_candidate_score'] = (
            (df['skill_match_pct'] * 0.4) + 
            (df['education_match'] * 0.2) + 
            (np.clip(df['cosine_similarity'], 0, 1) * 0.4)
        )
        
        # Select explicit structural columns for ML training
        feature_cols = [
            'years_experience', 'expected_salary', 'skill_match_pct', 
            'exp_gap', 'education_match', 'resume_length', 
            'num_projects', 'cosine_similarity', 'overall_candidate_score'
        ]
        
        X_dense = df[feature_cols].copy()
        
        # Append numerical TF-IDF properties for high dimensionality scaling
        X_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])], index=df.index)
        X_final = pd.concat([X_dense, X_tfidf], axis=1)
        
        logger.info(f"Feature Space Matrix completed: Shape {X_final.shape}")
        return X_final, df