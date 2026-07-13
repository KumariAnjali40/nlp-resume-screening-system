# src/preprocessing.py
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from src.utils import logger

# Lazy load NLTK resources
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)

class DataPreprocessor:
    """Validates, cleans, and standardizes structural text/numeric inputs."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        """Tokenizes, cleans, and normalizes prose fields."""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(f"[^a-zA-Z0-9\s,\.\-]", "", text)
        words = text.split()
        cleaned_words = [w for w in words if w not in self.stop_words]
        return " ".join(cleaned_words)

    def clean_skills(self, skills_str: str) -> list:
        """Transforms inconsistent skill fields into standardized arrays."""
        if not isinstance(skills_str, str) or pd.isna(skills_str):
            return []
        # Split by commas or semi-colons, strip whitespace
        skills = re.split(r'[;,]', skills_str.lower())
        return [s.strip() for s in skills if s.strip()]

    def fit_transform(self, resume_df: pd.DataFrame, jd_df: pd.DataFrame) -> pd.DataFrame:
        """Cleans anomalies, drops duplicates, and normalizes features across sources."""
        logger.info("Starting structural data cleaning and normalization pipeline...")
        
        # Deduplicate
        resume_df = resume_df.drop_duplicates(subset=['candidate_id']).copy()
        
        # Validate Experience values
        resume_df['years_experience'] = pd.to_numeric(resume_df['years_experience'], errors='coerce').fillna(0)
        resume_df['years_experience'] = resume_df['years_experience'].apply(lambda x: max(0, min(x, 50)))
        
        # Clean expected salary
        resume_df['expected_salary'] = pd.to_numeric(resume_df['expected_salary'], errors='coerce').fillna(0)

        # Cross-join or map against the primary target Job ID (assuming evaluating for a single JD at a time)
        target_jd = jd_df.iloc[0]
        
        # Standardize arrays & text
        resume_df['cleaned_resume_text'] = resume_df['resume_text'].apply(self.clean_text)
        resume_df['parsed_skills'] = resume_df['skills'].apply(self.clean_skills)
        
        # Broadcast JD specifications to resume rows for parallel vector processing
        resume_df['jd_required_skills'] = Skinner = [self.clean_skills(target_jd['required_skills'])] * len(resume_df)
        resume_df['jd_min_experience'] = target_jd['minimum_experience']
        resume_df['jd_education_requirement'] = str(target_jd['education_requirement']).lower()
        resume_df['jd_description_cleaned'] = self.clean_text(target_jd['job_description'])
        
        logger.info("Data preprocessing completed successfully.")
        return resume_df