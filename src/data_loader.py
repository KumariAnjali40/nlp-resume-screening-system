# src/data_loader.py
import pandas as pd
from src.utils import logger

class DataLoader:
    """Handles memory-efficient data ingestion for large candidate pools."""
    
    @staticmethod
    def load_resumes(file_path: str, chunk_size: int = 20000) -> pd.DataFrame:
        """Loads resumes using chunking to scale to 100,000+ rows efficiently."""
        logger.info(f"Loading resume dataset from: {file_path}")
        try:
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunks.append(chunk)
            df = pd.concat(chunks, axis=0, ignore_index=True)
            logger.info(f"Successfully loaded {len(df)} resume records.")
            return df
        except FileNotFoundError:
            logger.error(f"Resume dataset file not found at {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading resume CSV: {str(e)}")
            raise

    @staticmethod
    def load_job_description(file_path: str) -> pd.DataFrame:
        """Loads the target job description metadata."""
        logger.info(f"Loading job descriptions from: {file_path}")
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} job descriptions.")
            return df
        except Exception as e:
            logger.error(f"Error loading job descriptions: {str(e)}")
            raise