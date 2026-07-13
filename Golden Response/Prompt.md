# Project Prompt: Intelligent Resume Screening & Candidate Ranking System

## Context

You are an experienced Python Machine Learning Engineer working for a Recruitment Technology company that helps HR teams automate the hiring process. The organization receives thousands of resumes for every job opening and needs an intelligent system to efficiently screen, evaluate, rank, and recommend the most suitable candidates.

Your responsibility is to design and implement a scalable, production-ready Resume Screening System that combines machine learning, NLP, feature engineering, and explainable AI to automate the first stage of candidate selection.

---

# Objective

Develop a complete, modular, production-quality Python application that:

- Processes resume and job description datasets.
- Cleans and validates candidate information.
- Extracts meaningful structured and textual features.
- Trains multiple machine learning models.
- Selects the best-performing model.
- Predicts candidate suitability.
- Ranks candidates based on their probability of being shortlisted.
- Generates explainable hiring recommendations.
- Produces interactive visualizations and downloadable reports.
- Follows software engineering best practices including modularity, documentation, scalability, and error handling.

---

# Input Data

## Resume Dataset (CSV)

Each candidate record contains:

- candidate_id
- candidate_name
- education
- years_experience
- skills
- certifications
- previous_job_title
- projects
- expected_salary
- location
- resume_text
- shortlisted (Target Variable)

---

## Job Description Dataset (CSV)

Each job description contains:

- job_id
- job_title
- required_skills
- preferred_skills
- minimum_experience
- education_requirement
- job_description

---

# Functional Requirements

The system must perform the following pipeline.

## 1. Data Ingestion

- Load resume dataset
- Load job description dataset
- Validate file existence
- Validate schema
- Validate required columns
- Handle corrupted CSV files
- Handle encoding issues

---

## 2. Data Cleaning

Perform comprehensive preprocessing including:

- Missing value handling
- Duplicate removal
- Invalid experience correction
- Text normalization
- Lowercasing
- Whitespace removal
- Skill normalization
- Education normalization
- Salary formatting
- Experience standardization

---

## 3. NLP Processing

Process resume text using NLP techniques including:

- Tokenization
- Stopword removal
- Lemmatization
- Text normalization
- Keyword extraction
- Resume cleaning

Use:

- NLTK
- spaCy

---

## 4. Feature Engineering

Generate meaningful predictive features including:

### Structured Features

- Skill Match Percentage
- Experience Gap
- Education Match Score
- Certification Score
- Resume Length
- Project Count
- Keyword Frequency
- Required Skill Coverage
- Preferred Skill Coverage

### Text Features

Generate TF-IDF vectors from:

- Resume Text
- Job Description

Calculate:

- Cosine Similarity

Generate:

- Overall Candidate Score

---

# Machine Learning Requirements

Implement multiple classification models including:

- Random Forest Classifier
- XGBoost Classifier (preferred)
- Gradient Boosting Classifier (optional)

Perform:

- Train/Test Split
- Cross Validation
- Hyperparameter Tuning
- GridSearchCV or RandomizedSearchCV

Automatically select the best-performing model.

Save the trained model using Joblib.

---

# Model Evaluation

Evaluate every model using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

Generate:

- Classification Report
- Feature Importance
- Model Comparison

Provide a brief explanation describing why the best model outperformed the others.

---

# Candidate Ranking

Predict the shortlisting probability for every candidate.

Generate rankings using:

- Suitability Score
- Shortlisting Probability
- Final Rank

Recommendation Categories:

- Strongly Recommended
- Recommended
- Consider
- Not Recommended

---

# Visualizations

Generate professional visualizations including:

- Skill Match Distribution
- Candidate Score Distribution
- Feature Importance
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix
- Top Ranked Candidates
- Experience Distribution
- Education Distribution

Save all visualizations.

---

# Reports

Generate reports in:

## CSV

Include:

- Candidate ID
- Candidate Name
- Suitability Score
- Probability
- Rank
- Recommendation

## JSON

Example

```json
{
  "candidate_id": 1045,
  "candidate_name": "John Doe",
  "ranking": 3,
  "probability": 0.94,
  "recommendation": "Strongly Recommended"
}
```

Generate:

- Evaluation Report
- Ranking Report
- Model Metrics Report

---

# Explainability

Provide transparent predictions.

Include:

- Feature Importance
- SHAP Explainability (optional)
- Top features affecting candidate selection

---

# Error Handling

Implement robust exception handling for:

- Missing datasets
- Invalid CSV files
- Missing columns
- Incorrect datatypes
- Empty resumes
- Runtime failures
- Invalid model files
- Prediction failures

Use structured logging throughout the application.

---

# Project Architecture

Follow the modular project structure below.

```
project/
│
├── data/
├── models/
├── reports/
├── visualizations/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── report_generator.py
│   └── utils.py
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

# Documentation Requirements

Provide:

- Comprehensive Docstrings
- Inline Comments
- Logging
- README Documentation
- Installation Guide
- Usage Guide
- Deployment Instructions

---

# Performance Requirements

The application should efficiently process:

- 100,000+ resumes
- Large job description datasets

Optimize:

- Memory usage
- CPU utilization
- Feature extraction
- Model training
- Prediction latency

Ensure reproducibility using fixed random seeds.

---

# Bonus Features

Implement one or more advanced capabilities.

- Resume PDF Parsing
- Sentence Transformer Embeddings
- Semantic Resume Search
- Candidate Clustering
- SHAP Explainability
- Streamlit Dashboard
- FastAPI REST API
- Resume Recommendation Engine
- Automated Interview Recommendation
- Batch Candidate Processing

---

# Technology Stack

## Programming Language

- Python

## Data Processing

- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- XGBoost

## NLP

- NLTK
- spaCy
- Sentence Transformers (Optional)

## Visualization

- Matplotlib
- Seaborn

## Explainability

- SHAP

## Deployment

- Streamlit
- FastAPI (Optional)

---

# Expected Deliverables

The final solution must include:

- Fully modular Python source code
- Production-ready architecture
- Trained ML models
- Saved model artifacts
- Evaluation reports
- Candidate ranking reports
- CSV reports
- JSON reports
- Data visualizations
- Interactive Streamlit dashboard
- README documentation
- Requirements file
- Sample datasets
- Unit tests

---

# Success Criteria

The solution will be considered complete only if it:

- Successfully processes resume and job description datasets.
- Generates meaningful predictive features.
- Produces explainable candidate rankings.
- Selects the best-performing ML model.
- Achieves strong evaluation metrics.
- Generates downloadable reports.
- Provides professional visualizations.
- Uses modular architecture.
- Includes complete documentation.
- Is deployment-ready for Streamlit or Render.