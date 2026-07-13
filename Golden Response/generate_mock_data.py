# generate_mock_data.py
import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)

# Generate Mock Job Profiles
jd_data = {
    'job_id': [101],
    'job_title': ['Senior Python Backend Engineer'],
    'required_skills': ['Python, SQL, Django, AWS, Git'],
    'preferred_skills': ['Docker, Kubernetes, FastAPI, Redis'],
    'minimum_experience': [5],
    'education_requirement': ['Bachelor'],
    'job_description': ['Looking for a senior Python software engineer with heavy database and systems optimization expertise using AWS and Django architecture patterns. Must write clean documentation and design robust automated microservices pipelines.']
}
pd.DataFrame(jd_data).to_csv('data/sample_job_descriptions.csv', index=False)

# Generate Mock Resumes
records = 150
mock_candidates = {
    'candidate_id': range(1000, 1000 + records),
    'candidate_name': [f"Candidate Alpha {i}" for i in range(records)],
    'education': np.random.choice(['BSc Computer Science', 'MSc Data Engineering', 'High School Diploma', 'BTech Electrical'], records, p=[0.4, 0.3, 0.1, 0.2]),
    'years_experience': np.random.choice([2, 3, 5, 7, 10], records, p=[0.2, 0.3, 0.2, 0.2, 0.1]),
    'skills': np.random.choice([
        'Python, SQL, Git, Django, AWS',
        'Java, Spring Boot, MySQL',
        'Python, HTML, CSS, JavaScript, Git',
        'Excel, Python, Tableau, SQL, AWS',
        'C++, Embedded Systems, Linux'
    ], records),
    'certifications': ['AWS Certified Developer' if i % 3 == 0 else 'None' for i in range(records)],
    'previous_job_title': np.random.choice(['Software Dev', 'Data Scientist', 'System Analyst'], records),
    'projects': np.random.choice([1, 2, 4, 5], records),
    'expected_salary': np.random.randint(70000, 140000, records),
    'location': np.random.choice(['New York', 'San Francisco', 'Remote'], records),
    'resume_text': [
        "Experienced software developer building backend apps using Python, structured databases using SQL, scaling applications inside cloud infrastructures like AWS, and version tracking via Git." 
        if i % 2 == 0 else "Customer service representative handling technical inquiries using basic automation toolsets." 
        for i in range(records)
    ]
}

df_resumes = pd.DataFrame(mock_candidates)
# Synthesize realistic labels based on clear conditions
df_resumes['shortlisted'] = ((df_resumes['years_experience'] >= 4) & (df_resumes['skills'].str.contains('Python'))).astype(int)
df_resumes.to_csv('data/sample_resumes.csv', index=False)
print("Mock production data generated within data/ paths.")