# 🧠 Intelligent Resume Screening System

An AI-powered Resume Screening System built using **Python, Streamlit, Scikit-learn, and XGBoost** that automatically ranks candidates based on how well their resumes match a given Job Description.

---

## 🚀 Live Demo

🔗 https://llmetharaai.onrender.com/

---

# 📌 Features

- 📄 Upload multiple resumes
- 💼 Upload Job Description
- 🧹 Automatic resume preprocessing
- 🔍 TF-IDF feature extraction
- 🤖 XGBoost-based candidate ranking
- 📊 Interactive dashboard
- 📈 Candidate score visualization
- 📑 Download ranked candidate report
- 📉 Model evaluation metrics
- 📂 CSV & JSON report generation

---

# 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Frontend | Streamlit |
| Backend | Python |
| Machine Learning | Scikit-learn, XGBoost |
| NLP | TF-IDF Vectorizer |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Serialization | Joblib |
| Deployment | Render |
| Version Control | Git, GitHub |

---

# 📂 Project Structure

```text
GeminiEtharaAiTraning/
│
├── app.py                 # Streamlit Frontend
├── main.py                # Model Training Pipeline
├── requirements.txt
│
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
├── data/
│
├── models/
│   ├── champion_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── reports/
│
└── visualizations/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

Go inside the project

```bash
cd YOUR_REPO
```

Create virtual environment

```bash
python -m venv .venv
```

Activate virtual environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Train the Model

Run

```bash
python main.py
```

This generates

```
models/
    champion_model.pkl
    tfidf_vectorizer.pkl
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

Open

```
http://localhost:8501
```

---

# ☁️ Deployment

This application is deployed using **Render**.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

# 📊 Model Workflow

```text
Resume Upload
       │
       ▼
Data Cleaning
       │
       ▼
Feature Engineering (TF-IDF)
       │
       ▼
XGBoost Model
       │
       ▼
Candidate Ranking
       │
       ▼
Dashboard & Reports
```

---

# 📸 Screenshots

## Dashboard

<img width="1497" height="727" alt="image" src="https://github.com/user-attachments/assets/bfec639b-564d-43c0-8f12-e5bc488dabaa" />

---

## Candidate Ranking

<img width="1497" height="615" alt="image" src="https://github.com/user-attachments/assets/7076656b-207f-4d2f-8b73-9f0ac22f4513" />


---

## Model Performance

<img width="1486" height="722" alt="image" src="https://github.com/user-attachments/assets/97dee066-49bd-4557-b8e5-8e7171ae5d7e" />


---

# 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

# 📄 Reports Generated

- Ranked Candidates CSV
- Ranked Candidates JSON
- Evaluation Metrics
- Feature Importance Graphs

---

# 🔮 Future Improvements

- Resume parsing using LLMs
- BERT embeddings
- Semantic Search
- Resume PDF upload
- Authentication
- Multi Job Description support
- Cloud Storage Integration

---

# 👨‍💻 Author

**Kumari Anjali**

---
