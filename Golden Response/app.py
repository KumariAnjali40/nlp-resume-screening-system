# # app.py
# import streamlit as st
# import pandas as pd
# import json
# import os

# # Set up page configurations
# st.set_page_config(
#     page_title="AI Candidate Screener & Ranker",
#     page_icon="💼",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Application paths based on the backend project structure
# CSV_REPORT_PATH = "reports/ranked_candidates.csv"
# METRICS_JSON_PATH = "reports/evaluation_metrics.json"
# VISUALS_DIR = "visualizations"

# st.title("💼 Intelligent Resume Screening & Candidate Ranking System")
# st.markdown("Automated initial candidate screening, suitability mapping, and model diagnostics pipeline.")
# st.write("---")

# # Verify that the backend pipeline has run successfully
# if not os.path.exists(CSV_REPORT_PATH):
#     st.error(f"❌ Could not find ranking data at `{CSV_REPORT_PATH}`. Please run `python main.py` first to generate pipeline outputs.")
#     st.stop()

# # Load Data Core
# @st.cache_data
# def load_hiring_data():
#     df = pd.read_csv(CSV_REPORT_PATH)
#     return df

# df_candidates = load_hiring_data()

# # ----------------- SIDEBAR CONTROLS -----------------
# st.sidebar.header("🔍 Recruiter Search & Filters")

# # Global Text Query Search
# search_query = st.sidebar.text_input("Search Candidate by Name or ID", "")

# # Tiered Classification Filter
# recommendation_types = df_candidates['recommendation'].unique().tolist()
# selected_tiers = st.sidebar.multiselect(
#     "Filter by Recommendation Level",
#     options=recommendation_types,
#     default=recommendation_types
# )

# # Probability Cut-off Slider
# min_prob = st.sidebar.slider(
#     "Minimum Match Confidence Probability",
#     min_value=0.0,
#     max_value=1.0,
#     value=0.0,
#     step=0.05
# )

# # Apply Filter Arrays
# filtered_df = df_candidates[
#     (df_candidates['recommendation'].isin(selected_tiers)) &
#     (df_candidates['shortlist_probability'] >= min_prob)
# ]

# if search_query:
#     filtered_df = filtered_df[
#         (filtered_df['candidate_name'].str.contains(search_query, case=False, na=False)) |
#         (filtered_df['candidate_id'].astype(str).str.contains(search_query))
#     ]

# # ----------------- MAIN METRICS SUMMARY -----------------
# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     st.metric("Total Applicants Evaluated", len(df_candidates))
# with col2:
#     st.metric("Filtered Candidates", len(filtered_df))
# with col3:
#     strong_count = len(df_candidates[df_candidates['recommendation'] == "Strongly Recommended"])
#     st.metric("Strong Prospects Available", strong_count)
# with col4:
#     # Attempt to load trained model cross-validated baseline statistics
#     if os.path.exists(METRICS_JSON_PATH):
#         with open(METRICS_JSON_PATH, 'r') as f:
#             metrics = json.load(f)
#         st.metric("Model Champion F1-Score", f"{metrics.get('f1_score', 0.0):.2%}")
#     else:
#         st.metric("Model Champion F1-Score", "N/A")

# st.write("---")

# # Create functional interface layout tabs
# tab_leaderboard, tab_diagnostics = st.tabs(["🏆 Candidate Leaderboard", "📊 Model Analytics & Explanations"])

# # ----------------- TAB 1: LEADERBOARD -----------------
# with tab_leaderboard:
#     st.subheader("Ranked Evaluation Pipeline")
#     st.markdown("Candidates sorted below by model selection probability and interaction matching scores.")
    
#     # Render customized clear table layout
#     st.dataframe(
#         filtered_df,
#         column_config={
#             "rank": st.column_config.NumberColumn("Rank", format="%d"),
#             "candidate_id": st.column_config.NumberColumn("Candidate ID", format="%d"),
#             "candidate_name": "Candidate Name",
#             "suitability_score": st.column_config.NumberColumn("Suitability Score", format="%.4f"),
#             "shortlist_probability": st.column_config.ProgressColumn(
#                 "Shortlist Probability",
#                 help="Prediction certainty percentage calculated by Champion Classifier",
#                 format="%.2f",
#                 min_value=0.0,
#                 max_value=1.0,
#             ),
#             "recommendation": "Recruiter Action Tier"
#         },
#         use_container_width=True,
#         hide_index=True
#     )
    
#     # Download actions
#     st.download_button(
#         label="📥 Export Current View Queue to CSV",
#         data=filtered_df.to_csv(index=False).encode('utf-8'),
#         file_name='filtered_hiring_shortlist.csv',
#         mime='text/csv'
#     )

# # ----------------- TAB 2: DIAGNOSTICS & EXPLANATIONS -----------------
# with tab_diagnostics:
#     st.subheader("Pipeline Calibration Diagnostics")
#     st.markdown("Inspect performance distributions and explainability curves mapped during pipeline validation.")
    
#     col_vis1, col_vis2 = st.columns(2)
    
#     with col_vis1:
#         img_score = f"{VISUALS_DIR}/candidate_score_histogram.png"
#         if os.path.exists(img_score):
#             st.image(img_score, caption="Candidate Population Score Layout", use_container_width=True)
            
#         img_roc = f"{VISUALS_DIR}/roc_curve.png"
#         if os.path.exists(img_roc):
#             st.image(img_roc, caption="Receiver Operating Characteristic Profile", use_container_width=True)

#     with col_vis2:
#         img_feat = f"{VISUALS_DIR}/feature_importance.png"
#         if os.path.exists(img_feat):
#             st.image(img_feat, caption="Top Model System Driving Criteria Weights", use_container_width=True)
            
#         img_pr = f"{VISUALS_DIR}/precision_recall_curve.png"
#         if os.path.exists(img_pr):
#             st.image(img_pr, caption="Precision vs. Recall Confidence Scaling Bound", use_container_width=True)


# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
from pypdf import PdfReader

# Import existing pipeline classes from your source directory
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer

# System-wide configuration settings
st.set_page_config(
    page_title="AI Candidate Screener & Ranker",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSV_REPORT_PATH = "reports/ranked_candidates.csv"
METRICS_JSON_PATH = "reports/evaluation_metrics.json"
MODEL_PATH = "models/champion_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
JD_PATH = "data/sample_job_descriptions.csv"
VISUALS_DIR = "visualizations"

st.title("💼 Intelligent Resume Screening & Candidate Ranking System")
st.markdown("Automated initial candidate screening, suitability mapping, and model diagnostics pipeline.")
st.write("---")

# Verify core data structures exist
if not os.path.exists(CSV_REPORT_PATH):
    st.error("❌ Could not find background ranking data. Please run `python main.py` first to generate pipeline assets.")
    st.stop()

# Helper logic to parse PDF strings safely
def extract_text_from_pdf(file_wrapper) -> str:
    try:
        reader = PdfReader(file_wrapper)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() or ""
        return extracted_text.strip()
    except Exception as e:
        st.error(f"Failed parsing PDF layout boundaries: {str(e)}")
        return ""

# Initialize UI Navigation Tabs
tab_upload, tab_leaderboard, tab_diagnostics = st.tabs([
    "📥 Upload & Screen Live", 
    "🏆 Batch Candidate Leaderboard", 
    "📊 Model Analytics & Explanations"
])

# ---------------------------------------------------------
# TAB 1: LIVE SINGLE RESUME UPLOAD AND SCORE INTERACTION
# ---------------------------------------------------------
with tab_upload:
    st.subheader("📝 Live Ad-Hoc Candidate Evaluation")
    st.markdown("Drop a brand-new candidate's PDF resume below to instantly cross-validate fit parameters, predict shortlisting probability, and check qualification matches in real-time.")
    
    # Prerequisite verification check for live model files
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(JD_PATH)):
        st.warning("⚠️ Live runtime models or job descriptions are missing from storage arrays. Run `python main.py` to compile saved model assets.")
    else:
        # File selector block
        uploaded_file = st.file_uploader("Upload Candidate Resume (PDF Format)", type=["pdf"])
        
        if uploaded_file is not None:
            with st.spinner("Extracting parameters and running inference networks..."):
                raw_resume_text = extract_text_from_pdf(uploaded_file)
                
                if not raw_resume_text:
                    st.error("The uploaded document appears empty or unreadable. Scanned image tracking is not supported.")
                else:
                    st.success("📄 Resume text extracted successfully!")
                    
                    # 1. Capture Recruiter Input details manually for fields not found inside a generic PDF text space
                    col_in1, col_in2, col_in3 = st.columns(3)
                    with col_in1:
                        cand_name = st.text_input("Candidate Name Reference", value="Walk-In Applicant")
                    with col_in2:
                        years_exp = st.number_input("Verified Years of Experience", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
                    with col_in3:
                        expected_sal = st.number_input("Expected Salary ($)", min_value=0, value=95000, step=5000)
                        
                    cand_skills = st.text_area("Extracted Skills Array (Comma separated)", value="Python, SQL, AWS, Git")
                    
                    # 2. Re-create candidate layout configuration dataframe mock block
                    single_candidate_df = pd.DataFrame([{
                        'candidate_id': 9999,
                        'candidate_name': cand_name,
                        'education': 'Bachelor of Computer Science', # Baseline heuristic string
                        'years_experience': years_exp,
                        'skills': cand_skills,
                        'certifications': 'None',
                        'previous_job_title': 'Software Engineer',
                        'projects': 3,
                        'expected_salary': expected_sal,
                        'location': 'Remote',
                        'resume_text': raw_resume_text,
                        'shortlisted': 0 # Target buffer variable
                    }])
                    
                    # 3. Load underlying tracking dependencies
                    jd_df = pd.read_csv(JD_PATH)
                    model = joblib.load(MODEL_PATH)
                    vectorizer = joblib.load(VECTORIZER_PATH)
                    
                    # 4. Transform data arrays using exact functional backend classes
                    preprocessor = DataPreprocessor()
                    processed_single = preprocessor.fit_transform(single_candidate_df, jd_df)
                    
                    fe = FeatureEngineer()
                    fe.vectorizer = vectorizer  # Inject pre-fitted vectorizer matrix state
                    X_single, final_single_df = fe.transform(processed_single, is_training=False)
                    
                    # 5. Model Inference Execution
                    prob = model.predict_proba(X_single)[0, 1]
                    suitability_score = final_single_df['overall_candidate_score'].iloc[0]
                    
                    # Compute categorical badge tier assignments
                    if prob >= 0.85: status, color = "Strongly Recommended", "🟩"
                    elif prob >= 0.65: status, color = "Recommended", "🟨"
                    elif prob >= 0.40: status, color = "Borderline Review", "🟧"
                    else: status, color = "Not Shortlisted", "🟥"
                    
                    # Display Results Layout Window UI elements
                    st.write("---")
                    st.subheader("📊 Candidate Evaluation Dashboard")
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Shortlist Confidence Match", f"{prob:.2%}")
                    with col_m2:
                        st.metric("Composite Suitability Rating Score", f"{suitability_score:.4f}")
                    with col_m3:
                        st.markdown(f"**Recruiter Assignment Action Status:**\n### {color} {status}")
                        
                    with st.expander("🔍 View Raw Parsed NLP Content Matrix Summary"):
                        st.json({
                            "Skill Match Percentage Metric": f"{final_single_df['skill_match_pct'].iloc[0]:.2%}",
                            "Text Cosine Similarity Space Mapped": f"{final_single_df['cosine_similarity'].iloc[0]:.4f}",
                            "Computed Experience Variance Gap": float(final_single_df['exp_gap'].iloc[0]),
                            "Education Requirement Baseline Satisfied": "Yes" if final_single_df['education_match'].iloc[0] == 1 else "No"
                        })

# ---------------------------------------------------------
# TAB 2: BATCH HISTORICAL DATABASE LEADERBOARD LEADS
# ---------------------------------------------------------
with tab_leaderboard:
    st.subheader("Ranked Processing Evaluation Pipeline")
    df_candidates = pd.read_csv(CSV_REPORT_PATH)
    
    # Sidebar Filtering Sync Controls Mirroring
    st.sidebar.header("🔍 Global Pipeline View Controls")
    search_query = st.sidebar.text_input("Find Candidate profile", "")
    
    filtered_df = df_candidates.copy()
    if search_query:
        filtered_df = filtered_df[
            (filtered_df['candidate_name'].str.contains(search_query, case=False, na=False)) |
            (filtered_df['candidate_id'].astype(str).str.contains(search_query))
        ]
        
    st.dataframe(
        filtered_df,
        column_config={
            "rank": st.column_config.NumberColumn("Pipeline Rank", format="%d"),
            "candidate_id": st.column_config.NumberColumn("ID Reference", format="%d"),
            "candidate_name": "Full Name",
            "suitability_score": st.column_config.NumberColumn("Engine Score Weight", format="%.4f"),
            "shortlist_probability": st.column_config.ProgressColumn(
                "Classification Confidence Metric",
                format="%.2f", min_value=0.0, max_value=1.0,
            ),
            "recommendation": "Assigned System Recommendation Action Level"
        },
        width="stretch", hide_index=True
    )

# ---------------------------------------------------------
# TAB 3: MODEL PERFORMANCE AND ANOMALY CHARTS
# ---------------------------------------------------------
with tab_diagnostics:
    st.subheader("Pipeline Calibration Diagnostics")
    
    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        if os.path.exists(f"{VISUALS_DIR}/candidate_score_histogram.png"):
            st.image(f"{VISUALS_DIR}/candidate_score_histogram.png", width="stretch")
        if os.path.exists(f"{VISUALS_DIR}/roc_curve.png"):
            st.image(f"{VISUALS_DIR}/roc_curve.png", width="stretch")
            
    with col_vis2:
        if os.path.exists(f"{VISUALS_DIR}/feature_importance.png"):
            st.image(f"{VISUALS_DIR}/feature_importance.png", width="stretch")
        if os.path.exists(f"{VISUALS_DIR}/precision_recall_curve.png"):
            st.image(f"{VISUALS_DIR}/precision_recall_curve.png", width="stretch")