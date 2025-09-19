import os
import pickle
import streamlit as st
import numpy as np
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2ca02c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .real-news {
        color: #2ca02c;
        font-weight: bold;
    }
    .fake-news {
        color: #d62728;
        font-weight: bold;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_DIR = os.path.join(BASE_DIR, "models")
        
        with open(os.path.join(MODEL_DIR, "svc_model.pkl"), "rb") as f:
            model = pickle.load(f)
        
        with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model, st.session_state.vectorizer = load_model()

if 'history' not in st.session_state:
    st.session_state.history = []

# Header
st.markdown('<h1 class="main-header">📰 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by Linear SVC • Accuracy: 98.82%</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This app uses a Linear Support Vector Classifier (SVC) model trained on thousands 
    of news articles to detect potentially fake content with high accuracy.
    """)
    
    st.header("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "98.82%")
        st.metric("Precision", "98.91%")
    with col2:
        st.metric("Recall", "98.75%")
        st.metric("F1 Score", "98.83%")
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

# Main content
tab1, tab2, tab3 = st.tabs(["Detector", "How It Works", "History"])

with tab1:
    st.header("Analyze News Content")
    
    input_method = st.radio(
        "Input method:",
        ["Paste text", "Enter URL"],
        horizontal=True
    )
    
    if input_method == "Paste text":
        user_input = st.text_area(
            "Paste news content here:",
            height=200,
            placeholder="Copy and paste the news article text you want to analyze..."
        )
    else:
        url_input = st.text_input(
            "Enter news article URL:",
            placeholder="https://example.com/news-article"
        )
        user_input = ""
    
    if st.button("Analyze News", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing content..."):
                try:
                    # Transform the input text
                    X = st.session_state.vectorizer.transform([user_input])
                    
                    # Get prediction and decision function
                    prediction = st.session_state.model.predict(X)[0]
                    decision_score = st.session_state.model.decision_function(X)[0]
                    
                    # CORRECTED PROBABILITY CALCULATION
                    # Convert decision score to probability using sigmoid
                    probability = 1 / (1 + np.exp(-decision_score))
                    
                    # For LinearSVC: 
                    # decision_score > 0 → prediction = 1 (Fake), higher probability for Fake
                    # decision_score < 0 → prediction = 0 (Real), higher probability for Real
                    if prediction == 1:  # Fake news
                        fake_prob = probability
                        real_prob = 1 - probability
                    else:  # Real news
                        real_prob = 1 - probability
                        fake_prob = probability
                    
                    # Use model's prediction directly
                    final_prediction = prediction
                    confidence = max(real_prob, fake_prob)
                    
                    # Store in history
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'text': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                        'prediction': final_prediction,
                        'real_prob': real_prob,
                        'fake_prob': fake_prob,
                        'confidence': confidence
                    })
                    
                    # Display results
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if final_prediction == 0:
                            st.markdown(f'<h2 class="real-news">✅ REAL NEWS</h2>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<h2 class="fake-news">❌ FAKE NEWS</h2>', unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Confidence", f"{confidence*100:.2f}%")
                    
                    # Probability breakdown - ALWAYS show higher probability first
                    st.subheader("Probability Breakdown")
                    
                    prob_col1, prob_col2 = st.columns(2)
                    
                    # Show the higher probability class first
                    if real_prob >= fake_prob:
                        with prob_col1:
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            st.metric("Real News Probability", f"{real_prob*100:.2f}%")
                            st.progress(real_prob, text=f"Real News confidence")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with prob_col2:
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            st.metric("Fake News Probability", f"{fake_prob*100:.2f}%")
                            st.progress(fake_prob, text=f"Fake News confidence")
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        with prob_col1:
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            st.metric("Fake News Probability", f"{fake_prob*100:.2f}%")
                            st.progress(fake_prob, text=f"Fake News confidence")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with prob_col2:
                            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                            st.metric("Real News Probability", f"{real_prob*100:.2f}%")
                            st.progress(real_prob, text=f"Real News confidence")
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Confidence message
                    if confidence > 0.9:
                        st.success("High confidence in this prediction!")
                    elif confidence > 0.75:
                        st.info("Moderate confidence in this prediction.")
                    else:
                        st.warning("Lower confidence. Please review this content carefully.")
                        
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.header("How This Fake News Detector Works")
    
    st.write("""
    This application uses a Linear Support Vector Classifier (SVC) model that was trained 
    on thousands of real and fake news articles to identify patterns that distinguish 
    credible content from misinformation.
    """)
    
    st.subheader("Technology Stack")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Linear SVC**")
        st.write("Support Vector Machine with linear kernel for optimal text classification")
    
    with col2:
        st.markdown("**TF-IDF Vectorization**")
        st.write("Converts text into numerical features while weighting important words")
    
    with col3:
        st.markdown("**Streamlit**")
        st.write("Web framework for building interactive data applications")
    
    st.subheader("Model Training")
    st.write("""
    The model was trained on a balanced dataset containing:
    - 22,449 real news articles (labeled as 0)
    - 22,449 fake news articles (labeled as 1)
    
    Text was processed using TF-IDF with n-grams (1-2 words) and limited to 20,000 features
    to capture important phrases and patterns.
    """)
    
    st.subheader("Why Linear SVC?")
    st.write("""
    Linear SVC is particularly effective for text classification because:
    - Excellent with high-dimensional sparse data (like TF-IDF features)
    - Finds optimal decision boundaries between classes
    - Fast training and prediction times
    - Handles large datasets efficiently
    """)
    
    st.subheader("Interpreting Results")
    st.write("""
    - **Real News Prediction**: The model classifies this as credible content
    - **Fake News Prediction**: The model identifies potential misinformation
    - **Confidence Score**: How certain the model is about its prediction
    - **Probability Breakdown**: Detailed view of how the model weighed the decision
    """)

with tab3:
    st.header("Analysis History")
    
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"{item['timestamp'].strftime('%Y-%m-%d %H:%M')} - {item['text']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    if item['prediction'] == 0:
                        st.markdown(f'<span class="real-news">✅ REAL NEWS</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="fake-news">❌ FAKE NEWS</span>', unsafe_allow_html=True)
                    
                    st.write(f"Confidence: {item['confidence']*100:.2f}%")
                
                with col2:
                    # Show probabilities in order of highest first
                    if item['real_prob'] >= item['fake_prob']:
                        st.write(f"Real Probability: {item['real_prob']*100:.2f}%")
                        st.write(f"Fake Probability: {item['fake_prob']*100:.2f}%")
                    else:
                        st.write(f"Fake Probability: {item['fake_prob']*100:.2f}%")
                        st.write(f"Real Probability: {item['real_prob']*100:.2f}%")
    else:
        st.info("No analysis history yet. Analyze some news to see your history here.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>This fake news detector uses machine learning to identify potential misinformation but should not be used as the sole source of truth.</p>
        <p>Always verify information through multiple reliable sources.</p>
    </div>
    """,
    unsafe_allow_html=True
)