import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- Configuration and Setup ---

# NOTE: For a real deployment, you would ensure these files are available
# We are assuming the model and vectorizer were saved by the original script
MODEL_FILENAME = 'suicide_predictor_model.pkl'
VECTORIZER_FILENAME = 'vectorizer.pkl'
SUICIDE_PREVENTION_HELPER_LINK = "https://chatgpt.com/g/g-rhMdGLYm4-suicide-prevention-helper"

# --- Utility Functions ---

@st.cache_resource # Cache the model loading for efficiency
def load_models():
    """Loads the trained Logistic Regression model and TfidfVectorizer."""
    try:
        # Load the models. These files must be present in the execution directory.
        model = joblib.load(MODEL_FILENAME)
        vectorizer = joblib.load(VECTORIZER_FILENAME)
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"Error: Model or Vectorizer file not found. Please ensure '{MODEL_FILENAME}' and '{VECTORIZER_FILENAME}' are in the same directory.")
        return None, None

def preprocess_text(text):
    """Converts text to lowercase."""
    if isinstance(text, str):
        return text.lower()
    return ""

def predict_risk(input_text, model, vectorizer):
    """Predicts the risk level for a given input text."""
    input_text_preprocessed = preprocess_text(input_text)
    input_text_vectorized = vectorizer.transform([input_text_preprocessed])
    prediction = model.predict(input_text_vectorized)[0]
    return prediction

def display_high_risk_suggestions(input_text):
    """Displays high-risk suggestions with a visual delay and link."""
    
    # 1. Initial immediate warning
    st.markdown("---")
    st.error("🚨 **IMMEDIATE ACTION RECOMMENDED** 🚨")
    st.write("This thought pattern is classified as **High Risk**.")
    st.info("Please wait a few seconds while we prepare immediate, safe support options...")

    # 2. Add a visual pause/delay (using st.empty and time.sleep)
    # This pauses the app's execution flow for 10 seconds before proceeding.
    delay_placeholder = st.empty()
    delay_placeholder.warning("Pausing for 5 seconds to ensure you have time to see this warning.")
    time.sleep(5)
    delay_placeholder.empty()

    # 3. Display critical resources after the delay
    st.markdown("### 📞 Critical Support & Resources")
    
    # Using detailed markdown/HTML to ensure the link opens in a new tab (target="_blank")
    # and provides explicit copy instructions.
    st.page_link(SUICIDE_PREVENTION_HELPER_LINK, label="Suicide Prevention Helper", icon="🧠")

    
    st.markdown("---")
    st.markdown("**If you are in immediate crisis, please use a helpline:**")
    st.markdown("* **US National Suicide Prevention Lifeline:** Dial 988")
    st.markdown("* **Crisis Text Line (US/Canada):** Text HOME to 741741")
    st.markdown("---")

def display_low_risk_suggestions():
    """Displays low-risk suggestions."""
    st.markdown("---")
    st.success("🌟 **LOW RISK (Non-Suicide)** 🌟")
    st.write("This is classified as Low Risk, but please remember to prioritize your mental well-being.")
    st.markdown("""
    * If you are feeling stressed or overwhelmed, consider talking to a friend, family member, or mental health professional.
    * **General Wellness Tip:** Practice mindfulness or take a short break.
    """)
    st.markdown("---")

# --- Streamlit UI and Logic ---

def main():
    st.set_page_config(
        page_title="Suicide Risk Predictor",
        layout="centered",
        initial_sidebar_state="auto"
    )
    
    st.title("🧠 Thought Risk Predictor")
    st.header("Using Logistic Regression and TF-IDF")
    
    model, vectorizer = load_models()

    if model is None or vectorizer is None:
        return # Stop execution if models fail to load

    st.markdown(
        """
        Enter a thought or text below to classify it as High Risk (Suicide) or Low Risk (Non-Suicide).
        
        **Disclaimer:** This is an experimental classification tool and not a substitute for professional medical or mental health advice. 
        Always seek help from qualified professionals in a crisis.
        """
    )
    
    # Text input area
    user_input = st.text_area(
        "Enter thought:",
        placeholder="e.g., I feel so hopeless and can't see a way out.",
        height=150
    )
    
    # Prediction button
    if st.button("Analyze Thought", type="primary"):
        if user_input.strip():
            # Get prediction
            risk_prediction_value = predict_risk(user_input, model, vectorizer)
            
            # Display results
            st.subheader("Classification Result:")
            
            if risk_prediction_value == 1:
                st.metric(label="Predicted Risk", value="High Risk", delta="Immediate Attention Needed", delta_color="inverse")
                # Pass the original input text to the suggestions function
                display_high_risk_suggestions(user_input)
            else:
                st.metric(label="Predicted Risk", value="Low Risk")
                display_low_risk_suggestions()
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()