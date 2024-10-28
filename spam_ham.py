import streamlit as st
import pickle 
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download specific datasets or models
nltk.download('punkt')           # Tokenizer for words and sentences
nltk.download('stopwords')       # Common stopwords like 'the', 'is', etc.
nltk.download('wordnet')         # WordNet lexical database for word meanings
nltk.download('averaged_perceptron_tagger')  # Part of Speech Tagging
nltk.download('omw-1.4')         # WordNet's optional multilingual data
nltk.download('vader_lexicon')   # For sentiment analysis
nltk.download('brown')           # General purpose corpus (optional)
nltk.download('names')  

# Page configuration
st.set_page_config(
    page_title="Spam Detective",
    page_icon="🕵️‍♂️",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title section with emoji and description
st.title('🕵️‍♂️ Spam Detective')
st.markdown("""
    <p style='text-align: center; color: #666666; font-size: 1.2em;'>
        Detect whether a message is spam or legitimate using advanced machine learning
    </p>
    """, unsafe_allow_html=True)

st.markdown("---")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load models
@st.cache_resource
def load_models():
    with open('pickle_files/bow_model.pkl', 'rb') as modelfile:
        model = pickle.load(modelfile)
    with open('pickle_files/bow.pkl', 'rb') as vectorizerfile:
        cv = pickle.load(vectorizerfile)
    return model, cv

model, cv = load_models()

# Input section
with st.container():
    st.markdown("### 📝 Enter your message")
    text = st.text_area(
        label="Message",
        placeholder="Type or paste your message here...",
        height=150,
        label_visibility="collapsed"
    )

# Classification section
if st.button('Analyze Message 🔍'):
    if not text:
        st.warning('Please enter a message to analyze')
    else:
        with st.spinner('Analyzing...'):
            # Text preprocessing
            regex = re.sub('[^a-zA-Z]', ' ', text)
            regex = regex.lower()
            tokenized = regex.split()
            clean_tokens = [lemmatizer.lemmatize(word) for word in tokenized if word not in set(stopwords.words('english'))]
            clean_text = ' '.join(clean_tokens)
            
            # Prediction
            vectorized = cv.transform([clean_text]).toarray()
            prediction = model.predict(vectorized)
            
            # Display result
            st.markdown("### 🎯 Result")
            if prediction[0] == 1:
                st.error('🚨 This message is likely SPAM!')
            else:
                st.success('✅ This message appears to be legitimate.')

# Add footer
st.markdown("---")
st.markdown("""
    <p style='text-align: center; color: #666666;'>
        Built with ❤️ using Streamlit and Machine Learning By Talha
    </p>
    """, unsafe_allow_html=True)
