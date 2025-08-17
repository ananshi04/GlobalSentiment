# 🌍 GlobalSentiment

**GlobalSentiment** is a **multilingual sentiment analysis application** built with **Streamlit**. It allows users to analyze text or CSV reviews in multiple languages, automatically detect the language, translate non-English text to English, and provide sentiment analysis with an AI-generated summary.

---

## 🚀 Features

- **Multilingual Support**: Detects and translates text from 100+ languages.  
- **Accurate Sentiment Analysis**: Uses HuggingFace's `twitter-xlm-roberta-base-sentiment` model.  
- **AI Summaries**: Generates concise insights and summaries of the input reviews using **Google Gemini**.  
- **Flexible Input**: Accepts text input or CSV uploads (with a `text` column).  
- **Download Results**: Export sentiment analysis results and AI summaries.  
- **Beautiful Interface**: Interactive sidebar with paired buttons, feature cards, and responsive layout.

---

## 📄 Pages

1. **🏠 Home**  
   - Overview of features and quick guidance.  

2. **📄 Input**  
   - Upload CSV or paste reviews.  
   - Run sentiment analysis.  

3. **📊 Analysis**  
   - View detailed sentiment results.  
   - Download results as CSV.  

4. **🧠 Summary**  
   - AI-generated summary of reviews.  
   - Editable and downloadable summary.

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd GlobalSentiment
