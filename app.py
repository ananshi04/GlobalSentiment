import streamlit as st
import pandas as pd
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import google.generativeai as genai
import pycountry

# -----------------------------------------------------
# ⚙️ Setup
# -----------------------------------------------------
st.set_page_config(
    page_title="GlobalSentiment",
    page_icon="🌐",
    layout="wide"
)

# 🔑 Configure Gemini API
genai.configure(api_key="AIzaSyAHEeMb5sWZVbCqFKC4fCZG0vSMamm8rJQ")

# -----------------------------------------------------
# 🔍 Helpers
# -----------------------------------------------------
@st.cache_resource
def load_sentiment_pipeline():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def detect_language(text: str):
    try:
        return detect(text)
    except Exception:
        return "unknown"

def get_language_name(lang_code: str):
    try:
        if lang_code == "unknown":
            return "Unknown"
        return pycountry.languages.get(alpha_2=lang_code).name
    except Exception:
        return lang_code

def translate_text(text: str, lang: str):
    if lang == "en" or lang == "unknown":
        return text
    try:
        prompt = f"Translate the following {get_language_name(lang)} text into English:\n\n{text}"
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return text

def analyze_texts(texts, pipe=None):
    if pipe is None:
        pipe = load_sentiment_pipeline()
    return pipe(texts, batch_size=16)

def summarize_reviews(texts):
    try:
        prompt = f"Summarize the overall sentiment and key points from these reviews:\n\n{texts}"
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return f"(Fallback) Total {len(texts)} reviews analyzed. Common themes observed."

# Session state
if "results" not in st.session_state:
    st.session_state["results"] = None
if "page" not in st.session_state:
    st.session_state["page"] = "🏠 Home"

# -----------------------------------------------------
# 🎨 Sidebar Styling + Full-width Buttons
# -----------------------------------------------------
st.markdown("""
<style>
/* Sidebar header */
.sidebar .sidebar-content h2 {
    text-align: center;
    color: #1f4e79;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 20px;
}

/* Full width sidebar buttons with thicker border */
.sidebar .stButton > button {
    width: 100% !important;
    margin: 8px 0;
    padding: 12px 0;
    background-color: #3498db;
    color: white;
    font-size: 16px;
    font-weight: bold;
    border-radius: 15x;
    border: 3px solid #1f4e79;  /* Thicker border */
}

/* Active button highlight */
.sidebar .stButton > button:focus {
    background-color: #2c3e50 !important;
    color: #fff !important;
    border: 3px solid #2980b9;  /* Thicker border on active */
}

/* Sidebar footer */
.sidebar .sidebar-footer {
    position: absolute;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    color: #7f8c8d;
}
</style>
""", unsafe_allow_html=True)


# Sidebar header
st.sidebar.markdown("<h2>🌍 GlobalSentiment</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Sidebar buttons
pages = ["🏠 Home", "📄 Input", "📊 Analysis", "🧠 Summary"]
for p in pages:
    if st.sidebar.button(p):
        st.session_state["page"] = p

# Sidebar footer

st.sidebar.info(f"🚀 Currently: **{st.session_state['page']}**")
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-footer">🚀 Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)

# -----------------------------------------------------
# 🏠 Home Page
# -----------------------------------------------------
if st.session_state["page"] == "🏠 Home":
    st.markdown("<h1 style='text-align:center; color:#3498db;'>🌍 GlobalSentiment</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:#2c3e50;'>Multilingual Sentiment Analysis Made Easy</h4>", unsafe_allow_html=True)
    st.markdown("---")

    # Feature cards
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("<div style='background-color:#3498db; color:white; padding:15px; text-align:center; border-radius:10px;'>📝<br>Input<br><small>Text / CSV</small></div>", unsafe_allow_html=True)
    col2.markdown("<div style='background-color:#2ecc71; color:white; padding:15px; text-align:center; border-radius:10px;'>🌐<br>Languages<br><small>100+ Supported</small></div>", unsafe_allow_html=True)
    col3.markdown("<div style='background-color:#f1c40f; color:white; padding:15px; text-align:center; border-radius:10px;'>😃<br>Sentiment<br><small>Positive/Negative</small></div>", unsafe_allow_html=True)
    col4.markdown("<div style='background-color:#e67e22; color:white; padding:15px; text-align:center; border-radius:10px;'>🧠<br>Insights<br><small>AI Summary</small></div>", unsafe_allow_html=True)

    st.markdown("### ✨ Key Features")
    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        st.success("✅ Auto language detection")
        st.info("🌍 Translate non-English → English")
    with feat_col2:
        st.success("✅ Accurate sentiment model (Roberta-XLM)")
        st.info("🧠 AI summaries via Gemini")

    st.markdown("---")
    st.markdown("📌 **Start by going to the '📄 Input' tab from the sidebar →**")
    
# -----------------------------------------------------
# 📄 Input Page
# -----------------------------------------------------
elif st.session_state["page"] == "📄 Input":
    st.header("📄 Provide Input")
    uploaded = st.file_uploader("Upload CSV (column: `text`)", type=["csv"])
    text_input = st.text_area("Or paste reviews (one per line)", height=150)

    if st.button("🚀 Run Analysis"):
        lines = []
        if uploaded:
            df = pd.read_csv(uploaded)
            lines = df['text'].dropna().astype(str).tolist()
        else:
            lines = [l.strip() for l in text_input.splitlines() if l.strip()]

        if not lines:
            st.warning("⚠️ Please provide some text input.")
        else:
            pipe = load_sentiment_pipeline()
            processed = []
            for text in lines:
                lang_code = detect_language(text)
                lang_name = get_language_name(lang_code)
                translated = translate_text(text, lang_code)
                processed.append((text, lang_code, lang_name, translated))

            sentiments = analyze_texts([p[3] for p in processed], pipe=pipe)
            results = []
            for (orig, code, lang, trans), sent in zip(processed, sentiments):
                results.append({
                    "original": orig,
                    "detected_lang": lang,
                    "translation": trans if code != "en" else "-",
                    "sentiment": sent["label"],
                    "score": round(sent["score"], 3)
                })
            st.session_state["results"] = pd.DataFrame(results)
            st.success("✅ Analysis completed! Go to 📊 Analysis tab.")

# -----------------------------------------------------
# 📊 Analysis Page
# -----------------------------------------------------
elif st.session_state["page"] == "📊 Analysis":
    st.header("📊 Sentiment Analysis Results")
    if st.session_state["results"] is None:
        st.warning("⚠️ No results yet. Please upload text in 📄 Input tab.")
    else:
        out_df = st.session_state["results"]
        st.dataframe(out_df, use_container_width=True)
        st.download_button("💾 Download results CSV", out_df.to_csv(index=False),
                           "results.csv", "text/csv")

# -----------------------------------------------------
# 🧠 Summary Page
# -----------------------------------------------------
elif st.session_state["page"] == "🧠 Summary":
    st.header("🧠 AI-generated Summary")
    if st.session_state["results"] is None:
        st.warning("⚠️ No data available. Please run analysis first.")
    else:
        texts = st.session_state["results"]["original"].tolist()
        summary = summarize_reviews(texts)
        st.success("✅ AI summary generated!")
        edited_summary = st.text_area("AI Summary", summary, height=250)
        st.download_button("💾 Download Summary as TXT", edited_summary,
                           file_name="summary.txt", mime="text/plain")
