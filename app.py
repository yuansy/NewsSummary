# streamlit run app.py

import os
import streamlit as st
from pipeline import NewsPipeline
from langchain.chat_models import init_chat_model

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

st.set_page_config(page_title="News Insight", layout="wide")

st.title("📊 Financial News Insight Tool")
st.write("Type a financial event or company and get instant news insights.")

query = st.text_input("Enter a query:", "Tesla Q2 2025 earnings")

if st.button("Run Analysis"):
    llm = init_chat_model("google_genai:gemini-2.0-flash")
    pipeline = NewsPipeline(llm)

    with st.spinner("Fetching and analyzing news..."):
        result = pipeline.run(query)

    st.subheader("🔗 Source")
    st.markdown(f"[{result['title']}]({result['url']})")

    st.subheader("📝 Summary")
    st.write(result["summary"])

    st.subheader("🧠 Topic")
    st.write(result["topic"])

    st.subheader("🏢 Entities")
    st.write(result["entities"])