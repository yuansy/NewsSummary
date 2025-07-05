import os
import config  # your config.py with API keys

# Set environment variables for API keys
os.environ["GOOGLE_API_KEY"] = config.GOOGLE_API_KEY

from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

llm = init_chat_model("google_genai:gemini-2.0-flash")

# Define prompt templates
SUMMARY_PROMPT = PromptTemplate.from_template("""
Summarize the following news article:

{article}

Summary:
""")

TOPIC_PROMPT = PromptTemplate.from_template("""
What is the main topic of the following article?

{article}

Topic:
""")

ENTITIES_PROMPT = PromptTemplate.from_template("""
List the key people, organizations, and locations mentioned in this article:

{article}

Entities:
""")

BIAS_PROMPT = PromptTemplate.from_template("""
Analyze the following article for subjective or biased language. Label as "None", "Mild", or "Moderate" bias and explain briefly:

{article}

Bias:
""")

TWEET_PROMPT = PromptTemplate.from_template("""
Write a tweet-sized summary of the following article (max 280 characters):

{summary}

Tweet:
""")

# Define each node as a function
def summarize(state: dict) -> dict:
    summary = llm.invoke(SUMMARY_PROMPT.format(article=state['article'])).content.strip()
    return {**state, "summary": summary}

def classify_topic(state: dict) -> dict:
    topic = llm.invoke(TOPIC_PROMPT.format(article=state['article'])).content.strip()
    return {**state, "topic": topic}

def extract_entities(state: dict) -> dict:
    entities = llm.invoke(ENTITIES_PROMPT.format(article=state['article'])).content.strip()
    return {**state, "entities": entities}

def detect_bias(state: dict) -> dict:
    bias_flag = llm.invoke(BIAS_PROMPT.format(article=state['article'])).content.strip()
    return {**state, "bias_flag": bias_flag}

def generate_tweet(state: dict) -> dict:
    tweet = llm.invoke(TWEET_PROMPT.format(summary=state['summary'])).content.strip()
    return {**state, "tweet": tweet}

# Define the graph
graph = StateGraph(dict)
graph.add_node("summarizer", summarize)
graph.add_node("classifier", classify_topic)
graph.add_node("extractor", extract_entities)
graph.add_node("bias_checker", detect_bias)
graph.add_node("tweet_gen", generate_tweet)

# Define edges
graph.set_entry_point("summarizer")
graph.add_edge("summarizer", "classifier")
graph.add_edge("classifier", "extractor")
graph.add_edge("extractor", "bias_checker")
graph.add_edge("bias_checker", "tweet_gen")
graph.add_edge("tweet_gen", END)

# Compile the graph
news_pipeline = graph.compile()

# Example run
if __name__ == "__main__":
    article_text = """UK Prime Minister Rishi Sunak has announced a surprise election date..."""
    result = news_pipeline.invoke({"article": article_text})
    print("\n--- OUTPUT ---")
    for k, v in result.items():
        print(f"{k}:\n{v}\n")
