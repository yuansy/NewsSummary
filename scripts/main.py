import os
import config  # your config.py with API keys

# Set environment variables for API keys
os.environ["GOOGLE_API_KEY"] = config.GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = config.TAVILY_API_KEY

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_tavily import TavilySearch

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


class NewsPipeline:
    def __init__(self, llm):
        """
        Initialize the NewsPipeline with a language model instance and build the processing pipeline graph.
        """
        self.llm = llm
        self.search_tool = TavilySearch()
        self.pipeline = self.build_pipeline()

    def build_pipeline(self):
        """
        Construct the StateGraph pipeline by defining nodes and their execution flow.
        Returns the compiled pipeline ready for invocation.
        """
        graph = StateGraph(dict)

        # Add nodes representing each step of the pipeline
        graph.add_node("search", self.search)
        graph.add_node("summarize", self.summarize)
        graph.add_node("identify_topic", self.identify_topic)
        graph.add_node("extract_entities", self.extract_entities)

        # Define the execution order of nodes
        graph.set_entry_point("search")
        graph.add_edge("search", "summarize")
        graph.add_edge("summarize", "identify_topic")
        graph.add_edge("identify_topic", "extract_entities")
        graph.add_edge("extract_entities", END)

        # Compile and return the executable graph
        pipeline = graph.compile()
        return pipeline

    def search(self, state):
        """
        Use Tavily to search for news articles based on the query.
        Input: state dict containing 'query'
        Output: updated state dict including 'article' and 'url'
        """
        query = state['query']
        results = self.search_tool.run(query)["results"]
        if results:
            # Pick the first result for now
            title = results[0]["title"]
            article = results[0]["content"]
            url = results[0]["url"]
        else:
            title = "No relevant articles found."
            article = ""
            url = ""
        return {**state, "title": title, "article": article, "url": url}

    def summarize(self, state):
        """
        Generate a summary of the article.
        Input: state dict containing 'article'
        Output: updated state dict including 'summary'
        """
        summary = self.llm.invoke(SUMMARY_PROMPT.format(article=state['article'])).content.strip()
        return {**state, "summary": summary}

    def identify_topic(self, state):
        """
        Identify the main topic of the article.
        Input: state dict containing 'article'
        Output: updated state dict including 'topic'
        """
        topic = self.llm.invoke(TOPIC_PROMPT.format(article=state['article'])).content.strip()
        return {**state, "topic": topic}

    def extract_entities(self, state):
        """
        Extract key people, organizations, and locations mentioned.
        Input: state dict containing 'article'
        Output: updated state dict including 'entities'
        """
        entities = self.llm.invoke(ENTITIES_PROMPT.format(article=state['article'])).content.strip()
        return {**state, "entities": entities}

    def run(self, query):
        """
        Run the entire pipeline on a single query.
        Returns the final result dictionary with all generated outputs.
        """
        return self.pipeline.invoke({"query": query})

    def print_output(self, result):
        """
        Nicely print the pipeline output dictionary.
        """
        for k, v in result.items():
            print(f"\n--- {k.upper()} ---\n{v}")

    def save_graph(self, path):
        """
        Save a visualization of the pipeline graph as a PNG file.
        """
        png_bytes = self.pipeline.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_bytes)

if __name__ == "__main__":
    # Search query input
    query = "Tesla earnings Q2 2025"

    # Initialize the language model with Google Gemini 2.0 flash model
    llm = init_chat_model("google_genai:gemini-2.0-flash")
    
    # Instantiate the news pipeline with the language model
    pipeline = NewsPipeline(llm)

    # Run the pipeline and get the result
    result = pipeline.run(query)

    # Print the results
    pipeline.print_output(result)

    # (Dev) Save the graph visualization to file
    pipeline.save_graph("graph.png")
