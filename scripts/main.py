import os
import config  # your config.py with API keys

# Set environment variables for API keys
os.environ["GOOGLE_API_KEY"] = config.GOOGLE_API_KEY

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate

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
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        """
        Construct the StateGraph pipeline by defining nodes and their execution flow.
        Returns the compiled pipeline ready for invocation.
        """
        graph = StateGraph(dict)

        # Add nodes representing each step of the pipeline
        graph.add_node("summarizer", self.summarize)
        graph.add_node("classifier", self.classify_topic)
        graph.add_node("extractor", self.extract_entities)

        # Define the execution order of nodes
        graph.set_entry_point("summarizer")
        graph.add_edge("summarizer", "classifier")
        graph.add_edge("classifier", "extractor")
        graph.add_edge("extractor", END)

        # Compile and return the executable graph
        pipeline = graph.compile()
        return pipeline

    def summarize(self, state):
        """
        Generate a summary of the article.
        Input: state dict containing 'article'
        Output: updated state dict including 'summary'
        """
        summary = self.llm.invoke(SUMMARY_PROMPT.format(article=state['article'])).content.strip()
        return {**state, "summary": summary}

    def classify_topic(self, state):
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

    def run(self, article):
        """
        Run the entire pipeline on a single article text.
        Returns the final result dictionary with all generated outputs.
        """
        return self.pipeline.invoke({"article": article})

    def print_output(self, result):
        """
        Nicely print the pipeline output dictionary.
        """
        print("\n--- OUTPUT ---")
        for k, v in result.items():
            print(f"{k}:\n{v}\n")

    def save_graph(self, path):
        """
        Save a visualization of the pipeline graph as a PNG file.
        """
        png_bytes = self.pipeline.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png_bytes)

if __name__ == "__main__":
    # Initialize the language model with Google Gemini 2.0 flash model
    llm = init_chat_model("google_genai:gemini-2.0-flash")
    
    # Instantiate the news pipeline with the language model
    pipeline = NewsPipeline(llm)

    # Example article text to process
    article_text = "UK Prime Minister Rishi Sunak has announced a surprise election date..."

    # Run the pipeline and get the result
    result = pipeline.run(article_text)

    # Print the results
    pipeline.print_output(result)

    # Save the graph visualization to file
    path = "scripts/graph.png"
    pipeline.save_graph(path)
