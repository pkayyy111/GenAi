import streamlit as st
import os
from langgraph.graph import StateGraph, END
from langchain_community.utilities import SerpAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

# Set API keys (consider loading these from a .env file in production)
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
os.environ["SERPAPI_API_KEY"] = os.getenv('SERPAPI_API_KEY')

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# Step 1: Perform web search using SerpAPI
def web_search_node(state: dict):
    topic = state.get("topic")  # Extract topic from input state
    search = SerpAPIWrapper()  # Initialize SerpAPI
    results = search.run(topic)  # Run the web search
    return {"raw_content": results}  # Output raw content for next node

# Step 2: Summarize search content using Gemini
def summarizer_node(state: dict):
    content = state.get("raw_content")  # Get web search results
    prompt = f"Summarize this content in one concise paragraph:\n\n{content}"
    response = llm.invoke(prompt)  # Generate summary with Gemini
    return {"summary": response.content}  # Pass summary to next node

# Step 3: Fact-check the summary using Gemini
def fact_checker_node(state: dict):
    summary = state.get("summary")  # Get the summary
    prompt = f"""
    Fact-check this summary and revise any incorrect or uncertain claims.  
    Provide the revised summary as plain text only:

    {summary}
    """
    response = llm.invoke(prompt)  # Gemini fact-checks and revises
    return {"verified_summary": response.content}  # Final output

# Build the LangGraph workflow
graph_builder = StateGraph(dict)
graph_builder.add_node("WebSearch", web_search_node)
graph_builder.add_node("Summarizer", summarizer_node)
graph_builder.add_node("FactChecker", fact_checker_node)
graph_builder.set_entry_point("WebSearch")
graph_builder.add_edge("WebSearch", "Summarizer")
graph_builder.add_edge("Summarizer", "FactChecker")
graph_builder.add_edge("FactChecker", END)
graph = graph_builder.compile()

# Streamlit UI setup
st.set_page_config(page_title="GenAI Web Summarizer", layout="centered")
st.title("Research assistant")

# Input field for user topic
topic = st.text_input("Enter a topic:", "Climate change in 2025")

# Trigger pipeline when button is clicked
if st.button("Generate Verified Summary"):
    with st.spinner("Processing..."):
        input_state = {"topic": topic}
        final_state = graph.invoke(input_state)

        # Display the final verified summary
        st.subheader("Verified Summary:")
        st.write(final_state["verified_summary"])
