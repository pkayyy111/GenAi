import os
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

# Initialize Gemini LLM with low temperature for focused output
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# Define prompt template for summarization
prompt = PromptTemplate(input_variables=["text"], template="Give me a summary in 1-2 sentences: {text}")
chain = prompt | llm

# ---------- Summarization Functions ----------
def summarize_text(text: str) -> str:
    docs = [Document(page_content=text)]
    splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    summary = ""
    for doc in split_docs:
        result = chain.invoke({"text": doc.page_content})
        summary += result.content.strip() + "\n"
    return summary.strip()

def summarize_pdf(pdf_file) -> str:
    if pdf_file is None:
        return "No PDF uploaded."
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            full_text += content + "\n"
    return summarize_text(full_text)

def summarize_text_file(txt_file) -> str:
    if txt_file is None:
        return "No text file uploaded."
    text = txt_file.read().decode("utf-8")
    return summarize_text(text)

def summarize_plain_text(text) -> str:
    if not text.strip():
        return "Please enter some text."
    return summarize_text(text)

# ---------- Streamlit UI ----------
st.title("📄 Gemini Summarizer")

option = st.sidebar.radio("Select Input Type:", ("PDF File", "Text File", "Plain Text"))

if option == "PDF File":
    st.header("Upload a PDF file")
    pdf_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if st.button("Summarize PDF"):
        with st.spinner("Summarizing PDF..."):
            summary = summarize_pdf(pdf_file)
        st.text_area("Summary", value=summary, height=300)

elif option == "Text File":
    st.header("Upload a Text file")
    txt_file = st.file_uploader("Choose a TXT file", type=["txt"])
    if st.button("Summarize Text File"):
        with st.spinner("Summarizing Text File..."):
            summary = summarize_text_file(txt_file)
        st.text_area("Summary", value=summary, height=300)

elif option == "Plain Text":
    st.header("Enter Plain Text")
    user_text = st.text_area("Enter your text here", height=300)
    if st.button("Summarize Text"):
        with st.spinner("Summarizing Plain Text..."):
            summary = summarize_plain_text(user_text)
        st.text_area("Summary", value=summary, height=300)
