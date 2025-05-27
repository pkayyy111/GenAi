import gradio as gr
#imports
import os
from google import genai
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader
from pinecone import Pinecone
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

#definitions:
client = genai.Client()
client.api_key = os.getenv('GOOGLE_API_KEY')


model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                                     google_api_key=os.getenv('GOOGLE_API_KEY'))
pc = Pinecone(os.getenv('pc_key'))   #connecting to pinecone
index = pc.Index("rag-1")  #selecting the index
vector_store = PineconeVectorStore(index=index, embedding=model)   #creating vector store

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template='''Answer the question based only on this context. 
    Give the answer in 2-3 sentences for each question. 
    If the answer is not relevant to the context, say: "Resources are irrelevant."
     Context:{context}
     Question: {question} '''
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", 
                             temperature=0.3, 
                             google_api_key=os.getenv('GOOGLE_API_KEY'))
chain = prompt_template | llm

#------ 
#step-8: RAG functions for Gradio

def load_pdf(pdf_file):
    reader = PdfReader(pdf_file.name)
    full_text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            full_text += content + "\n"
    docs = [Document(page_content=full_text)]
    splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    vector_store.add_documents(split_docs)
    return "pdf uploaded"

def answer(query):
    results = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in results])
    response = chain.invoke({"context": context, "question": query})
    return response.content

with gr.Blocks(title="RAG Chatbot PDF Assistant") as demo:
    gr.Markdown(
        """
        #  RAG Chatbot with Gemini
        Upload your own PDF and ask questions. The chatbot reads your document and returns intelligent answers.
        """
    )

    with gr.Accordion("Upload and Index PDF", open=True):
        with gr.Row():
            pdf_input = gr.File(label="Upload your PDF", file_types=[".pdf"])
            upload_status = gr.Textbox(
                label="Upload Status",
                placeholder="No file uploaded yet...",
                interactive=False,
                lines=1
            )
        upload_button = gr.Button(" Index PDF")
        upload_button.click(fn=load_pdf, inputs=pdf_input, outputs=upload_status)

    gr.Markdown("### Ask a Question")

    with gr.Row():
        question = gr.Textbox(
            label="Your Question",
            placeholder="e.g., What is the main topic discussed on page 2?",
            lines=2
        )

    ans = gr.Textbox(
        label="Answer",
        placeholder="Answer will appear here...",
        lines=4,
        interactive=False
    )

    ask_button = gr.Button("Get Answer")
    ask_button.click(fn=answer, inputs=question, outputs=ans)

demo.launch(share=True)
