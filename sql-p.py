import streamlit as st
from typing import Annotated
from langchain_community.utilities import SQLDatabase
from typing_extensions import TypedDict
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

from dotenv import load_dotenv

load_dotenv()  # take environment variables

# -- Setup 
# DB Connection
db = SQLDatabase.from_uri(
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@localhost:3306/chinook"
)


# Google API Key
os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')

# LLM setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0.7,
    max_tokens=1024
)

# Prompt template
system_message = """
Given a question, write a correct {dialect} SQL query to find the answer.
If the question doesn't say how many results to show, limit it to {top_k} rows.
Sort the results by a useful column to show the most important data.
Don't select all columns from a table—only pick the columns needed for the question.
Use only column names that exist in the database schema.
Only use the following tables:
{table_info}
"""
user_prompt = "Question: {input}"
query_prompt_template = ChatPromptTemplate([("system", system_message), ("user", user_prompt)])


# -- Data Types 
class State(TypedDict):       #type of container instead of calling each attribute
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    # This tells the LLM to return a dictionary with one key: "query"
    # The value must be a valid SQL query (string format)
    query: Annotated[str, ..., "Syntactically valid SQL query."]



# --Functional Logic 

def write_query(state: State):
    # Generates SQL query to fetch information.
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    print(result)
    return {"query": result["query"]}

from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)          # tool used to execute query in mysql
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State):
    # Answers question using retrieved information as context.
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

# -- Streamlit UI

st.set_page_config(page_title="Natural Language - SQL ", layout="centered")
st.title("Natural Language - SQL ")
st.write("Ask your SQL question in plain English.")

# Input from user
user_input = st.text_input("Enter your question:", placeholder="e.g. List the customer ids of people in Dublin.")

if st.button("Submit") and user_input:
    # Initialize state
    state: State = {
        "question": user_input,
        "query": "",
        "result": "",
        "answer": ""
    }

    with st.spinner("Processing your request..."):
        state.update(write_query(state))
        state.update(execute_query(state))
        state.update(generate_answer(state))

    # Display results
    st.subheader("Generated SQL Query")
    st.code(state["query"], language="sql")

    st.subheader("SQL Result")
    st.code(state["result"])

    st.subheader("Final Answer")
    st.success(state["answer"])
