import streamlit as st
import os
import json
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


file_path = 'movie-metadata.json'

# Load the GROQ and OpenAI API keys from env/secrets
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize the llm_model model - Llama3 and gemini embedding model
llm_model = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


#### Load the Data helper function
loader = JSONLoader(
    file_path=file_path,
    jq_schema='.',  # jq_schema as per your JSON structure
    text_content=False
)

# Load JSON
docs = loader.load()

### Split the text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs)

### Load the vector store using gemini embeddings on the loaded text
knowledgeBase = FAISS.from_documents(final_documents, embeddings)

# Streamlit UI
st.title('Personalized Movie Recommendations using Advanced RAG Techniques')

# User input for query
user_query = st.text_input('I am Personalized Movie Recommender, How can I help you?')

if user_query:
    # Defining the prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the questions based on the provided context only.
        Please provide the response based on the question.
        
        Handle typos and variations in questions asked.
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )

    # document chain - stuff doc chain to pass the prompts along with model
    documents_chain = create_stuff_documents_chain(llm_model, prompt)
    kg_retriever = knowledgeBase.as_retriever()
    retrieval_chain = create_retrieval_chain(kg_retriever, documents_chain)

    # Perform retrieval and display results
    response = retrieval_chain.invoke({'input': user_query})
    st.subheader('Top Recommendations for your query:\n')

    st.write("")
    st.markdown(response['answer'])
