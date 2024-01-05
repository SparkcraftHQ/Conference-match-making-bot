import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import chromadb
import os


def generate_response(uploaded_file, openai_api_key, query_text):


    import pandas as pd
    df = pd.read_csv('1126_1416_dev_sheet.csv')
    column_names = df.columns.tolist()
    print('column names: ',column_names,'\n---')
    profiles = 'Name: ' + df['Event Attendees Names'] + ' \n Details' + df['Details about the event attendee']
    profiles = profiles[:10] # TODO: remove this line 
    from langchain.schema import Document
    docs = []
    for profile in profiles: 
        docs.append(Document(page_content=profile, metadata={"source": "local"}))
    
    # Select embeddings
    embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # load it into Chroma
    db = Chroma.from_documents(docs, embedding_function)
    
    # Create retriever interface
    retriever = db.as_retriever()
    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
    return qa.run(query_text)



# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload
uploaded_file = st.file_uploader('Upload an article', type='csv')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.')

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    # get information with hard rules
    # openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    # submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    
    # get information with default input
    openai_api_key = st.text_input('OpenAI API Key', type='password')
    submitted = st.form_submit_button('Submit', disabled=not(query_text))

    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
