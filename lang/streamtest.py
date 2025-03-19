
import streamlit as st


from langchain_ollama import ChatOllama
import os
import faiss
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.prompts import (
                                        SystemMessagePromptTemplate,
                                        HumanMessagePromptTemplate,
                                        ChatPromptTemplate,
                                        MessagesPlaceholder
                                        )
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyMuPDFLoader



load_dotenv('./../.env')
parse = StrOutputParser()
base_url = "127.0.0.1:11434"
# chatbot = "deepseek-r1:1.5b"
chatbot = "llama3.2:latest"
db_path = "./docs"
db_name = "cerebro"
folder_path = "./files"
st.title("💻 Work Wise")
st.subheader('Bem vindo ao Work Wise, usa plataforma interativa!')




base_url = "127.0.0.1:11434"
model = "llama3.2:latest"
user_id = "user_id"



embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url=base_url)

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store= FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)



def on_change():
    #pega todos os arquivos pdf. passa para ser cortada em chunks
    pdfs = []
    for root, dirs, files in os.walk("./files"):
        print(root, dirs, files)
        for file in files:
            if file.endswith(".pdf"):
                pdfs.append(os.path.join(root,file))
    docs = []
    for pdf in pdfs:
        loader = PyMuPDFLoader(pdf)
        temp = loader.load()
        docs.extend(temp)
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    chunks = text_splitter.split_documents(docs)
    # print(chunks)
    vector_store.add_documents(documents=chunks)
    vector_store.save_local(folder_path=db_path,index_name=db_name)
    
on_change()
    
def save_uploaded_file(uploadedfile, folder_path):
    with open(os.path.join(folder_path, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success(f"Saved file :{uploadedfile.name} in {folder_path}")

uploaded_file = st.file_uploader("Upload a file", type=[ "pdf"])

if st.button("Submit"):
   if uploaded_file is not None:
        save_uploaded_file(uploaded_file, folder_path)
        on_change()


def format_text(docs):
    print("doc passed")
    text = '\n\n'.join([doc.page_content for doc in docs])  
    if text:
        print("text exists")
    return text



def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("Start New Conversation"):
    st.session_state.chat_history = []
    history = get_session_history(user_id)
    history.clear()


for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


### LLM Setup
llm = ChatOllama(base_url=base_url, model=model)



_prompt = """Anwer from context provided only. If you don't have the information to answer a question, say 'I don't know'. 
### Context:
{context}
### Question:
{input}
### Answer"""




system = SystemMessagePromptTemplate.from_template("You are helpful assistant.")
human = HumanMessagePromptTemplate.from_template(_prompt)

messages = [system, human]

prompt = ChatPromptTemplate(messages)

retriever = vector_store.as_retriever(
       search_type = "mmr", search_kwargs= {'k':5, 'fetch_k': 20, 'lambda_mult': .6}
    )

# chain = prompt | llm | StrOutputParser()
chain = (
    {"context" : retriever | format_text , "input": RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)

# runnable_with_history = RunnableWithMessageHistory(chain, get_session_history, 
#                                                    input_messages_key='input', 
#                                                    history_messages_key='history')

def chat_with_llm(session_id, input):
    for output in chain.stream(input, config={'configurable': {'session_id': session_id}}):
        yield output



prompt = st.chat_input("What is up?")
# st.write(prompt)

if prompt:
    # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",prompt)
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(chat_with_llm(user_id, prompt))

    st.session_state.chat_history.append({'role': 'assistant', 'content': response})