import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
# from langchain.llms import openai
from htmlTemplate import css, bot_template, user_template
from langchain_community.llms import huggingface_hub
# from langchain_community.chat_models import hu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
from langchain_community.llms import huggingface_pipeline
import torch



checkpoint = "/Users/llm/T-YT/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype = torch.float32
)

@st.cache_resource
def llm_pipline():
    pipe = pipeline(
        "text2text-generation",
        model = base_model,
        tokenizer=tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p = 0.95
    )
    local_llm = huggingface_pipeline(pipeline = pipe)
    return local_llm





def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_text_chunks(text):
    text_spilitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_spilitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # llm = huggingface_hub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    llm = llm_pipline()
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True)
    conversation_chain = create_history_aware_retriever(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain



def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    # st.session_state.chat_history = response['chat_history']

    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    #     else:
    #         st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    st.write(response)

def main():
   load_dotenv()
   st.set_page_config(page_title='Chat with multiple PDFs', page_icon=":books:")

   st.write(css, unsafe_allow_html=True)
   
   if "conversation" not in st.session_state:
       st.session_state.conversation = None
#    if "chat_history" not in st.session_state:
#        st.session_state.chat_history = None

   st.header("Chat with multiple PDFs :books:")
   user_question = st.text_input("Ask your questions")
   if user_question:
       handle_user_input(user_question)

   st.write(user_template.replace("{{MSG}}", "hello robot"), unsafe_allow_html=True)
   st.write(bot_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True)

   with st.sidebar:
       st.subheader("Your documents")
       pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process' ", accept_multiple_files=True)
       if st.button("Process"):
           with st.spinner("Processing"):
               #get pdfs
               raw_text = get_pdf_text(pdf_docs)
            #    st.write(raw_text)

               #get chunks
               text_chunks = get_text_chunks(raw_text)
            #    st.write(text_chunks)

               #create vector store
               vectorstore = get_vectorstore(text_chunks)
               
            #    #conversation chain
               st.session_state.conversation = get_conversation_chain(vectorstore)

    
    

if __name__ == '__main__':
    main()