import getpass
import os
from dotenv import load_dotenv


os.environ["MISTRAL_API_KEY"] = "jL1BDiR52K7rDho3rontdYioto5Yb5n9"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(model="mistral-large-latest")

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings


load_dotenv()

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)



   

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")

