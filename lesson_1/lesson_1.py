import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


OLLAMA_EMBEDDING_BASE_URL = 'http://127.0.0.1:11434'
OLLAMA_OPENAI_BASE_URL = 'http://127.0.0.1:11434/v1'

EMBEDDING_MODEL = 'hf.co/Casual-Autopsy/snowflake-arctic-embed-l-v2.0-gguf:Q4_K_M'
LLM_MODEL = 'hf.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M'

persist_directory = "./chroma_db"
os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

loader = WebBaseLoader('https://antarcticwallet.com/faq')
docs = loader.load()
print(f"Total characters: {len(docs[0].page_content)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_EMBEDDING_BASE_URL,
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=persist_directory,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Ты помощник, который ОТВЕЧАЕТ СТРОГО НА РУССКОМ ЯЗЫКЕ. "
        "Даже если контекст или вопрос частично на других языках (английский, немецкий и т.п.), "
        "ты ВСЁ РАВНО отвечаешь только на русском, используя кириллицу. "
        "Не используй слова и фразы на других языках, кроме общепринятых имен собственных и названий. "
        "Используй следующие фрагменты контекста для ответа на вопрос. "
        "Если ты не знаешь ответа, просто скажи, что не знаешь. "
        "Используй максимум три предложения и будь лаконичным."
    ),
    (
        "human",
        "Контекст: {context}\n\nВопрос: {question}"
    ),
])

llm = ChatOpenAI(
    api_key='None',
    base_url=OLLAMA_OPENAI_BASE_URL,
    model=LLM_MODEL
)

rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("Что такое Antarctic Wallet?"))
print(rag_chain.invoke("Как начать пользвоваться Antarctic Wallet?"))
print(rag_chain.invoke("Какие комиссии есть в Antarctic Wallet?"))
print(rag_chain.invoke("Нужно ли проходить KYC?"))
print(rag_chain.invoke("Назови часы работы технической поддержки"))
