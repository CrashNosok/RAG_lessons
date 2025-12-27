from dotenv import load_dotenv

load_dotenv()

# ================================================================================================================

# Загрузка документов
import os
from langchain_community.document_loaders import SitemapLoader, RecursiveUrlLoader

os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

SITEMAP_URL = "https://antarcticwallet.com/sitemap.xml"
ROOT_URL = "https://antarcticwallet.com/"

# # 1) Загружаем все страницы из sitemap
# sitemap_loader = SitemapLoader(
#     web_path=SITEMAP_URL,
#     filter_urls=[ROOT_URL],  # на всякий случай ограничиваем доменом
# )
# sitemap_docs = sitemap_loader.load()
#
# # 2) Дополнительно рекурсивно обходим сайт от корня
# recursive_loader = RecursiveUrlLoader(
#     url=ROOT_URL,
#     max_depth=2,          # глубину при желании можно увеличить
#     prevent_outside=True  # не выходим за пределы домена
# )
# recursive_docs = recursive_loader.load()
#
# # 3) Объединяем всё в один список документов для RAG
# docs = sitemap_docs + recursive_docs

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader('https://antarcticwallet.com/faq')
docs = loader.load()

print(f"Total documents: {len(docs)}")
print(f"Total characters: {sum(len(doc.page_content) for doc in docs)}")

# ================================================================================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.HTML,   # учитываем структуру HTML
    chunk_size=1200,          # немного больше, т.к. структура сохраняется лучше
    chunk_overlap=200,
)

splits = text_splitter.split_documents(docs)

# ================================================================================================================

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

# Базовая модель
base_embeddings = HuggingFaceEmbeddings(
    model_name="ai-forever/ru-en-RoSBERTa"
)

class PrefixedEmbeddings(Embeddings):
    def __init__(self, base, query_prefix="", doc_prefix=""):
        self.base = base
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def embed_documents(self, texts):
        texts_prefixed = [self.doc_prefix + t for t in texts]
        return self.base.embed_documents(texts_prefixed)

    def embed_query(self, text):
        return self.base.embed_query(self.query_prefix + text)

embeddings = PrefixedEmbeddings(
    base_embeddings,
    query_prefix="search_query: ",
    doc_prefix="search_document: ",
)

# ================================================================================================================

from pathlib import Path
from langchain_chroma import Chroma

persist_directory = "./chroma_db"

if Path(persist_directory).exists():
    # Индекс уже есть — просто загружаем
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
else:
    # Первый запуск — создаём индекс
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

# ================================================================================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",  # вместо простого similarity
    search_kwargs={
        "k": 8,          # сколько документов вернуть в итоге
        "fetch_k": 32,   # из скольких кандидатов выбирать (больше = разнообразнее)
        # при желании можно добавить lambda_mult для тонкой настройки
        # lambda_mult – это параметр MMR, который задаёт баланс между релевантностью документов запросу и их разнообразием: чем ближе значение к 1, тем сильнее приоритет близости к запросу, чем ближе к 0 – тем важнее разнообразие результатов.
        # "lambda_mult": 0.8,
    },
)

# ================================================================================================================

def format_docs(docs, max_chars: int = 8000):
    formatted = []
    total_len = 0

    for doc in docs:
        source = doc.metadata.get("source", "unknown_source")
        page = doc.metadata.get("page", None)

        header = f"Source: {source}"
        if page is not None:
            header += f" | Page: {page}"

        text = doc.page_content.strip()
        block = f"{header}\n{text}"

        # если следующий блок слишком раздует контекст — останавливаемся
        if total_len + len(block) > max_chars:
            break

        formatted.append(block)
        total_len += len(block)

    return "\n\n---\n\n".join(formatted)

# ================================================================================================================

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Ты помощник, который отвечает СТРОГО НА РУССКОМ ЯЗЫКЕ. "
        "Используй только информацию из предоставленного контекста, не придумывай факты. "
        "Если ответа в контексте нет или данных недостаточно, честно скажи, что не нашёл ответа в базе. "
        "При необходимости можешь упоминать источник в формате из заголовка (Source и Page). "
        "Отвечай кратко и по делу, обычно до 5–7 предложений."
    ),
    # историю можно не заполнять, но структура уже есть
    MessagesPlaceholder("history"),
    (
        "human",
        "Контекст:\n{context}\n\nВопрос: {question}"
    ),
])

# ================================================================================================================

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key="None",
    base_url="http://127.0.0.1:11434/v1",
    # model="hf.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M",
    model="gemma3:4b",

    # важные параметры для RAG
    temperature=0.2,      # меньше фантазии
    max_tokens=512,       # контролируем длину ответа
    top_p=0.9,
)

# ================================================================================================================

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def ensure_context(input_dict: dict) -> dict:
    """
    Если retriever не нашёл ничего полезного и контекст пустой,
    явно помечаем это в контексте, чтобы модель не фантазировала.
    """
    context = input_dict.get("context", "").strip()
    if not context:
        input_dict["context"] = (
            "Контекст пуст: ретривер не нашёл ни одного подходящего фрагмента. "
            "Если ответ важен, лучше явно сказать пользователю об этом."
        )
    return input_dict

rag_chain = (
    {
        # контекст и вопрос теперь приходят извне в виде dict
        "context": lambda d: d.get("context", ""),
        "question": lambda d: d.get("question", ""),
        "history": lambda _: [],  # пока истории нет – передаём пустой список
    }
    | RunnableLambda(ensure_context)   # защита от пустого контекста
    | prompt
    | llm
    | StrOutputParser()
).with_config(run_name="rag_chain")

# ================================================================================================================

from langsmith import traceable


@traceable(name="AW_answer_question")
def answer_question(question: str, context: str) -> str:
    """
    Основная точка входа в RAG.
    Эту функцию мы будем отслеживать в LangSmith как корневой run.
    В INPUT корневого run'а будут поля question и context.
    """
    inputs = {
        "question": question,
        "context": context,
    }
    return rag_chain.invoke(inputs)

# ================================================================================================================

if __name__ == "__main__":
    questions = [
        "Как записать видос с первого раза и заработать 1000000$?",
        "Кто чаще всего использует Antarctic Wallet",
        "Для каких целей чаще всего используется Antarctic Wallet",
        "Что такое Antarctic Wallet?",
        "Как начать пользоваться Antarctic Wallet?",
        # "Какие комиссии есть в Antarctic Wallet?",
        # "Нужно ли проходить KYC?",
        # "Назови часы работы технической поддержки",
    ]

    for q in questions:
        print("Вопрос:", q)

        # 1) достаём документы из ретривера
        retrieved_docs = retriever.invoke(q)
        # 2) форматируем их в строку контекста
        ctx = format_docs(retrieved_docs)

        answer = answer_question(question=q, context=ctx)
        print("Ответ:", answer)
        print("================================================================")
