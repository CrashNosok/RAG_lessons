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

from langchain_core.prompts import ChatPromptTemplate


# # добавляет лишнюю информацию (при использовании hf.co/bartowski/Mistral-Nemo-Instruct)
# question_rewrite_prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "Ты AI-агент, который подготавливает пользовательские вопросы для RAG-системы, "
#         "отвечающей на вопросы по сервису Antarctic Wallet на основе его публичных веб-ресурсов "
#         "(сайт, FAQ, справка).\n\n"
#         "Твоя задача:\n"
#         "1) Понять, о чём именно вопрос пользователя.\n"
#         "2) Оценить, подходит ли формулировка для семантического поиска по документации.\n"
#         "3) Если вопрос слишком общий, разговорный, короткий или содержит местоимения "
#         "без контекста (например, «оно», «там», «это»), переписать его в чёткий, "
#         "формальный и самодостаточный вопрос.\n\n"
#         "Правила переписывания:\n"
#         "- Всегда явно упоминай «Antarctic Wallet», если это релевантно.\n"
#         "- Сохраняй исходный смысл вопроса, не добавляй новых фактов.\n"
#         "- Формулируй вопрос так, чтобы по нему можно было найти ответ в документации или FAQ.\n"
#         "- Используй нейтральный, технический стиль без разговорных выражений.\n\n"
#         "Если исходный вопрос уже хорошо подходит для поиска — верни его без изменений.\n"
#         "Отвечай ТОЛЬКО итоговой формулировкой вопроса, без кавычек, пояснений и комментариев."
#     ),
#     (
#         "human",
#         "Исходный вопрос: {question}"
#     )
# ])

question_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ты подготавливаешь вопрос для RAG по Antarctic Wallet.\n"
     "Верни результат СТРОГО в JSON без лишнего текста.\n"
     "Формат: {{\"question\": \"...\"}}\n"
     "Правила:\n"
     "- Если вопрос хороший — верни его как есть.\n"
     "- Если плохой — перепиши в чёткий самодостаточный вопрос.\n"
     "- Не добавляй префиксы вроде 'Формулировка для поиска:'\n"
     "- Не используй кавычки-ёлочки «» и двойные кавычки вокруг всего вопроса.\n"
     "- Всегда упоминай 'Antarctic Wallet', если уместно.\n"
    ),
    ("human", "{question}")
])

# ================================================================================================================

# from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

question_rewrite_chain = (
    question_rewrite_prompt
    | llm
    # | StrOutputParser()
    | JsonOutputParser() # используем JsonOutputParser так как возвращаем JSON
)

# ================================================================================================================

def rewrite_question_if_needed(question: str) -> str:
    data = question_rewrite_chain.invoke({"question": question})
    return data["question"].strip()

    # rewritten = question_rewrite_chain.invoke({"question": question}).strip()
    # return rewritten

# ================================================================================================================

from langchain.tools import tool

@tool
def rewrite_question(question: str) -> str:
    """Optimize user question for RAG search"""
    return rewrite_question_if_needed(question)

# ================================================================================================================

if __name__ == "__main__":
    print(rewrite_question_if_needed('какие комиссии?'))
    print(rewrite_question_if_needed('нужно ли kyc'))
    print(rewrite_question_if_needed('часы работы поддержки'))
