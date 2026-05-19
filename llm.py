from google import genai
from google.genai import types

GENERATION_MODEL = "gemini-3-flash-preview"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768

client = genai.Client()


def embed_documents(texts: list[str]) -> list[list[float]]:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )

    return [embedding.values for embedding in result.embeddings]


def embed_query(text: str) -> list[float]:
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[text],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )

    return result.embeddings[0].values


def generate_rag_answer(user_question: str, retrieved_docs: list[str]) -> str:
    context = "\n\n".join(retrieved_docs)

    prompt = f"""
あなたは社内業務アシスタントです。
必ず参考文書だけを根拠に回答してください。
参考文書にない内容は推測せず、
必ず「文書内に該当情報がありません」と返してください。

【参考文書】
{context}

【質問】
{user_question}
"""

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    if hasattr(response, "text") and response.text:
        return response.text

    return "文書内に該当情報がありません"


def build_extract_answer(retrieved_docs: list[str]) -> str:
    if not retrieved_docs:
        return "文書内に該当情報がありません。"

    lines = ["生成AIによる要約を利用できないため、関連文書をそのまま表示します。"]

    for i, doc in enumerate(retrieved_docs[:2], start=1):
        lines.append(f"\n【関連文書{i}】\n{doc}")

    return "\n".join(lines)