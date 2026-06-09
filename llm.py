from openai import OpenAI

GENERATION_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 768

client = OpenAI()


def embed_documents(texts: list[str]) -> list[list[float]]:
    result = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSION,
    )

    return [item.embedding for item in result.data]


def embed_query(text: str) -> list[float]:
    result = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
        dimensions=EMBEDDING_DIMENSION,
    )

    return result.data[0].embedding


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

    response = client.responses.create(
        model=GENERATION_MODEL,
        input=prompt,
    )

    if response.output_text:
        return response.output_text

    return "文書内に該当情報がありません"


def build_extract_answer(retrieved_docs: list[str]) -> str:
    if not retrieved_docs:
        return "文書内に該当情報がありません。"

    lines = ["生成AIによる要約を利用できないため、関連文書をそのまま表示します。"]

    for i, doc in enumerate(retrieved_docs[:2], start=1):
        lines.append(f"\n【関連文書{i}】\n{doc}")

    return "\n".join(lines)