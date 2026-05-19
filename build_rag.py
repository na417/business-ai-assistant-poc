import os

import chromadb

from llm import embed_documents

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "manual_chunks"
TEXT_FILE = "docs/manual.txt"


def split_text(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()] #空行区切りの段落単位で分割する
    return chunks


def main():
    if not os.getenv("GEMINI_API_KEY"):
        raise EnvironmentError("GEMINI_API_KEY が設定されていません。")

    if not os.path.exists(TEXT_FILE):
        raise FileNotFoundError(f"{TEXT_FILE} が見つかりません。")

    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        raise ValueError("manual.txt が空です。内容を確認してください。")

    chunks = split_text(text)

    if not chunks:
        raise ValueError("チャンクが作成できませんでした。manual.txt の内容を確認してください。")

    embeddings = embed_documents(chunks) #段落ごとに埋め込みベクトルを作っている。

    if len(chunks) != len(embeddings):
        raise ValueError("チャンク数と埋め込み数が一致しません。")

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": TEXT_FILE, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print("ChromaDB に文書を登録しました。")
    print(f"登録チャンク数: {len(chunks)}")


if __name__ == "__main__":
    main()