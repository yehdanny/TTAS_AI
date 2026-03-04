"""
build_rag_v3.py
從 data/chunks/ 下的手動 Python chunk 檔案建立 ChromaDB 向量索引。
Collection name: ttas_v3

chunk 格式：
    {
        "metadata": {"category": ..., "sub_category": ..., "target": ...},
        "query_text": "精簡主訴關鍵詞",
        "content": "- 一級 (復甦急救)：...\n- 二級 (危急)：...",
        "remarks": "可調節分級項目 : ..."
    }

執行方式：
    python build_rag_v3.py
"""

import os
import sys

import chromadb
from sentence_transformers import SentenceTransformer

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_DIR = os.path.join(BASE_DIR, "data", "chunks")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# ── 參數 ───────────────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "ttas_v3"
BATCH_SIZE = 32
INSTRUCTION_PREFIX = "为这个句子生成表示以用于检索相关文章："

# ── 載入手動 chunks ────────────────────────────────────────────────────────────
sys.path.insert(0, CHUNKS_DIR)

from chunks_list_A01_A13_non_trauma import non_trauma_chunks
from chunks_list_P01_P13_pediatric import pediatric_chunks
from chunks_list_T01_14_trauma import trauma_chunks
from chunks_list_E01_environment import environment_chunks
from chunk_list_adjustment import adjustment_chunks


def chunk_to_text(chunk: dict) -> str:
    """
    將 chunk dict 轉為純文字供 LLM 參考。
    格式：
        【主訴】昆蟲螫傷
        - 一級 (復甦急救)：...
        - 二級 (危急)：...
        可調節分級項目 : ...
    """
    query = chunk.get("query_text", "").strip()
    content = chunk.get("content", "").strip()
    remarks = chunk.get("remarks", "").strip()

    parts = [f"【主訴】{query}"]
    if content:
        parts.append(content)
    if remarks:
        parts.append(remarks)

    return "\n".join(parts)


def main():
    # 1. 合併所有 chunks
    all_chunks = (
        non_trauma_chunks
        + pediatric_chunks
        + trauma_chunks
        + environment_chunks
        + adjustment_chunks
    )
    print(f"成人非外傷: {len(non_trauma_chunks)}")
    print(f"兒童: {len(pediatric_chunks)}")
    print(f"外傷: {len(trauma_chunks)}")
    print(f"環境: {len(environment_chunks)}")
    print(f"調節變數(總表): {len(adjustment_chunks)}")
    print(f"合計: {len(all_chunks)} chunks\n")

    # 2. 載入嵌入模型
    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("Embedding model loaded.\n")

    # 3. 初始化 ChromaDB
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' exists. Deleting and rebuilding...")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 4. 建立文字與 metadata 列表
    all_texts = []
    all_ids = []
    all_metadatas = []

    for i, chunk in enumerate(all_chunks):
        text = chunk_to_text(chunk)
        meta = chunk.get("metadata", {})
        # ChromaDB metadata 只能有 str/int/float/bool
        clean_meta = {
            "source": str(meta.get("category", "")),
            "sub_category": str(meta.get("sub_category", "")),
            "target": str(meta.get("target", "")),
            "query_text": str(chunk.get("query_text", "")),
        }
        all_texts.append(text)
        all_ids.append(f"v3_{i:04d}")
        all_metadatas.append(clean_meta)

    # 5. 嵌入（加指令前綴）
    total = len(all_texts)
    print(f"Embedding {total} chunks (instruction prefix added)...")
    prefixed = [f"{INSTRUCTION_PREFIX}{t}" for t in all_texts]
    embeddings = embedder.encode(
        prefixed,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    ).tolist()

    # 6. 批次寫入 ChromaDB
    print("Writing to ChromaDB...")
    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)
        collection.add(
            ids=all_ids[i:end],
            documents=all_texts[i:end],
            embeddings=embeddings[i:end],
            metadatas=all_metadatas[i:end],
        )

    total_stored = collection.count()
    print(f"\nRAG v3 index built successfully. {total_stored} chunks in ChromaDB.")
    print(f"Collection: '{COLLECTION_NAME}'")
    print(f"Index saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
