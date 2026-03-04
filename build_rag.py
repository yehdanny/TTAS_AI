"""
build_rag.py  (v2)
Step 1: 將 TTAS PDF 指引解析並建立 ChromaDB 向量索引。

v2 改進：
1. 主訴分級表：改用 extract_tables() 取結構化數據，以「主訴」為單位生成 chunk
2. 總表（調節變數）：每頁一個 chunk，保留完整語意
3. 修正版 PDF 覆蓋原版對應條目（按編碼 key），總表修正版覆蓋表六

執行方式：
    python build_rag.py
"""

import os
import re

import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "data", "rag_knowledge")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# ── 參數 ───────────────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "ttas_guidelines"
BATCH_SIZE = 32
INSTRUCTION_PREFIX = "为这个句子生成表示以用于检索相关文章："

# ── 主訴分級表 PDF 配置 ───────────────────────────────────────────────────────
# 每個元素: (主版本檔名, 修正版檔名 or None, 來源標籤)
COMPLAINT_PDFS = [
    (
        "急診五級檢傷分類基準修正版-成人非外傷 (1).pdf",
        "急診五級檢傷分類基準修正版-成人非外傷(1080611).pdf",
        "成人非外傷",
    ),
    (
        "急診五級檢傷分類基準修正版-兒童.pdf",
        "急診五級檢傷分類基準修正版-兒童(第二次修正).pdf",
        "兒童",
    ),
    (
        "急診五級檢傷分類基準修正版-外傷.pdf",
        None,
        "外傷",
    ),
    (
        "急診五級檢傷分類基準修正版-環境.pdf",
        None,
        "環境",
    ),
]

# ── 總表 PDF 配置 ─────────────────────────────────────────────────────────────
# (主版本, 修正版 or None)  — 修正版覆蓋「表六」（兒童首要調節變數）
REFERENCE_PDFS = (
    "急診五級檢傷分類基準修正版-總表.pdf",
    "急診五級檢傷分類基準修正版-總表(第二次修正).pdf",
)


# ── 工具函式 ──────────────────────────────────────────────────────────────────
def _cell(row, idx: int) -> str:
    """安全取得表格欄位並轉為 str，None 或超界回傳空字串。"""
    try:
        v = row[idx]
    except IndexError:
        return ""
    return "" if v is None else str(v).strip()


# ── 主訴分級表解析 ────────────────────────────────────────────────────────────
def parse_complaint_pdf(pdf_path: str) -> dict:
    """
    解析主訴分級表 PDF，用 extract_tables() 提取結構化數據。

    PDF 表格欄位結構（各版本列數不同，但以下位置固定）：
        row[0]  = 大分類名稱（可能 None，需繼承上一行）
        row[1]  = 標準主訴名稱（可能 None，需繼承上一行）
        row[2]  = 判定依據中文名稱
        row[-3] = TTAS 檢傷級數（"1"–"5"）
        row[-2] = 次要調節變數（"＊" 或空白）
        row[-1] = 編碼（如 "E010101"）

    回傳 dict: { code: {category, complaint, criteria, level, secondary, code} }
    """
    entries = {}
    last_category = ""
    last_complaint = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    if not row or len(row) < 6:
                        continue

                    code = _cell(row, -1)
                    # 有效編碼：第1字元為字母、第2字元為數字，總長度 >= 6（如 "E010101"）
                    if len(code) < 6 or not (code[0].isalpha() and code[1].isdigit()):
                        continue

                    raw_cat = _cell(row, 0)
                    raw_comp = _cell(row, 1)
                    raw_crit = _cell(row, 2)
                    raw_level = _cell(row, -3)
                    raw_sec = _cell(row, -2)

                    # 清理換行符（表格跨行合併）
                    cat = raw_cat.replace("\n", "").replace(" ", "").strip()
                    comp = raw_comp.replace("\n", "").strip()
                    crit = raw_crit.replace("\n", "").strip()

                    # 繼承上一行的大分類 / 主訴（None 或空白則繼承）
                    if cat:
                        last_category = cat
                    else:
                        cat = last_category

                    if comp:
                        last_complaint = comp
                    else:
                        comp = last_complaint

                    # 只接受有效的 TTAS 級數（1–5）
                    if raw_level not in ("1", "2", "3", "4", "5"):
                        continue

                    # 過濾表頭行或無效判定依據
                    if not crit or "TTAS" in crit or "判定依據" in crit:
                        continue

                    has_secondary = "＊" in raw_sec or "*" in raw_sec

                    entries[code] = {
                        "category": cat,
                        "complaint": comp,
                        "criteria": crit,
                        "level": int(raw_level),
                        "secondary": has_secondary,
                        "code": code,
                    }

    return entries


def merge_complaint_pdfs(base_path: str, rev_path: str | None) -> dict:
    """讀主版本，若有修正版則用修正版條目覆蓋（相同編碼 key）。"""
    entries = parse_complaint_pdf(base_path)
    if rev_path and os.path.exists(rev_path):
        rev_entries = parse_complaint_pdf(rev_path)
        print(f"    修正版覆蓋 {len(rev_entries)} 個條目")
        entries.update(rev_entries)
    return entries


def entries_to_chunks(entries: dict, source_label: str) -> list[dict]:
    """
    將條目按「主訴」分組，每個主訴生成一個 chunk。
    格式：
        【主訴】E0101昆蟲螫傷（大分類：E01環境）
        判定依據 → TTAS級數：
        - 重度呼吸窘迫(<90%) → 1級
        - 過去曾出現嚴重過敏反應 → 2級（★需查次要調節變數規則）
        ...
    """
    groups: dict[tuple, list] = {}
    for entry in entries.values():
        key = (entry["category"], entry["complaint"])
        groups.setdefault(key, []).append(entry)

    chunks = []
    for (cat, comp), group_entries in groups.items():
        group_entries.sort(key=lambda x: (x["level"], x["code"]))

        lines = [f"【主訴】{comp}（大分類：{cat}）", "判定依據 → TTAS級數："]
        for e in group_entries:
            suffix = "（★需查次要調節變數規則）" if e["secondary"] else ""
            lines.append(f"- {e['criteria']} → {e['level']}級{suffix}")

        chunks.append({
            "text": "\n".join(lines),
            "source": source_label,
        })

    return chunks


# ── 總表（調節變數）解析 ──────────────────────────────────────────────────────
def parse_reference_pdf(pdf_path: str, source_label: str) -> list[dict]:
    """
    解析總表（首要/次要調節變數）PDF，每頁作為一個獨立 chunk。
    用 extract_text() 取得完整文字，過濾純頁碼行。
    """
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text() or ""
            lines = raw_text.strip().split("\n")
            # 過濾純數字頁碼行
            lines = [ln for ln in lines if not re.fullmatch(r"\s*\d+\s*", ln)]
            text = "\n".join(lines).strip()
            if len(text) < 30:
                continue

            chunks.append({
                "text": text,
                "source": source_label,
                "title": lines[0] if lines else f"page{i+1}",
                "page": i + 1,
            })

    return chunks


def merge_reference_pdfs(base_path: str, rev_path: str | None) -> list[dict]:
    """
    讀取總表主版本，用修正版替換「表六」相關頁面。
    總表(第二次修正) 包含更新後的兒童首要調節變數（表六），覆蓋原版的表六頁。
    """
    base_chunks = parse_reference_pdf(base_path, "總表")

    if rev_path and os.path.exists(rev_path):
        rev_chunks = parse_reference_pdf(rev_path, "總表(第二次修正)")

        # 找到 base_chunks 中所有「表六」相關頁（兒童首要調節變數）
        table6_indices = [
            i for i, c in enumerate(base_chunks)
            if "表六" in c["text"][:80]
        ]
        if table6_indices:
            start = min(table6_indices)
            end = max(table6_indices) + 1
            n_orig = end - start
            print(f"    總表修正版覆蓋 page{start+1}–{end}（表六，{n_orig}頁 → {len(rev_chunks)}頁）")
            base_chunks[start:end] = rev_chunks
        else:
            print("    [警告] 找不到總表表六頁面，跳過修正版覆蓋")

    return base_chunks


# ── 嵌入 ──────────────────────────────────────────────────────────────────────
def embed_with_instruction(
    embedder: SentenceTransformer, texts: list[str]
) -> list[list[float]]:
    prefixed = [f"{INSTRUCTION_PREFIX}{t}" for t in texts]
    embeddings = embedder.encode(
        prefixed,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    )
    return embeddings.tolist()


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    # 1. 載入嵌入模型
    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("Embedding model loaded.\n")

    # 2. 初始化 ChromaDB
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

    # 3. 解析所有主訴分級表
    all_texts: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict] = []

    print("=== 解析主訴分級表 ===")
    for base_fname, rev_fname, label in COMPLAINT_PDFS:
        base_path = os.path.join(PDF_DIR, base_fname)
        rev_path = os.path.join(PDF_DIR, rev_fname) if rev_fname else None
        print(f"\n[{label}] {base_fname}")

        entries = merge_complaint_pdfs(base_path, rev_path)
        chunks = entries_to_chunks(entries, label)
        print(f"  → {len(entries)} 個條目，{len(chunks)} 個主訴 chunk")

        for ci, chunk in enumerate(chunks):
            all_texts.append(chunk["text"])
            all_ids.append(f"{label}_{ci:04d}")
            all_metadatas.append({"source": label})

    # 4. 解析總表
    print("\n=== 解析總表（調節變數）===")
    base_path = os.path.join(PDF_DIR, REFERENCE_PDFS[0])
    rev_path = os.path.join(PDF_DIR, REFERENCE_PDFS[1]) if REFERENCE_PDFS[1] else None
    ref_chunks = merge_reference_pdfs(base_path, rev_path)
    print(f"  → {len(ref_chunks)} 個總表 chunk")

    for ci, chunk in enumerate(ref_chunks):
        all_texts.append(chunk["text"])
        all_ids.append(f"總表_{ci:04d}")
        all_metadatas.append({"source": chunk["source"]})

    # 5. 嵌入並寫入 ChromaDB
    total = len(all_texts)
    print(f"\n共 {total} 個 chunk")
    print("Embedding all chunks (this may take 5–10 minutes on CPU)...")

    embeddings = embed_with_instruction(embedder, all_texts)

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
    print(f"\nRAG index built successfully. {total_stored} chunks in ChromaDB.")
    print(f"Index saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
