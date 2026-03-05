"""
build_rag_v7.py
Step 1 (v7): 建立 ChromaDB 向量索引。

v7 改進（相對於 v2）：
- 主訴 chunk 的 **嵌入向量** 只用「大分類名稱 + 標準主訴名稱」
  （與病人主訴的語意更直接比對，排除判定依據文字的干擾）
- Document 內容仍保留完整 chunk（大分類 + 主訴 + 判定依據 + 級數 + 次要調節）
  供 LLM 用病患生理數值與判定依據比對後決定等級
- 總表 chunk 嵌入方式不變（全文）

執行方式：
    python build_rag_v7.py
"""

import os
import re

import pdfplumber

# 去除「大分類名稱」與「標準主訴名稱」前面的英數代碼（如 A02、A0203、T12）
_CODE_PREFIX_RE = re.compile(r"^[A-Za-z]\d+\s*")


def _strip_code(text: str) -> str:
    return _CODE_PREFIX_RE.sub("", text).strip()
import chromadb
from sentence_transformers import SentenceTransformer

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "data", "rag_knowledge")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db_v7")

# ── 參數 ───────────────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "ttas_v7"
BATCH_SIZE = 32
INSTRUCTION_PREFIX = "为这个句子生成表示以用于检索相关文章："

# ── 主訴分級表 PDF 配置 ───────────────────────────────────────────────────────
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
REFERENCE_PDFS = (
    "急診五級檢傷分類基準修正版-總表.pdf",
    "急診五級檢傷分類基準修正版-總表(第二次修正).pdf",
)


# ── 工具函式 ──────────────────────────────────────────────────────────────────
def _cell(row, idx: int) -> str:
    try:
        v = row[idx]
    except IndexError:
        return ""
    return "" if v is None else str(v).strip()


# ── 主訴分級表解析 ────────────────────────────────────────────────────────────
def parse_complaint_pdf(pdf_path: str) -> dict:
    """
    解析主訴分級表 PDF。

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
                    if len(code) < 6 or not (code[0].isalpha() and code[1].isdigit()):
                        continue

                    raw_cat = _cell(row, 0)
                    raw_comp = _cell(row, 1)
                    raw_crit = _cell(row, 2)
                    raw_level = _cell(row, -3)
                    raw_sec = _cell(row, -2)

                    cat = raw_cat.replace("\n", "").replace(" ", "").strip()
                    comp = raw_comp.replace("\n", "").strip()
                    crit = raw_crit.replace("\n", "").strip()

                    if cat:
                        last_category = cat
                    else:
                        cat = last_category

                    if comp:
                        last_complaint = comp
                    else:
                        comp = last_complaint

                    if raw_level not in ("1", "2", "3", "4", "5"):
                        continue

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
    entries = parse_complaint_pdf(base_path)
    if rev_path and os.path.exists(rev_path):
        rev_entries = parse_complaint_pdf(rev_path)
        print(f"    修正版覆蓋 {len(rev_entries)} 個條目")
        entries.update(rev_entries)
    return entries


def entries_to_chunks(entries: dict, source_label: str) -> list[dict]:
    """
    將條目按「主訴」分組，每個主訴生成一個 chunk。

    v7 新增：每個 chunk 同時輸出：
      - embed_text: 「大分類名稱 + 標準主訴名稱」（作為嵌入向量來源）
      - text: 完整 chunk（大分類 + 主訴 + 判定依據 + 級數 + 次要調節），作為 LLM context

    格式（text）：
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

        # 去除代碼前綴（A02→心臟血管系統, A0203→心悸/不規則心跳）
        cat_clean = _strip_code(cat)
        comp_clean = _strip_code(comp)

        # v7: embed_text 只用標準主訴名稱（去代碼後），與 EXACT 比對基準一致
        embed_text = comp_clean

        # document text: 新格式「大分類-主訴」，供 LLM 比對判定依據
        lines = [f"{cat_clean}-{comp_clean}", "判定依據 → TTAS級數："]
        for e in group_entries:
            suffix = "（★需查次要調節變數規則）" if e["secondary"] else ""
            lines.append(f"- {e['criteria']} → {e['level']}級{suffix}")

        chunks.append({
            "embed_text": embed_text,
            "text": "\n".join(lines),
            "source": source_label,
        })

    return chunks


# ── 總表（調節變數）解析 ──────────────────────────────────────────────────────
def parse_reference_pdf(pdf_path: str, source_label: str) -> list[dict]:
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text() or ""
            lines = raw_text.strip().split("\n")
            lines = [ln for ln in lines if not re.fullmatch(r"\s*\d+\s*", ln)]
            text = "\n".join(lines).strip()
            if len(text) < 30:
                continue

            chunks.append({
                "embed_text": text,  # 總表: embed_text = text（全文）
                "text": text,
                "source": source_label,
                "title": lines[0] if lines else f"page{i+1}",
                "page": i + 1,
            })

    return chunks


def merge_reference_pdfs(base_path: str, rev_path: str | None) -> list[dict]:
    base_chunks = parse_reference_pdf(base_path, "總表")

    if rev_path and os.path.exists(rev_path):
        rev_chunks = parse_reference_pdf(rev_path, "總表(第二次修正)")

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
    all_embed_texts: list[str] = []
    all_doc_texts: list[str] = []
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
        print(f"  embed_text 範例: '{chunks[0]['embed_text'][:60]}'")

        for ci, chunk in enumerate(chunks):
            all_embed_texts.append(chunk["embed_text"])
            all_doc_texts.append(chunk["text"])
            all_ids.append(f"{label}_{ci:04d}")
            all_metadatas.append({"source": label})

    # 4. 解析總表
    print("\n=== 解析總表（調節變數）===")
    base_path = os.path.join(PDF_DIR, REFERENCE_PDFS[0])
    rev_path = os.path.join(PDF_DIR, REFERENCE_PDFS[1]) if REFERENCE_PDFS[1] else None
    ref_chunks = merge_reference_pdfs(base_path, rev_path)
    print(f"  → {len(ref_chunks)} 個總表 chunk")

    for ci, chunk in enumerate(ref_chunks):
        all_embed_texts.append(chunk["embed_text"])
        all_doc_texts.append(chunk["text"])
        all_ids.append(f"總表_{ci:04d}")
        all_metadatas.append({"source": chunk["source"]})

    # 5. 嵌入（使用 embed_text）並寫入 ChromaDB（存 doc_text 為 document）
    total = len(all_embed_texts)
    print(f"\n共 {total} 個 chunk")
    print("Embedding (embed_text only)...")

    embeddings = embed_with_instruction(embedder, all_embed_texts)

    print("Writing to ChromaDB (document = full chunk text)...")
    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)
        collection.add(
            ids=all_ids[i:end],
            documents=all_doc_texts[i:end],   # LLM 看完整 chunk
            embeddings=embeddings[i:end],      # 向量來自 embed_text（大分類+主訴名稱）
            metadatas=all_metadatas[i:end],
        )

    total_stored = collection.count()
    print(f"\nRAG index (v7) built successfully. {total_stored} chunks in ChromaDB.")
    print(f"Index saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
