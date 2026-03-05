"""
main_v8.py
Step 2 (v8): RAG TOP-5 + LLM Reranking + 結構化兩階段推理。

v8 相較於 v7 的改進：
1. RAG 取 TOP-5 主訴 chunks（而非 TOP-3）
2. 新增 LLM Reranking step：給 LLM 看 5 個候選主訴分類，選出最符合病患主訴的一個
3. 後續流程與 v7 相同（criterion selection → rule-based grade → Stage 2 if ★）

執行方式：
    python main_v8.py           # 350 筆分層抽樣
    python main_v8.py --full    # 全量推理
"""

import matplotlib
matplotlib.use("Agg")

import os
import re
import sys
import logging
import warnings
from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    cohen_kappa_score,
)
from sentence_transformers import SentenceTransformer
import chromadb
from llama_cpp import Llama

warnings.filterwarnings("ignore")

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "patient_data", "total_data.xls")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db_v7")   # 沿用 v7 index
MODEL_PATH = os.path.join(BASE_DIR, "model", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
RESULTS_DIR = os.path.join(BASE_DIR, "results_v8")

# ── 參數 ───────────────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "ttas_v7"
INSTRUCTION_PREFIX = "为这个句子生成表示以用于检索相关文章："
SAMPLE_PER_LEVEL = 70
RANDOM_STATE = 42

# ── RAG 參數 ──────────────────────────────────────────────────────────────────
PEDIATRIC_AGE_CUTOFF = 18
N_RETRIEVAL_COMPLAINT = 5          # v8: TOP-5 供 reranking
ADULT_COMPLAINT_SOURCES = ["外傷", "成人非外傷", "環境"]
PEDIATRIC_COMPLAINT_SOURCES = ["外傷", "兒童", "環境"]
ALL_COMPLAINT_SOURCES = ["外傷", "成人非外傷", "兒童", "環境"]
N_RETRIEVAL_REFERENCE = 2
REFERENCE_SOURCES = ["總表", "總表(第二次修正)"]

# ── Prompt 模板 ────────────────────────────────────────────────────────────────

# Reranking：LLM 從 TOP-5 候選中選出最符合病患主訴的主訴分類
SYSTEM_RERANK = """\
你是一位台灣急診室的資深檢傷護理師，專精於TTAS（台灣急診五級檢傷分類制度）。
根據病患主訴，從下列 TTAS 主訴分類候選中選出最符合的一項。
只回答編號（一個整數），不要輸出任何其他文字。"""

USER_RERANK = """\
【病患主訴】
{complaint}

【TTAS 主訴分類候選清單】
{candidates}

哪一項最符合此病患主訴？只回答編號："""

# Criterion Selection：LLM 逐條判斷可成立的依據，取最緊急的一條
SYSTEM_SELECTION = """\
你是一位台灣急診室的資深檢傷護理師，專精於TTAS（台灣急診五級檢傷分類制度）。
你的任務是：
1. 逐條檢查判定依據清單，找出所有根據病患生理資料可以成立的條目。
2. 在所有可成立的條目中，選出級數數字最小的那一條（最緊急）。
3. 只回答該條的編號（一個整數），不要輸出任何其他文字。"""

USER_SELECTION = """\
【病患生理資料】
{vitals}

【判定依據清單】
{criteria_list}

請逐條判斷哪些條目對此病患可成立，並回答其中級數最小（最緊急）的那一條編號："""

# Stage 2：次要調節變數
SYSTEM_SECONDARY = """\
你是一位台灣急診室的資深檢傷護理師，專精於TTAS次要調節變數規則。
你的任務是：根據病患生理資料與次要調節變數規則，決定最終的 TTAS 等級（1–5）。
只輸出一個阿拉伯數字（1–5），不要輸出其他文字。"""

USER_SECONDARY = """\
【病患生理資料】
{vitals}

【已匹配的判定依據】
{selected_criteria}（初步等級：{initial_level}級）

【次要調節變數規則】
{reference_context}

根據以上次要調節變數規則，此病患的最終 TTAS 等級為幾級？
只輸出數字1、2、3、4或5："""


# ── 工具函式 ──────────────────────────────────────────────────────────────────
def safe_val(val, default="不詳") -> str:
    if pd.isna(val):
        return default
    s = str(val).strip()
    return s if s else default


def _parse_yyyymmdd(val) -> "date | None":
    try:
        s = str(int(val))
        if len(s) == 8:
            return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except (ValueError, TypeError):
        pass
    return None


def _calc_age(birth_val, emergency_val) -> "int | None":
    bd = _parse_yyyymmdd(birth_val)
    ed = _parse_yyyymmdd(emergency_val)
    if bd is None or ed is None:
        return None
    age = ed.year - bd.year - ((ed.month, ed.day) < (bd.month, bd.day))
    return max(age, 0)


GENDER_MAP = {"M": "男", "F": "女", "m": "男", "f": "女"}


def build_rag_query(row: pd.Series) -> str:
    return safe_val(row.get("病人主訴", np.nan))


def build_vitals_text(row: pd.Series) -> str:
    parts = []

    parts.append(f"主訴：{safe_val(row.get('病人主訴', np.nan))}")

    age = _calc_age(row.get("生日"), row.get("急診日期"))
    if age is not None:
        group = "兒童" if age < PEDIATRIC_AGE_CUTOFF else "成人"
        parts.append(f"年齡：{age}歲（{group}）")

    gender_raw = safe_val(row.get("性別", np.nan))
    if gender_raw != "不詳":
        parts.append(f"性別：{GENDER_MAP.get(gender_raw, gender_raw)}")

    for field, label, unit in [
        ("體溫", "體溫", "°C"), ("收縮壓", "收縮壓", "mmHg"),
        ("舒張壓", "舒張壓", "mmHg"), ("脈搏", "脈搏", "次/分"),
        ("呼吸", "呼吸", "次/分"), ("SAO2", "血氧", "%"),
    ]:
        v = safe_val(row.get(field, np.nan))
        if v != "不詳":
            parts.append(f"{label}：{v}{unit}")

    # MAP（平均動脈壓）= (SBP - DBP) / 3 + DBP
    try:
        sbp = float(row.get("收縮壓"))
        dbp = float(row.get("舒張壓"))
        if not (np.isnan(sbp) or np.isnan(dbp)):
            map_val = (sbp - dbp) / 3 + dbp
            parts.append(f"MAP（平均動脈壓）：{map_val:.1f}mmHg")
    except (TypeError, ValueError):
        pass

    all_gcs_missing = (
        pd.isna(row.get("GCS_E"))
        and pd.isna(row.get("GCS_V"))
        and pd.isna(row.get("GCS_M"))
    )
    if not all_gcs_missing:
        try:
            gcs = int(
                float(row.get("GCS_E") or 0)
                + float(row.get("GCS_V") or 0)
                + float(row.get("GCS_M") or 0)
            )
            parts.append(f"GCS：{gcs}")
        except (TypeError, ValueError):
            pass

    for field, label, unit in [("身高", "身高", "cm"), ("體重", "體重", "kg")]:
        v = safe_val(row.get(field, np.nan))
        if v != "不詳":
            parts.append(f"{label}：{v}{unit}")

    pl = safe_val(row.get("瞳孔左", np.nan))
    pr = safe_val(row.get("瞳孔右", np.nan))
    if pl != "不詳" or pr != "不詳":
        parts.append(f"瞳孔：左{pl}，右{pr}")

    return "\n".join(parts)


def is_pediatric_from_row(row: pd.Series) -> "bool | None":
    age = _calc_age(row.get("生日"), row.get("急診日期"))
    if age is None:
        return None
    return age < PEDIATRIC_AGE_CUTOFF


# ── RAG 檢索 ──────────────────────────────────────────────────────────────────
def retrieve_complaint_chunks(
    rag_query: str,
    embedder: SentenceTransformer,
    collection,
    is_pediatric: "bool | None" = None,
) -> list[str]:
    """取 TOP-5 主訴 chunks（依年齡過濾 source）。"""
    prefixed = f"{INSTRUCTION_PREFIX}{rag_query}"
    embedding = embedder.encode(prefixed, normalize_embeddings=True).tolist()

    if is_pediatric is None:
        sources = ALL_COMPLAINT_SOURCES
    elif is_pediatric:
        sources = PEDIATRIC_COMPLAINT_SOURCES
    else:
        sources = ADULT_COMPLAINT_SOURCES

    result = collection.query(
        query_embeddings=[embedding],
        n_results=N_RETRIEVAL_COMPLAINT,
        where={"source": {"$in": sources}},
    )
    return result["documents"][0]


def retrieve_reference_chunks(
    rag_query: str,
    embedder: SentenceTransformer,
    collection,
) -> list[str]:
    """Stage 2：查詢次要調節變數（總表）chunks。"""
    prefixed = f"{INSTRUCTION_PREFIX}{rag_query}"
    embedding = embedder.encode(prefixed, normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=[embedding],
        n_results=N_RETRIEVAL_REFERENCE,
        where={"source": {"$in": REFERENCE_SOURCES}},
    )
    return result["documents"][0]


# ── Reranking ─────────────────────────────────────────────────────────────────
def format_rerank_candidates(docs: list[str]) -> str:
    """
    從每個 chunk 取第一行（【主訴】...大分類...）作為候選顯示，
    讓 LLM 判斷哪個最符合病患主訴，不需要看完整判定依據。
    """
    lines = []
    for i, doc in enumerate(docs):
        first_line = doc.split("\n")[0].strip()
        lines.append(f"{i + 1}. {first_line}")
    return "\n".join(lines)


def rerank_chunks(
    complaint: str,
    docs: list[str],
    llm: Llama,
) -> "tuple[int, str]":
    """
    LLM reranking：從 TOP-N 候選中選出最符合病患主訴的 chunk。
    回傳 (selected_index_0based, raw_response)。
    """
    candidates_str = format_rerank_candidates(docs)
    msg = USER_RERANK.format(complaint=complaint, candidates=candidates_str)
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_RERANK},
            {"role": "user", "content": msg},
        ],
        temperature=0.1,
        max_tokens=16,
        top_p=0.9,
        repeat_penalty=1.1,
    )
    raw = resp["choices"][0]["message"]["content"].strip()
    sel = parse_small_int(raw, len(docs))
    return sel - 1, raw   # 轉為 0-based index


# ── 判定依據解析（rule-based）────────────────────────────────────────────────
CRITERIA_PATTERN = re.compile(
    r"-\s*(.+?)\s*→\s*(\d)級(（★需查次要調節變數規則）)?"
)


def parse_criteria(chunk_text: str) -> list[dict]:
    criteria = []
    for i, m in enumerate(CRITERIA_PATTERN.finditer(chunk_text)):
        criteria.append({
            "id": i + 1,
            "criteria": m.group(1).strip(),
            "level": int(m.group(2)),
            "has_star": m.group(3) is not None,
        })
    return criteria


def format_criteria_list(criteria: list[dict]) -> str:
    return "\n".join(f"{c['id']}. {c['criteria']}" for c in criteria)


# ── LLM 回應解析 ──────────────────────────────────────────────────────────────
def parse_small_int(text: str, n_max: int, fallback: int = 1) -> int:
    """解析 1–n_max 範圍內的整數；無法解析回傳 fallback。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\b([1-9]\d*)\b", text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= n_max:
            return val
    return fallback


def parse_grade(text: str) -> int:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cn_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
    m = re.search(r"第([一二三四五1-5])級", text)
    if m:
        c = m.group(1)
        return cn_map.get(c, int(c) if c.isdigit() else -1)
    m = re.search(r"\b([1-5])\b", text)
    if m:
        return int(m.group(1))
    m = re.search(r"[1-5]", text)
    if m:
        return int(m.group())
    return -1


# ── 評估輸出 ──────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true: list, y_pred: list, save_path: str):
    labels = [1, 2, 3, 4, 5]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted Level")
    axes[0].set_ylabel("True Level")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Oranges",
                xticklabels=labels, yticklabels=labels, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (Row-Normalized %)")
    axes[1].set_xlabel("Predicted Level")
    axes[1].set_ylabel("True Level")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logging.info(f"Confusion matrix saved: {save_path}")


def compute_metrics(y_true: list, y_pred: list) -> dict:
    valid_mask = [p != -1 for p in y_pred]
    y_true_v = [y for y, m in zip(y_true, valid_mask) if m]
    y_pred_v = [p for p, m in zip(y_pred, valid_mask) if m]
    n_total = len(y_true)
    n_valid = len(y_true_v)
    accuracy = accuracy_score(y_true_v, y_pred_v) if y_true_v else 0.0
    adj_correct = sum(abs(t - p) <= 1 for t, p in zip(y_true_v, y_pred_v))
    adjacent_accuracy = adj_correct / n_valid if n_valid else 0.0
    kappa = (
        cohen_kappa_score(y_true_v, y_pred_v, weights="linear")
        if len(set(y_true_v)) > 1 else 0.0
    )
    labels = [1, 2, 3, 4, 5]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_v, y_pred_v, labels=labels, zero_division=0
    )
    return {
        "n_total": n_total, "n_valid": n_valid, "n_parse_failed": n_total - n_valid,
        "accuracy": accuracy, "adjacent_accuracy": adjacent_accuracy,
        "kappa": kappa, "labels": labels,
        "precision": precision, "recall": recall, "f1": f1, "support": support,
    }


def save_metrics_report(metrics: dict, save_path: str):
    lines = [
        "=" * 60, "TTAS 檢傷分級 LLM 評估報告", "=" * 60,
        f"總樣本數   : {metrics['n_total']}",
        f"有效預測數 : {metrics['n_valid']}",
        f"解析失敗數 : {metrics['n_parse_failed']}",
        "",
        f"Accuracy          : {metrics['accuracy']:.4f}",
        f"Adjacent Accuracy : {metrics['adjacent_accuracy']:.4f}  (|true-pred| <= 1)",
        f"Linear Kappa      : {metrics['kappa']:.4f}",
        "", "─" * 40, "各級指標（僅計有效預測）：",
        f"{'Level':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}",
    ]
    for i, lv in enumerate(metrics["labels"]):
        lines.append(
            f"  Lv{lv}   {metrics['precision'][i]:>10.4f}  "
            f"{metrics['recall'][i]:>8.4f}  "
            f"{metrics['f1'][i]:>8.4f}  "
            f"{int(metrics['support'][i]):>8d}"
        )
    lines.append("=" * 60)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"Metrics report saved: {save_path}")
    logging.info("\n".join(lines))


def save_per_level_csv(metrics: dict, save_path: str):
    rows = [
        {"level": lv, "precision": metrics["precision"][i],
         "recall": metrics["recall"][i], "f1": metrics["f1"][i],
         "support": int(metrics["support"][i])}
        for i, lv in enumerate(metrics["labels"])
    ]
    pd.DataFrame(rows).to_csv(save_path, index=False, encoding="utf-8-sig")
    logging.info(f"Per-level metrics saved: {save_path}")


# ── Logging 設定 ──────────────────────────────────────────────────────────────
def setup_logging(results_dir: str, suffix: str = ""):
    log_path = os.path.join(results_dir, f"run{suffix}.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    return log_path


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    FULL_RUN = "--full" in sys.argv
    suffix = "_full" if FULL_RUN else ""
    log_path = setup_logging(RESULTS_DIR, suffix)
    logging.info(f"Log file: {log_path}")

    # ── 1. 載入資料並分層抽樣 ────────────────────────────────────────────────
    logging.info(f"Loading patient data: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH, engine="xlrd")
    logging.info(f"Total records: {len(df)}")

    triage_col = None
    for col in df.columns:
        if "檢傷" in col and "分級" in col:
            triage_col = col
            break
    if triage_col is None:
        raise ValueError("找不到「檢傷分級」欄位")

    df[triage_col] = pd.to_numeric(df[triage_col], errors="coerce")
    df = df[df[triage_col].isin([1, 2, 3, 4, 5])].copy()
    df[triage_col] = df[triage_col].astype(int)
    logging.info(f"Records after filtering: {len(df)}")
    logging.info(f"Level distribution:\n{df[triage_col].value_counts().sort_index()}")

    if FULL_RUN:
        sample = df.reset_index(drop=True)
        logging.info(f"\nFull run: {len(sample)} records.")
    else:
        sample = (
            df.groupby(triage_col, group_keys=False)
            .apply(lambda x: x.sample(n=min(SAMPLE_PER_LEVEL, len(x)), random_state=RANDOM_STATE))
            .reset_index(drop=True)
        )
        logging.info(f"\nSampled {len(sample)} records ({SAMPLE_PER_LEVEL} per level).")

    # ── 2. 載入嵌入模型 ───────────────────────────────────────────────────────
    logging.info(f"\nLoading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    logging.info("Embedding model loaded.")

    # ── 3. 載入 ChromaDB ──────────────────────────────────────────────────────
    logging.info(f"Loading ChromaDB from: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    logging.info(f"ChromaDB loaded: {collection.count()} chunks")

    # ── 4. 載入 LLM ───────────────────────────────────────────────────────────
    logging.info(f"\nLoading LLM: {MODEL_PATH}")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_batch=512,
        n_threads=4,
        verbose=False,
        seed=42,
    )
    logging.info("LLM loaded.")

    # ── 5. 推理迴圈 ───────────────────────────────────────────────────────────
    y_true, y_pred, records = [], [], []
    logging.info(f"\nStarting inference on {len(sample)} samples...\n")

    for idx, (_, row) in enumerate(sample.iterrows()):
        true_level = int(row[triage_col])
        is_ped = is_pediatric_from_row(row)
        rag_query = build_rag_query(row)
        vitals_text = build_vitals_text(row)

        # ── RAG: 取 TOP-5 主訴 chunks ─────────────────────────────────────────
        complaint_docs = retrieve_complaint_chunks(rag_query, embedder, collection, is_ped)

        # ── Reranking: LLM 選出最符合主訴的 chunk ────────────────────────────
        rerank_idx, raw_rerank = rerank_chunks(rag_query, complaint_docs, llm)
        selected_doc = complaint_docs[rerank_idx]

        # ── Rule-based: parse 判定依據清單 ────────────────────────────────────
        criteria = parse_criteria(selected_doc)

        # ── Criterion Selection: LLM 選出最緊急可成立的判定依據 ───────────────
        raw_selection = ""
        selected_criterion = None
        stage2_triggered = False
        final_level = -1

        if criteria:
            criteria_list_str = format_criteria_list(criteria)
            sel_resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_SELECTION},
                    {"role": "user", "content": USER_SELECTION.format(
                        vitals=vitals_text,
                        criteria_list=criteria_list_str,
                    )},
                ],
                temperature=0.1,
                max_tokens=16,
                top_p=0.9,
                repeat_penalty=1.1,
            )
            raw_selection = sel_resp["choices"][0]["message"]["content"].strip()
            sel_idx = parse_small_int(raw_selection, len(criteria))
            selected_criterion = criteria[sel_idx - 1]

            # Rule-based: 取出等級和 ★ flag
            initial_level = selected_criterion["level"]
            stage2_triggered = selected_criterion["has_star"]

            if stage2_triggered:
                # Stage 2: 次要調節變數
                ref_docs = retrieve_reference_chunks(rag_query, embedder, collection)
                ref_context = "\n---\n".join(ref_docs)
                sec_resp = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_SECONDARY},
                        {"role": "user", "content": USER_SECONDARY.format(
                            vitals=vitals_text,
                            selected_criteria=selected_criterion["criteria"],
                            initial_level=initial_level,
                            reference_context=ref_context,
                        )},
                    ],
                    temperature=0.1,
                    max_tokens=64,
                    top_p=0.9,
                    repeat_penalty=1.1,
                )
                raw_secondary = sec_resp["choices"][0]["message"]["content"].strip()
                final_level = parse_grade(raw_secondary)
                raw_selection = f"{raw_selection} | Stage2: {raw_secondary}"
            else:
                final_level = initial_level
        else:
            # criteria 解析失敗 → fallback LLM
            context_str = "\n---\n".join(complaint_docs)
            fallback_resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "你是台灣急診室資深檢傷護理師，只輸出一個數字1–5代表TTAS等級。"},
                    {"role": "user", "content": f"【病患生理資料】\n{vitals_text}\n\n【TTAS參考】\n{context_str}\n\n等級："},
                ],
                temperature=0.1,
                max_tokens=8,
                top_p=0.9,
                repeat_penalty=1.1,
            )
            raw_selection = fallback_resp["choices"][0]["message"]["content"].strip()
            final_level = parse_grade(raw_selection)

        y_true.append(true_level)
        y_pred.append(final_level)

        selected_str = (
            f"{selected_criterion['criteria']} → {selected_criterion['level']}級"
            f"{'（★）' if selected_criterion['has_star'] else ''}"
            if selected_criterion else "（無法解析）"
        )

        # reranking 候選清單（僅 header）供 log 顯示
        rerank_candidates_str = format_rerank_candidates(complaint_docs)

        records.append({
            "idx": idx,
            "true_level": true_level,
            "pred_level": final_level,
            "selected_criterion": selected_str,
            "raw_rerank": raw_rerank,
            "rerank_selected": rerank_idx + 1,
            "raw_selection": raw_selection,
            "stage2_triggered": stage2_triggered,
            "rag_query": rag_query,
            "selected_doc": selected_doc,
            "vitals": vitals_text.replace("\n", " | "),
        })

        # ── 每筆完整 log ──────────────────────────────────────────────────────
        logging.info(
            f"\n[{idx + 1}/{len(sample)}] GT={true_level} | Pred={final_level}\n"
            f"調整後預測（Rule-based）: {selected_str}\n"
            f"RAW 預測: {raw_selection[:120]}\n"
            f"Stage2(調節變數): {'YES' if stage2_triggered else 'no'}\n"
            f"\n{'-'*30}\n"
            f"{vitals_text}\n"
            f"{'-'*30}\n"
            f"Reranking 候選（TOP-5）:\n{rerank_candidates_str}\n"
            f"Reranking 選擇: {rerank_idx + 1}（RAW: {raw_rerank}）\n"
            f"Reranked TOP-1:\n{selected_doc}\n"
            f"Query: {rag_query}\n"
        )

    logging.info(f"\nInference complete. Total: {len(y_true)} samples.")

    # ── 6. 儲存結果 & 評估 ────────────────────────────────────────────────────
    pred_path = os.path.join(RESULTS_DIR, f"predictions{suffix}.csv")
    pd.DataFrame(records).to_csv(pred_path, index=False, encoding="utf-8-sig")
    logging.info(f"Predictions saved: {pred_path}")

    if not any(p != -1 for p in y_pred):
        logging.error("ERROR: No valid predictions.")
        return

    metrics = compute_metrics(y_true, y_pred)

    y_true_v = [t for t, p in zip(y_true, y_pred) if p != -1]
    y_pred_v = [p for p in y_pred if p != -1]

    cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix{suffix}.png")
    plot_confusion_matrix(y_true_v, y_pred_v, cm_path)

    report_path = os.path.join(RESULTS_DIR, f"metrics_report{suffix}.txt")
    save_metrics_report(metrics, report_path)

    per_level_path = os.path.join(RESULTS_DIR, f"per_level_metrics{suffix}.csv")
    save_per_level_csv(metrics, per_level_path)

    logging.info(f"\n=== All done! Results saved to: {RESULTS_DIR} ===")


if __name__ == "__main__":
    main()
