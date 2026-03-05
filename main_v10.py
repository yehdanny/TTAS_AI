"""
main_v10.py
Step 2 (v10): B+C+D 改善。

v10 相較於 v9 的改進：
[B] Rule-based Vital Pre-filter：criterion selection 前先用生理數值
    刪除明顯不成立的條目（SpO2/GCS/MAP 門檻），避免 LLM 錯選嚴重條目
[C] Prompt 改為「有直接證據才選」：不再「最緊急優先」，改為「有具體數值
    或症狀支撐才選，生命徵象穩定時不選需要血行動力不穩定的條目」
[D] EXACT 多重命中先 reranking：命中多個標準主訴 chunk 時，先讓 LLM 選
    最相關的那一個，再做 criterion selection（避免合併 50+ 條同時送入）

執行方式：
    python main_v10.py           # 350 筆分層抽樣
    python main_v10.py --full    # 全量推理
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
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db_v7")
MODEL_PATH = os.path.join(BASE_DIR, "model", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
RESULTS_DIR = os.path.join(BASE_DIR, "results_v10")

# ── 參數 ───────────────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "ttas_v7"
INSTRUCTION_PREFIX = "为这个句子生成表示以用于检索相关文章："
SAMPLE_PER_LEVEL = 70
RANDOM_STATE = 42

PEDIATRIC_AGE_CUTOFF = 18
N_RETRIEVAL_COMPLAINT = 5
ADULT_COMPLAINT_SOURCES = ["外傷", "成人非外傷", "環境"]
PEDIATRIC_COMPLAINT_SOURCES = ["外傷", "兒童", "環境"]
ALL_COMPLAINT_SOURCES = ["外傷", "成人非外傷", "兒童", "環境"]
N_RETRIEVAL_REFERENCE = 2
REFERENCE_SOURCES = ["總表", "總表(第二次修正)"]
MIN_NAME_LEN = 2

# ── Prompt 模板 ────────────────────────────────────────────────────────────────

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

# [C] 改為「有直接證據才選」，而非「最緊急優先」
SYSTEM_SELECTION = """\
你是一位台灣急診室的資深檢傷護理師，專精於TTAS（台灣急診五級檢傷分類制度）。

【判定原則】
1. 依照病患的具體生理數值與主訴描述，找出有直接證據支持的判定依據。
2. 若生命徵象穩定（血壓/MAP 正常、血氧正常、意識清楚），不應選擇需要
   血行動力不穩定、意識改變、重度呼吸窘迫等嚴重條件的條目，即使它們
   的級數看起來較緊急。
3. 若無法確定某嚴重條件是否成立，以穩定的生命徵象為準，選擇較保守
   （數字較大）的條目。
4. 在有直接證據支持的條目中，選出最符合病患目前狀態的那一條。
5. 只回答一個整數（條目編號），不要輸出任何其他文字。"""

USER_SELECTION = """\
【病患生理資料】
{vitals}

【判定依據清單】（已依生命徵象預先過濾明顯不適用的條目）
{criteria_list}

根據以上生理資料，哪一條判定依據有直接證據支持？只回答編號："""

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
COMPLAINT_HEADER_RE = re.compile(r"^([^-\n]+)-([^\n]+)", re.MULTILINE)


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


def extract_numeric_vitals(row: pd.Series) -> dict:
    """提取數值生命徵象，供 rule-based 前置過濾使用。"""
    vitals = {}
    try:
        sao2 = float(row.get("SAO2"))
        if not np.isnan(sao2):
            vitals["sao2"] = sao2
    except (TypeError, ValueError):
        pass

    try:
        e = float(row.get("GCS_E") or 0)
        v = float(row.get("GCS_V") or 0)
        m = float(row.get("GCS_M") or 0)
        if not (pd.isna(row.get("GCS_E")) and pd.isna(row.get("GCS_V")) and pd.isna(row.get("GCS_M"))):
            vitals["gcs"] = int(e + v + m)
    except (TypeError, ValueError):
        pass

    try:
        sbp = float(row.get("收縮壓"))
        dbp = float(row.get("舒張壓"))
        if not (np.isnan(sbp) or np.isnan(dbp)):
            vitals["sbp"] = sbp
            vitals["dbp"] = dbp
            vitals["map"] = (sbp - dbp) / 3 + dbp
    except (TypeError, ValueError):
        pass

    return vitals


def is_pediatric_from_row(row: pd.Series) -> "bool | None":
    age = _calc_age(row.get("生日"), row.get("急診日期"))
    if age is None:
        return None
    return age < PEDIATRIC_AGE_CUTOFF


# ── [B] Rule-based Vital Pre-filter ──────────────────────────────────────────
# 每條規則：(criterion 關鍵字 pattern, vital key, 保留條件 lambda)
# 若 vital 有值且條件不成立 → 刪除該 criterion
# 若 vital 缺失 → 保留（無法排除）
_VITAL_FILTER_RULES: list[tuple] = [
    # 重度呼吸窘迫 <90%
    (re.compile(r"<\s*90\s*%|重度呼吸窘迫"), "sao2", lambda v: v < 90),
    # 中度呼吸窘迫 <92%
    (re.compile(r"<\s*92\s*%|中度呼吸窘迫"), "sao2", lambda v: v < 92),
    # 輕度呼吸窘迫 92-94%
    (re.compile(r"92.94\s*%|輕度呼吸窘迫"), "sao2", lambda v: v < 94),
    # 無意識 GCS 3-8
    (re.compile(r"GCS\s*3.8|無意識"), "gcs", lambda v: v <= 8),
    # 意識程度改變 GCS 9-13
    (re.compile(r"GCS\s*9.13|意識程度改變"), "gcs", lambda v: v <= 13),
    # 血行動力循環不足 / 休克（MAP < 65 或 SBP < 90）
    (re.compile(r"血行動力循環不足|休克(?!後)"), "map", lambda v: v < 65),
]


def filter_criteria_by_vitals(
    criteria: list[dict], numeric_vitals: dict
) -> "tuple[list[dict], list[str]]":
    """
    [B] 依生命徵象數值刪除明顯不成立的判定依據。

    回傳 (filtered_criteria, removed_reasons)。
    若過濾後清單為空，回傳原始清單（安全保底）。
    """
    kept, removed_reasons = [], []

    for c in criteria:
        remove = False
        for pattern, vital_key, condition in _VITAL_FILTER_RULES:
            if pattern.search(c["criteria"]) and vital_key in numeric_vitals:
                if not condition(numeric_vitals[vital_key]):
                    removed_reasons.append(
                        f"  刪除「{c['criteria']}」"
                        f"（{vital_key}={numeric_vitals[vital_key]:.1f} 不符條件）"
                    )
                    remove = True
                    break
        if not remove:
            kept.append(c)

    # 重新編號
    for i, c in enumerate(kept):
        c["id"] = i + 1

    return (kept if kept else criteria), removed_reasons


# ── 標準主訴名稱索引 ──────────────────────────────────────────────────────────
def build_complaint_name_index(collection) -> dict[str, list[dict]]:
    COMPLAINT_SOURCES = {"外傷", "成人非外傷", "兒童", "環境"}
    all_data = collection.get(include=["documents", "metadatas"])
    docs = all_data["documents"]
    metas = all_data["metadatas"]

    index: dict[str, list[dict]] = {}
    for doc, meta in zip(docs, metas):
        source = meta.get("source", "")
        if source not in COMPLAINT_SOURCES:
            continue
        m = COMPLAINT_HEADER_RE.search(doc)
        if not m:
            continue
        name = m.group(2).strip()
        if len(name) < MIN_NAME_LEN:
            continue
        index.setdefault(name, []).append({"doc": doc, "source": source})

    logging.info(f"Complaint name index built: {len(index)} unique 標準主訴名稱")
    return index


def match_complaint_names(
    patient_complaint: str,
    name_index: dict[str, list[dict]],
    is_pediatric: "bool | None",
) -> "tuple[list[str], list[str]]":
    """Substring 精確比對，依年齡過濾，長名稱優先，避免重複覆蓋。"""
    if is_pediatric is True:
        valid_sources = set(PEDIATRIC_COMPLAINT_SOURCES)
    elif is_pediatric is False:
        valid_sources = set(ADULT_COMPLAINT_SOURCES)
    else:
        valid_sources = set(ALL_COMPLAINT_SOURCES)

    sorted_names = sorted(name_index.keys(), key=len, reverse=True)
    matched_docs: list[str] = []
    matched_names: list[str] = []
    covered_spans: list[tuple[int, int]] = []

    for name in sorted_names:
        start = 0
        while True:
            pos = patient_complaint.find(name, start)
            if pos == -1:
                break
            end = pos + len(name)
            overlapped = any(cs <= pos and end <= ce for cs, ce in covered_spans)
            if not overlapped:
                age_filtered = [
                    e["doc"] for e in name_index[name]
                    if e["source"] in valid_sources
                ]
                if age_filtered:
                    covered_spans.append((pos, end))
                    matched_names.append(name)
                    for d in age_filtered:
                        if d not in matched_docs:
                            matched_docs.append(d)
            start = end

    return matched_docs, matched_names


# ── RAG 檢索 ──────────────────────────────────────────────────────────────────
def retrieve_complaint_chunks(
    rag_query: str, embedder: SentenceTransformer, collection,
    is_pediatric: "bool | None" = None,
) -> list[str]:
    prefixed = f"{INSTRUCTION_PREFIX}{rag_query}"
    embedding = embedder.encode(prefixed, normalize_embeddings=True).tolist()
    sources = (
        ALL_COMPLAINT_SOURCES if is_pediatric is None
        else PEDIATRIC_COMPLAINT_SOURCES if is_pediatric
        else ADULT_COMPLAINT_SOURCES
    )
    result = collection.query(
        query_embeddings=[embedding],
        n_results=N_RETRIEVAL_COMPLAINT,
        where={"source": {"$in": sources}},
    )
    return result["documents"][0]


def retrieve_reference_chunks(
    rag_query: str, embedder: SentenceTransformer, collection,
) -> list[str]:
    prefixed = f"{INSTRUCTION_PREFIX}{rag_query}"
    embedding = embedder.encode(prefixed, normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=[embedding],
        n_results=N_RETRIEVAL_REFERENCE,
        where={"source": {"$in": REFERENCE_SOURCES}},
    )
    return result["documents"][0]


# ── Reranking ─────────────────────────────────────────────────────────────────
def format_chunk_headers(docs: list[str]) -> str:
    return "\n".join(
        f"{i + 1}. {doc.split(chr(10))[0].strip()}"
        for i, doc in enumerate(docs)
    )


def rerank_chunks(complaint: str, docs: list[str], llm: Llama) -> "tuple[int, str]":
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_RERANK},
            {"role": "user", "content": USER_RERANK.format(
                complaint=complaint,
                candidates=format_chunk_headers(docs),
            )},
        ],
        temperature=0.1, max_tokens=16, top_p=0.9, repeat_penalty=1.1,
    )
    raw = resp["choices"][0]["message"]["content"].strip()
    return parse_small_int(raw, len(docs)) - 1, raw   # 0-based


# ── 判定依據解析 ──────────────────────────────────────────────────────────────
CRITERIA_PATTERN = re.compile(
    r"-\s*(.+?)\s*→\s*(\d)級(（★需查次要調節變數規則）)?"
)


def parse_criteria(chunk_text: str) -> list[dict]:
    return [
        {
            "id": i + 1,
            "criteria": m.group(1).strip(),
            "level": int(m.group(2)),
            "has_star": m.group(3) is not None,
        }
        for i, m in enumerate(CRITERIA_PATTERN.finditer(chunk_text))
    ]


def format_criteria_list(criteria: list[dict]) -> str:
    return "\n".join(f"{c['id']}. {c['criteria']}" for c in criteria)


# ── LLM 回應解析 ──────────────────────────────────────────────────────────────
def parse_small_int(text: str, n_max: int, fallback: int = 1) -> int:
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

    # ── 1. 載入資料 ───────────────────────────────────────────────────────────
    logging.info(f"Loading patient data: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH, engine="xlrd")

    triage_col = next((c for c in df.columns if "檢傷" in c and "分級" in c), None)
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

    # ── 2. 載入模型與資料庫 ───────────────────────────────────────────────────
    logging.info(f"\nLoading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    logging.info("Embedding model loaded.")

    logging.info(f"Loading ChromaDB from: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    logging.info(f"ChromaDB loaded: {collection.count()} chunks")

    name_index = build_complaint_name_index(collection)

    logging.info(f"\nLoading LLM: {MODEL_PATH}")
    llm = Llama(
        model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=4096,
        n_batch=512, n_threads=4, verbose=False, seed=42,
    )
    logging.info("LLM loaded.")

    # ── 3. 推理迴圈 ───────────────────────────────────────────────────────────
    y_true, y_pred, records = [], [], []
    n_exact_hit = 0
    logging.info(f"\nStarting inference on {len(sample)} samples...\n")

    for idx, (_, row) in enumerate(sample.iterrows()):
        true_level = int(row[triage_col])
        is_ped = is_pediatric_from_row(row)
        rag_query = build_rag_query(row)
        vitals_text = build_vitals_text(row)
        numeric_vitals = extract_numeric_vitals(row)

        # ── 路徑 A：EXACT 比對 ────────────────────────────────────────────────
        matched_docs, matched_names = match_complaint_names(rag_query, name_index, is_ped)
        retrieval_method = ""
        rerank_info = ""
        selected_doc = ""

        if matched_docs:
            n_exact_hit += 1
            retrieval_method = "EXACT"
            # [D] 多個命中 chunk 時先 reranking，選最相關的一個
            if len(matched_docs) > 1:
                rerank_idx, raw_rerank = rerank_chunks(rag_query, matched_docs, llm)
                selected_doc = matched_docs[rerank_idx]
                rerank_info = (
                    f"EXACT 命中 {len(matched_docs)} 個 → Rerank 選 {rerank_idx+1}"
                    f"（RAW: {raw_rerank}）"
                )
            else:
                selected_doc = matched_docs[0]
                rerank_info = f"EXACT 命中 1 個（直接使用）"
        else:
            # ── 路徑 B：RAG TOP-5 + Reranking ────────────────────────────────
            retrieval_method = "RAG+RERANK"
            complaint_docs = retrieve_complaint_chunks(rag_query, embedder, collection, is_ped)
            rerank_idx, raw_rerank = rerank_chunks(rag_query, complaint_docs, llm)
            selected_doc = complaint_docs[rerank_idx]
            rerank_info = (
                f"RAG TOP-5 候選:\n{format_chunk_headers(complaint_docs)}\n"
                f"Rerank 選 {rerank_idx+1}（RAW: {raw_rerank}）"
            )

        # ── Rule-based: parse + [B] vital pre-filter ─────────────────────────
        raw_criteria = parse_criteria(selected_doc)
        filtered_criteria, removed_reasons = filter_criteria_by_vitals(
            raw_criteria, numeric_vitals
        )

        # ── [C] Criterion Selection ───────────────────────────────────────────
        raw_selection = ""
        selected_criterion = None
        stage2_triggered = False
        final_level = -1

        if filtered_criteria:
            sel_resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_SELECTION},
                    {"role": "user", "content": USER_SELECTION.format(
                        vitals=vitals_text,
                        criteria_list=format_criteria_list(filtered_criteria),
                    )},
                ],
                temperature=0.1, max_tokens=16, top_p=0.9, repeat_penalty=1.1,
            )
            raw_selection = sel_resp["choices"][0]["message"]["content"].strip()
            sel_idx = parse_small_int(raw_selection, len(filtered_criteria))
            selected_criterion = filtered_criteria[sel_idx - 1]

            initial_level = selected_criterion["level"]
            stage2_triggered = selected_criterion["has_star"]

            if stage2_triggered:
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
                    temperature=0.1, max_tokens=64, top_p=0.9, repeat_penalty=1.1,
                )
                raw_secondary = sec_resp["choices"][0]["message"]["content"].strip()
                final_level = parse_grade(raw_secondary)
                raw_selection = f"{raw_selection} | Stage2: {raw_secondary}"
            else:
                final_level = initial_level
        else:
            # fallback
            fallback_resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "你是台灣急診室資深檢傷護理師，只輸出一個數字1–5代表TTAS等級。"},
                    {"role": "user", "content": f"【病患生理資料】\n{vitals_text}\n\n【TTAS參考】\n{selected_doc}\n\n等級："},
                ],
                temperature=0.1, max_tokens=8, top_p=0.9, repeat_penalty=1.1,
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

        records.append({
            "idx": idx, "true_level": true_level, "pred_level": final_level,
            "retrieval_method": retrieval_method,
            "matched_names": ", ".join(matched_names),
            "selected_criterion": selected_str,
            "raw_selection": raw_selection,
            "stage2_triggered": stage2_triggered,
            "n_raw_criteria": len(raw_criteria),
            "n_filtered_criteria": len(filtered_criteria),
            "rag_query": rag_query,
            "vitals": vitals_text.replace("\n", " | "),
        })

        # ── 每筆 log ──────────────────────────────────────────────────────────
        filter_summary = (
            f"Criteria: {len(raw_criteria)} → {len(filtered_criteria)} "
            f"（過濾 {len(raw_criteria)-len(filtered_criteria)} 條）"
        )
        if removed_reasons:
            filter_summary += "\n" + "\n".join(removed_reasons)

        logging.info(
            f"\n[{idx+1}/{len(sample)}] GT={true_level} | Pred={final_level} [{retrieval_method}]\n"
            f"調整後預測（Rule-based）: {selected_str}\n"
            f"RAW 預測: {raw_selection[:120]}\n"
            f"Stage2(調節變數): {'YES' if stage2_triggered else 'no'}\n"
            f"\n{'-'*30}\n"
            f"{vitals_text}\n"
            f"{'-'*30}\n"
            f"{rerank_info}\n"
            f"Selected chunk: {selected_doc.split(chr(10))[0]}\n"
            f"{filter_summary}\n"
            f"Query: {rag_query}\n"
        )

    logging.info(f"\nInference complete. Total: {len(y_true)} samples.")
    logging.info(
        f"EXACT hit: {n_exact_hit}/{len(y_true)} ({n_exact_hit/len(y_true)*100:.1f}%)\n"
        f"RAG fallback: {len(y_true)-n_exact_hit}/{len(y_true)} "
        f"({(len(y_true)-n_exact_hit)/len(y_true)*100:.1f}%)"
    )

    # ── 4. 儲存結果 & 評估 ────────────────────────────────────────────────────
    pred_path = os.path.join(RESULTS_DIR, f"predictions{suffix}.csv")
    pd.DataFrame(records).to_csv(pred_path, index=False, encoding="utf-8-sig")
    logging.info(f"Predictions saved: {pred_path}")

    if not any(p != -1 for p in y_pred):
        logging.error("ERROR: No valid predictions.")
        return

    metrics = compute_metrics(y_true, y_pred)
    y_true_v = [t for t, p in zip(y_true, y_pred) if p != -1]
    y_pred_v = [p for p in y_pred if p != -1]

    plot_confusion_matrix(y_true_v, y_pred_v,
                          os.path.join(RESULTS_DIR, f"confusion_matrix{suffix}.png"))
    save_metrics_report(metrics, os.path.join(RESULTS_DIR, f"metrics_report{suffix}.txt"))
    save_per_level_csv(metrics, os.path.join(RESULTS_DIR, f"per_level_metrics{suffix}.csv"))

    logging.info(f"\n=== All done! Results saved to: {RESULTS_DIR} ===")


if __name__ == "__main__":
    main()
