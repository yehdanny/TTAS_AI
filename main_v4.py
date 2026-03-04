"""
main_v4.py
Step 2 (v4): 加入年齡/兒童判斷、身高體重、瞳孔欄位後重跑評估。

執行方式：
    python main_v4.py
"""

import matplotlib
matplotlib.use("Agg")   # 無顯示器環境，必須在 import pyplot 前設定

import os
import re
import csv
import sys
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
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
MODEL_PATH = os.path.join(BASE_DIR, "model", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
RESULTS_DIR = os.path.join(BASE_DIR, "results_v4")

# ── 參數 ───────────────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "ttas_guidelines"
INSTRUCTION_PREFIX = "为这个句子生成表示以用于检索相关文章："
SAMPLE_PER_LEVEL = 70
RANDOM_STATE = 42
N_RETRIEVAL = 5

# ── Prompt ────────────────────────────────────────────────────────────────────
PEDIATRIC_AGE_CUTOFF = 18

SYSTEM_PROMPT = """\
你是一位台灣急診室的資深檢傷護理師，專精於TTAS（台灣急診五級檢傷分類制度）。

【年齡分組——重要】
- 兒童（< 18 歲）：使用「兒童 TTAS」判定標準。兒童生命徵象正常值與成人不同（心跳、呼吸頻率較快，血壓較低），請以兒童版參考指引為準。
- 成人（≥ 18 歲）：使用「成人 TTAS」判定標準。

【TTAS 判定流程】
1. 依據年齡分組，在參考指引中找到最匹配病人主訴與生命徵象的判定依據及 TTAS 級數。
2. 若判定依據標有「★需查次要調節變數規則」，需進一步參考次要調節變數規則，決定是否調整級數。
3. 若無★標記，直接輸出查到的 TTAS 級數。

【各級定義】
第1級（復甦急救）：生命徵象極不穩定，需立即搶救，如心跳停止、呼吸停止、嚴重休克。
第2級（危急）：生命徵象不穩定，有立即生命危險，需在10分鐘內處置，如急性心肌梗塞、嚴重呼吸困難。
第3級（緊急）：生命徵象尚穩定，但症狀明顯，需在30分鐘內處置，如中度疼痛、發燒合併感染徵象。
第4級（次緊急）：生命徵象穩定，輕度不適，可在60分鐘內處置，如輕微外傷、輕度發燒。
第5級（非緊急）：生命徵象穩定，非急性或慢性問題，可在120分鐘內處置，如一般感冒症狀、慢性病複查。

請只輸出一個阿拉伯數字（1–5），不要輸出任何其他文字。
/no_think"""

USER_TEMPLATE = """\
【病人資料】
{query}

【TTAS參考指引】
{context}

根據以上資料，此病人的檢傷等級為幾級？
請只輸出數字1、2、3、4或5："""


# ── 工具函式 ──────────────────────────────────────────────────────────────────
def safe_val(val, default="不詳") -> str:
    """將 NaN / None / 空白值轉為預設字串。"""
    if pd.isna(val):
        return default
    s = str(val).strip()
    return s if s else default


def _parse_yyyymmdd(val) -> "date | None":
    """將 YYYYMMDD 整數（如 20251001）轉為 date；失敗回傳 None。"""
    try:
        s = str(int(val))
        if len(s) == 8:
            return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except (ValueError, TypeError):
        pass
    return None


def _calc_age(birth_val, emergency_val) -> "int | None":
    """從 YYYYMMDD 欄位計算年齡（歲）。"""
    bd = _parse_yyyymmdd(birth_val)
    ed = _parse_yyyymmdd(emergency_val)
    if bd is None or ed is None:
        return None
    age = ed.year - bd.year - ((ed.month, ed.day) < (bd.month, bd.day))
    return max(age, 0)


GENDER_MAP = {"M": "男", "F": "女", "m": "男", "f": "女"}


def build_query(row: pd.Series) -> str:
    """從病患資料列建立查詢字串（v4：含年齡/分組/身高體重/瞳孔）。"""
    # GCS
    gcs_total = "不詳"
    try:
        e = float(row.get("GCS_E", 0) or 0)
        v = float(row.get("GCS_V", 0) or 0)
        m = float(row.get("GCS_M", 0) or 0)
        if not (pd.isna(row.get("GCS_E")) and pd.isna(row.get("GCS_V")) and pd.isna(row.get("GCS_M"))):
            gcs_total = str(int(e + v + m))
    except (TypeError, ValueError):
        pass

    # 年齡與分組
    age = _calc_age(row.get("生日"), row.get("急診日期"))
    if age is not None:
        group = "兒童" if age < PEDIATRIC_AGE_CUTOFF else "成人"
        age_str = f"{age}歲（{group}）"
    else:
        age_str = "不詳"

    # 性別
    gender_raw = safe_val(row.get("性別", np.nan))
    gender = GENDER_MAP.get(gender_raw, gender_raw)

    chief_complaint = safe_val(row.get("病人主訴", np.nan))
    temp  = safe_val(row.get("體溫", np.nan))
    sbp   = safe_val(row.get("收縮壓", np.nan))
    dbp   = safe_val(row.get("舒張壓", np.nan))
    pulse = safe_val(row.get("脈搏", np.nan))
    resp  = safe_val(row.get("呼吸", np.nan))
    sao2  = safe_val(row.get("SAO2", np.nan))
    height = safe_val(row.get("身高", np.nan))
    weight = safe_val(row.get("體重", np.nan))
    pl    = safe_val(row.get("瞳孔左", np.nan))
    pr    = safe_val(row.get("瞳孔右", np.nan))

    return (
        f"主訴：{chief_complaint}。"
        f"年齡：{age_str}。"
        f"性別：{gender}。"
        f"體溫：{temp}°C。"
        f"收縮壓：{sbp}mmHg。"
        f"舒張壓：{dbp}mmHg。"
        f"脈搏：{pulse}次/分。"
        f"呼吸：{resp}次/分。"
        f"血氧：{sao2}%。"
        f"GCS：{gcs_total}。"
        f"身高：{height}cm。"
        f"體重：{weight}kg。"
        f"瞳孔：左{pl}，右{pr}。"
    )


def parse_response(text: str) -> int:
    """
    從 LLM 輸出中解析檢傷等級（1–5）。
    無法解析回傳 -1。
    """
    # 1. 移除 <think>...</think> 殘留（Qwen3 思考模式未完全關閉時）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # 2. 比對「第X級」（中文序數）
    cn_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
    m = re.search(r"第([一二三四五1-5])級", text)
    if m:
        c = m.group(1)
        return cn_map.get(c, int(c) if c.isdigit() else -1)

    # 3. 比對獨立數字 \b[1-5]\b
    m = re.search(r"\b([1-5])\b", text)
    if m:
        return int(m.group(1))

    # 4. 最寬鬆 fallback：任意 [1-5]
    m = re.search(r"[1-5]", text)
    if m:
        return int(m.group())

    return -1   # 無法解析


def retrieve_context(query: str, embedder: SentenceTransformer, collection) -> str:
    """RAG 檢索，回傳格式化的上下文字串。"""
    prefixed = f"{INSTRUCTION_PREFIX}{query}"
    embedding = embedder.encode(prefixed, normalize_embeddings=True)
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=N_RETRIEVAL,
    )
    docs = results["documents"][0]
    return "\n---\n".join(docs)


# ── 評估輸出 ──────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true: list, y_pred: list, save_path: str):
    """繪製原始次數 + 列正規化百分比雙圖並排混淆矩陣。"""
    labels = [1, 2, 3, 4, 5]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 原始次數
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted Level")
    axes[0].set_ylabel("True Level")

    # 列正規化百分比
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Oranges",
        xticklabels=labels, yticklabels=labels,
        ax=axes[1],
        vmin=0, vmax=1,
    )
    axes[1].set_title("Confusion Matrix (Row-Normalized %)")
    axes[1].set_xlabel("Predicted Level")
    axes[1].set_ylabel("True Level")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def compute_metrics(y_true: list, y_pred: list):
    """計算並回傳各項評估指標。"""
    valid_mask = [p != -1 for p in y_pred]
    y_true_v = [y for y, m in zip(y_true, valid_mask) if m]
    y_pred_v = [p for p, m in zip(y_pred, valid_mask) if m]

    n_total = len(y_true)
    n_valid = len(y_true_v)
    n_failed = n_total - n_valid

    accuracy = accuracy_score(y_true_v, y_pred_v) if y_true_v else 0.0

    adj_correct = sum(abs(t - p) <= 1 for t, p in zip(y_true_v, y_pred_v))
    adjacent_accuracy = adj_correct / n_valid if n_valid else 0.0

    kappa = cohen_kappa_score(y_true_v, y_pred_v, weights="linear") if len(set(y_true_v)) > 1 else 0.0

    labels = [1, 2, 3, 4, 5]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_v, y_pred_v, labels=labels, zero_division=0
    )

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_parse_failed": n_failed,
        "accuracy": accuracy,
        "adjacent_accuracy": adjacent_accuracy,
        "kappa": kappa,
        "labels": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def save_metrics_report(metrics: dict, save_path: str):
    """將評估指標寫入文字報告。"""
    lines = [
        "=" * 60,
        "TTAS 檢傷分級 LLM 評估報告",
        "=" * 60,
        f"總樣本數   : {metrics['n_total']}",
        f"有效預測數 : {metrics['n_valid']}",
        f"解析失敗數 : {metrics['n_parse_failed']}",
        "",
        f"Accuracy          : {metrics['accuracy']:.4f}",
        f"Adjacent Accuracy : {metrics['adjacent_accuracy']:.4f}  (|true-pred| <= 1)",
        f"Linear Kappa      : {metrics['kappa']:.4f}",
        "",
        "─" * 40,
        "各級指標（僅計有效預測）：",
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

    print(f"Metrics report saved: {save_path}")
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        print("\n".join(lines).encode("ascii", errors="replace").decode("ascii"))


def save_per_level_csv(metrics: dict, save_path: str):
    """儲存各級指標 CSV。"""
    rows = []
    for i, lv in enumerate(metrics["labels"]):
        rows.append({
            "level": lv,
            "precision": metrics["precision"][i],
            "recall": metrics["recall"][i],
            "f1": metrics["f1"][i],
            "support": int(metrics["support"][i]),
        })
    pd.DataFrame(rows).to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Per-level metrics saved: {save_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. 載入資料並分層抽樣 ────────────────────────────────────────────────
    print(f"Loading patient data: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH, engine="xlrd")
    print(f"Total records: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # 找出檢傷分級欄位（可能有空格或不同命名）
    triage_col = None
    for col in df.columns:
        if "檢傷" in col and "分級" in col:
            triage_col = col
            break
    if triage_col is None:
        raise ValueError("找不到「檢傷分級」欄位，請確認 Excel 欄位名稱。")

    print(f"Triage column: '{triage_col}'")

    # 確保分級欄位為整數，過濾無效值
    df[triage_col] = pd.to_numeric(df[triage_col], errors="coerce")
    df = df[df[triage_col].isin([1, 2, 3, 4, 5])].copy()
    df[triage_col] = df[triage_col].astype(int)
    print(f"Records after filtering valid triage levels: {len(df)}")
    print(f"Level distribution:\n{df[triage_col].value_counts().sort_index()}")

    FULL_RUN = "--full" in sys.argv
    if FULL_RUN:
        sample = df.reset_index(drop=True)
        print(f"\nFull run: using all {len(sample)} records.")
    else:
        sample = (
            df.groupby(triage_col, group_keys=False)
            .apply(lambda x: x.sample(n=min(SAMPLE_PER_LEVEL, len(x)), random_state=RANDOM_STATE))
            .reset_index(drop=True)
        )
        print(f"\nSampled {len(sample)} records ({SAMPLE_PER_LEVEL} per level).")

    # ── 2. 載入嵌入模型 ───────────────────────────────────────────────────────
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("Embedding model loaded.")

    # ── 3. 載入 ChromaDB ──────────────────────────────────────────────────────
    print(f"Loading ChromaDB from: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    chunk_count = collection.count()
    print(f"ChromaDB loaded: {chunk_count} chunks")

    # ── 4. 載入 LLM ───────────────────────────────────────────────────────────
    print(f"\nLoading LLM: {MODEL_PATH}")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,    # 全部 offload 到 GPU
        n_ctx=4096,
        n_batch=512,
        n_threads=4,
        verbose=False,
        seed=42,
    )
    print("LLM loaded.")

    # ── 5. 推理迴圈 ───────────────────────────────────────────────────────────
    y_true = []
    y_pred = []
    records = []

    print(f"\nStarting inference on {len(sample)} samples...\n")

    for idx, (_, row) in enumerate(sample.iterrows()):
        true_level = int(row[triage_col])
        query = build_query(row)
        context = retrieve_context(query, embedder, collection)

        user_msg = USER_TEMPLATE.format(query=query, context=context)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=20,
            top_p=0.9,
            repeat_penalty=1.1,
        )
        raw = response["choices"][0]["message"]["content"].strip()
        pred_level = parse_response(raw)

        y_true.append(true_level)
        y_pred.append(pred_level)
        records.append({
            "idx": idx,
            "true_level": true_level,
            "pred_level": pred_level,
            "raw_response": raw,
            "query": query,
        })

        # 前 5 筆印出 raw response 供驗證
        if idx < 5:
            print(f"[{idx+1}] True={true_level} | Pred={pred_level} | Raw='{raw}'")
        elif idx % 50 == 0:
            done = idx + 1
            pct = done / len(sample) * 100
            acc_so_far = sum(t == p for t, p in zip(y_true, y_pred) if p != -1) / max(1, sum(p != -1 for p in y_pred))
            print(f"Progress: {done}/{len(sample)} ({pct:.1f}%) | Running accuracy: {acc_so_far:.3f}")

    print(f"\nInference complete. Total: {len(y_true)} samples.")

    # ── 6. 儲存預測結果 ───────────────────────────────────────────────────────
    suffix = "_full" if FULL_RUN else ""
    pred_path = os.path.join(RESULTS_DIR, f"predictions{suffix}.csv")
    pd.DataFrame(records).to_csv(pred_path, index=False, encoding="utf-8-sig")
    print(f"Predictions saved: {pred_path}")

    # ── 7. 過濾有效預測後計算指標 ─────────────────────────────────────────────
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != -1]
    if not valid_pairs:
        print("ERROR: No valid predictions. Check LLM output format.")
        return

    metrics = compute_metrics(y_true, y_pred)

    # ── 8. 混淆矩陣 ───────────────────────────────────────────────────────────
    y_true_v = [t for t, p in zip(y_true, y_pred) if p != -1]
    y_pred_v = [p for p in y_pred if p != -1]

    cm_path = os.path.join(RESULTS_DIR, f"confusion_matrix{suffix}.png")
    plot_confusion_matrix(y_true_v, y_pred_v, cm_path)

    # ── 9. 文字報告 & 各級 CSV ────────────────────────────────────────────────
    report_path = os.path.join(RESULTS_DIR, f"metrics_report{suffix}.txt")
    save_metrics_report(metrics, report_path)

    per_level_path = os.path.join(RESULTS_DIR, f"per_level_metrics{suffix}.csv")
    save_per_level_csv(metrics, per_level_path)

    print("\n=== All done! Results saved to:", RESULTS_DIR, "===")


if __name__ == "__main__":
    main()
