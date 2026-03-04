"""
api_server.py
TTAS 檢傷分級 REST API 服務。

POST /triage   → 輸入病患資料 JSON，回傳檢傷等級 JSON
GET  /health   → 確認服務狀態

啟動方式：
    .venv\\Scripts\\python.exe -m uvicorn api_server:app --host 0.0.0.0 --port 8000

或使用熱重載（開發用）：
    .venv\\Scripts\\python.exe -m uvicorn api_server:app --reload --port 8000
"""

import re
import threading
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, date
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
from llama_cpp import Llama

warnings.filterwarnings("ignore")

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR        = os.path.join(BASE_DIR, "data", "chroma_db")
MODEL_PATH        = os.path.join(BASE_DIR, "model", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
EMBED_MODEL       = "BAAI/bge-m3"
COLLECTION_NAME   = "ttas_guidelines"   # v2 推薦版
INSTRUCTION_PREFIX = "为这个句子生成表示以用于检索相关文章："
N_RETRIEVAL       = 5
PEDIATRIC_AGE_CUTOFF = 18   # 未滿 18 歲使用兒童 TTAS 標準

# 兩階段 RAG 參數
N_RETRIEVAL_COMPLAINT  = 3
ADULT_COMPLAINT_SOURCES     = ["外傷", "成人非外傷", "環境"]
PEDIATRIC_COMPLAINT_SOURCES = ["外傷", "兒童",       "環境"]
ALL_COMPLAINT_SOURCES       = ["外傷", "成人非外傷", "兒童", "環境"]

N_RETRIEVAL_REFERENCE = 2
REFERENCE_SOURCES = ["總表", "總表(第二次修正)"]

SYSTEM_PROMPT = """\
你是一位台灣急診室的資深檢傷護理師，專精於TTAS（台灣急診五級檢傷分類制度）。

【參考指引說明】
提供給你的參考指引已依病患年齡預先篩選（成人版或兒童版），直接使用即可。
若參考指引後段附有次要調節變數規則，表示本案例需要套用，請參照後決定最終級數。

【判定步驟】
1. 從參考指引中找到與病患主訴最匹配的判定依據與 TTAS 級數。
2. 若該判定依據標有「★」，次要調節變數規則已附於參考指引後段，直接套用並決定最終級數。
3. 若無「★」，直接輸出查到的 TTAS 級數。

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


# ── 全域模型物件 ───────────────────────────────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None
_collection = None
_llm: Optional[Llama] = None
_lock = threading.Lock()   # llama_cpp 非執行緒安全，序列化推理請求


# ── FastAPI lifespan ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _collection, _llm

    print(f"[startup] Loading embedding model: {EMBED_MODEL}")
    _embedder = SentenceTransformer(EMBED_MODEL)

    print(f"[startup] Loading ChromaDB: {CHROMA_DIR} / {COLLECTION_NAME}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    _collection = client.get_collection(name=COLLECTION_NAME)
    print(f"[startup] ChromaDB loaded: {_collection.count()} chunks")

    print(f"[startup] Loading LLM: {MODEL_PATH}")
    _llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=4096,
        n_batch=512,
        n_threads=4,
        verbose=False,
        seed=42,
    )
    print("[startup] All models ready. API is serving.")
    yield
    print("[shutdown] Releasing models.")


app = FastAPI(
    title="TTAS 檢傷分級 API",
    description="輸入病患主訴與生命徵象，回傳 TTAS 五級檢傷等級。",
    version="1.1.0",
    lifespan=lifespan,
)


# ── Pydantic 資料模型 ─────────────────────────────────────────────────────────
class PatientInput(BaseModel):
    # 主訴（必填）
    chief_complaint: str = Field(..., description="病人主訴", examples=["胸痛"])

    # 年齡（三擇一：直接給 age，或給 birth_date + emergency_date 讓系統計算）
    age: Optional[int] = Field(
        None, description="年齡（歲），若提供則優先使用，不再計算", examples=[45]
    )
    birth_date: Optional[str] = Field(
        None,
        description="生日，支援西元（1990-05-20）或民國（079/05/20）格式",
        examples=["1990-05-20"],
    )
    emergency_date: Optional[str] = Field(
        None,
        description="急診日期，支援西元或民國格式；不填預設為今日",
        examples=["2024-01-15"],
    )

    # 基本生命徵象
    gender: Optional[str] = Field(None, description="性別（男/女）", examples=["男"])
    temperature: Optional[float] = Field(None, description="體溫（°C）", examples=[36.5])
    systolic_bp: Optional[float] = Field(None, description="收縮壓（mmHg）", examples=[120.0])
    diastolic_bp: Optional[float] = Field(None, description="舒張壓（mmHg）", examples=[80.0])
    pulse: Optional[float] = Field(None, description="脈搏（次/分）", examples=[75.0])
    respiration: Optional[float] = Field(None, description="呼吸（次/分）", examples=[18.0])
    sao2: Optional[float] = Field(None, description="血氧飽和度（%）", examples=[98.0])

    # GCS
    gcs_e: Optional[int] = Field(None, description="GCS 睜眼（1–4）", examples=[4])
    gcs_v: Optional[int] = Field(None, description="GCS 語言（1–5）", examples=[5])
    gcs_m: Optional[int] = Field(None, description="GCS 運動（1–6）", examples=[6])

    # 體型
    height: Optional[float] = Field(None, description="身高（cm）", examples=[170.0])
    weight: Optional[float] = Field(None, description="體重（kg）", examples=[65.0])

    # 瞳孔光反應
    # 常見值：「+」有光反應、「-」無光反應、「+C」白內障手術史；
    # 亦可含大小如「3+」（3 mm 有光反應）
    pupil_left: Optional[str] = Field(
        None, description='左瞳孔光反應，如 "+"、"-"、"+C"、"3+"', examples=["+"]
    )
    pupil_right: Optional[str] = Field(
        None, description='右瞳孔光反應，如 "+"、"-"、"+C"、"3+"', examples=["+"]
    )


class TriageResponse(BaseModel):
    triage_level: int = Field(..., description="TTAS 檢傷等級（1–5）；-1 表示解析失敗")
    parse_success: bool = Field(..., description="是否成功解析 LLM 輸出")
    raw_response: str = Field(..., description="LLM 原始輸出文字")
    age_computed: Optional[int] = Field(None, description="計算後的年齡（歲）；無法計算時為 null")
    is_pediatric: bool = Field(..., description=f"是否為兒童（< {PEDIATRIC_AGE_CUTOFF} 歲）")
    star_triggered: bool = Field(..., description="是否觸發次要調節變數二階段查詢（主訴 chunk 含 ★）")
    query: str = Field(..., description="送進 RAG + LLM 的查詢字串")
    retrieved_chunks: list[str] = Field(..., description="RAG 取回的 TTAS 指引片段（主訴 + 調節變數）")


# ── 工具函式 ──────────────────────────────────────────────────────────────────
def _safe(val, default: str = "不詳") -> str:
    if val is None:
        return default
    s = str(val).strip()
    return s if s else default


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """
    解析日期字串，同時支援西元與民國（ROC）曆。
    民國判斷：以 '/' 或 '-' 分隔，首段數字 < 200 則視為民國年，加 1911 轉換。
    """
    if not date_str:
        return None
    s = str(date_str).strip()
    # 嘗試拆分成 [年, 月, 日]
    parts = re.split(r"[/\-]", s)
    if len(parts) >= 3:
        try:
            y = int(parts[0])
            if y < 200:          # 民國年
                y += 1911
            s = f"{y:04d}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
        except ValueError:
            pass
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _calc_age(patient: PatientInput) -> Optional[int]:
    """從 patient.age（直接給）或 birth_date + emergency_date 計算年齡。"""
    if patient.age is not None:
        return patient.age
    bd = _parse_date(patient.birth_date)
    if bd is None:
        return None
    ed = _parse_date(patient.emergency_date) or date.today()
    age = ed.year - bd.year - ((ed.month, ed.day) < (bd.month, bd.day))
    return max(age, 0)


def _pupil_desc(val: Optional[str]) -> str:
    """將瞳孔欄位值轉為易讀描述。"""
    if val is None:
        return "不詳"
    v = str(val).strip()
    lookup = {
        "+":  "+(有光反應)",
        "-":  "-(無光反應)",
        "+C": "+C(白內障手術史)",
        "-C": "-C(無光反應，白內障手術史)",
    }
    return lookup.get(v, v)   # 其他原始值（如 "3+"）直接回傳


def build_query(patient: PatientInput, age: Optional[int]) -> str:
    # GCS 加總
    gcs_total = "不詳"
    if any(x is not None for x in [patient.gcs_e, patient.gcs_v, patient.gcs_m]):
        gcs_total = str((patient.gcs_e or 0) + (patient.gcs_v or 0) + (patient.gcs_m or 0))

    # 年齡 + 分組標籤
    if age is not None:
        group = "兒童" if age < PEDIATRIC_AGE_CUTOFF else "成人"
        age_str = f"{age}歲（{group}）"
    else:
        age_str = "不詳"

    # 瞳孔
    pl = _pupil_desc(patient.pupil_left)
    pr = _pupil_desc(patient.pupil_right)
    pupil_str = f"左{pl}，右{pr}"

    return (
        f"主訴：{_safe(patient.chief_complaint)}。"
        f"年齡：{age_str}。"
        f"性別：{_safe(patient.gender)}。"
        f"體溫：{_safe(patient.temperature)}°C。"
        f"收縮壓：{_safe(patient.systolic_bp)}mmHg。"
        f"舒張壓：{_safe(patient.diastolic_bp)}mmHg。"
        f"脈搏：{_safe(patient.pulse)}次/分。"
        f"呼吸：{_safe(patient.respiration)}次/分。"
        f"血氧：{_safe(patient.sao2)}%。"
        f"GCS：{gcs_total}。"
        f"身高：{_safe(patient.height)}cm。"
        f"體重：{_safe(patient.weight)}kg。"
        f"瞳孔：{pupil_str}。"
    )


def parse_response(text: str) -> int:
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


# ── API 端點 ──────────────────────────────────────────────────────────────────
@app.get("/health", summary="服務健康檢查")
def health():
    if _collection is None:
        raise HTTPException(status_code=503, detail="Models not yet loaded.")
    return {
        "status": "ok",
        "collection": COLLECTION_NAME,
        "chunks": _collection.count(),
        "model": os.path.basename(MODEL_PATH),
    }


@app.post("/triage", response_model=TriageResponse, summary="病患檢傷分級")
def triage(patient: PatientInput):
    """
    輸入病患主訴與生命徵象，回傳 TTAS 五級檢傷等級。

    - **triage_level**: 1（最緊急）至 5（非緊急）；-1 表示 LLM 輸出無法解析
    - **is_pediatric**: 依年齡自動判斷是否適用兒童 TTAS 標準（< 18 歲）
    - **retrieved_chunks**: RAG 取回的 TTAS 指引原文（供除錯 / 解釋用）
    """
    if _embedder is None or _collection is None or _llm is None:
        raise HTTPException(status_code=503, detail="Models not yet loaded.")

    # 1. 計算年齡
    age = _calc_age(patient)
    is_pediatric = (age is not None and age < PEDIATRIC_AGE_CUTOFF)

    # 2. 建立查詢字串
    query = build_query(patient, age)

    # 3. 兩階段 RAG 檢索
    prefixed = f"{INSTRUCTION_PREFIX}{query}"
    embedding = _embedder.encode(prefixed, normalize_embeddings=True).tolist()

    # Stage 1：主訴 chunk（依年齡選 source）
    if age is None:
        complaint_sources = ALL_COMPLAINT_SOURCES
    elif is_pediatric:
        complaint_sources = PEDIATRIC_COMPLAINT_SOURCES
    else:
        complaint_sources = ADULT_COMPLAINT_SOURCES

    stage1 = _collection.query(
        query_embeddings=[embedding],
        n_results=N_RETRIEVAL_COMPLAINT,
        where={"source": {"$in": complaint_sources}},
    )
    docs: list[str] = stage1["documents"][0]

    # Stage 2：次要調節變數（條件觸發）
    star_triggered = any("★" in doc for doc in docs)
    if star_triggered:
        stage2 = _collection.query(
            query_embeddings=[embedding],
            n_results=N_RETRIEVAL_REFERENCE,
            where={"source": {"$in": REFERENCE_SOURCES}},
        )
        docs = docs + stage2["documents"][0]
    context = "\n---\n".join(docs)

    # 4. LLM 推理（加鎖確保序列執行）
    user_msg = USER_TEMPLATE.format(query=query, context=context)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    with _lock:
        response = _llm.create_chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=20,
            top_p=0.9,
            repeat_penalty=1.1,
        )

    raw = response["choices"][0]["message"]["content"].strip()

    # 5. 解析輸出
    level = parse_response(raw)

    return TriageResponse(
        triage_level=level,
        parse_success=(level != -1),
        raw_response=raw,
        age_computed=age,
        is_pediatric=is_pediatric,
        star_triggered=star_triggered,
        query=query,
        retrieved_chunks=docs,
    )


# ── 直接執行 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
