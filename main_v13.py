"""
main_v13.py
Step 2 (v13): 改進休克 proxy 偵測、疼痛欄位動態支援、移除需臨床判斷才能確定的 rule。

v13 核心改進（相對於 v12）：
[L] 休克 proxy 偵測（has_shock_signs）：
    總表定義休克徵象（意識模糊/呼吸急促/脈搏微弱/低血壓/缺氧）
    可從 CSV vitals 近似：MAP<65 OR SpO2<92 OR GCS<=13 OR RR>24 OR HR>120
    SBP Lv1 = SBP<70 OR (SBP<90 AND has_shock_signs)
    HR  Lv1 = (HR<50 or >140) AND (SBP<70 OR has_shock_signs)
[M] 移除需臨床判斷才能確定的 rule（不影響模型性能）：
    - 移除 Temp >38 Lv2/3（需「病容/免疫缺陷/SIRS」，CSV 無法判定）
    - 移除高血壓 SBP>=200/DBP>=110（表五「附加選項」屬主訴項目，不是 vital modifier）
[N] 疼痛 rule-based（動態）：
    若 CSV/API 有疼痛分數欄位（NRS/疼痛指數等）自動啟用：
      成人：重度(8-10)→Lv2；中度(4-7)→Lv3；輕度(<4)→Lv4（以中樞型保守估算）
      兒童：重度→Lv2；中度→Lv3；輕度→Lv4（不分中樞/周邊）
    若無疼痛欄位 → 跳過疼痛 rule-based，完全交 LLM 判斷

繼承 v12：I(官方閾值)、J(雙軌制)、K(VitalAlert 等級顯示)
繼承 v11：E(Vitals Assessment)、F(標注)、G(Pre-filter)、H(強制約束)
繼承 v10：B(pre-filter)、C(prompt 有據才選)、D(EXACT 多命中 reranking)

執行方式：
    python main_v13.py
    python main_v13.py --full
"""

import matplotlib
matplotlib.use("Agg")

import os
import re
import sys
import logging
import warnings
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_recall_fscore_support, cohen_kappa_score,
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
RESULTS_DIR = os.path.join(BASE_DIR, "results_v13")

# ── 參數 ───────────────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "ttas_v7"
INSTRUCTION_PREFIX = "为这个句子生成表示以用于检索相关文章："
SAMPLE_PER_LEVEL = 70
RANDOM_STATE = 42

PEDIATRIC_AGE_CUTOFF = 18
N_RETRIEVAL_COMPLAINT = 5
ADULT_COMPLAINT_SOURCES   = ["外傷", "成人非外傷", "環境"]
PEDIATRIC_COMPLAINT_SOURCES = ["外傷", "兒童", "環境"]
ALL_COMPLAINT_SOURCES     = ["外傷", "成人非外傷", "兒童", "環境"]
N_RETRIEVAL_REFERENCE = 2
REFERENCE_SOURCES = ["總表", "總表(第二次修正)"]
MIN_NAME_LEN = 2

# ── [E] 生命徵象正常範圍定義 ──────────────────────────────────────────────────
# (lower_bound, upper_bound)
# 成人正常範圍
VITAL_NORMAL_RANGES = {
    "sao2":  (95.0, 100.0),
    "sbp":   (90.0, 140.0),
    "dbp":   (60.0, 90.0),
    "map":   (65.0, 110.0),
    "hr":    (60.0, 100.0),
    "rr":    (12.0, 20.0),
    "temp":  (36.0, 37.5),
    "gcs":   (14.0, 15.0),   # ≥14 為正常
}

# 小兒正常範圍（< 18 歲）
# 兒童心跳、呼吸偏快，血壓偏低，參考 TTAS 兒童分類基準與 PALS 標準
PEDIATRIC_VITAL_NORMAL_RANGES = {
    "sao2":  (95.0, 100.0),   # 同成人
    "sbp":   (80.0, 120.0),   # 兒童血壓偏低
    "dbp":   (50.0, 80.0),
    "map":   (55.0, 90.0),
    "hr":    (60.0, 140.0),   # 兒童心跳正常範圍較寬
    "rr":    (15.0, 30.0),    # 兒童呼吸速率偏快
    "temp":  (36.0, 37.5),    # 同成人
    "gcs":   (14.0, 15.0),    # 同成人
}

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

SYSTEM_SELECTION = """\
你是一位台灣急診室的資深檢傷護理師，專精於TTAS（台灣急診五級檢傷分類制度）。

【判定原則】
1. 依照病患的具體生理數值與主訴描述，找出有直接證據支持的判定依據。
2. 生理資料中每個數值已標注「正常/偏低/偏高/異常」，請直接參考這些標注。
3. 若整體生命徵象穩定，不應選擇需要血行動力不穩定、意識改變、
   重度呼吸窘迫等嚴重條件的條目。
4. 若無法確定某嚴重條件是否成立，以穩定的生命徵象為準，選擇較保守的條目。
5. 只回答一個整數（條目編號），不要輸出任何其他文字。"""

USER_SELECTION = """\
【病患生理資料】
{vitals}

【判定依據清單】（已依生命徵象預先過濾明顯不適用的條目）
{criteria_list}

根據以上生理資料，哪一條判定依據有直接數值或症狀證據支持？只回答編號："""

# [H] 強制約束模板：vital 已強烈指示緊急度時使用
USER_SELECTION_CONSTRAINED = """\
【病患生理資料】
{vitals}

【生命徵象警示】
{vital_alerts}

【判定依據清單】（已預先限制為符合生命徵象嚴重度的條目）
{criteria_list}

根據以上生命徵象警示與判定依據，選出最符合的條目編號："""

SYSTEM_SECONDARY = """\
你是一位台灣急診室的資深檢傷護理師，專精於TTAS次要調節變數規則。
根據病患生理資料與次要調節變數規則，決定最終的 TTAS 等級（1–5）。
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


# ── 基本工具函式 ──────────────────────────────────────────────────────────────
def safe_val(val, default="不詳") -> str:
    if pd.isna(val):
        return default
    s = str(val).strip()
    return s if s else default


def _parse_yyyymmdd(val) -> "Optional[date]":
    try:
        s = str(int(val))
        if len(s) == 8:
            return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except (ValueError, TypeError):
        pass
    return None


def _calc_age(birth_val, emergency_val) -> "Optional[int]":
    bd = _parse_yyyymmdd(birth_val)
    ed = _parse_yyyymmdd(emergency_val)
    if bd is None or ed is None:
        return None
    age = ed.year - bd.year - ((ed.month, ed.day) < (bd.month, bd.day))
    return max(age, 0)


GENDER_MAP = {"M": "男", "F": "女", "m": "男", "f": "女"}
COMPLAINT_HEADER_RE = re.compile(r"^([^-\n]+)-([^\n]+)", re.MULTILINE)

def is_pediatric_from_row(row: pd.Series) -> "Optional[bool]":
    age = _calc_age(row.get("生日"), row.get("急診日期"))
    if age is None:
        return None
    return age < PEDIATRIC_AGE_CUTOFF


# ── [E] 生命徵象評估系統 ──────────────────────────────────────────────────────
@dataclass
class VitalAlert:
    vital: str          # 指標名稱
    value: float        # 實際數值
    status: str         # "正常" / "偏低" / "偏高" / "異常"
    min_level: int      # 此異常最低建議 TTAS 等級（1=最緊急）
    reason: str         # 說明文字

    def label(self) -> str:
        val_str = f"{self.value:.1f}" if isinstance(self.value, float) else str(int(self.value))
        return f"{self.vital}={val_str}（{self.min_level}級）"


def extract_numeric_vitals(row: pd.Series) -> dict:
    """
    提取所有可用的數值生命徵象。

    異常值過濾規則：
    - NaN / 無法轉 float → 視為缺失
    - 值 <= 0 → 無效（體重0、身高0、SBP=0 等），視為缺失
    - GCS：三個分量（E/V/M）必須全部有效且各 >= 1，缺任一項不計算 GCS
      （避免部分缺失導致 GCS 被嚴重低估，誤觸 Lv1 警示）
    """
    v = {}

    def _get(field) -> "Optional[float]":
        """取數值；NaN 或非數值 → None。"""
        try:
            val = float(row.get(field))
            return None if np.isnan(val) else val
        except (TypeError, ValueError):
            return None

    def _get_positive(field) -> "Optional[float]":
        """取正數值；NaN、0 或負數 → None。"""
        val = _get(field)
        return val if (val is not None and val > 0) else None

    if (sao2 := _get_positive("SAO2")) is not None:
        v["sao2"] = sao2
    if (sbp := _get_positive("收縮壓")) is not None:
        v["sbp"] = sbp
    if (dbp := _get_positive("舒張壓")) is not None:
        v["dbp"] = dbp
    if sbp is not None and dbp is not None:
        v["map"] = (sbp - dbp) / 3 + dbp
    if (hr := _get_positive("脈搏")) is not None:
        v["hr"] = hr
    if (rr := _get_positive("呼吸")) is not None:
        v["rr"] = rr
    if (temp := _get_positive("體溫")) is not None:
        v["temp"] = temp

    # GCS：三個分量必須全部有效且各 >= 1（最小有效值）
    ge = _get("GCS_E")
    gv = _get("GCS_V")
    gm = _get("GCS_M")
    if (ge is not None and ge >= 1 and
            gv is not None and gv >= 1 and
            gm is not None and gm >= 1):
        v["gcs"] = int(ge + gv + gm)

    return v


def has_shock_signs(nv: dict) -> bool:
    """
    [L] 依總表定義，用可用的 vitals 近似判斷是否存在「休克徵象」。
    休克典型徵象（TTAS 表一/表六）：
      意識模糊（GCS≤13）、呼吸急促（RR>24）、缺氧（SpO2<92）、
      心搏過速（HR>120）、直接低血壓（MAP<65）

    滿足任一項即視為有休克徵象（保守策略，避免漏判）。
    """
    if nv.get("map", 999) < 65:
        return True
    if nv.get("sao2", 100) < 92:
        return True
    if nv.get("gcs", 15) <= 13:
        return True
    if nv.get("rr", 15) > 24:
        return True
    if nv.get("hr", 80) > 120:
        return True
    return False


def assess_vitals(
    nv: dict,
    is_pediatric: "Optional[bool]" = False,
    age_years: "Optional[float]" = None,
    pain_score: "Optional[float]" = None,
) -> "list[VitalAlert]":
    """
    [I] 依官方 TTAS 成人/兒童基準產生 VitalAlert 清單。

    === 成人標準（成人判斷標準.txt）===
      SBP:  1級<70, 2級<90, 3級≥220, 4級200~220
      MAP:  2級<65
      DBP:  3級≥130, 4級110~130
      SpO2: 1級<90, 2級<92, 3級92~94%
      RR:   1級<10
      HR:   1級=(HR<50 or >140) AND SBP<70; 2級=HR<50 or >140
      Temp: 1級>41 or <32; 2級>38 or 32~35
      GCS:  1級3~8; 2級9~13

    === 兒童標準（小兒判斷標準.txt）===
      SBP:  1級<70 (>1歲)
      SpO2: 同成人
      RR:   1級<10
      HR（依年齡）：
        <3月:    1級 <90 or ≥190; 2級 <110 or ≥170
        3月~3歲: 1級 <80 or ≥170; 2級 <90 or ≥150
        >3歲:    1級 <50 or ≥150; 2級 <60 or ≥130
      Temp:
        全年齡: 1級>41 or <32
        <3月:   2級>38 or 32~36
        ≥3月:   2級32~35
      GCS:  同成人
    """
    alerts = []
    ped = bool(is_pediatric)
    sbp = nv.get("sbp")
    shock = has_shock_signs(nv)  # [L] 休克 proxy

    # ── SpO2（成人兒童相同）─────────────────────────────────────────────────────
    if "sao2" in nv:
        s = nv["sao2"]
        if s < 90:
            alerts.append(VitalAlert("SpO2", s, "異常", 1, f"SpO2={s:.1f}% < 90% → 至少1級"))
        elif s < 92:
            alerts.append(VitalAlert("SpO2", s, "異常", 2, f"SpO2={s:.1f}% < 92% → 至少2級"))
        elif s < 95:
            alerts.append(VitalAlert("SpO2", s, "偏低", 3, f"SpO2={s:.1f}% 92-94% → 至少3級"))

    # ── GCS（成人兒童相同）──────────────────────────────────────────────────────
    if "gcs" in nv:
        g = nv["gcs"]
        if g <= 8:
            alerts.append(VitalAlert("GCS", g, "異常", 1, f"GCS={g} 3~8分 → 至少1級"))
        elif g <= 13:
            alerts.append(VitalAlert("GCS", g, "異常", 2, f"GCS={g} 9~13分 → 至少2級"))

    # ── 血壓（SBP / MAP）────────────────────────────────────────────────────────
    # [L] SBP Lv1 = SBP<70（絕對低血壓）OR SBP<90 且有休克徵象
    # [M] 移除高血壓規則（表五附加選項，屬主訴項目非 vital modifier）
    if ped:
        if sbp is not None:
            age_gt1 = (age_years is None) or (age_years > 1)
            if sbp < 70 and age_gt1:
                alerts.append(VitalAlert("SBP", sbp, "異常", 1, f"SBP={sbp:.0f} < 70 → 兒童1級"))
    else:
        if sbp is not None:
            if sbp < 70:
                alerts.append(VitalAlert("SBP", sbp, "異常", 1,
                    f"SBP={sbp:.0f} < 70（絕對低血壓）→ 1級"))
            elif sbp < 90 and shock:
                alerts.append(VitalAlert("SBP", sbp, "異常", 1,
                    f"SBP={sbp:.0f} < 90 且有休克徵象 → 1級"))
            elif sbp < 90:
                alerts.append(VitalAlert("SBP", sbp, "偏低", 2,
                    f"SBP={sbp:.0f} < 90（無休克徵象）→ 2級"))
        # 成人 MAP
        if "map" in nv:
            m = nv["map"]
            if m < 65:
                alerts.append(VitalAlert("MAP", m, "偏低", 2, f"MAP={m:.1f} < 65 → 2級"))

    # ── HR ──────────────────────────────────────────────────────────────────────
    # [L] 成人 HR Lv1 = (HR<50 or >140) AND (SBP<70 OR 有休克徵象)
    if "hr" in nv:
        h = nv["hr"]
        if ped:
            age_m = (age_years * 12) if age_years is not None else None
            if age_m is not None and age_m < 3:
                if h < 90 or h >= 190:
                    alerts.append(VitalAlert("HR", h, "異常", 1,
                        f"HR={h:.0f} 嬰兒(<3月)1級 (<90 or ≥190)"))
                elif h < 110 or h >= 170:
                    alerts.append(VitalAlert("HR", h, "異常", 2,
                        f"HR={h:.0f} 嬰兒(<3月)2級 (<110 or ≥170)"))
            elif age_m is not None and age_m < 36:
                if h < 80 or h >= 170:
                    alerts.append(VitalAlert("HR", h, "異常", 1,
                        f"HR={h:.0f} 幼兒(3月~3歲)1級 (<80 or ≥170)"))
                elif h < 90 or h >= 150:
                    alerts.append(VitalAlert("HR", h, "異常", 2,
                        f"HR={h:.0f} 幼兒(3月~3歲)2級 (<90 or ≥150)"))
            else:
                if h < 50 or h >= 150:
                    alerts.append(VitalAlert("HR", h, "異常", 1,
                        f"HR={h:.0f} 兒童(>3歲)1級 (<50 or ≥150)"))
                elif h < 60 or h >= 130:
                    alerts.append(VitalAlert("HR", h, "異常", 2,
                        f"HR={h:.0f} 兒童(>3歲)2級 (<60 or ≥130)"))
        else:
            hr_abnormal = (h < 50 or h > 140)
            if hr_abnormal and (sbp is not None and sbp < 70 or shock):
                alerts.append(VitalAlert("HR", h, "異常", 1,
                    f"HR={h:.0f} 異常 AND 休克徵象 → 1級"))
            elif hr_abnormal:
                alerts.append(VitalAlert("HR", h, "異常", 2,
                    f"HR={h:.0f} <50 or >140（無休克）→ 2級"))

    # ── RR（<10 → 1級，成人兒童相同）───────────────────────────────────────────
    if "rr" in nv:
        r = nv["rr"]
        if r < 10:
            alerts.append(VitalAlert("RR", r, "異常", 1, f"RR={r:.0f} < 10 → 1級"))

    # ── Temp ────────────────────────────────────────────────────────────────────
    # [M] 移除 Temp >38 Lv2/3 規則（需病容/免疫缺陷/SIRS，CSV 無法判定）
    # 保留純數值可確定的：>41/<32 → Lv1，兒童<3月 低體溫 → Lv2
    if "temp" in nv:
        t = nv["temp"]
        if t > 41 or t < 32:
            alerts.append(VitalAlert("Temp", t, "異常", 1,
                f"體溫={t:.1f}°C >41 or <32 → 1級"))
        elif ped:
            age_m = (age_years * 12) if age_years is not None else None
            if age_m is not None and age_m < 3 and 32 <= t <= 36:
                alerts.append(VitalAlert("Temp", t, "偏低", 2,
                    f"體溫={t:.1f}°C 嬰兒(<3月) 32~36°C → 2級"))
            elif 32 <= t <= 35:
                # ≥3月 低體溫（環境暴露引起，保守觸發）
                alerts.append(VitalAlert("Temp", t, "偏低", 2,
                    f"體溫={t:.1f}°C 兒童 32~35°C（低體溫）→ 2級"))

    # ── [N] 疼痛（動態：僅在有疼痛分數時觸發）──────────────────────────────────
    if pain_score is not None:
        if ped:
            # 兒童：不分中樞/周邊
            if pain_score >= 8:
                alerts.append(VitalAlert("疼痛", pain_score, "重度", 2,
                    f"疼痛={pain_score:.0f}分（重度 8-10）→ 兒童至少2級"))
            elif pain_score >= 4:
                alerts.append(VitalAlert("疼痛", pain_score, "中度", 3,
                    f"疼痛={pain_score:.0f}分（中度 4-7）→ 兒童至少3級"))
            elif pain_score > 0:
                alerts.append(VitalAlert("疼痛", pain_score, "輕度", 4,
                    f"疼痛={pain_score:.0f}分（輕度 <4）→ 兒童至少4級"))
        else:
            # 成人：無疼痛類型資料，採中樞型保守估算
            if pain_score >= 8:
                alerts.append(VitalAlert("疼痛", pain_score, "重度", 2,
                    f"疼痛={pain_score:.0f}分（重度 8-10，依中樞型估算）→ 至少2級"))
            elif pain_score >= 4:
                alerts.append(VitalAlert("疼痛", pain_score, "中度", 3,
                    f"疼痛={pain_score:.0f}分（中度 4-7）→ 至少3級"))
            elif pain_score > 0:
                alerts.append(VitalAlert("疼痛", pain_score, "輕度", 4,
                    f"疼痛={pain_score:.0f}分（輕度 <4，依中樞型估算）→ 至少4級"))

    return alerts


def vital_min_level(alerts: "list[VitalAlert]") -> int:
    """從 alert 清單取最緊急的建議等級（最小數字）。無 alert → 5。"""
    return min((a.min_level for a in alerts), default=5)


def is_vitals_stable(nv: dict, is_pediatric: "Optional[bool]" = False) -> bool:
    """
    判斷生命徵象是否整體穩定（全部在正常範圍內）。
    若任一有值的指標超出正常範圍則回傳 False。
    若沒有任何數值則回傳 None（未知）。
    """
    if not nv:
        return None
    ranges = PEDIATRIC_VITAL_NORMAL_RANGES if is_pediatric else VITAL_NORMAL_RANGES
    for key, (lo, hi) in ranges.items():
        if key in nv:
            val = nv[key]
            if val < lo or val > hi:
                return False
    return True


# ── [F] 生命徵象標注文字 ──────────────────────────────────────────────────────
def _vital_label(value: float, key: str, is_pediatric: "Optional[bool]" = False) -> str:
    """回傳「正常/偏低/偏高/異常」標注，依成人/兒童使用不同正常範圍。"""
    ranges = PEDIATRIC_VITAL_NORMAL_RANGES if is_pediatric else VITAL_NORMAL_RANGES
    lo, hi = ranges.get(key, (None, None))
    if lo is None:
        return "正常"
    if value < lo:
        diff = (lo - value) / lo
        return "異常偏低" if diff > 0.15 else "偏低"
    if value > hi:
        diff = (value - hi) / hi
        return "異常偏高" if diff > 0.15 else "偏高"
    return "正常"


def build_vitals_text(
    row: pd.Series, nv: dict, alerts: "list[VitalAlert]",
    is_pediatric: "Optional[bool]" = False,
) -> str:
    """
    [F] 生理數值字串，每個數值附正常/異常標注（依年齡群使用對應正常範圍）。
    在最後加上「整體生命徵象」穩定性摘要。
    """
    parts = []
    parts.append(f"主訴：{safe_val(row.get('病人主訴', np.nan))}")

    age = _calc_age(row.get("生日"), row.get("急診日期"))
    if age is not None:
        group = "兒童" if age < PEDIATRIC_AGE_CUTOFF else "成人"
        parts.append(f"年齡：{age}歲（{group}）")

    gender_raw = safe_val(row.get("性別", np.nan))
    if gender_raw != "不詳":
        parts.append(f"性別：{GENDER_MAP.get(gender_raw, gender_raw)}")

    vital_fields = [
        ("體溫",  "temp",  "°C"),
        ("收縮壓", "sbp",  "mmHg"),
        ("舒張壓", "dbp",  "mmHg"),
        ("脈搏",  "hr",   "次/分"),
        ("呼吸",  "rr",   "次/分"),
        ("SAO2",  "sao2", "%"),
    ]
    for col, key, unit in vital_fields:
        v = safe_val(row.get(col, np.nan))
        if v != "不詳" and key in nv:
            label = _vital_label(nv[key], key, is_pediatric)
            parts.append(f"{col}：{v}{unit}（{label}）")

    # MAP
    if "map" in nv:
        label = _vital_label(nv["map"], "map", is_pediatric)
        parts.append(f"MAP：{nv['map']:.1f}mmHg（{label}）")

    # GCS
    if "gcs" in nv:
        g = nv["gcs"]
        label = _vital_label(g, "gcs", is_pediatric)
        parts.append(f"GCS：{g}（{label}）")

    for field, label, unit in [("身高", "身高", "cm"), ("體重", "體重", "kg")]:
        v = safe_val(row.get(field, np.nan))
        if v != "不詳":
            parts.append(f"{label}：{v}{unit}")

    pl = safe_val(row.get("瞳孔左", np.nan))
    pr = safe_val(row.get("瞳孔右", np.nan))
    if pl != "不詳" or pr != "不詳":
        parts.append(f"瞳孔：左{pl}，右{pr}")

    # 整體生命徵象摘要
    stable = is_vitals_stable(nv, is_pediatric)
    if not nv:
        parts.append("★整體生命徵象：資料缺失，無法評估")
    elif stable is True:
        parts.append("★整體生命徵象：穩定（所有指標在正常範圍內）")
    elif stable is False and alerts:
        alert_strs = "、".join(a.label() for a in alerts)
        v_min_disp = min(a.min_level for a in alerts)
        parts.append(f"★整體生命徵象：不穩定（{alert_strs}）→ vital_min={v_min_disp}級")
    elif stable is False:
        # 有值但略超出正常範圍，未達到 TTAS 警示門檻
        parts.append("★整體生命徵象：大致穩定（部分指標略超出正常範圍，未達緊急處置門檻）")
    else:
        parts.append("★整體生命徵象：部分資料缺失，無法完整評估")

    return "\n".join(parts)


# ── [G] 擴充 Criteria Pre-filter ─────────────────────────────────────────────
CRITERIA_PATTERN = re.compile(
    r"-\s*(.+?)\s*→\s*(\d)級(（★需查次要調節變數規則）)?"
)

# 各規則：(criterion 比對 pattern, vital_key, 保留條件 lambda, 說明)
_VITAL_FILTER_RULES: list[tuple] = [
    # SpO2
    (re.compile(r"<\s*90\s*%|重度呼吸窘迫"),  "sao2", lambda v: v < 90,  "SpO2<90"),
    (re.compile(r"<\s*92\s*%|中度呼吸窘迫"),  "sao2", lambda v: v < 92,  "SpO2<92"),
    (re.compile(r"92.94\s*%|輕度呼吸窘迫"),   "sao2", lambda v: v < 95,  "SpO2<95"),
    # GCS
    (re.compile(r"GCS\s*3.8|無意識"),         "gcs",  lambda v: v <= 8,  "GCS≤8"),
    (re.compile(r"GCS\s*9.13|意識程度改變"),   "gcs",  lambda v: v <= 13, "GCS≤13"),
    # MAP/SBP
    (re.compile(r"血行動力循環不足|休克(?!後)"), "map",  lambda v: v < 65,  "MAP<65"),
    # HR：若 HR 60-100 且 SBP 正常 → 刪除「血壓或心跳有異於平常數值」catch-all
    (re.compile(r"血壓或心跳有異於病人之平常數值"), "hr_sbp_normal",
     lambda v: v == False, "HR/SBP 正常"),
    # RR
    (re.compile(r"呼吸速率?[>＞]\s*24|呼吸急促"),  "rr",   lambda v: v > 24, "RR>24"),
    (re.compile(r"呼吸速率?[<＜]\s*10|呼吸過慢"),  "rr",   lambda v: v < 10, "RR<10"),
    # Temp（成人 >38 → 2級，故 ≤38 可過濾含「高燒/發燒」條目）
    (re.compile(r"高燒|[>＞]\s*38|體溫.*38"),     "temp", lambda v: v > 38.0, "Temp>38"),
    (re.compile(r"低體溫|[<＜]\s*35"),            "temp", lambda v: v < 35.0, "Temp<35"),
]

# Lv5「生命徵象正常」條件驗證 pattern
_LV5_NORMAL_PATTERN = re.compile(r"生命徵象正常|無發燒|無疼痛不適")
_FEVER_PATTERN = re.compile(r"無發燒")


def _make_extended_vitals(nv: dict, is_pediatric: "Optional[bool]" = False) -> dict:
    """
    建立擴充的 vitals dict，加入複合標誌供 filter rule 使用。
    hr_sbp_normal=False 表示「HR 或 SBP 異常」→ 條目應保留。
    成人標準：HR 50-140、SBP 90-219；兒童(>3歲)：HR 60-129、SBP ≥70
    """
    ev = dict(nv)
    if is_pediatric:
        # 兒童 >3 歲近似正常範圍（最保守）
        hr_normal = 60 <= nv.get("hr", 100) <= 129 if "hr" in nv else None
        sbp_normal = nv.get("sbp", 100) >= 70 if "sbp" in nv else None
        map_normal = True  # 兒童不單獨判斷 MAP
    else:
        # 成人：HR 50-140、SBP 90-219、MAP ≥65
        hr_normal = 50 <= nv.get("hr", 80) <= 140 if "hr" in nv else None
        sbp_normal = 90 <= nv.get("sbp", 120) <= 219 if "sbp" in nv else None
        map_normal = nv.get("map", 80) >= 65 if "map" in nv else None

    if hr_normal is not None and sbp_normal is not None and map_normal is not None:
        # 三者皆正常 → hr_sbp_normal = True → 應刪除「血壓/心跳有異」criterion
        ev["hr_sbp_normal"] = hr_normal and sbp_normal and map_normal
    return ev


def filter_criteria_by_vitals(
    criteria: list[dict], nv: dict, is_pediatric: "Optional[bool]" = False
) -> "tuple[list[dict], list[str]]":
    """
    [G] Rule-based 過濾，回傳 (filtered_criteria, removed_reasons)。
    若過濾後為空，回傳原始清單（安全保底）。
    """
    ev = _make_extended_vitals(nv, is_pediatric)
    stable = is_vitals_stable(nv, is_pediatric)

    kept, removed = [], []
    for c in criteria:
        remove = False
        reason_parts = []

        # 規則一：一般 vital threshold 規則
        for pattern, vk, condition, desc in _VITAL_FILTER_RULES:
            if pattern.search(c["criteria"]) and vk in ev:
                if not condition(ev[vk]):
                    reason_parts.append(desc)
                    remove = True
                    break

        # 規則二：Lv5「生命徵象正常，無發燒，無疼痛不適」條件驗證
        if not remove and c["level"] == 5 and _LV5_NORMAL_PATTERN.search(c["criteria"]):
            # (a) 有發燒 → 刪除「無發燒」條目
            if _FEVER_PATTERN.search(c["criteria"]) and nv.get("temp", 36) >= 37.5:
                reason_parts.append(f"Temp={nv['temp']:.1f}≥37.5 與「無發燒」矛盾")
                remove = True
            # (b) 生命徵象不穩定 → 刪除「生命徵象正常」條目
            elif stable is False:
                reason_parts.append("生命徵象不穩定 與「生命徵象正常」矛盾")
                remove = True

        if remove:
            removed.append(
                f"  刪除[Lv{c['level']}]「{c['criteria'][:40]}」（{', '.join(reason_parts)}）"
            )
        else:
            kept.append(c)

    # 重新編號
    for i, c in enumerate(kept):
        c["id"] = i + 1

    return (kept if kept else criteria), removed


# ── [H] Vital 強制約束 ────────────────────────────────────────────────────────
def constrain_criteria_by_vital_level(
    criteria: list[dict], v_min_level: int
) -> "tuple[list[dict], bool]":
    """
    [H] 若 vital_min_level ≤ 3，將 criteria 限制在 ≤ (v_min_level+1) 級。
    例：vital 指示至少 2 級 → 只保留 Lv1/Lv2 的 criteria。
    回傳 (constrained_criteria, was_constrained)。
    """
    if v_min_level >= 4:
        return criteria, False

    max_allowed = min(v_min_level + 1, 5)   # 略寬一個等級，避免過度限制
    constrained = [c for c in criteria if c["level"] <= max_allowed]
    if not constrained:
        return criteria, False

    for i, c in enumerate(constrained):
        c["id"] = i + 1
    return constrained, True


# ── 標準主訴名稱索引 ──────────────────────────────────────────────────────────
def build_complaint_name_index(collection) -> dict[str, list[dict]]:
    COMPLAINT_SOURCES = {"外傷", "成人非外傷", "兒童", "環境"}
    all_data = collection.get(include=["documents", "metadatas"])
    index: dict[str, list[dict]] = {}
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        if meta.get("source", "") not in COMPLAINT_SOURCES:
            continue
        m = COMPLAINT_HEADER_RE.search(doc)
        if not m:
            continue
        name = m.group(2).strip()
        if len(name) >= MIN_NAME_LEN:
            index.setdefault(name, []).append({"doc": doc, "source": meta["source"]})
    logging.info(f"Complaint name index: {len(index)} 標準主訴名稱")
    return index


def match_complaint_names(
    patient_complaint: str, name_index: dict, is_pediatric: "Optional[bool]"
) -> "tuple[list[str], list[str]]":
    if is_pediatric is True:
        valid_sources = set(PEDIATRIC_COMPLAINT_SOURCES)
    elif is_pediatric is False:
        valid_sources = set(ADULT_COMPLAINT_SOURCES)
    else:
        valid_sources = set(ALL_COMPLAINT_SOURCES)

    sorted_names = sorted(name_index.keys(), key=len, reverse=True)
    matched_docs, matched_names, covered = [], [], []

    for name in sorted_names:
        start = 0
        while True:
            pos = patient_complaint.find(name, start)
            if pos == -1:
                break
            end = pos + len(name)
            if not any(cs <= pos and end <= ce for cs, ce in covered):
                age_filtered = [e["doc"] for e in name_index[name] if e["source"] in valid_sources]
                if age_filtered:
                    covered.append((pos, end))
                    matched_names.append(name)
                    for d in age_filtered:
                        if d not in matched_docs:
                            matched_docs.append(d)
            start = end

    return matched_docs, matched_names


# ── RAG 檢索 ──────────────────────────────────────────────────────────────────
def retrieve_complaint_chunks(
    rag_query: str, embedder: SentenceTransformer, collection,
    is_pediatric: "Optional[bool]" = None,
) -> list[str]:
    prefixed = f"{INSTRUCTION_PREFIX}{rag_query}"
    embedding = embedder.encode(prefixed, normalize_embeddings=True).tolist()
    sources = (ALL_COMPLAINT_SOURCES if is_pediatric is None
               else PEDIATRIC_COMPLAINT_SOURCES if is_pediatric
               else ADULT_COMPLAINT_SOURCES)
    return collection.query(
        query_embeddings=[embedding],
        n_results=N_RETRIEVAL_COMPLAINT,
        where={"source": {"$in": sources}},
    )["documents"][0]


def retrieve_reference_chunks(rag_query: str, embedder, collection) -> list[str]:
    prefixed = f"{INSTRUCTION_PREFIX}{rag_query}"
    embedding = embedder.encode(prefixed, normalize_embeddings=True).tolist()
    return collection.query(
        query_embeddings=[embedding],
        n_results=N_RETRIEVAL_REFERENCE,
        where={"source": {"$in": REFERENCE_SOURCES}},
    )["documents"][0]


# ── Reranking ─────────────────────────────────────────────────────────────────
def format_chunk_headers(docs: list[str]) -> str:
    return "\n".join(
        f"{i+1}. {doc.split(chr(10))[0].strip()}"
        for i, doc in enumerate(docs)
    )


def rerank_chunks(complaint: str, docs: list[str], llm: Llama) -> "tuple[int, str]":
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_RERANK},
            {"role": "user", "content": USER_RERANK.format(
                complaint=complaint, candidates=format_chunk_headers(docs))},
        ],
        temperature=0.1, max_tokens=16, top_p=0.9, repeat_penalty=1.1,
    )
    raw = resp["choices"][0]["message"]["content"].strip()
    return parse_small_int(raw, len(docs)) - 1, raw


# ── 判定依據解析 ──────────────────────────────────────────────────────────────
def parse_criteria(chunk_text: str) -> list[dict]:
    return [
        {"id": i+1, "criteria": m.group(1).strip(),
         "level": int(m.group(2)), "has_star": m.group(3) is not None}
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
def plot_confusion_matrix(y_true, y_pred, save_path):
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


def compute_metrics(y_true, y_pred):
    valid_mask = [p != -1 for p in y_pred]
    yt = [y for y, m in zip(y_true, valid_mask) if m]
    yp = [p for p, m in zip(y_pred, valid_mask) if m]
    n = len(y_true)
    nv = len(yt)
    acc = accuracy_score(yt, yp) if yt else 0.0
    adj = sum(abs(t-p) <= 1 for t, p in zip(yt, yp)) / nv if nv else 0.0
    kappa = cohen_kappa_score(yt, yp, weights="linear") if len(set(yt)) > 1 else 0.0
    labels = [1, 2, 3, 4, 5]
    prec, rec, f1, sup = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    return {"n_total": n, "n_valid": nv, "n_parse_failed": n-nv,
            "accuracy": acc, "adjacent_accuracy": adj, "kappa": kappa,
            "labels": labels, "precision": prec, "recall": rec, "f1": f1, "support": sup}


def save_metrics_report(metrics, save_path):
    lines = [
        "="*60, "TTAS 檢傷分級 LLM 評估報告", "="*60,
        f"總樣本數   : {metrics['n_total']}",
        f"有效預測數 : {metrics['n_valid']}",
        f"解析失敗數 : {metrics['n_parse_failed']}",
        "",
        f"Accuracy          : {metrics['accuracy']:.4f}",
        f"Adjacent Accuracy : {metrics['adjacent_accuracy']:.4f}  (|true-pred| <= 1)",
        f"Linear Kappa      : {metrics['kappa']:.4f}",
        "", "─"*40, "各級指標（僅計有效預測）：",
        f"{'Level':>6}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}",
    ]
    for i, lv in enumerate(metrics["labels"]):
        lines.append(
            f"  Lv{lv}   {metrics['precision'][i]:>10.4f}  "
            f"{metrics['recall'][i]:>8.4f}  {metrics['f1'][i]:>8.4f}  "
            f"{int(metrics['support'][i]):>8d}"
        )
    lines.append("="*60)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"Metrics report saved: {save_path}")
    logging.info("\n".join(lines))


def save_per_level_csv(metrics, save_path):
    pd.DataFrame([
        {"level": lv, "precision": metrics["precision"][i],
         "recall": metrics["recall"][i], "f1": metrics["f1"][i],
         "support": int(metrics["support"][i])}
        for i, lv in enumerate(metrics["labels"])
    ]).to_csv(save_path, index=False, encoding="utf-8-sig")
    logging.info(f"Per-level metrics saved: {save_path}")


# ── Logging ───────────────────────────────────────────────────────────────────
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

    # [N] 疼痛欄位偵測（動態，有才啟用 pain rule-based）
    PAIN_SCORE_COL_CANDIDATES = ["疼痛指數", "疼痛分數", "NRS疼痛分數", "NRS", "VAS", "pain_score", "疼痛評分"]

    logging.info(f"Loading patient data: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH, engine="xlrd")
    triage_col = next((c for c in df.columns if "檢傷" in c and "分級" in c), None)
    if triage_col is None:
        raise ValueError("找不到「檢傷分級」欄位")

    df[triage_col] = pd.to_numeric(df[triage_col], errors="coerce")
    df = df[df[triage_col].isin([1, 2, 3, 4, 5])].copy()
    df[triage_col] = df[triage_col].astype(int)
    logging.info(f"Records: {len(df)}")
    logging.info(f"Level dist:\n{df[triage_col].value_counts().sort_index()}")

    pain_col = next((c for c in PAIN_SCORE_COL_CANDIDATES if c in df.columns), None)
    if pain_col:
        logging.info(f"[N] 疼痛分數欄位偵測到: {pain_col} → 啟用疼痛 rule-based")
    else:
        logging.info("[N] 未偵測到疼痛分數欄位 → 疼痛判斷全交 LLM")

    if FULL_RUN:
        sample = df.reset_index(drop=True)
    else:
        sample = (
            df.groupby(triage_col, group_keys=False)
            .apply(lambda x: x.sample(n=min(SAMPLE_PER_LEVEL, len(x)), random_state=RANDOM_STATE))
            .reset_index(drop=True)
        )
    logging.info(f"\nSampled {len(sample)} records.")

    logging.info(f"\nLoading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    logging.info(f"Loading ChromaDB: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    logging.info(f"ChromaDB: {collection.count()} chunks")

    name_index = build_complaint_name_index(collection)

    logging.info(f"Loading LLM: {MODEL_PATH}")
    llm = Llama(
        model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=4096,
        n_batch=512, n_threads=4, verbose=False, seed=42,
    )
    logging.info("LLM loaded.\n")

    y_true, y_pred, records = [], [], []
    n_exact_hit = 0

    for idx, (_, row) in enumerate(sample.iterrows()):
        true_level = int(row[triage_col])
        is_ped = is_pediatric_from_row(row)
        age_years = _calc_age(row.get("生日"), row.get("急診日期"))
        rag_query = safe_val(row.get("病人主訴", np.nan))

        # [I][L][N] 生命徵象評估（官方標準 + 休克 proxy + 疼痛動態）
        nv = extract_numeric_vitals(row)
        pain_score = None
        if pain_col:
            try:
                ps = float(row.get(pain_col))
                if not pd.isna(ps) and 0 <= ps <= 10:
                    pain_score = ps
            except (TypeError, ValueError):
                pass
        alerts = assess_vitals(nv, is_ped, age_years=age_years, pain_score=pain_score)
        v_min = vital_min_level(alerts)
        vitals_text = build_vitals_text(row, nv, alerts, is_ped)

        # ── 路徑選擇：EXACT 或 RAG+RERANK ─────────────────────────────────────
        matched_docs, matched_names = match_complaint_names(rag_query, name_index, is_ped)
        retrieval_method = ""
        rerank_info = ""
        selected_doc = ""

        if matched_docs:
            n_exact_hit += 1
            retrieval_method = "EXACT"
            if len(matched_docs) > 1:
                rerank_idx, raw_rerank = rerank_chunks(rag_query, matched_docs, llm)
                selected_doc = matched_docs[rerank_idx]
                rerank_info = f"EXACT {len(matched_docs)}個 → Rerank選{rerank_idx+1}（{raw_rerank}）"
            else:
                selected_doc = matched_docs[0]
                rerank_info = "EXACT 1個（直接使用）"
        else:
            retrieval_method = "RAG+RERANK"
            complaint_docs = retrieve_complaint_chunks(rag_query, embedder, collection, is_ped)
            rerank_idx, raw_rerank = rerank_chunks(rag_query, complaint_docs, llm)
            selected_doc = complaint_docs[rerank_idx]
            rerank_info = (
                f"RAG TOP-5:\n{format_chunk_headers(complaint_docs)}\n"
                f"Rerank選{rerank_idx+1}（{raw_rerank}）"
            )

        # [G] Criteria 解析 + Rule-based 過濾（依年齡群使用對應閾值）
        raw_criteria = parse_criteria(selected_doc)
        filtered_criteria, removed_reasons = filter_criteria_by_vitals(raw_criteria, nv, is_ped)

        # [H] Vital 強制約束
        constrained_criteria, was_constrained = constrain_criteria_by_vital_level(
            filtered_criteria, v_min
        )

        # Criterion Selection
        raw_selection = ""
        selected_criterion = None
        stage2_triggered = False
        final_level = -1

        criteria_to_use = constrained_criteria

        if criteria_to_use:
            # [H] 有 vital alert 時加入警示訊息
            if alerts and was_constrained:
                alert_strs = "\n".join(f"  • {a.reason}" for a in alerts)
                user_msg = USER_SELECTION_CONSTRAINED.format(
                    vitals=vitals_text,
                    vital_alerts=alert_strs,
                    criteria_list=format_criteria_list(criteria_to_use),
                )
            else:
                user_msg = USER_SELECTION.format(
                    vitals=vitals_text,
                    criteria_list=format_criteria_list(criteria_to_use),
                )

            sel_resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_SELECTION},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1, max_tokens=16, top_p=0.9, repeat_penalty=1.1,
            )
            raw_selection = sel_resp["choices"][0]["message"]["content"].strip()
            sel_idx = parse_small_int(raw_selection, len(criteria_to_use))
            selected_criterion = criteria_to_use[sel_idx - 1]

            initial_level = selected_criterion["level"]
            stage2_triggered = selected_criterion["has_star"]

            if stage2_triggered:
                ref_docs = retrieve_reference_chunks(rag_query, embedder, collection)
                sec_resp = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": SYSTEM_SECONDARY},
                        {"role": "user", "content": USER_SECONDARY.format(
                            vitals=vitals_text,
                            selected_criteria=selected_criterion["criteria"],
                            initial_level=initial_level,
                            reference_context="\n---\n".join(ref_docs),
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
            fallback_resp = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "你是台灣急診室資深檢傷護理師，只輸出一個數字1–5代表TTAS等級。"},
                    {"role": "user", "content": f"【病患生理資料】\n{vitals_text}\n\n【TTAS參考】\n{selected_doc}\n\n等級："},
                ],
                temperature=0.1, max_tokens=8, top_p=0.9, repeat_penalty=1.1,
            )
            raw_selection = fallback_resp["choices"][0]["message"]["content"].strip()
            final_level = parse_grade(raw_selection)

        # ── 雙軌制：Vital 軌 + LLM 軌，取 min ─────────────────────────────────
        llm_level = final_level   # LLM 軌結果（含 Stage2）
        if llm_level != -1:
            final_level = min(v_min, llm_level)
        else:
            final_level = -1

        y_true.append(true_level)
        y_pred.append(final_level)

        selected_str = (
            f"{selected_criterion['criteria']} → {selected_criterion['level']}級"
            f"{'（★）' if selected_criterion['has_star'] else ''}"
            if selected_criterion else "（無法解析）"
        )
        correct_mark = "[OK]" if final_level == true_level else "[NG]"

        records.append({
            "idx": idx, "true_level": true_level, "pred_level": final_level,
            "vital_track": v_min, "llm_track": llm_level,
            "retrieval_method": retrieval_method,
            "matched_names": ", ".join(matched_names),
            "selected_criterion": selected_str,
            "raw_selection": raw_selection,
            "stage2_triggered": stage2_triggered,
            "vital_min_level": v_min,
            "vital_alerts": "; ".join(a.label() for a in alerts),
            "n_raw": len(raw_criteria),
            "n_filtered": len(filtered_criteria),
            "n_constrained": len(constrained_criteria),
            "was_constrained": was_constrained,
            "rag_query": rag_query,
            "vitals": vitals_text.replace("\n", " | "),
        })

        # ── Log ───────────────────────────────────────────────────────────────
        SEP = "─" * 50
        alert_lines = (
            "  " + "、".join(a.label() for a in alerts) + f" → vital_min={v_min}級"
            if alerts else "  （無異常警示）"
        )
        criteria_flow = (
            f"  Criteria: {len(raw_criteria)} → filter→{len(filtered_criteria)}"
            f" → constrain→{len(constrained_criteria)}"
            f"{'（強制約束）' if was_constrained else ''}"
        )

        logging.info(
            f"\n{'═'*50}\n"
            f"[{idx+1}/{len(sample)}] GT={true_level} | Vital={v_min} | LLM={llm_level} | "
            f"Final=min({v_min},{llm_level})={final_level}  [{retrieval_method}] {correct_mark}\n"
            f"{'═'*50}\n"
            f"【病患】\n"
            f"{vitals_text}\n"
            f"  ⚡ 生命徵象警示：{alert_lines}\n"
            f"{SEP}\n"
            f"【RAG/{retrieval_method}】\n"
            f"{rerank_info}\n"
            f"  Chunk: {selected_doc.split(chr(10))[0]}\n"
            f"{criteria_flow}\n"
            f"  Selected: {selected_str}\n"
            f"  Stage2: {'YES → ' + raw_selection.split('Stage2:')[-1].strip() if stage2_triggered else 'no'}\n"
            f"  RAW: {raw_selection[:80]}\n"
            f"{SEP}\n"
            f"【最終】min(vital={v_min}, LLM={llm_level}) = {final_level}  {correct_mark}\n"
        )

    logging.info(f"\nDone. Total: {len(y_true)}")
    logging.info(f"EXACT: {n_exact_hit}/{len(y_true)} ({n_exact_hit/len(y_true)*100:.1f}%)")

    pred_path = os.path.join(RESULTS_DIR, f"predictions{suffix}.csv")
    pd.DataFrame(records).to_csv(pred_path, index=False, encoding="utf-8-sig")

    if not any(p != -1 for p in y_pred):
        logging.error("No valid predictions.")
        return

    metrics = compute_metrics(y_true, y_pred)
    yt = [t for t, p in zip(y_true, y_pred) if p != -1]
    yp = [p for p in y_pred if p != -1]

    plot_confusion_matrix(yt, yp, os.path.join(RESULTS_DIR, f"confusion_matrix{suffix}.png"))
    save_metrics_report(metrics, os.path.join(RESULTS_DIR, f"metrics_report{suffix}.txt"))
    save_per_level_csv(metrics, os.path.join(RESULTS_DIR, f"per_level_metrics{suffix}.csv"))
    logging.info(f"\n=== Done! Results: {RESULTS_DIR} ===")


if __name__ == "__main__":
    main()
