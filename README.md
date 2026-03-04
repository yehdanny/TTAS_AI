# TTAS — 台灣急診五級檢傷分級本地推理系統

基於 **RAG + 本地 LLM** 的急診檢傷分類評估工具。
以 TTAS（Taiwan Triage and Acuity Scale）官方 PDF 指引為知識庫，對病患主訴與生命徵象進行自動分級（Level 1–5），並計算多項評估指標。

---

## 環境需求

| 項目 | 規格 |
|---|---|
| GPU | NVIDIA RTX 2070（8 GB VRAM），CUDA 12.4 |
| Python | 3.10.x |
| OS | Windows 11 |

---

## 檔案結構

```
TTAS/
├── README.md
├── requirements.txt             # 套件清單（不含 torch / llama-cpp-python）
│
├── build_rag.py                 # 索引建立 v2（PDF → 以主訴為單位 chunk）★ 推薦
├── build_rag_v3.py              # 索引建立 v3（手動 Python chunks 匯入）
├── main.py                      # 推理評估（對應 v2 索引）★ 推薦
├── main_v3.py                   # 推理評估（對應 v3 索引）
│
├── data/
│   ├── rag_knowledge/           # 8 份 TTAS 官方 PDF 指引（知識庫來源）
│   │   ├── 急診五級檢傷分類基準修正版-兒童.pdf
│   │   ├── 急診五級檢傷分類基準修正版-外傷.pdf
│   │   ├── 急診五級檢傷分類基準修正版-成人非外傷(1080611).pdf
│   │   └── ... (共 8 份)
│   ├── patient_data/
│   │   └── total_data.xls       # 病患資料（9653 筆，BIFF8 格式）
│   ├── chunks/                  # 手動切割的 Python chunk 檔案（v3 用）
│   │   ├── chunks_list_A01_A13_non_trauma.py
│   │   ├── chunks_list_P01_P13_pediatric.py
│   │   ├── chunks_list_T01_14_trauma.py
│   │   ├── chunks_list_E01_environment.py
│   │   └── chunk_list_adjustment.py
│   └── chroma_db/               # 由 build_rag.py 自動生成，勿手動修改
│                                #   ttas_guidelines (v2, 320 chunks)
│                                #   ttas_v3         (v3, 318 chunks)
│
├── model/
│   └── Qwen3-4B-Instruct-2507-Q4_K_M.gguf   # 本地量化 LLM
│
├── results/                     # v2 推理結果（由 main.py 生成）
│   ├── predictions.csv
│   ├── confusion_matrix.png
│   ├── metrics_report.txt
│   └── per_level_metrics.csv
│
└── results_v3/                  # v3 推理結果（由 main_v3.py 生成）
    ├── predictions.csv
    ├── confusion_matrix.png
    ├── metrics_report.txt
    └── per_level_metrics.csv
```

---

## 安裝流程

### 1. 建立虛擬環境

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. 安裝基礎套件

```powershell
pip install -r requirements.txt
```

### 3. 安裝 PyTorch（GPU 版）

```powershell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 4. 安裝 llama-cpp-python（CUDA 編譯版）

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

> **驗證 GPU 支援**
> ```powershell
> python -c "from llama_cpp import llama_supports_gpu_offload; print(llama_supports_gpu_offload())"
> # 應輸出 True
> ```

---

## 執行流程

### Step 1：建立 RAG 向量索引（只需執行一次）

```powershell
python build_rag.py
```

預估時間約 10–15 分鐘（BGE-M3 CPU 嵌入為主要瓶頸）。

成功標誌：
```
RAG index built successfully. 320 chunks in ChromaDB.
```

### Step 2：執行推理與評估

#### 分層抽樣（各級 70 筆，共 350 筆）

```powershell
python main.py
```

#### 全量預測（全部 9653 筆）

```powershell
python main.py --full
```

---

## 三版 RAG Chunking 策略說明

本專案對 RAG 的 chunking 方式進行了三輪實驗，逐步改進知識庫的組織方式。

### v1：固定長度切割（原始版）

- **方式**：用 `pdfplumber.extract_text()` 取純文字，以雙換行分段，超過 400 字硬切（50 字 overlap）
- **問題**：TTAS PDF 表格無雙換行，導致所有 PDF 只有 1 段，所有內容被硬切為隨機片段，破壞了「主訴 → 判定依據 → 級數」的結構
- **chunk 數**：~400 個
- **collection**：無（已被 v2 取代）

### v2：以主訴為單位結構化切割（推薦版）

- **方式**：改用 `pdfplumber.extract_tables()` 提取結構化表格，以「主訴」為單位將所有判定依據聚合為一個 chunk
- **修正版覆蓋**：成人非外傷 (1080611) 覆蓋對應條目；兒童(第二次修正) 覆蓋相關主訴；總表(第二次修正) 覆蓋表六（兒童首要調節變數）
- **★ 標記**：帶次要調節變數（`＊`）的條目標記「★需查次要調節變數規則」，Prompt 中說明判定流程
- **chunk 格式**：
  ```
  【主訴】E0101昆蟲螫傷（大分類：E01環境）
  判定依據 → TTAS級數：
  - 重度呼吸窘迫(<90%) → 1級
  - 過去曾出現嚴重過敏反應 → 2級（★需查次要調節變數規則）
  ...
  ```
- **chunk 數**：320 個（302 主訴 + 18 總表調節變數）
- **collection**：`ttas_guidelines`

### v3：手動精選 chunk（對照版）

- **方式**：從 `data/chunks/` 下的 Python 檔案匯入，共 5 個類別，每個主訴一個 dict
- **chunk 格式**：
  ```
  【主訴】昆蟲螫傷
  - 一級 (復甦急救)：重度呼吸窘迫(<90%)。休克。無意識(GCS3-8)。
  - 二級 (危急)：中度呼吸窘迫(<92%)。...
  可調節分級項目 : 過去曾出現嚴重過敏反應。...
  ```
- **chunk 數**：318 個（成人非外傷 125 + 兒童 127 + 外傷 41 + 環境 11 + 調節 14）
- **collection**：`ttas_v3`

---

## 實驗結果比較

所有版本均使用相同條件：分層抽樣 350 筆（各級各 70 筆）、Qwen3-4B-Instruct GGUF、BGE-M3 嵌入、Top-5 檢索。

### 整體指標

| 指標 | v1（硬切 400 字） | v2（主訴結構化）★ | v3（手動精選） |
|---|:---:|:---:|:---:|
| **Accuracy** | 52.86% | **55.71%** | 51.71% |
| **Adjacent Accuracy** | **93.43%** | 90.57% | 92.86% |
| **Linear Kappa** | 0.6313 | **0.6344** | 0.6201 |
| Parse Failure | 0/350 | 0/350 | 0/350 |

### 各級 F1

| Level | v1 硬切 | v2 主訴結構化 ★ | v3 手動精選 |
|---|:---:|:---:|:---:|
| Lv1（復甦急救） | 0.762 | **0.702** | 0.648 |
| Lv2（危急） | 0.580 | 0.576 | **0.579** |
| Lv3（緊急） | 0.441 | 0.449 | **0.450** |
| Lv4（次緊急） | 0.373 | **0.410** | 0.310 |
| Lv5（非緊急） | 0.515 | **0.661** | 0.598 |

---

## 版本選擇結論

**推薦使用 v2（`build_rag.py` + `main.py`）**。

### 理由

**v2 在大多數指標上表現最好：**
- Accuracy 55.71%，是三版最高，較 v1 提升 +2.85%、較 v3 提升 +4.00%
- Linear Kappa 0.6344 最高，達到 Substantial agreement 水準
- Lv4 F1（0.410 vs v1 的 0.373、v3 的 0.310）與 Lv5 F1（0.661 vs v1 的 0.515、v3 的 0.598）改善最顯著

**v3 手動 chunks 的侷限：**
- Lv4（次緊急）F1 僅 0.310，是三版最差。手動 chunk 的 `query_text` 為精簡關鍵詞，次緊急案例主訴語意模糊時 RAG 難以精準匹配
- 整體 Accuracy 反而比 v1 更低，顯示人工整理的格式不一定優於從原始 PDF 結構化提取

**v1 的問題：**
- 硬切破壞表格結構，同一主訴的判定依據被拆散到不同 chunk，導致 Lv5 F1 特別低（0.515）
- Accuracy 和 Kappa 均弱於 v2

**v3 的唯一優勢：**
- Adjacent Accuracy 92.86%，略高於 v2 的 90.57%，表示 v3 的預測誤差更集中在相鄰一級。若臨床應用中「預測偏一級」可接受，v3 也是合理選擇

### 如需進一步改善

目前瓶頸在 Lv3（緊急）F1 約 0.45，是最難分辨的類別，Lv4 其次。可考慮：
1. 增加 Top-K 從 5 提升至 7–10（較多參考指引）
2. 在 Prompt 中加入更明確的生命徵象判斷規則（如具體呼吸次數、血壓閾值）
3. 針對 Lv3/Lv4 易混淆對（緊急 vs 次緊急）設計專屬的 re-ranking 或 fallback 機制

---

## main.py 內部流程詳解

```
載入病患資料 (total_data.xls)
    │
    ▼
分層抽樣 / 全量使用
    │  每級各 70 筆（抽樣模式）或全部 9653 筆（--full 模式）
    ▼
載入嵌入模型 BAAI/bge-m3（CPU）
    │
    ▼
載入 ChromaDB（data/chroma_db/，collection: ttas_guidelines）
    │
    ▼
載入 Qwen3-4B LLM → GPU offload（n_gpu_layers=-1）
    │
    ▼
對每筆病患資料執行以下流程：
    │
    ├─ 1. 組合查詢字串
    │       主訴 + 性別 + 體溫 + 收縮壓 + 舒張壓 + 脈搏 + 呼吸 + 血氧 + GCS
    │       缺值填「不詳」
    │
    ├─ 2. RAG 檢索
    │       加上 BGE-M3 指令前綴後嵌入查詢字串
    │       從 ChromaDB 取回最相關的 5 個 TTAS 指引片段（Top-5）
    │
    ├─ 3. LLM 推理
    │       System prompt：檢傷護理師角色 + 次要調節變數流程 + 各級定義 + /no_think
    │       User prompt：病患資料 + RAG 參考片段
    │       參數：temperature=0.1, max_tokens=20
    │
    └─ 4. 解析輸出
            移除 <think>...</think> 殘留
            → 比對「第X級」中文序數
            → 比對獨立數字 \b[1-5]\b
            → 比對任意 [1-5]
            → 以上皆失敗 → 記錄為 -1（parse failure）
    │
    ▼
計算評估指標
    ├─ Accuracy（嚴格完全命中）
    ├─ Adjacent Accuracy（|誤差| ≤ 1 級）
    ├─ Linear Weighted Cohen's Kappa
    └─ 各級 Precision / Recall / F1
    │
    ▼
輸出結果至 results/
```

---

## 評估指標說明

| 指標 | 說明 | 臨床意義 |
|---|---|---|
| **Accuracy** | 預測完全正確的比例 | 嚴格標準，五級分類天花板約 60–70% |
| **Adjacent Accuracy** | \|真實 − 預測\| ≤ 1 的比例 | 臨床上相鄰一級誤差可接受 |
| **Linear Kappa** | 加權一致性（0=隨機, 1=完美） | >0.6 為 Substantial agreement |
| **Precision / Recall / F1** | 各級別獨立評估 | 找出哪一級最難分類 |

---

## 重要技術備註

| 問題 | 解法 |
|---|---|
| Excel 為 BIFF8 格式 | `pd.read_excel(..., engine="xlrd")` |
| PyTorch CPU-only 無法做 BGE-M3 GPU 加速 | 嵌入跑 CPU，約 5–10 分鐘，可接受 |
| Qwen3 `/no_think` 偶爾失效 | Parser 第一步清除 `<think>` 區塊 |
| Windows 終端機編碼 | 報告檔案用 UTF-8 寫入；print 遇錯自動 fallback ASCII |
| chromadb 版本 | 固定 1.0.21，1.5.x 以上 API 有 breaking change |
| matplotlib 無顯示器 | 腳本開頭 `matplotlib.use("Agg")` |
| n_ctx 設定 | 設為 4096，v2 每個主訴 chunk 含 20–30 條判定依據，2048 會溢出 |
