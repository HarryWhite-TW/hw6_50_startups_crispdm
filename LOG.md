# HW6 Project LOG

本文件記錄 HW6 專案的 AI-assisted workflow 與交付過程。專案主題為 `50 Startups Profit Prediction + CRISP-DM`，使用 AI-generated synthetic 50 Startups-style dataset 進行教育用途的機器學習流程展示。

## 2026-06-12 Project Start

- 從零建立 `hw6_50_startups_crispdm` 專案。
- 專案定位為 GitHub homework showcase，主題是 Startup Profit Prediction。
- 採用 CRISP-DM 架構整理作業內容，包含 Business Understanding、Data Understanding、Data Preparation、Modeling、Evaluation、Deployment。
- 主要產出檔案：`README.md`、`hw6.md`、`docs/HW6_TECHNICAL_WHITEPAPER.md`、`src/`、`data/`、`outputs/`。

## Dataset Decision

- 老師沒有提供真實 dataset。
- 因此使用 `src/generate_synthetic_startups.py` 產生 AI-generated synthetic 50 Startups-style dataset。
- 產出資料檔案為 `data/50_Startups.csv`，包含 50 筆資料。
- 欄位包含 `R&D Spend`、`Administration`、`Marketing Spend`、`State`、`Profit`。
- 專案明確聲明：此資料只用於 educational CRISP-DM workflow demonstration，不是真實商業研究資料，也不做真實世界因果推論。

## Core ML Workflow

- 新增並使用 `src/generate_synthetic_startups.py` 產生 synthetic dataset。
- 新增並使用 `src/solve_50_startups_crispdm.py` 執行主要 ML 流程。
- 模型使用 `LinearRegression` 作為 baseline regression model。
- `State` 使用 `OneHotEncoder`，避免把類別變數誤當成有大小順序的數值。
- 評估指標包含 R2、MAE、RMSE、5-fold cross-validation mean R2、5-fold cross-validation std。
- 目前主要模型結果：
  - Test R2 Score: 0.9460
  - Test MAE: 5578.56
  - Test RMSE: 6971.82
  - 5-fold CV mean R2: 0.9694
  - 5-fold CV std: 0.0165

## README Showcase

- 在 README 開頭加入 `assets/readme_hero.png` 作為主要 hero infographic。
- 加入 `outputs/feature_selection_performance_allinone.png` 作為進階 feature selection comparison 展示圖。
- 在 README 最上方加入 GitHub Pages interactive dashboard 連結。
- README 保留 synthetic dataset notice，避免過度宣稱模型結果。

## Whitepaper

- 新增 `docs/HW6_TECHNICAL_WHITEPAPER.md`。
- 白皮書用於補充資料來源、CRISP-DM 流程、資料處理、模型、評估、限制與解讀。
- 文件目標是達成老師要求的 20,000 中文字以上白皮規格書。
- 後續只追加延伸章節，避免重寫整份白皮書造成版本混亂。

## Advanced Feature Selection

- 新增 `src/feature_selection_comparison.py`。
- 比較方法包含：
  - Sequential Forward Selection (SFS / Forward)
  - RFE
  - SelectKBest
  - Lasso
  - Random Forest feature importance
- `State` 先做 one-hot encoding 並使用 `drop_first=True`。
- 每個方法產生完整 feature ranking，並測試 top 1 到 top 5 features。
- 使用同一個 `LinearRegression` 模型比較 RMSE 與 R2。
- 產出圖檔：`outputs/feature_selection_performance_allinone.png`。
- 目前最佳觀察結果為 SFS / Forward 使用 4 個 features。

## Interactive Website

- 新增 `index.html` 作為 GitHub Pages 靜態入口。
- 新增 `site/style.css` 控制展示頁視覺設計。
- 新增 `site/app.js` 提供互動資料篩選、Chart.js 圖表與 feature selection comparison dashboard。
- 網站不使用 npm、Vite、React、Streamlit、PPT 或影片。
- 網站主題包含：
  - 50 Startups Profit Prediction
  - CRISP-DM Machine Learning Workflow
  - Feature Selection Comparison Dashboard
- 網站明確標註 synthetic dataset limitation、educational workflow demonstration、not real business research、no causal claim。

## Visual Gallery

- 新增 `src/generate_visual_gallery.py`。
- 使用現有 `data/50_Startups.csv` 產生多張 README 與網站可用圖片。
- 新增輸出圖：
  - `outputs/dataset_overview_dashboard.png`
  - `outputs/spending_profit_relationships.png`
  - `outputs/correlation_heatmap.png`
  - `outputs/model_metrics_summary.png`
- 保留既有圖：
  - `outputs/feature_selection_performance.png`
  - `outputs/feature_selection_performance_allinone.png`
- 圖片風格保持乾淨、英文標題、適合 GitHub 展示。

## GitHub Delivery

- 專案交付流程以 GitHub repo 為保存位置。
- 每次完成一組可檢查的變更後，應透過 Git commit 記錄變更，再 push 到 GitHub 保存。
- 本次補件依照使用者限制尚未 commit、尚未 push。
- 建議 commit 範圍包含新增網站、LOG、visual gallery script、輸出圖與文件更新。

## Limitations

- Dataset 是 AI-generated synthetic 50 Startups-style dataset，不是原始 Kaggle dataset，也不是真實公司資料。
- 資料只有 50 rows，模型評估容易受到 split 與 synthetic formula 影響。
- Feature ranking 只代表不同方法在此 synthetic dataset 上的模型行為。
- 本專案不宣稱 R&D Spend、Marketing Spend、State 或其他欄位會在真實世界中造成 profit 改變。
- 專案用途是 educational CRISP-DM workflow demonstration only。
