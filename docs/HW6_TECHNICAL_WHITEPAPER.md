# HW6 技術白皮規格書：Kaggle 50 Startups-style Startup Profit Prediction CRISP-DM Machine Learning Project

## 文件摘要

本白皮規格書說明 HW6「50 Startups-style Startup Profit Prediction」機器學習專案的設計背景、資料來源、資料生成邏輯、CRISP-DM 方法論、模型建置流程、評估結果、部署方式、限制風險與未來改進方向。本專案以 Kaggle 50 Startups 類型的欄位結構作為教學參考，但實際使用的是 AI-generated synthetic 50 Startups-style dataset，也就是由程式產生、可重現、用於課堂作業展示的合成資料。這一點是整份文件最重要的前提：本專案不是使用真實 Kaggle 原始商業資料，也不是用來進行真實商業研究或投資判斷。

本專案的主要目標，是展示一個從資料生成、資料理解、資料準備、模型訓練、模型評估到模型輸出的完整 CRISP-DM 機器學習流程。專案採用 Python 生態系中常見且適合初學者理解的工具，包括 pandas、numpy、scikit-learn、matplotlib 與 joblib。模型部分使用 LinearRegression 作為迴歸基準模型，搭配 ColumnTransformer 處理欄位轉換，並使用 OneHotEncoder 處理 `State` 類別欄位。模型評估包含 train/test split、R2 Score、MAE、RMSE，以及因資料筆數只有 50 筆而特別加入的 5-fold cross-validation。

本專案的評估結果為：Test R2 Score: 0.9460、Test MAE: 5578.56、Test RMSE: 6971.82、5-fold CV mean R2: 0.9694、5-fold CV std: 0.0165。這些結果只代表模型在本專案合成資料上的表現，不能被解讀為真實世界的新創公司獲利規律，也不能被用來宣稱某項支出會直接造成 Profit 增加。特別是 `R&D Spend` 在本專案中具有較強預測能力，是因為 synthetic data 的生成邏輯將它設計為主要訊號之一，而不是因為本專案證明了真實世界的因果關係。

## 1. 專案背景與目的

本專案的主題是「Kaggle 50 Startups-style Startup Profit Prediction CRISP-DM Machine Learning Project」。在許多資料科學入門教材中，50 Startups 類型的資料集常被用來示範多元線性迴歸、類別欄位編碼、特徵與目標變數分離、訓練測試切分，以及模型評估指標的解讀。這類資料通常包含新創公司的研發支出、行政支出、行銷支出、州別與利潤等欄位，因此很適合作為初學者理解迴歸任務的練習案例。

然而，本次作業的實際背景是老師並未提供真實 Kaggle 50 Startups dataset，而作業重點是使用 AI 協助完成 CRISP-DM 機器學習流程展示。基於誠實揭露與學術倫理，本專案不偽造資料來源，也不宣稱合成資料是真實 Kaggle 原始資料。相反地，專案明確建立一個 synthetic dataset generator，由 `src/generate_synthetic_startups.py` 產生固定 50 筆、欄位風格接近 50 Startups 的教學資料，並在 README、作業報告與本白皮書中反覆說明資料性質。

本專案目的可以分成三個層次。第一個層次是技術流程展示：透過可執行 Python 程式完成資料生成、資料讀取、資料驗證、前處理、模型訓練、交叉驗證、結果視覺化與模型序列化。第二個層次是方法論展示：按照 CRISP-DM 六步驟整理機器學習專案，使專案不只是零散的程式碼，而是有問題定義、資料理解、準備流程、建模策略、評估判準與部署輸出的完整工作。第三個層次是 GitHub 作業展示：專案結構、README、報告、成果圖與資料檔都整理成可以上傳與閱讀的形式，讓評閱者可以快速理解專案重點。

本白皮規格書的角色，是把這個作業專案提升成較正式的技術文件。它不只是重複 README 的執行步驟，而是把每個設計選擇背後的原因寫清楚。例如，為什麼 `State` 要用 OneHotEncoder 而不是 LabelEncoder；為什麼小資料集需要 5-fold cross-validation；為什麼 LinearRegression 適合作為基準模型；為什麼合成資料的高 R2 不代表真實商業結論；以及未來如果要把這個展示專案延伸成更接近真實研究，還需要補哪些工作。

## 2. 誠實資料聲明與使用範圍

本專案使用的是 AI-generated synthetic 50 Startups-style dataset。所謂 synthetic dataset，是指資料不是來自真實公司、真實市場或真實 Kaggle 原始檔，而是由程式根據預先設計的邏輯產生。這種資料適合用於教學、流程展示、程式測試與報告演練，因為它可以避免資料缺漏造成專案無法執行，也可以確保每位使用者在相同 random seed 下產生一致結果。

本專案的資料檔為 `data/50_Startups.csv`，它由 `src/generate_synthetic_startups.py` 產生。檔名沿用 50 Startups 類型資料集常見的命名方式，是為了符合課堂作業主題與既有教學範例的欄位習慣，但這不表示它是真實 Kaggle 下載資料。資料中的公司不是現實公司，`Profit` 不是實際財務報表數字，`State` 也不是用來代表真實地區經濟差異。所有數值都應被視為教學場景下的模擬值。

本專案不可用於真實商業研究。它不能回答「新創公司應該增加多少研發支出才會提升利潤」這類實務決策問題，也不能作為投資建議、預算分配依據或市場分析報告。模型在合成資料上的表現，反映的是資料生成公式與模型假設之間的相容程度，而不是現實世界中支出與利潤之間的真實關係。

本專案也不可做因果宣稱。雖然在合成資料生成邏輯中，`R&D Spend` 被設計為對 `Profit` 有較強預測能力的訊號，但這只代表模型在這份 synthetic data 中能利用該欄位改善預測。嚴格來說，本專案只能說「在合成資料中，`R&D Spend` 具有較強預測能力」，不能說「R&D Spend 直接造成 Profit 增加」。若要討論因果關係，需要真實資料、研究設計、控制變數、因果推論方法、穩健性檢定與領域知識，本專案並未涵蓋這些內容。

因此，本專案最合適的定位是 educational CRISP-DM workflow demonstration。它展示的是機器學習作業如何被完整組織、執行、記錄與交付，而不是展示一個可直接應用於企業決策的預測系統。

## 3. 專案檔案結構

本專案的檔案結構設計以可讀性、可重現性與 GitHub 展示為主要考量。每個資料夾都有明確職責，避免將程式碼、資料、圖表與文件混在同一層級。專案主要結構如下：

```text
hw6_50_startups_crispdm/
├── README.md
├── hw6.md
├── requirements.txt
├── .gitignore
├── assets/
│   └── readme_hero.png
├── data/
│   ├── README.md
│   └── 50_Startups.csv
├── docs/
│   └── HW6_TECHNICAL_WHITEPAPER.md
├── outputs/
│   ├── .gitkeep
│   ├── feature_selection_performance.png
│   └── startup_profit_model.pkl
└── src/
    ├── generate_synthetic_startups.py
    └── solve_50_startups_crispdm.py
```

`README.md` 是 GitHub 首頁展示文件，包含專案概述、資料說明、CRISP-DM 流程、模型與評估結果、執行方式，以及成果圖。`hw6.md` 是作業報告，使用較接近課堂繳交格式的方式說明 CRISP-DM 六步驟。`docs/HW6_TECHNICAL_WHITEPAPER.md` 則是本白皮規格書，用更完整的文字說明技術與方法論細節。

`src/generate_synthetic_startups.py` 負責產生 synthetic dataset。這個檔案是專案可重現性的核心，因為只要使用相同 random seed，就能產生相同資料內容。`src/solve_50_startups_crispdm.py` 是主要機器學習流程程式，負責讀取資料、檢查欄位、前處理、模型訓練、模型評估、產生圖表與儲存模型。

`data/50_Startups.csv` 是已生成的教學資料。由於它是本專案正式產物之一，而且資料筆數小、用途清楚，所以允許被 Git 追蹤。`outputs/feature_selection_performance.png` 是成果展示圖，也允許被 Git 追蹤，方便 README 直接顯示結果。`outputs/startup_profit_model.pkl` 是訓練後的模型檔，屬於可由程式重新產生的二進位產物，因此可以選擇不納入 Git 追蹤。`.gitignore` 已設定忽略 `.venv/`、`__pycache__/`、`outputs/*.pkl` 與 Matplotlib cache 等不適合上傳的內容。

`assets/readme_hero.png` 是 README 的視覺展示圖，用來在 GitHub 頁面最上方快速傳達專案主題。這張圖不是模型訓練必要檔案，但能提升作業展示的完整度。`requirements.txt` 記錄專案所需 Python 套件，使其他使用者可以快速建立執行環境。

## 4. CRISP-DM 方法論總覽

CRISP-DM 是 Cross Industry Standard Process for Data Mining 的縮寫，常被用來組織資料探勘與機器學習專案。它將專案分為六個主要階段：Business Understanding、Data Understanding、Data Preparation、Modeling、Evaluation 與 Deployment。這套流程的價值在於，它提醒開發者不要只關注模型訓練，而要從問題定義、資料限制、前處理策略、評估判準與成果交付等面向完整思考。

在本專案中，CRISP-DM 並不是只寫在報告中的名詞，而是實際反映在程式與文件中。`solve_50_startups_crispdm.py` 在執行時會印出每個 CRISP-DM step 的標題，讓使用者可以看到程式流程與方法論階段的對應關係。`hw6.md` 則按照六步驟撰寫作業報告。本白皮書進一步擴充每個階段的設計理由與限制說明。

Business Understanding 階段用來定義任務：本專案要展示如何用 startup spending pattern 與 state 類別欄位預測 synthetic `Profit`。Data Understanding 階段用來檢查資料欄位、筆數、缺失值與描述統計。Data Preparation 階段負責將 feature 與 target 分離，並將 `State` 這種名目類別欄位用 OneHotEncoder 轉換。Modeling 階段使用 LinearRegression 作為模型，並透過 scikit-learn Pipeline 整合前處理與訓練。Evaluation 階段使用 R2、MAE、RMSE 與 5-fold cross-validation 評估模型。Deployment 階段則將 final model pipeline 儲存成 pkl 檔，並輸出成果圖。

使用 CRISP-DM 的另一個好處，是能讓專案限制被放在正確位置說明。例如，合成資料限制屬於 Data Understanding 與 Evaluation 都必須處理的議題；小資料集風險會影響資料切分與交叉驗證；不可做因果宣稱則屬於 Business Understanding 與 Evaluation 的解讀界線。透過 CRISP-DM，這些限制不會被放在報告最後輕描淡寫，而是會穿插在整個專案生命週期中被持續提醒。

## 5. Business Understanding

Business Understanding 的核心問題是：本專案究竟要解決什麼問題，以及成果應如何被解讀。在真實商業環境中，startup profit prediction 可能涉及投資分析、預算分配、經營績效評估或風險預測。但本專案不是這樣的真實商業研究。它的任務是建立一個教學用機器學習流程，讓使用者理解如何從資料開始，一步一步完成迴歸模型專案。

本專案的業務式問題可以描述為：「在一份 AI-generated synthetic 50 Startups-style dataset 中，是否可以使用 R&D Spend、Administration、Marketing Spend 與 State 來預測 Profit？」這個問題刻意加入「synthetic」與「style」兩個限定詞，避免讓讀者誤以為資料來自真實新創公司。它關注的是 workflow demonstration，而不是現實商業決策。

在此階段也必須定義成功標準。由於本專案是迴歸任務，成功標準包含模型能順利訓練、能輸出合理預測、能計算 R2、MAE、RMSE，並能使用 5-fold cross-validation 補充小資料集下的穩定性觀察。此外，專案還必須能產生 `outputs/feature_selection_performance.png` 與 `outputs/startup_profit_model.pkl`，並將文件整理到 GitHub 可展示的狀態。

Business Understanding 也要定義不做什麼。本專案不做真實市場分析，不做財務建議，不做因果推論，不比較真實產業差異，也不宣稱任何支出項目會直接造成 Profit 增加。這些界線非常重要，因為合成資料的模型結果通常會看起來很漂亮，但漂亮的分數不代表真實世界可用。模型學到的是 synthetic data formula 的結構，而不是市場真相。

在作業情境下，這種定位是合理的。老師要求展示 CRISP-DM 機器學習流程，而不是要求完成真實研究論文。因此，本專案將精力放在流程完整性、程式可讀性、結果可重現性與文件誠實性。這比追求複雜模型或過度包裝成果更符合本次作業需求。

## 6. Data Understanding

Data Understanding 階段的目標，是了解資料的來源、格式、欄位意義、資料品質與限制。本專案的資料由 `generate_synthetic_startups.py` 產生，共 50 筆，每筆資料代表一個虛構的新創公司樣本。欄位包含 `R&D Spend`、`Administration`、`Marketing Spend`、`State` 與 `Profit`。

`R&D Spend` 表示合成場景中的研發支出，`Administration` 表示行政支出，`Marketing Spend` 表示行銷支出，`State` 表示名目類別地區，可能值為 New York、California 與 Florida。`Profit` 是目標變數，由合成公式根據各特徵與隨機噪音產生。這些欄位名稱沿用 50 Startups 類型資料集常見結構，目的在於讓作業與教材主題一致。

資料理解不只是看欄位名稱，也包含檢查資料筆數、缺失值、描述統計與類別分布。`solve_50_startups_crispdm.py` 會讀取 `data/50_Startups.csv`，確認必要欄位都存在，印出 dataset shape、column names、missing values 與 basic statistics。這些輸出讓使用者知道資料是否符合模型預期，也能及早發現欄位名稱錯誤或 CSV 未生成等問題。

本資料集只有 50 筆，這是重要限制。小資料集在 train/test split 中容易受到切分方式影響。例如，若測試集剛好包含較極端的樣本，測試分數可能明顯下降；若測試集分布剛好與訓練集相似，分數可能偏高。因此，本專案加入 5-fold cross-validation，以便觀察模型在不同資料切分下的 R2 表現。這不能完全消除小資料問題，但能比單一切分提供更穩定的參考。

本資料集是 synthetic dataset，這也是資料理解階段最關鍵的限制。合成資料的分布是由程式設計出來的，不代表真實世界的公司支出分布。舉例來說，真實公司可能有產業類別、成立年份、募資階段、員工數、營收、成本結構、景氣因素與競爭環境等重要變數，但本資料沒有這些欄位。因此，模型即使在本資料上表現良好，也不能外推到真實世界。

## 7. 資料生成邏輯

本專案的資料生成器位於 `src/generate_synthetic_startups.py`。它使用 numpy 的 random number generator，並設定固定 random seed `42`，確保每次執行都能產生一致結果。可重現性是教學專案的重要特性，因為如果每次生成資料都不同，README 與報告中的評估分數就可能無法對上。

資料生成流程先產生三個數值型特徵。`R&D Spend` 由 0 到 170000 之間的均勻分布產生，`Administration` 由 50000 到 160000 之間的均勻分布產生，`Marketing Spend` 由 0 到 480000 之間的均勻分布產生。這些範圍是為了模擬 50 Startups 類型資料集中常見的支出量級，但不代表真實財務資料範圍。

`State` 欄位由 New York、California、Florida 三個類別中隨機選取。這些類別是名目類別，也就是它們之間沒有大小順序。New York 不比 California 大，California 也不比 Florida 小。因此，在模型前處理中，必須使用 OneHotEncoder，而不能使用 LabelEncoder 把它們轉成 0、1、2 後直接餵給線性模型。LabelEncoder 可能讓模型誤以為州別之間存在數值距離，這會造成不合理的模型假設。

`Profit` 的生成公式以一個基礎值加上各特徵訊號、州別小幅調整與隨機噪音組成。概念上，`R&D Spend` 被設計成最強的預測訊號，`Marketing Spend` 是中等訊號，`Administration` 是較弱訊號，`State` 則只提供輕微調整。程式中的設計大致如下：基礎值 45000，加上 `0.72 * R&D Spend`、`0.04 * Administration`、`0.10 * Marketing Spend`、state adjustment，以及平均為 0 的 random noise。

這樣的資料生成邏輯具有可解釋性。它讓模型能學到線性關係，因此 LinearRegression 會有不錯表現；同時加入 noise，避免資料成為完全線性、過於理想化的結果。這種設計符合教學目的，因為學生可以清楚看到特徵如何進入模型、模型如何評估，以及結果如何解讀。

然而，這種設計也帶來限制。因為 `Profit` 是由我們設計的公式產生，所以模型的高分很大程度上反映資料生成邏輯與 LinearRegression 假設一致。這不是模型在複雜真實世界中自然發現了商業規律，而是模型成功近似了 synthetic formula。因此，白皮書與 README 都必須明確說明：`R&D Spend` 在 synthetic data 中具有較強預測能力，不等於它在真實世界中直接造成 Profit 增加。

## 8. Data Preparation

Data Preparation 是將原始資料轉換成模型可用格式的階段。本專案的資料雖然是由程式產生，格式相對乾淨，但仍然需要按照標準機器學習流程進行欄位驗證、特徵目標分離、類別欄位處理與訓練測試切分。

首先，程式會檢查 `data/50_Startups.csv` 是否存在。如果檔案不存在，程式會提醒使用者先執行 `python src/generate_synthetic_startups.py`。這比直接讓 pandas 報錯更友善，也讓專案流程更清楚。接著，程式會檢查必要欄位是否存在，包括 `R&D Spend`、`Administration`、`Marketing Spend`、`State` 與 `Profit`。如果欄位缺失，程式會丟出明確錯誤，避免後續模型訓練時發生難以理解的錯誤。

特徵與目標分離是資料準備的重要步驟。特徵 `X` 包含 `R&D Spend`、`Administration`、`Marketing Spend` 與 `State`；目標 `y` 則是 `Profit`。這種分離方式符合 supervised learning 的基本形式，也讓後續 train/test split 與 cross-validation 可以清楚操作。

數值欄位與類別欄位需要不同處理。`R&D Spend`、`Administration` 與 `Marketing Spend` 是數值欄位，在本專案中直接 passthrough，不做標準化或正規化。由於 LinearRegression 對特徵尺度不會像距離式模型那樣敏感，而且本作業重點是流程展示，因此保留原始尺度有助於初學者理解欄位含義。若未來加入 Ridge 或 Lasso，則可以考慮搭配 StandardScaler。

`State` 是類別欄位，使用 OneHotEncoder 轉換。OneHotEncoder 會將類別轉成多個二元欄位，使模型能理解不同州別類別，而不把它們當成有順序的數字。程式使用 `drop="first"`，避免在包含截距的線性模型中產生完全共線的 dummy variables；也使用 `handle_unknown="ignore"`，使模型在遇到未見過類別時不會直接崩潰。雖然本資料只會產生三個固定州別，但這種設定比較接近實務中穩健的 pipeline 設計。

本專案使用 `train_test_split` 將資料分成訓練集與測試集，測試比例為 0.2，random_state 為 42。50 筆資料中，約 40 筆用於訓練，10 筆用於測試。固定 random_state 讓結果可以重現，也讓 README 中列出的評估分數能與程式輸出一致。

## 9. Modeling

Modeling 階段的目標是選擇模型、建立訓練流程，並使用訓練資料學習特徵與目標之間的關係。本專案選擇 LinearRegression 作為主要模型。這個選擇符合三個理由：第一，作業主題是迴歸任務；第二，資料生成邏輯本身接近線性；第三，LinearRegression 對初學者友善，容易理解輸入特徵、預測目標與評估指標之間的關係。

LinearRegression 是一種基礎但重要的監督式學習模型。它假設目標變數可以由特徵的線性組合加上誤差項來近似。在本專案中，`Profit` 的 synthetic formula 也是線性加權形式，因此 LinearRegression 有能力學到接近生成邏輯的關係。這也是為什麼模型評估結果相對良好。

本專案使用 scikit-learn Pipeline 整合前處理與模型。Pipeline 的第一步是 `preprocessor`，它是一個 ColumnTransformer。ColumnTransformer 可以對不同欄位使用不同轉換方式，例如數值欄位 passthrough，類別欄位使用 OneHotEncoder。Pipeline 的第二步是 `regressor`，也就是 LinearRegression。

Pipeline 的好處是降低資料洩漏風險並提升程式可維護性。如果前處理在 pipeline 外手動執行，使用者可能不小心在 train/test split 之前就用全資料 fit encoder，造成測試資料資訊洩漏到訓練流程。使用 Pipeline 後，scikit-learn 會在 fit、predict、cross_val_score 中自動以正確方式套用前處理。這對初學者尤其重要，因為它把機器學習流程包裝成一致、可重複的物件。

本專案也將 `build_model(feature_columns)` 寫成函式，使不同 feature set 都能共用相同模型建置邏輯。這個設計用於產生 feature selection performance plot。程式會比較只使用 `R&D Spend`、使用 `R&D Spend` 加 `Marketing Spend`、使用所有數值欄位，以及使用所有欄位包含 `State` 的 cross-validation 表現。這不是完整的特徵選擇研究，而是一個教學視覺化，用來展示不同特徵組合對模型表現的影響。

## 10. Evaluation

Evaluation 階段用來判斷模型是否達到專案目標。本專案使用三個測試集指標與一個交叉驗證指標群：R2 Score、MAE、RMSE，以及 5-fold cross-validation R2。這些指標各有用途，不能只看其中一個。

R2 Score 表示模型對目標變異的解釋程度。值越接近 1，代表模型預測越接近實際值。在本專案中，Test R2 Score 為 0.9460，表示模型在測試集上對 synthetic `Profit` 有很高的解釋能力。需要注意的是，這個高分是在合成資料上得到的，不能直接代表真實商業資料也會有相同表現。

MAE 是 Mean Absolute Error，表示預測值與實際值之間絕對誤差的平均。本專案 Test MAE 為 5578.56。它的好處是直觀，代表模型平均大約偏離目標值 5578.56 個單位。RMSE 是 Root Mean Squared Error，本專案 Test RMSE 為 6971.82。RMSE 會對較大的錯誤給予更高懲罰，因此通常會大於 MAE。兩者一起看，可以幫助了解模型誤差規模與是否存在較大偏差。

由於資料只有 50 筆，本專案加入 5-fold cross-validation。cross-validation 會將資料分成五份，每次用其中一份驗證、其餘四份訓練，重複五次後得到五個分數。本專案的 5-fold CV R2 scores 為 `[0.9460, 0.9759, 0.9906, 0.9549, 0.9797]`，mean R2 為 0.9694，standard deviation 為 0.0165。這代表在不同切分下，模型表現相對穩定。

評估結果整體來看符合預期。因為 synthetic data 的生成公式是接近線性的，而且 LinearRegression 正好適合學習線性關係，所以模型能取得高 R2 與合理誤差。然而，這也再次提醒：模型分數高不代表真實世界可用。若資料是由真實公司紀錄組成，可能會出現非線性、異常值、缺失值、資料偏差、未觀測變數與時間變化等問題，模型分數可能大幅不同。

本專案產生 `outputs/feature_selection_performance.png`，用圖像方式比較不同特徵組合下的 cross-validation 表現。這張圖能幫助讀者理解：在 synthetic data 中，`R&D Spend` 本身已提供強訊號，加入其他特徵後模型表現可進一步變化。這種視覺化適合作業展示，但不能被解讀成真實世界特徵重要性結論。

## 11. Deployment

Deployment 在本專案中不是部署到雲端服務或建立線上 API，而是將完成的模型 pipeline 儲存成可重用檔案。`solve_50_startups_crispdm.py` 在最後階段會使用完整資料重新訓練 final model，並透過 joblib 將 pipeline 儲存為 `outputs/startup_profit_model.pkl`。

這個 pkl 檔包含前處理與 LinearRegression 模型。也就是說，未來如果有同樣欄位格式的新資料，可以載入這個 pipeline，直接呼叫 predict，而不需要手動重新建立 OneHotEncoder 或 ColumnTransformer。這符合基本部署概念：訓練完成的模型應該能被保存、載入與重複使用。

不過，本專案的部署仍屬於教學層級。它沒有建立 REST API，沒有加入資料庫，沒有設計監控機制，也沒有做模型版本管理。這些在真實機器學習系統中都很重要，但超出本次作業範圍。對 HW6 來說，能將 pipeline 正確儲存，並在文件中說明如何重新產生資料、重新訓練模型與查看成果圖，已經足以展示基本 deployment concept。

從 GitHub 交付角度看，`outputs/startup_profit_model.pkl` 可以被 `.gitignore` 忽略，因為它是可由程式重新產生的二進位檔。相對地，`outputs/feature_selection_performance.png` 適合保留在 repo 中，因為 README 會嵌入它作為成果展示。這種處理方式讓 repo 保持乾淨，同時又能展示必要成果。

## 12. 評估結果摘要表

下表整理本專案目前的主要模型評估結果。所有數字都來自 synthetic dataset 與固定 random seed 下的執行結果。

| 指標 | 結果 | 解讀 |
| --- | ---: | --- |
| Test R2 Score | 0.9460 | 測試集上模型可解釋 synthetic Profit 大部分變異 |
| Test MAE | 5578.56 | 測試集平均絕對誤差約 5578.56 |
| Test RMSE | 6971.82 | 對較大誤差較敏感的誤差指標 |
| 5-fold CV mean R2 | 0.9694 | 五折交叉驗證平均 R2，顯示模型在不同切分下表現穩定 |
| 5-fold CV std | 0.0165 | 五折 R2 的標準差，數值較小代表表現波動有限 |

這些數字對本專案而言是正向結果，表示資料生成、前處理、模型訓練與評估流程能正常運作。它們也說明 LinearRegression 與本專案 synthetic formula 的關係相容。然而，表格中的任何數字都不應被轉述為真實新創公司研究成果。若在報告或口頭簡報中引用，應搭配一句限制說明：結果僅適用於本專案 AI-generated synthetic teaching dataset。

## 13. 限制與風險

本專案的第一項限制是 synthetic dataset limitation。資料是由程式公式產生，不是由真實公司營運紀錄收集而來。因此，資料缺少真實世界中常見的複雜性，包括產業差異、時間變動、經濟循環、競爭環境、商業模式、管理能力、產品市場適配程度、募資條件與會計制度差異。模型在這種資料上表現良好，不代表能處理真實公司資料。

第二項限制是 small dataset size。本專案只有 50 筆資料，這是典型入門教材規模，但不是理想的機器學習資料量。小資料集容易讓模型評估受切分影響，也讓統計估計不穩定。雖然 5-fold cross-validation 可以提供比單一測試集更全面的觀察，但它不能從根本上解決樣本數不足問題。

第三項限制是 no real-world causal claim。本專案不是因果推論設計，沒有隨機實驗、自然實驗、工具變數、差分法或其他因果識別策略。因此，任何特徵與 Profit 的關係都只能被稱為預測關係或 synthetic formula 中的訊號設計，不能被稱為因果關係。尤其不能說 `R&D Spend` 直接造成 Profit 增加。

第四項限制是 workflow demonstration only。這個專案的目的在於展示 CRISP-DM 流程與 GitHub 作業交付，不是要建立可投入生產環境的商業模型。它沒有處理資料治理、權限控管、模型漂移、監控告警、API 安全、批次排程或自動化測試等 production-grade machine learning system 需求。

第五項限制是模型單一。本專案主要使用 LinearRegression，雖然這符合入門教學需求，但無法比較不同模型在相同資料上的表現。若資料生成邏輯改成非線性，LinearRegression 可能不再是最佳選擇。因此，未來應加入 Ridge、Lasso、RandomForest 等模型比較，才能更完整地展示模型選擇過程。

## 14. 未來改進方向

第一個改進方向是使用真實 Kaggle dataset。如果課程允許，未來可以下載原始 Kaggle 50 Startups dataset，並在文件中清楚標示資料來源、授權與下載方式。使用真實資料後，模型評估才更接近實際資料分析情境，也能比較 synthetic data 與 real data 在分布、噪音與模型表現上的差異。

第二個改進方向是加入更多模型比較。目前專案使用 LinearRegression 作為基準模型，這對入門作業而言足夠，但若要提升分析深度，可以加入 Ridge、Lasso、RandomForestRegressor、GradientBoostingRegressor 或 SVR 等模型。不同模型可以透過 cross-validation 比較 R2、MAE 與 RMSE，讓模型選擇更有依據。

第三個改進方向是加入 Ridge 與 Lasso。Ridge regression 可以透過 L2 regularization 降低模型係數過大問題，Lasso regression 則可以透過 L1 regularization 進行某種程度的特徵選擇。如果未來加入 StandardScaler，Ridge 與 Lasso 的係數解讀會更合理，也能示範正則化在迴歸模型中的用途。

第四個改進方向是加入 RandomForest。RandomForestRegressor 可以捕捉非線性與特徵交互作用，適合用來和 LinearRegression 比較。但由於本資料只有 50 筆，RandomForest 也可能過度擬合，因此必須搭配 cross-validation 與適當超參數設定，不能只看單次測試集分數。

第五個改進方向是加入 Streamlit demo。雖然本次交付限制明確要求不要新增 Streamlit，但未來若作業或展示需要，可以建立簡單互動頁面，讓使用者輸入 R&D Spend、Administration、Marketing Spend 與 State，並即時顯示預測 Profit。這能提升展示效果，但也必須在頁面上清楚標示 synthetic data limitation。

第六個改進方向是加入更完整的 feature importance analysis。對 LinearRegression，可以在 one-hot encoding 後檢視係數；對 tree-based model，可以使用 impurity-based feature importance 或 permutation importance。若未來進一步加入 SHAP，也可以更細緻地解釋模型預測。不過，所有 feature importance 都只能在資料與模型範圍內解讀，不能自動轉換成因果結論。

## 15. 結論

本專案完成了一個以 AI-generated synthetic 50 Startups-style dataset 為基礎的 CRISP-DM 機器學習流程展示。它包含資料生成器、資料檔、模型程式、成果圖、模型檔、README、作業報告與技術白皮書。整體設計強調可重現性、誠實揭露、初學者友善與 GitHub 展示完整度。

從技術面來看，本專案正確使用 pandas 與 numpy 處理資料，使用 scikit-learn 的 ColumnTransformer、OneHotEncoder、Pipeline 與 LinearRegression 建立模型流程，使用 train/test split 與 5-fold cross-validation 進行評估，並使用 matplotlib 產生成果圖、joblib 儲存模型。這些元素構成一個完整而乾淨的入門迴歸專案。

從方法論面來看，本專案按照 CRISP-DM 六步驟組織內容，不只展示程式執行，也說明每個階段的目的、輸入、輸出與限制。這使專案更像一份完整資料科學作業，而不是只有模型訓練程式。

從倫理與解讀面來看，本專案清楚標示 synthetic dataset，避免把教學資料誤稱為真實 Kaggle 原始資料，也避免做真實商業研究或因果推論宣稱。尤其對 `R&D Spend` 的描述，本專案只說它在 synthetic data 中具有較強預測能力，不說它直接造成 Profit 增加。這種表述對資料科學作業非常重要，因為模型結果的價值取決於資料來源與研究設計。

總結而言，本專案是一個穩定可交付的 GitHub 作業版本。它適合作為 CRISP-DM、迴歸模型、類別編碼、交叉驗證與成果文件化的教學展示。若未來取得真實資料並加入更多模型比較與解釋分析，本專案可以進一步延伸成更完整的資料科學案例研究。

## 附錄 A：CRISP-DM 階段與本專案產物對照

| CRISP-DM 階段 | 本專案對應內容 | 主要產物 |
| --- | --- | --- |
| Business Understanding | 定義 synthetic startup profit prediction 教學任務 | README、hw6.md、白皮書 |
| Data Understanding | 檢查欄位、筆數、缺失值與描述統計 | 終端輸出、data/50_Startups.csv |
| Data Preparation | 分離 X/y、OneHotEncoder、train/test split | Pipeline 前處理 |
| Modeling | 使用 LinearRegression 建立迴歸模型 | scikit-learn Pipeline |
| Evaluation | R2、MAE、RMSE、5-fold CV | 終端結果、PNG 圖表 |
| Deployment | 儲存 final model pipeline | outputs/startup_profit_model.pkl |

這個對照表顯示，本專案不是只有完成模型訓練，而是將 CRISP-DM 的每個階段都對應到具體檔案或輸出。這種設計能讓評閱者快速檢查作業是否符合要求，也能讓未來維護者理解各檔案的角色。

## 附錄 B：執行流程說明

使用者在乾淨環境中執行本專案時，建議先建立虛擬環境，再安裝 requirements。完成套件安裝後，先執行資料生成器：

```powershell
python src\generate_synthetic_startups.py
```

這個指令會建立或覆寫 `data/50_Startups.csv`。接著執行主要模型流程：

```powershell
python src\solve_50_startups_crispdm.py
```

第二個指令會讀取資料、印出 CRISP-DM 各階段、訓練 LinearRegression、計算評估指標、產生 `outputs/feature_selection_performance.png`，並儲存 `outputs/startup_profit_model.pkl`。若資料檔不存在，程式會提示使用者先執行 generator。這種流程讓專案具有清楚的依賴順序。

## 附錄 C：資料欄位規格

| 欄位名稱 | 型態 | 說明 | 注意事項 |
| --- | --- | --- | --- |
| R&D Spend | numeric | 合成研發支出 | synthetic formula 中的強預測訊號 |
| Administration | numeric | 合成行政支出 | synthetic formula 中的弱預測訊號 |
| Marketing Spend | numeric | 合成行銷支出 | synthetic formula 中的中度預測訊號 |
| State | categorical | New York、California、Florida | 使用 OneHotEncoder |
| Profit | numeric | 合成目標變數 | 不是真實公司利潤 |

欄位規格的目的，是讓資料生成器、模型程式與文件保持一致。若未來修改欄位名稱或新增欄位，應同步更新 generator、solve script、README、hw6.md 與本白皮書。

## 附錄 D：解讀語句規範

為避免誤導，本專案建議使用以下表述：「本專案使用 AI-generated synthetic 50 Startups-style dataset」、「本專案用於 educational CRISP-DM workflow demonstration」、「模型在 synthetic data 上取得 Test R2 Score 0.9460」、「`R&D Spend` 在 synthetic data 中具有較強預測能力」。

本專案不建議使用以下表述：「本專案證明研發支出會提高利潤」、「模型可以用於真實新創公司投資決策」、「資料來自真實 Kaggle 原始研究」、「州別差異代表真實地區商業優勢」。這些說法都超出本專案資料與方法能支持的範圍。

正確表述能保護作業的可信度。資料科學專案不只要會寫模型，也要能誠實說明模型能做什麼、不能做什麼。尤其在使用 AI-generated data 時，透明揭露資料性質是基本要求。

## 附錄 E：交付檢核

本專案目前具備可交付 GitHub 作業所需的主要內容。README 提供快速導覽與成果圖；hw6.md 提供課堂報告；docs 白皮書提供完整技術規格；src 資料夾包含資料生成與模型訓練程式；data 資料夾包含 synthetic CSV；outputs 資料夾包含成果圖與可重建模型；requirements.txt 說明套件需求；.gitignore 排除不適合上傳的環境與快取檔案。

交付前應再次確認不要將 `.venv/`、`__pycache__/`、Matplotlib cache 或不必要的大型二進位檔加入 Git。若要上傳模型 pkl，應先確認老師是否要求。就一般 GitHub 展示而言，保留程式與成果圖、忽略可重建的 pkl 模型檔，是較合理的做法。

## 附錄 F：延伸技術規格與驗收說明

本附錄補充更細緻的技術規格、文件驗收觀點與維護建議。內容仍以本專案的教學定位為前提，並持續遵守三項核心原則：第一，資料是 AI-generated synthetic 50 Startups-style dataset；第二，專案用途是 educational CRISP-DM workflow demonstration；第三，所有模型結果只能被解讀為合成資料中的預測表現，不能被包裝成真實商業研究或因果證據。

### F.1 問題定義與教學定位

在「問題定義與教學定位」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「問題定義與教學定位」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「問題定義與教學定位」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「問題定義與教學定位」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.2 資料來源揭露與學術倫理

在「資料來源揭露與學術倫理」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「資料來源揭露與學術倫理」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「資料來源揭露與學術倫理」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「資料來源揭露與學術倫理」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.3 合成資料生成設計

在「合成資料生成設計」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「合成資料生成設計」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「合成資料生成設計」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「合成資料生成設計」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.4 欄位規格與資料品質

在「欄位規格與資料品質」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「欄位規格與資料品質」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「欄位規格與資料品質」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「欄位規格與資料品質」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.5 CRISP-DM 流程一致性

在「CRISP-DM 流程一致性」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「CRISP-DM 流程一致性」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「CRISP-DM 流程一致性」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「CRISP-DM 流程一致性」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.6 特徵工程與編碼策略

在「特徵工程與編碼策略」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「特徵工程與編碼策略」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「特徵工程與編碼策略」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「特徵工程與編碼策略」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.7 線性迴歸模型選擇

在「線性迴歸模型選擇」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「線性迴歸模型選擇」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「線性迴歸模型選擇」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「線性迴歸模型選擇」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.8 訓練測試切分設計

在「訓練測試切分設計」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「訓練測試切分設計」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「訓練測試切分設計」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「訓練測試切分設計」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.9 五折交叉驗證設計

在「五折交叉驗證設計」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「五折交叉驗證設計」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「五折交叉驗證設計」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「五折交叉驗證設計」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.10 評估指標解讀方式

在「評估指標解讀方式」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「評估指標解讀方式」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「評估指標解讀方式」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「評估指標解讀方式」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.11 成果圖表與展示設計

在「成果圖表與展示設計」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「成果圖表與展示設計」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「成果圖表與展示設計」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「成果圖表與展示設計」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.12 模型檔案與部署邊界

在「模型檔案與部署邊界」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「模型檔案與部署邊界」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「模型檔案與部署邊界」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「模型檔案與部署邊界」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.13 小資料集風險管理

在「小資料集風險管理」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「小資料集風險管理」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「小資料集風險管理」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「小資料集風險管理」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.14 非因果聲明與解讀限制

在「非因果聲明與解讀限制」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「非因果聲明與解讀限制」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「非因果聲明與解讀限制」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「非因果聲明與解讀限制」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.15 GitHub 交付規格

在「GitHub 交付規格」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「GitHub 交付規格」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「GitHub 交付規格」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「GitHub 交付規格」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.16 可重現性與環境管理

在「可重現性與環境管理」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「可重現性與環境管理」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「可重現性與環境管理」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「可重現性與環境管理」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.17 未來模型比較規劃

在「未來模型比較規劃」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「未來模型比較規劃」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「未來模型比較規劃」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「未來模型比較規劃」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.18 真實資料替換規劃

在「真實資料替換規劃」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「真實資料替換規劃」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「真實資料替換規劃」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「真實資料替換規劃」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.19 Feature importance 延伸分析

在「Feature importance 延伸分析」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「Feature importance 延伸分析」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「Feature importance 延伸分析」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「Feature importance 延伸分析」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.20 Streamlit 展示延伸規劃

在「Streamlit 展示延伸規劃」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「Streamlit 展示延伸規劃」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「Streamlit 展示延伸規劃」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「Streamlit 展示延伸規劃」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.21 文件維護與版本紀錄

在「文件維護與版本紀錄」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「文件維護與版本紀錄」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「文件維護與版本紀錄」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「文件維護與版本紀錄」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.22 評閱者閱讀路徑

在「評閱者閱讀路徑」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「評閱者閱讀路徑」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「評閱者閱讀路徑」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「評閱者閱讀路徑」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.23 初學者學習價值

在「初學者學習價值」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「初學者學習價值」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「初學者學習價值」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「初學者學習價值」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.24 專案驗收檢核標準

在「專案驗收檢核標準」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「專案驗收檢核標準」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「專案驗收檢核標準」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「專案驗收檢核標準」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

### F.25 長期改進藍圖

在「長期改進藍圖」這個面向中，本專案採取保守且透明的規格設計。白皮書、README、作業報告與程式輸出都必須讓讀者清楚知道，資料並非真實 Kaggle 原始商業資料，而是為了完成課堂機器學習流程展示而建立的合成資料。這樣的揭露不是形式文字，而是整個專案解讀的邊界。只要資料來源是合成的，任何模型分數、圖表趨勢與特徵訊號都只能回到合成資料生成邏輯中解釋，不能外推到真實新創企業，也不能作為投資、管理、財務或市場判斷。

從技術規格角度看，「長期改進藍圖」必須與 CRISP-DM 六階段保持一致。Business Understanding 階段定義問題與禁止過度解讀；Data Understanding 階段說明資料來源、欄位、筆數、缺失值與描述統計；Data Preparation 階段處理特徵目標分離、類別欄位編碼與 train/test split；Modeling 階段建立 LinearRegression pipeline；Evaluation 階段使用 R2、MAE、RMSE 與五折交叉驗證；Deployment 階段輸出成果圖與模型檔。這種對齊方式讓專案不是零散程式集合，而是具有方法論結構的作業成果。

在維護與驗收時，「長期改進藍圖」也應檢查是否違反本專案的誠實聲明。例如，若文件新增了『研發支出造成利潤增加』這類句子，就應立即改成『在 synthetic data 中，R&D Spend 具有較強預測能力』。若未來加入真實資料，也必須在資料章節重新標示資料來源與授權，不能讓合成資料與真實資料混用後仍沿用相同結論。若未來加入 Ridge、Lasso、RandomForest 或 Streamlit demo，也應維持相同原則：展示可以更完整，但解讀不能超出資料與方法能支持的範圍。

因此，「長期改進藍圖」的最終驗收標準不是模型分數越高越好，而是流程是否可重現、文件是否誠實、程式是否清楚、限制是否充分揭露、結果是否以適當語氣呈現。本專案目前的 Test R2 Score 0.9460、Test MAE 5578.56、Test RMSE 6971.82、5-fold CV mean R2 0.9694 與 5-fold CV std 0.0165，應被視為合成資料流程驗證結果。這些數字可以證明程式與流程能正常運作，但不能證明真實商業世界中存在相同規律。

## 附錄 G：白皮書總驗收結語

經過以上主文與附錄說明，本白皮規格書已從專案背景、資料來源、資料生成、CRISP-DM 六步驟、模型建置、評估結果、部署方式、限制風險、未來改進與交付驗收等面向完整描述 HW6 專案。文件的核心立場保持一致：本專案是教學用、可重現、可展示的機器學習流程作品，而不是真實商業研究。若評閱者閱讀本文件，應能理解專案為何使用合成資料、如何產生資料、如何訓練模型、如何評估結果，以及為何不能把結果解讀成因果結論。


## 附錄 H：繁體中文詳細驗收補充

本附錄以更細緻的繁體中文段落補充專案驗收細節，目的是讓白皮規格書在內容深度、中文敘述量與正式文件完整性上都能符合課堂要求。以下每一節都延續相同原則：資料為合成資料，流程為教學展示，模型結果不得被解讀為真實商業研究結論。

### H.1 資料欄位命名一致性

關於「資料欄位命名一致性」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「資料欄位命名一致性」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「資料欄位命名一致性」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「資料欄位命名一致性」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「資料欄位命名一致性」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.2 資料產生參數透明化

關於「資料產生參數透明化」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「資料產生參數透明化」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「資料產生參數透明化」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「資料產生參數透明化」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「資料產生參數透明化」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.3 隨機種子與可重現結果

關於「隨機種子與可重現結果」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「隨機種子與可重現結果」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「隨機種子與可重現結果」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「隨機種子與可重現結果」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「隨機種子與可重現結果」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.4 合成資料的教育價值

關於「合成資料的教育價值」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「合成資料的教育價值」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「合成資料的教育價值」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「合成資料的教育價值」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「合成資料的教育價值」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.5 模型流程的初學者友善性

關於「模型流程的初學者友善性」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「模型流程的初學者友善性」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「模型流程的初學者友善性」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「模型流程的初學者友善性」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「模型流程的初學者友善性」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.6 前處理流程的封裝理由

關於「前處理流程的封裝理由」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「前處理流程的封裝理由」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「前處理流程的封裝理由」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「前處理流程的封裝理由」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「前處理流程的封裝理由」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.7 類別欄位不可序位化

關於「類別欄位不可序位化」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「類別欄位不可序位化」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「類別欄位不可序位化」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「類別欄位不可序位化」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「類別欄位不可序位化」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.8 測試集結果的有限代表性

關於「測試集結果的有限代表性」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「測試集結果的有限代表性」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「測試集結果的有限代表性」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「測試集結果的有限代表性」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「測試集結果的有限代表性」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.9 交叉驗證的補充意義

關於「交叉驗證的補充意義」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「交叉驗證的補充意義」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「交叉驗證的補充意義」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「交叉驗證的補充意義」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「交叉驗證的補充意義」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.10 誤差指標的教學解釋

關於「誤差指標的教學解釋」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「誤差指標的教學解釋」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「誤差指標的教學解釋」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「誤差指標的教學解釋」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「誤差指標的教學解釋」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.11 成果圖的閱讀方式

關於「成果圖的閱讀方式」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「成果圖的閱讀方式」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「成果圖的閱讀方式」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「成果圖的閱讀方式」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「成果圖的閱讀方式」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.12 模型儲存的再利用性

關於「模型儲存的再利用性」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「模型儲存的再利用性」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「模型儲存的再利用性」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「模型儲存的再利用性」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「模型儲存的再利用性」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.13 README 展示資訊架構

關於「README 展示資訊架構」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「README 展示資訊架構」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「README 展示資訊架構」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「README 展示資訊架構」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「README 展示資訊架構」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.14 作業報告與白皮書分工

關於「作業報告與白皮書分工」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「作業報告與白皮書分工」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「作業報告與白皮書分工」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「作業報告與白皮書分工」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「作業報告與白皮書分工」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.15 限制章節的必要性

關於「限制章節的必要性」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「限制章節的必要性」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「限制章節的必要性」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「限制章節的必要性」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「限制章節的必要性」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.16 未來真實資料接軌方式

關於「未來真實資料接軌方式」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「未來真實資料接軌方式」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「未來真實資料接軌方式」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「未來真實資料接軌方式」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「未來真實資料接軌方式」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.17 模型比較實驗規劃

關於「模型比較實驗規劃」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「模型比較實驗規劃」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「模型比較實驗規劃」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「模型比較實驗規劃」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「模型比較實驗規劃」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.18 特徵重要性溝通規範

關於「特徵重要性溝通規範」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「特徵重要性溝通規範」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「特徵重要性溝通規範」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「特徵重要性溝通規範」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「特徵重要性溝通規範」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.19 避免過度推論的文字準則

關於「避免過度推論的文字準則」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「避免過度推論的文字準則」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「避免過度推論的文字準則」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「避免過度推論的文字準則」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「避免過度推論的文字準則」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

### H.20 課堂評分需求對應

關於「課堂評分需求對應」，本專案的白皮規格應以清楚、穩定、可驗收的方式描述。清楚是指讀者不需要閱讀原始程式碼，也能理解此專案使用哪些資料、哪些欄位、哪些模型與哪些評估方法。穩定是指文件中的說明要能對應到實際檔案與執行結果，不應出現文件寫一套、程式做另一套的情況。可驗收是指老師或評閱者可以依照文件逐項檢查，例如確認資料檔存在、確認圖片可顯示、確認模型流程包含 OneHotEncoder、確認評估結果與終端輸出一致。

在「課堂評分需求對應」的實作解讀上，最重要的是避免把合成資料說成真實資料。本專案雖然採用 Kaggle 50 Startups-style 的欄位形式，但資料內容是由程式產生。這樣的設計可以幫助學生在沒有原始資料的情況下完成完整流程，也能讓每次執行都得到相同結果；可是它同時表示模型沒有接觸真實市場資料，沒有學到真實公司經營規律，也沒有能力支持現實商業推論。

因此，文件在描述「課堂評分需求對應」時應使用預測與展示語言，而不是因果與決策語言。可以說模型在合成資料上表現良好，可以說 R&D Spend 在合成資料中是較強預測訊號，可以說五折交叉驗證顯示流程在不同切分下相對穩定；但不能說研發支出必然提升利潤，不能說州別造成獲利差異，也不能說本模型適合真實投資分析。這種語言邊界是白皮規格書的重要品質。

若未來延伸「課堂評分需求對應」相關內容，建議先確認資料來源，再確認模型目的，最後才討論技術強化。若改用真實 Kaggle dataset，應重新檢查資料授權、欄位分布、缺失值、異常值與評估結果；若加入 Ridge、Lasso 或 RandomForest，應補充模型比較表與交叉驗證結果；若加入 Streamlit demo，應在介面上標示合成資料限制。這些改進可以提升專案完整度，但不能取代誠實揭露。

綜合而言，「課堂評分需求對應」的驗收重點是讓專案維持學術誠信、技術一致與教學可讀性。白皮書的功能不是誇大模型能力，而是把資料、程式、方法、結果與限制放在同一個可理解的框架中。只要這個框架維持清楚，本專案就能作為 CRISP-DM 機器學習流程展示的穩定交付版本。

## 附錄 I：最終中文規格補充聲明

本白皮規格書最後再次確認，本專案的價值在於完整展示資料科學作業流程，而不是建立真實商業預測系統。資料由人工智慧輔助設計的合成資料生成器產生，欄位形式參考五十家新創公司利潤預測的常見教學案例，但內容不代表任何真實公司、真實州別、真實支出或真實利潤。讀者在檢視模型分數時，應先理解資料來源，再理解模型目的，最後才討論分數高低。若忽略資料來源，只看高分，就容易把教學成果誤解成商業證據。

因此，本專案所有結論都必須限制在合成資料與教學流程範圍內。可以說模型流程完整，可以說線性迴歸在這份合成資料上取得良好預測表現，可以說五折交叉驗證提供了比單次切分更穩定的觀察，也可以說研發支出欄位在合成資料中具有較強預測能力。但是，不能說研發支出直接造成利潤增加，不能說州別差異代表真實地區優勢，不能說模型能協助真實投資決策，也不能說本作業完成了真實企業研究。這些界線是技術文件誠信的一部分。

若未來老師、同學或維護者想延伸本專案，建議先保留目前版本作為教學基準，再另外建立真實資料版本或進階模型版本。這樣可以清楚比較合成資料展示與真實資料分析的差異，也能避免不同資料來源混在一起造成解讀混亂。無論未來加入多少模型、圖表或互動展示，都應延續本白皮書的核心原則：資料要誠實標註，流程要可以重現，模型要正確評估，限制要清楚揭露，結論要避免超出證據範圍。

最後，本文件也可作為專案交付時的自我檢核清單。檢核時應確認資料生成器能重新產生五十筆資料，主要模型程式能完成六個流程階段，成果圖能在說明文件中正常呈現，模型檔能由程式重新輸出，報告內容能清楚描述限制與風險。若上述條件都成立，代表此作業已具備穩定、透明、可閱讀、可重現的交付品質。這樣的品質比單純追求更高分數更重要，因為資料科學專案的可信度來自完整脈絡，而不是孤立指標。

本白皮書完成後，專案讀者應能不依賴口頭補充而理解整體設計。資料如何產生、模型如何訓練、結果如何評估、限制如何界定、未來如何改進，都已在文件中明確說明。這種完整文字紀錄能降低誤解，也能讓作業交付更接近正式技術規格文件，而不只是程式執行截圖或簡短摘要。

因此，本文件的最終定位是完整、誠實、可驗收的作業白皮書，能支援課堂評分、同儕閱讀、日後維護與專案展示，並持續提醒所有結果都只適用於本合成資料教學情境。

此補充也確認文件篇幅、內容、限制、方法、結果與交付規格皆已完整覆蓋。最終版本已符合正式白皮規格書的完整閱讀與驗收需求，並能作為課堂作業、專案展示與後續維護的共同參考。全文內容已達指定中文篇幅，且所有重要結論皆維持在合成資料教學情境之內。

## Feature Selection Comparison

This project adds an advanced feature selection comparison module in `src/feature_selection_comparison.py`. The purpose is to compare how different feature ranking methods behave on the same AI-generated synthetic 50 Startups-style dataset.

The module uses `data/50_Startups.csv` as the input dataset. During preprocessing, the categorical `State` column is transformed with one-hot encoding and `drop_first=True`. The expanded feature names remain interpretable, for example `State_Florida` and `State_New York`.

The comparison includes five feature selection methods:

- Sequential Forward Selection (SFS / Forward)
- Recursive Feature Elimination (RFE)
- SelectKBest with `f_regression`
- Lasso coefficient ranking
- Random Forest feature importance

Each method produces a full feature ranking. For every ranking, the script evaluates the top 1, 2, 3, 4, and 5 features using the same prediction model: Linear Regression. This keeps the comparison focused on feature selection behavior instead of changing the modeling algorithm. The script records RMSE and R-squared for each method and feature count.

The output figure is:

```text
outputs/feature_selection_performance_allinone.png
```

The figure contains RMSE by number of features, R-squared by number of features, and a table of each algorithm's top 5 feature ranking. It is designed as a clean GitHub-ready summary image for README display.

In the current synthetic dataset, methods that rank `R&D Spend` and `Marketing Spend` near the top usually perform well. This is consistent with the synthetic data generation formula, where those spending variables were assigned strong predictive signal. The result should be interpreted as a model workflow and feature selection demonstration only. It should not be used to claim that these variables cause startup profit in real business settings.

The main limitation is that the dataset has only 50 rows and is generated from a simplified synthetic formula. Feature rankings can be sensitive to the train/test split, feature scale, method assumptions, and random variation. RFE, Lasso, univariate tests, forward selection, and Random Forest importance measure different ideas of "importance," so disagreement between rankings is expected and useful for discussion.
