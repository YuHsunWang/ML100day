# ML 百日馬拉松 — 學習歷程與技能總覽

這份文件整理我完成「機器學習百日馬拉松」課程的學習成果，涵蓋資料前處理、統計分析、傳統機器學習、非監督式學習到深度學習的完整脈絡，並附上各階段使用的核心技術與工具。

---

## 技能摘要

| 領域 | 技術 |
|------|------|
| 資料處理 | NumPy, Pandas, 特徵工程, 缺值處理, 異常值偵測 |
| 視覺化 | Matplotlib, Seaborn (Heatmap, PairPlot, KDE, ECDF) |
| 傳統 ML | Scikit-learn, XGBoost, Random Forest, Logistic Regression |
| 非監督式 | K-Means, Hierarchical Clustering, PCA, t-SNE |
| 深度學習 | Keras / TensorFlow (MLP, CNN, Transfer Learning) |
| 大數據 | PySpark MLlib |
| 其他 | Python Generator, Data Augmentation, Ensemble / Blending |

---

## 第一階段：資料處理與探索性分析（Day 1–20）

這個階段建立資料分析的基本功，從理解資料品質到視覺化呈現洞察。

### Day 1 — 回歸評估指標
- 手刻 MSE（均方誤差）與 MAE（平均絕對誤差），理解不同評估指標對預測誤差的敏感度差異。
- **技術：** NumPy, 評估指標設計

### Day 5 — 資料建立與操作
- 練習 DataFrame 的建構、欄位操作與資料型態轉換。
- **技術：** Pandas

### Day 9 — 異常值偵測與處理
- 使用 **Home Credit 貸款資料集**（122 個特徵），以 ECDF、箱型圖統計方法識別並處理資料中的異常值。
- **技術：** ECDF, IQR, Pandas, Seaborn

### Day 14 — 相關性分析
- 計算特徵間的皮爾森相關係數，建立特徵選擇的基礎判斷依據。
- **技術：** Pandas `.corr()`, Seaborn heatmap

### Day 16–17 — 分布視覺化
- 對連續與類別特徵繪製 Bar chart 與 KDE 密度圖，分析資料分布型態。
- **技術：** Matplotlib, Seaborn KDE

### Day 20 — 多變量視覺化
- 對矩陣型資料繪製 Heatmap 與 PairPlot，直觀呈現多變量間的關聯結構。
- **技術：** Seaborn PairPlot, Heatmap

---

## 第二階段：特徵工程與監督式學習（Day 22–50）

以 **Titanic 生存預測**（Kaggle）與 **房價預測** 作為主線任務，逐步完整走完一個 ML 專案流程。

### Day 22–26 — 特徵工程（Titanic）
- 對缺值補值策略（中位數/眾數/模型預測）進行比較；對類別特徵做 Label Encoding 與 One-Hot Encoding。
- **技術：** Scikit-learn preprocessing, Pandas

### Day 27–28 — 迴歸實作（計程車費用預測）
- 端對端建立迴歸模型，對地理座標特徵進行工程設計。
- **技術：** Linear Regression, Feature Engineering

### Day 29–32 — 分類模型（Titanic）
- 建立 Logistic Regression 與決策樹分類器，並對模型輸出做交叉驗證與結果分析。
- **技術：** Logistic Regression, Decision Tree, Cross-Validation

### Day 34 — 不平衡資料處理
- 使用分層抽樣（Stratified Sampling）確保訓練集與測試集保有相同類別比例。
- **技術：** `train_test_split(stratify=y)`

### Day 44 — 整合式學習（Random Forest）
- 對 **Wine 資料集**（13 特徵, 178 筆）比較單一決策樹與隨機森林的準確率，展示 Ensemble 的優勢。
- **技術：** Random Forest, GridSearchCV, Hyperparameter Tuning

### Day 49 — Blending 模型融合（Titanic Kaggle）
- 以 Logistic Regression、Gradient Boosting、Random Forest 三個模型做加權平均融合（Blending），應用於 Kaggle Titanic 競賽，展示 Ensemble 在競賽場景的完整流程。
- **技術：** Blending Ensemble, Kaggle Pipeline

---

## 第三階段：非監督式學習（Day 54–65）

從分群到降維，探索資料內部的結構。

### Day 56 — K-Means 分群與 Silhouette 分析
- 對 Gaussian Blob 合成資料（500 筆, 5 群）執行 K-Means，並用 **Silhouette Coefficient** 自動選取最佳 k 值（k=2~8）。
- **技術：** K-Means, Silhouette Score, Scikit-learn

### Day 58 — 階層式分群（Hierarchical Clustering）
- 對 Two Moons、Blobs 等 2D 玩具資料集（1,500 筆）比較 Ward、Complete、Average 三種 linkage 方法的分群結果。
- **技術：** Agglomerative Clustering, Dendrogram, Scikit-learn

### Day 60 — 主成分分析（PCA）
- 對手寫數字資料集（64 維特徵）進行 PCA 降維，比較不同保留主成分數（4~64）對後續分類準確率的影響。
- **技術：** PCA, SGDClassifier, Explained Variance Ratio

### Day 62 — t-SNE 流形學習
- 對 S-curve 流形資料（300 筆）調整 perplexity 參數（4~100），視覺化非線性高維結構。
- **技術：** t-SNE, Manifold Learning, Matplotlib 3D

---

## 第四階段：深度學習基礎（Day 66–89）

從神經網路原理到實作訓練技巧，建立深度學習完整知識體系。

### Day 70 — MLP on MNIST
- 建立 2 層 MLP（256 hidden units, ReLU），在 MNIST 手寫數字資料集達到 **97.89% 測試準確率**。
- **技術：** Keras Sequential, Dense, Dropout, Categorical Crossentropy

### Day 71 — 損失函數比較
- 在 **CIFAR-10** 上比較 Categorical Crossentropy、MSE、Binary Crossentropy 對 CNN 收斂速度與最終準確率的影響。
- **技術：** Keras, Loss Function 設計

### Day 73–74 — 梯度下降實驗
- 對 y=(x+5)² 函數以不同學習率（0.1 / 0.001 / 0.0001）執行梯度下降，可視化收斂曲線，直觀理解 learning rate 的影響。
- **技術：** Gradient Descent, Learning Rate Sensitivity

### Day 75 — 反向傳播手刻
- 對 XOR 問題建立 3 層神經網路，**手動實作 Backpropagation**（Forward pass → Loss → Gradient → Weight update），深化對自動微分的理解。
- **技術：** NumPy, Sigmoid, Backpropagation from scratch

### Day 76 — 優化器比較
- 在 **CIFAR-10** 上比較 SGD、RMSprop、Adam 三種優化器的收斂速度與準確率（Adam 達 **78.11%**），搭配 Data Augmentation。
- **技術：** Keras Optimizers, Data Augmentation, CIFAR-10

### Day 77–89 — 深度學習訓練技巧
涵蓋 Regularization（L1/L2, Dropout）、Batch Normalization、Early Stopping、Learning Rate Scheduling 等訓練穩定化技術的實驗與比較。
- **技術：** Keras Callbacks, BatchNorm, Dropout, Regularization

---

## 第五階段：卷積神經網路（Day 90–100）

從影像特徵提取原理到遷移學習的完整 CNN 學習路徑。

### Day 90 — 色彩直方圖
- 對影像抽取 RGB Color Histogram 作為特徵，用於影像相似度分析。
- **技術：** OpenCV, Color Histogram

### Day 94 — 卷積運算原理
- 手刻單步卷積（single-step convolution），理解 filter / stride / padding 對 feature map 形狀的影響。
- **技術：** NumPy, Convolution from scratch

### Day 95 — Pooling 與 Padding
- 實驗 Max Pooling / Average Pooling 與 Same / Valid Padding 對特徵圖的縮放效果。
- **技術：** Keras Conv2D, MaxPooling2D

### Day 97 — CNN vs DNN 比較
- 在 **CIFAR-10** 上直接比較 DNN（47.1%）與 CNN（77.1%），量化說明卷積結構對影像任務的優勢。
- **技術：** Keras CNN, DNN, CIFAR-10

### Day 98 — Python Generator
- 撰寫自訂 Generator 實現批次資料載入，解決大型影像資料集無法一次載入記憶體的問題。
- **技術：** Python Generator, `yield`, Keras `fit_generator`

### Day 99 — 資料增強（Data Augmentation）
- 使用 `ImageDataGenerator` 對 CIFAR-10 套用旋轉、平移、水平翻轉等增強策略，提升模型泛化能力。
- **技術：** Keras ImageDataGenerator

### Day 100 — 遷移學習（Transfer Learning）
- 基於自訂 CNN 架構在 **CIFAR-10** 上實作遷移學習，搭配 `ReduceLROnPlateau` 動態調整學習率，最終達到 **81.09% 測試準確率**。
- **技術：** Transfer Learning, ReduceLROnPlateau, Keras Callbacks

---

## 補充：大數據機器學習（Logistic.ipynb）

- 以 **Apache Spark / PySpark MLlib** 對點擊率預測資料建立 Logistic Regression 模型。
- 完整的 Feature Engineering Pipeline：StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler → LogisticRegression。
- **技術：** PySpark MLlib, Feature Pipeline, Big Data ML

---

## 課程結業成果

| 指標 | 數值 |
|------|------|
| 完成 Notebooks | 95+ |
| 課程天數 | 100 天 |
| 主要競賽 | Kaggle Titanic, House Prices |
| 最高模型準確率 | MNIST MLP 97.89% / CIFAR-10 CNN 81.09% |
| 涵蓋技術棧 | NumPy, Pandas, Scikit-learn, Keras/TF, PySpark |
