# 🎵 Spotify Song Popularity Classifier

> Binary classification of Spotify track popularity using logistic regression, unsupervised clustering, and stratified cross-validation.

**Course:** CMPINF 2100 — Final Project  
**Author:** Andrew Hornyak  
**Model Type:** Logistic Regression (Binary Classification)  
**Dataset:** 32,833 Spotify tracks × 23 features

---

## 📌 Problem Statement

Given Spotify's audio feature data, can we predict whether a track is **popular or not**?

The continuous `track_popularity` score is converted into a binary target using the dataset **median as a threshold**:

| Label | Meaning | Condition |
|-------|---------|-----------|
| `1` | Popular | `track_popularity ≥ median` |
| `0` | Not Popular | `track_popularity < median` |

Logistic regression is applied to learn the relationship between audio features and popularity, outputting a probability via the sigmoid function. Predicted class is assigned at a **0.50 probability threshold**.

---

## 📂 Repository Structure

```
├── Hornyak_Andrew_Final.ipynb                        # Main analysis notebook
├── Hornyak_Andrew_Final.html                         # Rendered notebook (HTML export)
├── Hornyak_Andrew_Final_Part_F_Cross_Validation.ipynb  # Supplemental CV deep-dive
├── Hornyak_Andrew_Final_Part_F_Cross_Validation.html   # Rendered supplemental notebook
├── index.html                                        # Portfolio showcase page
└── README.md
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Visualized distributions and feature correlations using bar charts, heatmaps, box plots, and scatter plots
- Identified strong positive correlation between **energy** and **loudness**
- Found negative trend between energy/duration and track popularity

### 2. Unsupervised Clustering
- Applied **KMeans** and **Hierarchical Agglomerative Clustering** on standardized audio features
- Both methods converged on **6 optimal clusters** — matching the 6 playlist genres in the dataset
- Cluster 0 dominated on *speechiness*; Cluster 1 dominated on *liveness*

### 3. Model Specification & Fitting
Three logistic regression formulas were built with increasing complexity:

| Model | Formula Type | # Coefficients |
|-------|-------------|----------------|
| `mod_d` | All inputs, linear additive | 28 |
| `mod_e` | Select pairwise interactions | 56 |
| `mod_f` | Higher-order interactions | 198 |

Fit using **statsmodels** (for p-values + coefficient interpretation) and **scikit-learn** (for pipeline + CV).

### 4. Prediction & Evaluation
- Confusion matrices, ROC curves, and AUC scores on training data
- Coefficient magnitude and significance analysis

### 5. Stratified Cross-Validation
Used `StratifiedKFold` with a full scikit-learn `Pipeline`:

```python
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

cv = StratifiedKFold(n_splits=5, shuffle=True)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
```

---

## 📊 Results

### Cross-Validation Model Comparison

| Model | # Coefficients | Avg Train Accuracy | Avg CV Test Accuracy | Overfitting | Verdict |
|-------|---------------|-------------------|---------------------|-------------|---------|
| `mod_d` ✅ | 28 | ~60.6% | **~60.5%** | None | **Selected** |
| `mod_e` | 56 | ~59.1% | ~58.7% | Moderate | Rejected |
| `mod_f` | 198 | ~57.7% | ~57.4% | Severe | Rejected |

*Accuracy figures from supplemental cross-validation notebook (`Part_F_Cross_Validation.ipynb`).*

**Best model: `mod_d`** — all inputs with linear additive features (28 coefficients).

As model complexity increased, the gap between training and test accuracy grew — a clear demonstration of the **bias-variance tradeoff**. `mod_f` memorized the training data but failed to generalize.

---

## 🔑 Key Findings

- **`playlist_genre` was the strongest predictor** — all 6 genre categories were statistically significant; Rock and Pop had the largest coefficients. This was reinforced by clustering independently finding 6 natural groupings.
- **Energy had a negative effect** on popularity, despite being highly correlated with loudness (which was positive). Counterintuitive but statistically consistent across all models.
- **Longer songs tend to be less popular** — `duration_ms` showed a consistent negative coefficient.
- **Simplest model generalized best** — 28 coefficients outperformed 56 and 198 on unseen data.

### Feature Importance Summary

| Feature | Direction | Relative Importance |
|---------|-----------|-------------------|
| `playlist_genre` (Rock, Pop) | ↑ Positive | ★★★★★ |
| `loudness` | ↑ Positive | ★★★★☆ |
| `energy` | ↓ Negative | ★★★★☆ |
| `duration_ms` | ↓ Negative | ★★★☆☆ |
| `danceability` | ↑ Positive | ★★☆☆☆ |
| `acousticness`, `valence`, `tempo` | Mixed | ★☆☆☆☆ |

---

## 🧰 Tech Stack

| Category | Libraries |
|----------|-----------|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `seaborn`, `matplotlib` |
| Statistical modeling | `statsmodels`, `patsy` |
| ML / Pipelines | `scikit-learn` |
| Clustering | `sklearn.cluster.KMeans`, `scipy.cluster.hierarchy` |
| Evaluation | `ConfusionMatrix`, `ROC/AUC`, `StratifiedKFold` |
| Environment | Python 3.8, Jupyter Lab |

---

## 📁 Dataset

**Spotify Songs** — publicly available via [TidyTuesday](https://github.com/rfordatascience/tidytuesday) / Kaggle.

- **32,833 tracks** across 6 playlist genres: `pop`, `rap`, `rock`, `latin`, `r&b`, `edm`
- **23 features** including Spotify-computed audio characteristics and track metadata
- Audio features include: `danceability`, `energy`, `loudness`, `tempo`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `duration_ms`

---

## 🔁 Supplemental: Cross-Validation Deep Dive

**Notebook:** `Hornyak_Andrew_Final_Part_F_Cross_Validation.ipynb`

After completing the main analysis, a supplemental notebook was built to explore and compare three different approaches to implementing stratified cross-validation, using the same three model formulas.

| Method | Approach | Framework |
|--------|----------|-----------|
| **Method 1** | Manual `StandardScaler` + `OneHotEncoder` applied inside each fold; updated formula strings to match encoded column names | statsmodels + sklearn preprocessing |
| **Method 2** | `patsy` design matrices → `ColumnTransformer` (scale numerics, pass-through dummies) → `Pipeline` → `LogisticRegression`; wide output format (train/test in separate columns) | sklearn Pipeline |
| **Method 3** | Same as Method 2 but stores train and test scores in a single `data_split` column — long format, directly compatible with seaborn `catplot` | sklearn Pipeline |

Method 1 was used in the final notebook. Methods 2 and 3 were built to experiment with sklearn Pipelines and explore different bookkeeping patterns for fold results. The supplemental notebook also notes an observed discrepancy in Model 2 train/test behavior between methods, which surfaced a meaningful question about how patsy-encoded interaction terms interact with pipeline preprocessing.

All three methods confirmed the same conclusion: **Model 0 (28 coefficients) generalizes best.**

---

## 💡 Takeaways

- Learned to implement the full ML workflow in Python: data wrangling → EDA → unsupervised learning → supervised modeling → validation
- Experienced the practical value of cross-validation over training-set accuracy alone
- Developed reusable scikit-learn pipelines with preprocessing and encoding integrated
- Applied skills immediately to real-world data engineering work: built an automated data pipeline connecting an IBM Cloud API to an Oracle database using Python and pandas

---

*CMPINF 2100 Final Project · Andrew Hornyak*
