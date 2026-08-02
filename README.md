# CV pH prediction Streamlit app

This app converts the supplied Colab workflow into an interactive pipeline for cyclic-voltammetry pH data.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens the app in your browser. Upload one Excel workbook per pH. Each filename must contain the pH (for example `pH 3.xlsx`), and current columns must be named `I4`, `I20`, `I30`, ..., `I80`, with the matching voltage column immediately before each current column.

## Included workflow

1. Upload and validate Excel CV data.
2. Stratified 70/15/15 split within each pH-temperature group.
3. Extract the electrochemical and signal-shape features from the supplied code.
4. Rank features using ANOVA F-score, mutual information, Random Forest importance, and permutation importance.
5. Select the top features and inspect ranking, correlation, and PCA plots.
6. Train and compare 11 regression models using validation and test metrics.
7. Inspect actual-versus-predicted, residual, and confusion-matrix plots.
8. Download feature tables, results, predictions, and the selected trained model.

For reliable validation and testing, provide at least three replicate CV curves for every pH-temperature combination.
