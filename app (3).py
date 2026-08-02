import io
import re
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.signal import peak_widths
from scipy.stats import entropy, kurtosis, skew
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
META = ["sample_id", "pH", "temperature_C", "replicate", "source_file"]

st.set_page_config(page_title="CV pH Prediction", page_icon="⚗️", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #f5fbff 0%, #fff8f2 52%, #f7f2ff 100%);}
[data-testid="stHeader"] {background: rgba(255,255,255,.72);}
.hero {padding: 1.45rem 1.6rem; border-radius: 22px; color: white;
 background: linear-gradient(115deg,#073b66,#087ea4 48%,#6f42c1);
 box-shadow: 0 12px 30px rgba(15,75,120,.22); margin-bottom: 1rem;}
.hero h1 {font-size: 2rem; margin:0 0 .25rem 0; color:white;}
.hero p {margin:0; opacity:.92; font-size:1.02rem;}
.info-card {border-radius:18px; padding:1rem 1.15rem; color:white; min-height:118px;
 box-shadow:0 8px 22px rgba(35,55,80,.16);}
.blue-card {background:linear-gradient(135deg,#0061a8,#00a6c7);}
.purple-card {background:linear-gradient(135deg,#6338a8,#b149b9);}
.green-card {background:linear-gradient(135deg,#047857,#12a884);}
.orange-card {background:linear-gradient(135deg,#d35a20,#f59e0b);}
.prediction-card {padding:1.3rem; border-radius:20px; text-align:center; color:white;
 background:linear-gradient(135deg,#5928a0,#9b2fae,#d94878); box-shadow:0 10px 28px rgba(92,40,150,.25);}
.prediction-card .value {font-size:3rem; font-weight:800; line-height:1.1;}
div[data-testid="stMetric"] {background:white; border:1px solid #dfe9f2; padding:14px;
 border-radius:16px; box-shadow:0 5px 15px rgba(20,60,90,.08);}
.stButton>button {border-radius:12px; font-weight:700;}
</style>
<div class="hero">
  <h1>⚗️ AI-TC pH Sensor App</h1>
  <p>AI-powered temperature-compensated electrochemical pH prediction</p>
</div>
""", unsafe_allow_html=True)
st.caption("Upload → feature extraction → intelligent selection → model training → smart prediction")


def ph_from_name(name):
    match = re.search(r"ph[\s_\-]*(\d+(?:\.\d+)?)", name, re.I)
    if not match:
        raise ValueError(f"Cannot identify pH in '{name}'. Rename it like pH 3.xlsx.")
    return float(match.group(1))


def read_cv_file(uploaded):
    ph = ph_from_name(uploaded.name)
    raw = pd.read_excel(uploaded, sheet_name=0)
    records, counters, columns = [], {}, list(raw.columns)
    for j, col in enumerate(columns):
        match = re.match(r"^I\s*(-?\d+(?:\.\d+)?)\b", str(col).strip(), re.I)
        if not match or j == 0:
            continue
        temperature = float(match.group(1))
        if temperature.is_integer():
            temperature = int(temperature)
        v = pd.to_numeric(raw[columns[j - 1]], errors="coerce")
        i = pd.to_numeric(raw[col], errors="coerce")
        valid = v.notna() & i.notna()
        v, i = v[valid].to_numpy(float), i[valid].to_numpy(float)
        if len(v) < 10:
            continue
        counters[temperature] = counters.get(temperature, 0) + 1
        rep = counters[temperature]
        records.append({"sample_id": f"pH{ph:g}_T{temperature}_R{rep}", "pH": ph,
                        "temperature_C": temperature, "replicate": rep,
                        "potential_V": v, "current_A": i, "source_file": uploaded.name})
    if not records:
        raise ValueError(f"No current columns found in '{uploaded.name}'. Expected I4, I20, I30, etc.")
    return records


def split_group(group):
    group = group.sample(frac=1, random_state=RANDOM_STATE).copy()
    n = len(group)
    if n < 3:
        group["dataset"] = "Training"
        return group
    n_test = max(1, round(n * .15)); n_val = max(1, round(n * .15))
    while n_test + n_val >= n:
        if n_val > 1: n_val -= 1
        elif n_test > 1: n_test -= 1
        else: break
    train_val, test = train_test_split(group, test_size=n_test, random_state=RANDOM_STATE)
    train, val = train_test_split(train_val, test_size=n_val, random_state=RANDOM_STATE)
    train["dataset"], val["dataset"], test["dataset"] = "Training", "Validation", "Testing"
    return pd.concat([train, val, test])


def extract_features(row):
    v = np.asarray(row["potential_V"], float); i = np.asarray(row["current_A"], float) * 1e9
    valid = np.isfinite(v) & np.isfinite(i); v, i = v[valid], i[valid]
    if len(v) < 10: raise ValueError(f"Too few points in {row['sample_id']}")
    ox, red = int(np.argmax(i)), int(np.argmin(i)); ipa, ipc, epa, epc = i[ox], i[red], v[ox], v[red]
    with np.errstate(all="ignore"):
        didv = np.gradient(i, v); d2 = np.gradient(didv, v)
    def width(signal, index):
        try:
            left, right = peak_widths(signal, [index], rel_height=.5)[2:4]
            x = np.arange(len(v)); return abs(np.interp(right[0], x, v) - np.interp(left[0], x, v))
        except Exception: return np.nan
    ox_area = abs(np.trapezoid(np.clip(i, 0, None), v)); red_area = abs(np.trapezoid(np.clip(-i, 0, None), v))
    turn, loop_area = int(np.argmax(v)), np.nan
    if 2 < turn < len(v) - 3:
        vf, iff = v[:turn + 1], i[:turn + 1]
        vr, indices = np.unique(v[turn:][::-1], return_index=True); irr = i[turn:][::-1][indices]
        lo, hi = max(vf.min(), vr.min()), min(vf.max(), vr.max())
        if lo < hi:
            grid = np.linspace(lo, hi, 300)
            loop_area = np.trapezoid(abs(np.interp(grid, vf, iff) - np.interp(grid, vr, irr)), grid)
    probability = (abs(i) + 1e-12); probability /= probability.sum()
    out = {"sample_id": row["sample_id"], "pH": row["pH"], "temperature_C": row["temperature_C"],
           "replicate": row["replicate"], "source_file": row["source_file"],
           "oxidation_peak_current_nA": ipa, "reduction_peak_current_nA": ipc,
           "oxidation_peak_voltage_V": epa, "reduction_peak_voltage_V": epc,
           "peak_separation_V": epa-epc, "peak_current_difference_nA": ipa-ipc,
           "peak_current_ratio": ipa/abs(ipc) if ipc else np.nan,
           "oxidation_FWHM_V": width(i, ox), "reduction_FWHM_V": width(-i, red),
           "oxidation_sharpness_nA_per_V2": abs(d2[ox]), "reduction_sharpness_nA_per_V2": abs(d2[red]),
           "oxidation_area_nA_V": ox_area, "reduction_area_nA_V": red_area,
           "total_peak_area_nA_V": ox_area+red_area, "CV_loop_area_nA_V": loop_area,
           "absolute_signal_area": np.trapezoid(abs(i), x=np.arange(len(i))),
           "maximum_current_nA": np.max(i), "minimum_current_nA": np.min(i), "mean_current_nA": np.mean(i),
           "median_current_nA": np.median(i), "current_range_nA": np.ptp(i), "current_std_nA": np.std(i, ddof=1),
           "current_variance_nA2": np.var(i, ddof=1), "current_RMS_nA": np.sqrt(np.mean(i**2)),
           "current_skewness": skew(i), "current_kurtosis": kurtosis(i),
           "maximum_slope_nA_per_V": np.nanmax(didv), "minimum_slope_nA_per_V": np.nanmin(didv),
           "mean_slope_nA_per_V": np.nanmean(didv), "slope_std_nA_per_V": np.nanstd(didv),
           "maximum_curvature": np.nanmax(abs(d2)), "mean_curvature": np.nanmean(abs(d2)),
           "voltage_range_V": np.ptp(v), "number_of_points": len(v),
           "zero_crossings": np.sum(np.diff(np.signbit(i)) != 0), "signal_energy_nA2": np.sum(i**2),
           "signal_entropy": entropy(probability),
           "curve_length": np.sum(np.sqrt(np.diff(v)**2 + np.diff(i)**2)), "roughness_nA": np.std(np.diff(i))}
    return out


def rank_features(train_features):
    candidates = [c for c in train_features.columns if c not in META and pd.api.types.is_numeric_dtype(train_features[c])]
    x = train_features[candidates].replace([np.inf, -np.inf], np.nan)
    x = x.drop(columns=[c for c in x if x[c].nunique(dropna=True) <= 1])
    clean = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x), columns=x.columns, index=x.index)
    y = train_features["pH"]
    fscore, pvalue = f_regression(clean, y)
    mi = mutual_info_regression(clean, y, random_state=RANDOM_STATE)
    forest = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1).fit(clean, y)
    pi = permutation_importance(forest, clean, y, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    ranking = pd.DataFrame({"Feature": clean.columns, "F_Score": fscore, "P_Value": pvalue,
                            "Adjusted_P_Value": np.minimum(pvalue*clean.shape[1], 1),
                            "Mutual_Information": mi, "RF_Importance": forest.feature_importances_,
                            "Permutation_Importance": pi.importances_mean})
    scores = ["F_Score", "Mutual_Information", "RF_Importance", "Permutation_Importance"]
    scaled = MinMaxScaler().fit_transform(ranking[scores].fillna(0))
    ranking["Combined_Score"] = scaled.mean(axis=1)
    ranking["Statistically_Significant"] = ranking["Adjusted_P_Value"] < .05
    ranking = ranking.sort_values("Combined_Score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "Rank", np.arange(1, len(ranking)+1))
    return ranking, clean


def model_set(feature_count, train_count):
    scaled = lambda model: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
    plain = lambda model: make_pipeline(SimpleImputer(strategy="median"), model)
    return {"Linear Regression": scaled(LinearRegression()), "Ridge Regression": scaled(Ridge(alpha=1)),
            "LASSO Regression": scaled(Lasso(alpha=.01, max_iter=20000)),
            "Elastic Net": scaled(ElasticNet(alpha=.01, l1_ratio=.5, max_iter=20000)),
            "PLS Regression": scaled(PLSRegression(n_components=max(1, min(5, feature_count, train_count-1)))),
            "SVR": scaled(SVR(kernel="rbf", C=10, epsilon=.1)),
            "KNN Regression": scaled(KNeighborsRegressor(n_neighbors=max(1, min(5, train_count)), weights="distance")),
            "Random Forest": plain(RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
            "Extra Trees": plain(ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
            "Gradient Boosting": plain(GradientBoostingRegressor(n_estimators=300, learning_rate=.05, max_depth=3, random_state=RANDOM_STATE)),
            "Histogram Gradient Boosting": plain(HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, random_state=RANDOM_STATE))}


def nearest(values, classes):
    values = np.asarray(values).reshape(-1); classes = np.asarray(classes)
    return classes[np.abs(values[:, None]-classes[None, :]).argmin(axis=1)]


def calculate_mean_cv(curves, points=400):
    """Return a mean CV and current SD while preserving scan direction."""
    voltage_curves, current_curves = [], []
    progress_grid = np.linspace(0, 1, points)
    for _, row in curves.iterrows():
        voltage = np.asarray(row["potential_V"], dtype=float)
        current = np.asarray(row["current_A"], dtype=float) * 1e9
        valid = np.isfinite(voltage) & np.isfinite(current)
        voltage, current = voltage[valid], current[valid]
        if len(voltage) < 10:
            continue
        scan_progress = np.linspace(0, 1, len(voltage))
        voltage_curves.append(np.interp(progress_grid, scan_progress, voltage))
        current_curves.append(np.interp(progress_grid, scan_progress, current))
    if not current_curves:
        return None
    voltage_curves = np.vstack(voltage_curves)
    current_curves = np.vstack(current_curves)
    mean_voltage = voltage_curves.mean(axis=0)
    mean_current = current_curves.mean(axis=0)
    current_sd = current_curves.std(axis=0, ddof=1) if len(current_curves) > 1 else np.zeros(points)
    return mean_voltage, mean_current, current_sd


def excel_bytes(sheets):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, data in sheets.items(): data.to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()


with st.sidebar:
    st.header("📍 Measurement context")
    location_name = st.text_input("Location name", "Cork, Ireland")
    latitude = st.number_input("Latitude", value=51.8985, format="%.5f")
    longitude = st.number_input("Longitude", value=-8.4756, format="%.5f")
    timezone_name = st.selectbox(
        "Time zone",
        ["Europe/Dublin", "Asia/Kolkata", "Europe/London", "UTC"],
        index=0,
    )
    try:
        local_now = datetime.now(ZoneInfo(timezone_name))
    except Exception:
        local_now = datetime.now(ZoneInfo("UTC"))
    st.markdown(
        f"""<div class="info-card blue-card">
        <div style="font-size:1.8rem">📍 🕒</div>
        <b>{location_name}</b><br>
        {local_now.strftime('%d %B %Y')}<br>
        <span style="font-size:1.25rem;font-weight:750">{local_now.strftime('%I:%M:%S %p')}</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=9)


upload_tab, feature_tab, selection_tab, training_tab, comparison_tab, prediction_tab = st.tabs([
    "1 · Upload", "2 · Extract", "3 · Select", "4 · Train", "5 · Compare", "6 · Smart prediction"])

with upload_tab:
    uploads = st.file_uploader("Upload one Excel file for each pH", type=["xlsx", "xls"], accept_multiple_files=True,
                               help="The filename must contain pH, for example pH 3.xlsx. Current columns should be I4, I20, I30, etc., with voltage immediately before current.")
    if st.button("Read uploaded CV data", type="primary", disabled=not uploads):
        try:
            records = []
            for uploaded in uploads: records.extend(read_cv_file(uploaded))
            data = pd.DataFrame(records)
            # Use an explicit loop instead of groupby.apply(). Pandas 3.x may
            # exclude grouping columns from the DataFrame passed to apply(),
            # which removes pH/temperature_C on Streamlit Cloud.
            split_parts = []
            for (ph_value, temperature), group in data.groupby(
                ["pH", "temperature_C"], sort=True
            ):
                group = group.copy()
                group["pH"] = ph_value
                group["temperature_C"] = temperature
                split_parts.append(split_group(group))
            split = pd.concat(split_parts, ignore_index=True)
            st.session_state.update(cv_data=data, split_data=split)
            for key in ["features", "ranking", "selected", "results", "models", "predictions"]: st.session_state.pop(key, None)
            st.success(f"Detected {len(data)} complete CV curves from {len(uploads)} files.")
        except Exception as exc: st.error(str(exc))
    if "cv_data" in st.session_state:
        data, split = st.session_state.cv_data, st.session_state.split_data
        a,b,c,d = st.columns(4)
        a.metric("CV curves", len(data)); b.metric("pH values", data.pH.nunique()); c.metric("Temperatures", data.temperature_C.nunique()); d.metric("Files", data.source_file.nunique())
        st.dataframe(pd.crosstab([data.pH], data.temperature_C), use_container_width=True)
        # Build the count table explicitly. This avoids an ambiguity in newer
        # pandas/seaborn versions when grouped data retains pH as an index level.
        split_counts = (
            split.reset_index(drop=True)
            .groupby(["pH", "dataset"], as_index=False)
            .size()
            .pivot(index="pH", columns="dataset", values="size")
            .fillna(0)
        )
        split_counts = split_counts.reindex(
            columns=["Training", "Validation", "Testing"], fill_value=0
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        split_counts.plot(kind="bar", ax=ax, color=["#2878B5", "#F2A104", "#D64550"])
        ax.set_xlabel("pH")
        ax.set_ylabel("CV curves")
        ax.set_title("Training, validation and testing distribution")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Dataset", frameon=False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Mean CV comparison at a selected temperature")
        selected_temperature = st.selectbox(
            "Select temperature",
            options=sorted(data["temperature_C"].unique()),
            key="mean_cv_selected_temperature",
            help="One graph compares the mean CV responses of pH 3, 4, 5, 6 and 7 at the same temperature.",
        )
        ph_targets = [3, 4, 5, 6, 7]
        ph_colours = {3:"#e63946", 4:"#f59e0b", 5:"#10b981", 6:"#168aad", 7:"#7b2cbf"}
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        plotted_ph = []
        for ph_value in ph_targets:
            subset = data[
                np.isclose(data["pH"].astype(float), ph_value)
                & np.isclose(data["temperature_C"].astype(float), selected_temperature)
            ]
            mean_result = calculate_mean_cv(subset)
            if mean_result is None:
                continue
            mean_voltage, mean_current, current_sd = mean_result
            colour = ph_colours[ph_value]
            ax.plot(
                mean_voltage, mean_current, color=colour, linewidth=2.5,
                label=f"pH {ph_value} (n={len(subset)})",
            )
            ax.fill_between(
                mean_voltage, mean_current-current_sd, mean_current+current_sd,
                color=colour, alpha=.10,
            )
            plotted_ph.append(ph_value)
        ax.set_title(
            f"Mean CV Responses at {selected_temperature:g} °C",
            fontsize=15, fontweight="bold",
        )
        ax.set_xlabel("Potential (V)", fontweight="bold")
        ax.set_ylabel("Mean current (nA)", fontweight="bold")
        ax.grid(alpha=.20)
        ax.legend(title="pH level", frameon=False, ncol=3)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        missing_ph = [ph for ph in ph_targets if ph not in plotted_ph]
        if missing_ph:
            st.warning(
                f"No CV curves were found at {selected_temperature:g} °C for pH: "
                + ", ".join(map(str, missing_ph))
            )

        chosen_ph = st.selectbox("Raw CV plot: pH", sorted(data.pH.unique()))
        fig, ax = plt.subplots(figsize=(8,5))
        for _, row in data[data.pH == chosen_ph].iterrows(): ax.plot(row.potential_V, row.current_A*1e9, alpha=.45, label=f"{row.temperature_C} °C")
        handles, labels = ax.get_legend_handles_labels(); unique = dict(zip(labels, handles)); ax.legend(unique.values(), unique.keys(), ncol=4)
        ax.set(xlabel="Potential (V)", ylabel="Current (nA)", title=f"Raw CV curves at pH {chosen_ph:g}"); st.pyplot(fig); plt.close(fig)

with feature_tab:
    if "split_data" not in st.session_state: st.info("Upload and read your Excel files first.")
    elif st.button("Extract CV features", type="primary"):
        with st.spinner("Extracting electrochemical and signal-shape features..."):
            features = pd.DataFrame(st.session_state.split_data.apply(extract_features, axis=1).tolist())
            features["dataset"] = st.session_state.split_data["dataset"].to_numpy()
            st.session_state.features = features
            for key in ["ranking", "selected", "results", "models", "predictions"]: st.session_state.pop(key, None)
    if "features" in st.session_state:
        features = st.session_state.features
        st.success(f"Extracted {len([c for c in features if c not in META+['dataset']])} features from {len(features)} curves.")
        st.dataframe(features, use_container_width=True, height=360)
        view_feature = st.selectbox("Plot an extracted feature", [c for c in features if c not in META+["dataset"]])
        fig, ax = plt.subplots(figsize=(8,4)); sns.boxplot(data=features, x="pH", y=view_feature, hue="temperature_C", ax=ax); ax.legend(ncol=4, fontsize=8); st.pyplot(fig)
        st.download_button("Download all extracted features", excel_bytes({"All_CV_Features": features}), "CV_extracted_features.xlsx")

with selection_tab:
    if "features" not in st.session_state: st.info("Complete feature extraction first.")
    elif st.button("Calculate feature ranking", type="primary"):
        with st.spinner("Running ANOVA, mutual information, Random Forest and permutation importance..."):
            train_f = st.session_state.features.query("dataset == 'Training'").reset_index(drop=True)
            ranking, clean = rank_features(train_f)
            st.session_state.ranking, st.session_state.clean_train = ranking, clean
    if "ranking" in st.session_state:
        ranking = st.session_state.ranking
        top_n = st.slider("Number of top features used for modelling", 1, len(ranking), min(15, len(ranking)))
        st.session_state.selected = ranking.Feature.head(top_n).tolist()
        st.dataframe(ranking, use_container_width=True, height=350)
        fig, ax = plt.subplots(figsize=(9, max(4, top_n*.32))); plot = ranking.head(top_n).sort_values("Combined_Score"); ax.barh(plot.Feature, plot.Combined_Score); ax.set_xlabel("Combined feature-selection score"); st.pyplot(fig)
        corr = st.session_state.clean_train[st.session_state.selected].corr()
        fig, ax = plt.subplots(figsize=(9,7)); sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax); ax.set_title("Selected-feature correlation"); st.pyplot(fig)
        if len(st.session_state.selected) >= 2:
            x = StandardScaler().fit_transform(st.session_state.clean_train[st.session_state.selected]); pca = PCA(n_components=min(3, x.shape[1])).fit_transform(x)
            fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111, projection="3d" if pca.shape[1] >= 3 else None)
            y = st.session_state.features.query("dataset == 'Training'").pH.to_numpy()
            if pca.shape[1] >= 3: sc=ax.scatter(pca[:,0],pca[:,1],pca[:,2],c=y,cmap="viridis"); ax.set_zlabel("PC3")
            else: sc=ax.scatter(pca[:,0],pca[:,1],c=y,cmap="viridis")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); fig.colorbar(sc, label="pH"); ax.set_title("PCA of selected features"); st.pyplot(fig)
        st.download_button("Download feature ranking", excel_bytes({"Feature_Ranking": ranking, "Selected_Features": pd.DataFrame({"Feature":st.session_state.selected})}), "CV_feature_selection.xlsx")

with training_tab:
    if "selected" not in st.session_state: st.info("Calculate a feature ranking and select features first.")
    else:
        selected_models = st.multiselect("Models to train", list(model_set(len(st.session_state.selected), 10)), default=list(model_set(len(st.session_state.selected), 10)))
        if st.button("Train and evaluate models", type="primary", disabled=not selected_models):
            f, selected = st.session_state.features, st.session_state.selected
            train, val, test = [f.query("dataset == @name").reset_index(drop=True) for name in ["Training","Validation","Testing"]]
            if val.empty or test.empty: st.error("Validation or test data is empty. Each pH-temperature group needs at least 3 curves.")
            else:
                xtr,ytr=train[selected].replace([np.inf,-np.inf],np.nan),train.pH; xv,yv=val[selected].replace([np.inf,-np.inf],np.nan),val.pH; xt,yt=test[selected].replace([np.inf,-np.inf],np.nan),test.pH
                models=model_set(len(selected),len(train)); rows=[]; fitted={}; predictions={}
                progress=st.progress(0)
                for idx,name in enumerate(selected_models):
                    try:
                        model=clone(models[name]).fit(xtr,ytr); pv=np.asarray(model.predict(xv)).reshape(-1); pt=np.asarray(model.predict(xt)).reshape(-1)
                        fitted[name]=model; predictions[name]={"validation":pv,"test":pt}
                        rows.append({"Model":name,"Validation_R2":r2_score(yv,pv),"Validation_RMSE":np.sqrt(mean_squared_error(yv,pv)),"Validation_MAE":mean_absolute_error(yv,pv),
                                     "Test_R2":r2_score(yt,pt),"Test_RMSE":np.sqrt(mean_squared_error(yt,pt)),"Test_MAE":mean_absolute_error(yt,pt)})
                    except Exception as exc: st.warning(f"{name} skipped: {exc}")
                    progress.progress((idx+1)/len(selected_models))
                results=pd.DataFrame(rows).sort_values(["Validation_RMSE","Validation_MAE"]).reset_index(drop=True); results.insert(0,"Rank",range(1,len(results)+1))
                st.session_state.update(results=results, models=fitted, predictions=predictions, test_frame=test)
        if "results" in st.session_state:
            st.success(f"Best validation model: {st.session_state.results.iloc[0].Model}")
            st.dataframe(st.session_state.results.style.format(precision=4), use_container_width=True)

with comparison_tab:
    if "results" not in st.session_state: st.info("Train the models first.")
    else:
        results, predictions, test = st.session_state.results, st.session_state.predictions, st.session_state.test_frame
        metric = st.selectbox("Comparison metric", ["Validation_RMSE","Validation_MAE","Validation_R2","Test_RMSE","Test_MAE","Test_R2"])
        fig, ax = plt.subplots(figsize=(9,5)); ordered=results.sort_values(metric,ascending="R2" not in metric); sns.barplot(data=ordered,x=metric,y="Model",ax=ax); st.pyplot(fig)
        name = st.selectbox("Detailed model plots", results.Model)
        actual=test.pH.to_numpy(); predicted=predictions[name]["test"]
        c1,c2=st.columns(2)
        with c1:
            fig,ax=plt.subplots(figsize=(5,5)); ax.scatter(actual,predicted,c=test.temperature_C,cmap="plasma"); lo=min(actual.min(),predicted.min()); hi=max(actual.max(),predicted.max()); ax.plot([lo,hi],[lo,hi],"k--"); ax.set(xlabel="Actual pH",ylabel="Predicted pH",title=f"{name}: actual vs predicted"); st.pyplot(fig)
        with c2:
            fig,ax=plt.subplots(figsize=(5,5)); residual=predicted-actual; ax.scatter(predicted,residual,c=test.temperature_C,cmap="plasma"); ax.axhline(0,color="black",ls="--"); ax.set(xlabel="Predicted pH",ylabel="Residual (predicted − actual)",title=f"{name}: residuals"); st.pyplot(fig)
        classes=np.sort(st.session_state.features.pH.unique()); classified=nearest(predicted,classes); cm=confusion_matrix(actual,classified,labels=classes)
        acc=accuracy_score(actual,classified); bal=balanced_accuracy_score(actual,classified)
        st.write(f"Nearest-pH classification accuracy: **{acc:.1%}** · Balanced accuracy: **{bal:.1%}**")
        fig,ax=plt.subplots(figsize=(6,5)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=classes,yticklabels=classes,ax=ax); ax.set(xlabel="Predicted pH class",ylabel="Actual pH class",title=f"{name}: confusion matrix"); st.pyplot(fig)
        table=pd.DataFrame({"sample_id":test.sample_id,"Temperature_C":test.temperature_C,"Actual_pH":actual,"Predicted_pH":predicted,"Absolute_Error":abs(predicted-actual)})
        st.dataframe(table,use_container_width=True)
        st.download_button("Download model results",excel_bytes({"Model_Comparison":results,"Test_Predictions":table}),"CV_model_results.xlsx")
        buffer=io.BytesIO(); joblib.dump({"model":st.session_state.models[name],"selected_features":st.session_state.selected,"pH_classes":classes},buffer)
        st.download_button(f"Download trained {name}",buffer.getvalue(),f"{name.replace(' ','_')}_model.joblib")

with prediction_tab:
    if "results" not in st.session_state:
        st.info("Complete model training first. This page will then predict one randomly selected test CV.")
    else:
        results = st.session_state.results
        test = st.session_state.test_frame.reset_index(drop=True)
        predictions = st.session_state.predictions
        best_model = results.iloc[0]["Model"]
        st.subheader("🎯 Random test-sample prediction")
        st.caption("A test curve that was not used for model fitting is selected for an independent demonstration.")

        control_1, control_2 = st.columns([2, 1])
        with control_1:
            prediction_model = st.selectbox(
                "Prediction model", results["Model"].tolist(),
                index=results["Model"].tolist().index(best_model), key="random_prediction_model"
            )
        with control_2:
            if st.button("🎲 Select random test CV", type="primary", use_container_width=True):
                previous = st.session_state.get("random_test_index", -1)
                rng = np.random.default_rng()
                choices = [i for i in range(len(test)) if i != previous] or [0]
                st.session_state.random_test_index = int(rng.choice(choices))
        if "random_test_index" not in st.session_state or st.session_state.random_test_index >= len(test):
            st.session_state.random_test_index = 0

        random_index = st.session_state.random_test_index
        sample = test.iloc[random_index]
        predicted_ph = float(predictions[prediction_model]["test"][random_index])
        actual_ph = float(sample["pH"])
        absolute_error = abs(predicted_ph - actual_ph)

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.markdown(f"""<div class="prediction-card"><div>AI PREDICTED pH</div>
            <div class="value">{predicted_ph:.3f}</div><small>{prediction_model}</small></div>""", unsafe_allow_html=True)
        p2.metric("🧪 Actual test pH", f"{actual_ph:g}")
        p3.metric("🌡️ Temperature", f"{float(sample['temperature_C']):g} °C")
        p4.metric("📏 Absolute error", f"{absolute_error:.3f}")

        raw_matches = st.session_state.split_data[
            st.session_state.split_data["sample_id"] == sample["sample_id"]
        ]
        left, right = st.columns([1.15, 1])
        with left:
            st.markdown("#### Raw test CV data plot")
            if not raw_matches.empty:
                raw = raw_matches.iloc[0]
                fig, ax = plt.subplots(figsize=(7, 4.6))
                ax.plot(raw["potential_V"], np.asarray(raw["current_A"]) * 1e9,
                        color="#7b2cbf", linewidth=2.3)
                ax.fill_between(raw["potential_V"], np.asarray(raw["current_A"]) * 1e9,
                                alpha=.15, color="#00a6c7")
                ax.set_xlabel("Potential (V)"); ax.set_ylabel("Current (nA)")
                ax.set_title(f"{sample['sample_id']} · Test measurement")
                ax.grid(alpha=.2); fig.tight_layout(); st.pyplot(fig)
        with right:
            st.markdown("#### Actual and predicted pH")
            all_model_values = pd.DataFrame({
                "Model": ["Actual pH"] + results["Model"].tolist(),
                "pH": [actual_ph] + [float(predictions[m]["test"][random_index]) for m in results["Model"]],
                "Type": ["Experimental"] + ["AI prediction"] * len(results),
            })
            fig, ax = plt.subplots(figsize=(7, 4.6))
            colors = ["#0f766e"] + ["#7b2cbf"] * len(results)
            bars = ax.barh(all_model_values["Model"], all_model_values["pH"], color=colors)
            for bar, value in zip(bars, all_model_values["pH"]):
                ax.text(bar.get_width() + .03, bar.get_y() + bar.get_height()/2,
                        f"{value:.2f}", va="center", fontsize=8)
            ax.axvline(actual_ph, color="#0f766e", linestyle="--", linewidth=1.5)
            ax.set_xlabel("pH"); ax.set_title("One random test sample: all-model prediction")
            ax.invert_yaxis(); fig.tight_layout(); st.pyplot(fig)

        st.markdown("#### Features extracted from this test CV")
        selected = st.session_state.selected
        test_feature_table = pd.DataFrame({
            "Rank": np.arange(1, len(selected) + 1),
            "Selected feature": selected,
            "Extracted value": [sample[f] for f in selected],
        })
        feature_left, feature_right = st.columns([1, 1.2])
        with feature_left:
            st.dataframe(test_feature_table, use_container_width=True, hide_index=True, height=430)
        with feature_right:
            train_features = st.session_state.features.query("dataset == 'Training'")[selected]
            medians = train_features.replace([np.inf, -np.inf], np.nan).median()
            scales = train_features.replace([np.inf, -np.inf], np.nan).std().replace(0, 1)
            standardized = ((sample[selected].astype(float) - medians) / scales).clip(-5, 5)
            feature_plot = pd.DataFrame({"Feature": selected, "Standardized value": standardized.values})
            feature_plot = feature_plot.sort_values("Standardized value")
            fig, ax = plt.subplots(figsize=(7, max(4.5, len(selected) * .3)))
            bar_colors = ["#ef476f" if x < 0 else "#06a77d" for x in feature_plot["Standardized value"]]
            ax.barh(feature_plot["Feature"], feature_plot["Standardized value"], color=bar_colors)
            ax.axvline(0, color="#333", linewidth=1); ax.set_xlabel("Standardized feature value (z-score)")
            ax.set_title("Test-CV feature profile relative to training data"); fig.tight_layout(); st.pyplot(fig)

        report = pd.DataFrame({
            "sample_id": [sample["sample_id"]], "location": [location_name],
            "date_time": [local_now.strftime("%Y-%m-%d %H:%M:%S %Z")],
            "latitude": [latitude], "longitude": [longitude], "model": [prediction_model],
            "actual_pH": [actual_ph], "predicted_pH": [predicted_ph],
            "absolute_error": [absolute_error], "temperature_C": [sample["temperature_C"]],
        })
        st.download_button(
            "⬇️ Download this smart prediction report",
            excel_bytes({"Prediction_Summary": report, "Extracted_Test_Features": test_feature_table}),
            "random_test_pH_prediction.xlsx",
            use_container_width=True,
        )

st.sidebar.header("Workflow status")
for label,key in [("Data uploaded","cv_data"),("Features extracted","features"),("Features selected","selected"),("Models trained","results")]:
    st.sidebar.write(("✅" if key in st.session_state else "○")+" "+label)
st.sidebar.info("Keep this browser tab open while processing. Results are retained while the Streamlit session remains active.")
