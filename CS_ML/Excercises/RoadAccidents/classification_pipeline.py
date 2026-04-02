# ============================================================
#  Road Accidents — CLASSIFICATION PIPELINE  (v2)
#  Target  : accident_risk_class  (Low / Medium / High)
#            (percentile-binned from accident_risk)
#  Models  : XGBoost  +  CatBoost  (tuned, ensembled)
#
#  What changed from v1:
#  [FIX]  Data-leakage bug patched: imputer/OHE now fitted
#         ONLY on train fold, never on val/test data.
#  [NEW]  XGBoost uses sklearn Pipeline + ColumnTransformer
#         (SO-post technique): preprocessor is encapsulated
#         so RandomizedSearchCV tunes it leakage-free across
#         all CV folds automatically.
#  [NEW]  OneHotEncoder(handle_unknown='ignore') replaces
#         pd.get_dummies + align — robustly handles unseen
#         test categories (SO-post technique).
#  [KEPT] RandomizedSearchCV + StratifiedKFold — already
#         better than the SO post's GridSearchCV.
#  [KEPT] No StandardScaler — trees are scale-invariant.
#  [KEPT] CatBoost native categoricals via Pool.
#  [KEPT] Probability-blended ensemble + early stopping.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV,
    cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, roc_auc_score
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

# ─────────────────────────────────────────────
#  0.  CONFIGURATION
# ─────────────────────────────────────────────
TRAIN_PATH   = "train.csv"
TEST_PATH    = "test.csv"
OUTPUT_DIR   = Path("./Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_REG   = "accident_risk"
TARGET_CLS   = "accident_risk_class"
ID_COL       = "id"
DROP_COLS    = ["road_signs_present", "id", "school_season", "num_lanes", "time_of_day"]
BIN_LABELS   = ["Low", "Medium", "High"]

RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5
TUNE_ITERS   = 40


# ─────────────────────────────────────────────
#  1.  DATA LOADING
# ─────────────────────────────────────────────
def load_data():
    print("=" * 60)
    print("  ROAD ACCIDENTS — CLASSIFICATION PIPELINE  (v2)")
    print("=" * 60)
    df      = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    print(f"\n[Data] Train shape : {df.shape}")
    print(f"[Data] Test  shape : {df_test.shape}")
    return df, df_test


# ─────────────────────────────────────────────
#  2.  TARGET CREATION
# ─────────────────────────────────────────────
def create_classification_target(df):
    """Percentile-based binning → balanced class distribution."""
    print("\n[Target] Creating classification target …")
    q33 = df[TARGET_REG].quantile(0.33)
    q66 = df[TARGET_REG].quantile(0.66)
    mn, mx = df[TARGET_REG].min(), df[TARGET_REG].max()
    bins = [mn - 1e-9, q33, q66, mx + 1e-9]
    df[TARGET_CLS] = pd.cut(df[TARGET_REG], bins=bins, labels=BIN_LABELS)
    print(f"[Target] Bin edges: {[round(b, 3) for b in bins]}")
    print(f"[Target] Class distribution:\n{df[TARGET_CLS].value_counts().sort_index()}")
    return df


# ─────────────────────────────────────────────
#  3.  LIGHT PREP  (no fitting — split comes first)
# ─────────────────────────────────────────────
def initial_prep(df, df_test):
    print("\n[Prep] Dropping unused columns …")
    test_ids = df_test[ID_COL].copy() if ID_COL in df_test.columns else None

    for col in DROP_COLS + [TARGET_REG]:
        if col in df:      df      = df.drop(col, axis=1)
        if col in df_test: df_test = df_test.drop(col, axis=1)

    X     = df.drop(TARGET_CLS, axis=1)
    y_raw = df[TARGET_CLS].copy()
    X_test = df_test.copy()

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    print(f"[Target] Encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"[Prep] Numeric   cols ({len(num_cols)}) : {num_cols}")
    print(f"[Prep] Categoric cols ({len(cat_cols)}) : {cat_cols}")

    return X, y, X_test, num_cols, cat_cols, le, n_classes, test_ids


# ─────────────────────────────────────────────
#  4.  SKLEARN PREPROCESSOR BUILDER
#      (ColumnTransformer pattern from SO post)
# ─────────────────────────────────────────────
def build_preprocessor(num_cols, cat_cols):
    """
    Numerics  → median imputation only (no scaler)
    Categorics → most_frequent imputation → OHE
    handle_unknown='ignore' is the SO-post improvement:
    handles categories in test set that weren't in train.
    """
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop"
    )


# ─────────────────────────────────────────────
#  5.  METRICS HELPER
# ─────────────────────────────────────────────
def print_metrics(y_true, y_pred, y_prob, label="Val", n_classes=3):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")
    print(f"  [{label}] Acc={acc:.4f}  F1(weighted)={f1:.4f}  AUC-OvR={auc:.4f}")
    print(f"\n  Classification Report [{label}]:")
    print(classification_report(y_true, y_pred, target_names=BIN_LABELS, zero_division=0))
    return acc, f1, auc


# ─────────────────────────────────────────────
#  6.  XGBOOST CLASSIFIER  (sklearn Pipeline)
# ─────────────────────────────────────────────
def train_xgboost(X_train, y_train, X_val, y_val,
                  X, y, num_cols, cat_cols, X_test, n_classes):
    """
    sklearn Pipeline(ColumnTransformer → XGBClassifier).
    RandomizedSearchCV + StratifiedKFold: tunes preprocessing
    and model together, leakage-free, class-balanced folds.
    """
    print("\n" + "─" * 60)
    print("  XGBoost Classifier  [sklearn Pipeline, leakage-free]")
    print("─" * 60)

    objective   = "multi:softprob" if n_classes > 2 else "binary:logistic"
    eval_metric = "mlogloss"       if n_classes > 2 else "logloss"

    xgb_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(num_cols, cat_cols)),
        ("xgb", XGBClassifier(
            objective=objective, eval_metric=eval_metric,
            tree_method="hist", use_label_encoder=False,
            random_state=RANDOM_STATE, n_jobs=-1,
            **({"num_class": n_classes} if n_classes > 2 else {})
        ))
    ])

    param_dist = {
        "xgb__n_estimators"     : [200, 300, 400, 500, 700],
        "xgb__max_depth"        : [3, 4, 5, 6, 7],
        "xgb__learning_rate"    : [0.01, 0.03, 0.05, 0.1, 0.15],
        "xgb__subsample"        : [0.6, 0.7, 0.8, 0.9, 1.0],
        "xgb__colsample_bytree" : [0.5, 0.6, 0.7, 0.8, 1.0],
        "xgb__min_child_weight" : [1, 3, 5, 7],
        "xgb__gamma"            : [0, 0.1, 0.2, 0.5],
        "xgb__reg_alpha"        : [0, 0.01, 0.1, 1.0],
        "xgb__reg_lambda"       : [1, 1.5, 2, 5],
    }

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"[XGBoost] RandomizedSearchCV ({TUNE_ITERS} iters, StratifiedKFold-{CV_FOLDS}) …")
    xgb_search = RandomizedSearchCV(
        estimator=xgb_pipeline, param_distributions=param_dist,
        n_iter=TUNE_ITERS, scoring="f1_weighted", cv=skf,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    xgb_search.fit(X_train, y_train)   # preprocessor fit on train fold only ✓

    best_params = xgb_search.best_params_
    print(f"[XGBoost] Best CV F1(weighted) : {xgb_search.best_score_:.4f}")
    print(f"[XGBoost] Best params          : {best_params}")

    xgb_params = {k.replace("xgb__", ""): v for k, v in best_params.items()}

    # Final model with early stopping (preprocessor fit on train only)
    prep = build_preprocessor(num_cols, cat_cols)
    X_tr_proc = prep.fit_transform(X_train, y_train)
    X_vl_proc = prep.transform(X_val)
    X_te_proc = prep.transform(X_test)

    xgb_model = XGBClassifier(
        **xgb_params,
        objective=objective, eval_metric=eval_metric,
        tree_method="hist", use_label_encoder=False,
        n_estimators=1000, early_stopping_rounds=50,
        random_state=RANDOM_STATE, n_jobs=-1,
        **({"num_class": n_classes} if n_classes > 2 else {})
    )
    xgb_model.fit(X_tr_proc, y_train,
                  eval_set=[(X_vl_proc, y_val)], verbose=False)

    val_pred = xgb_model.predict(X_vl_proc)
    val_prob = xgb_model.predict_proba(X_vl_proc)
    print_metrics(y_val, val_pred, val_prob, "XGBoost Val", n_classes)

    # Leakage-free CV on full data via Pipeline
    cv_pipe = Pipeline([
        ("preprocessor", build_preprocessor(num_cols, cat_cols)),
        ("xgb", XGBClassifier(**xgb_params, objective=objective,
                               eval_metric=eval_metric, tree_method="hist",
                               use_label_encoder=False, random_state=RANDOM_STATE,
                               n_jobs=-1))
    ])
    cv = cross_val_score(cv_pipe, X, y, cv=skf, scoring="f1_weighted", n_jobs=-1)
    print(f"[XGBoost] CV F1(weighted) = {cv.mean():.4f} ± {cv.std():.4f}")

    try:
        ohe_names = list(prep.named_transformers_["cat"]
                         .named_steps["ohe"].get_feature_names_out(cat_cols))
        feat_names = num_cols + ohe_names
    except Exception:
        feat_names = [f"f{i}" for i in range(xgb_model.n_features_in_)]
    imp = pd.Series(xgb_model.feature_importances_, index=feat_names)
    print("[XGBoost] Top 15 features:")
    print(imp.sort_values(ascending=False).head(15).to_string())

    return xgb_model, prep, X_vl_proc, X_te_proc, xgb_params


# ─────────────────────────────────────────────
#  7.  CATBOOST CLASSIFIER  (native cats, leakage-fixed)
# ─────────────────────────────────────────────
def train_catboost(X_train, y_train, X_val, y_val,
                   X, y, cat_cols, X_test, n_classes):
    """
    CatBoost with native categorical handling via Pool.
    v2 fix: imputers fit ONLY on X_train, not full X.
    """
    print("\n" + "─" * 60)
    print("  CatBoost Classifier  [native cats, leakage-fixed]")
    print("─" * 60)

    num_cols_raw = X_train.select_dtypes(include=["number"]).columns.tolist()
    loss_fn = "MultiClass" if n_classes > 2 else "Logloss"

    num_imp = SimpleImputer(strategy="median")
    cat_imp = SimpleImputer(strategy="most_frequent")

    def _apply(df_in, fit=False):
        out = df_in.copy()
        if fit:
            out[num_cols_raw] = num_imp.fit_transform(out[num_cols_raw])
            if cat_cols:
                out[cat_cols] = cat_imp.fit_transform(out[cat_cols])
        else:
            out[num_cols_raw] = num_imp.transform(out[num_cols_raw])
            if cat_cols:
                out[cat_cols] = cat_imp.transform(out[cat_cols])
        if cat_cols:
            out[cat_cols] = out[cat_cols].fillna("missing")
        return out

    X_tr = _apply(X_train, fit=True)   # imputers fitted here only ✓
    X_vl = _apply(X_val)
    X_te = _apply(X_test)

    cat_fi = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]
    train_pool = Pool(X_tr, y_train, cat_features=cat_fi)
    val_pool   = Pool(X_vl, y_val,   cat_features=cat_fi)

    param_dist = {
        "iterations"       : [300, 500, 700, 1000],
        "learning_rate"    : [0.01, 0.03, 0.05, 0.1],
        "depth"            : [4, 5, 6, 7, 8],
        "l2_leaf_reg"      : [1, 3, 5, 7, 10],
        "min_data_in_leaf" : [1, 5, 10, 20],
        "subsample"        : [0.6, 0.7, 0.8, 1.0],
        "colsample_bylevel": [0.6, 0.8, 1.0],
    }

    base_cat = CatBoostClassifier(loss_function=loss_fn,
                                   random_seed=RANDOM_STATE,
                                   verbose=0, cat_features=cat_fi)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"[CatBoost] RandomizedSearchCV ({TUNE_ITERS} iters, StratifiedKFold-{CV_FOLDS}) …")
    cat_search = RandomizedSearchCV(
        estimator=base_cat, param_distributions=param_dist,
        n_iter=TUNE_ITERS, scoring="f1_weighted", cv=skf,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    cat_search.fit(X_tr, y_train)

    best_params = cat_search.best_params_
    print(f"[CatBoost] Best CV F1(weighted) : {cat_search.best_score_:.4f}")
    print(f"[CatBoost] Best params          : {best_params}")

    cat_model = CatBoostClassifier(
        **best_params, loss_function=loss_fn,
        random_seed=RANDOM_STATE, verbose=0,
        cat_features=cat_fi, early_stopping_rounds=50
    )
    cat_model.fit(train_pool, eval_set=val_pool)

    val_pred = cat_model.predict(X_vl).ravel()
    val_prob = cat_model.predict_proba(X_vl)
    print_metrics(y_val, val_pred, val_prob, "CatBoost Val", n_classes)

    cv = cross_val_score(
        CatBoostClassifier(**best_params, loss_function=loss_fn,
                           random_seed=RANDOM_STATE, verbose=0, cat_features=cat_fi),
        X_tr, y_train, cv=skf, scoring="f1_weighted", n_jobs=-1
    )
    print(f"[CatBoost] CV F1(weighted) = {cv.mean():.4f} ± {cv.std():.4f}")

    imp = pd.Series(cat_model.get_feature_importance(), index=X_tr.columns)
    print("[CatBoost] Top 15 features:")
    print(imp.sort_values(ascending=False).head(15).to_string())

    return cat_model, num_imp, cat_imp, X_vl, X_te, best_params, cat_fi


# ─────────────────────────────────────────────
#  8.  WEIGHTED ENSEMBLE  (probability blend)
# ─────────────────────────────────────────────
def build_ensemble(xgb_model, cat_model,
                   X_vl_proc, X_vl_cat, y_val,
                   X_te_proc, X_te_cat, n_classes):
    print("\n" + "─" * 60)
    print("  Weighted Ensemble  (XGBoost + CatBoost)")
    print("─" * 60)

    xgb_prob_val = xgb_model.predict_proba(X_vl_proc)
    cat_prob_val = cat_model.predict_proba(X_vl_cat)

    best_w, best_f1 = 0.5, -np.inf
    for w in np.arange(0.05, 1.0, 0.05):
        blend_pred = np.argmax(w * xgb_prob_val + (1 - w) * cat_prob_val, axis=1)
        f1 = f1_score(y_val, blend_pred, average="weighted")
        if f1 > best_f1:
            best_f1, best_w = f1, w

    print(f"[Ensemble] Optimal XGB weight = {best_w:.2f}  →  Val F1 = {best_f1:.4f}")
    blend_prob_val = best_w * xgb_prob_val + (1 - best_w) * cat_prob_val
    print_metrics(y_val, np.argmax(blend_prob_val, axis=1),
                  blend_prob_val, "Ensemble Val", n_classes)

    xgb_prob_te    = xgb_model.predict_proba(X_te_proc)
    cat_prob_te    = cat_model.predict_proba(X_te_cat)
    blend_test     = np.argmax(best_w * xgb_prob_te + (1 - best_w) * cat_prob_te, axis=1)
    return blend_test, xgb_model.predict(X_te_proc), cat_model.predict(X_te_cat).ravel()


# ─────────────────────────────────────────────
#  9.  SUBMISSION
# ─────────────────────────────────────────────
def save_submissions(blend_pred, xgb_pred, cat_pred, le, test_ids):
    def _save(preds_int, name):
        labels = le.inverse_transform(preds_int)
        out = pd.DataFrame({TARGET_CLS: labels})
        if test_ids is not None:
            out.insert(0, ID_COL, test_ids.values)
        path = OUTPUT_DIR / name
        out.to_csv(path, index=False)
        print(f"  Saved → {path}  |  dist: {pd.Series(labels).value_counts().to_dict()}")

    print("\n[Submission] Saving …")
    _save(blend_pred, "submission_ensemble_classification.csv")
    _save(xgb_pred,   "submission_xgboost_classification.csv")
    _save(cat_pred,   "submission_catboost_classification.csv")


# ─────────────────────────────────────────────
#  10. MAIN
# ─────────────────────────────────────────────
def main():
    df, df_test = load_data()
    df = create_classification_target(df)
    X, y, X_test, num_cols, cat_cols, le, n_classes, test_ids = initial_prep(df, df_test)

    # Split BEFORE any preprocessing → guarantees leakage-free fitting
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[Split] Train={len(X_train)}  Val={len(X_val)}")

    # Train both models
    xgb_model, xgb_prep, X_vl_proc, X_te_proc, xgb_params = train_xgboost(
        X_train, y_train, X_val, y_val, X, y, num_cols, cat_cols, X_test, n_classes
    )
    cat_model, cat_num_imp, cat_cat_imp, X_vl_cat, X_te_cat, cat_params, cat_fi = train_catboost(
        X_train, y_train, X_val, y_val, X, y, cat_cols, X_test, n_classes
    )

    # Retrain on FULL data
    print("\n[Final] Retraining on full train data …")

    # XGBoost full retrain via Pipeline
    objective   = "multi:softprob" if n_classes > 2 else "binary:logistic"
    eval_metric = "mlogloss"       if n_classes > 2 else "logloss"
    full_pipe = Pipeline([
        ("preprocessor", build_preprocessor(num_cols, cat_cols)),
        ("xgb", XGBClassifier(**xgb_params, objective=objective,
                               eval_metric=eval_metric, tree_method="hist",
                               use_label_encoder=False,
                               random_state=RANDOM_STATE, n_jobs=-1))
    ])
    full_pipe.fit(X, y)
    X_te_proc_full = full_pipe.named_steps["preprocessor"].transform(X_test)
    xgb_full       = full_pipe.named_steps["xgb"]

    # CatBoost full retrain
    num_cols_raw = X.select_dtypes(include=["number"]).columns.tolist()
    X_full_cat = X.copy()
    X_te_cat_f = X_test.copy()
    X_full_cat[num_cols_raw] = cat_num_imp.fit_transform(X_full_cat[num_cols_raw])
    X_te_cat_f[num_cols_raw] = cat_num_imp.transform(X_te_cat_f[num_cols_raw])
    if cat_cols:
        X_full_cat[cat_cols] = cat_cat_imp.fit_transform(X_full_cat[cat_cols])
        X_te_cat_f[cat_cols] = cat_cat_imp.transform(X_te_cat_f[cat_cols])
        for df_ in [X_full_cat, X_te_cat_f]:
            df_[cat_cols] = df_[cat_cols].fillna("missing")

    cat_model.set_params(early_stopping_rounds=None)
    cat_model.fit(Pool(X_full_cat, y, cat_features=cat_fi))

    # Ensemble & submit
    blend_pred, xgb_pred, cat_pred = build_ensemble(
        xgb_full, cat_model,
        X_vl_proc, X_vl_cat, y_val,
        X_te_proc_full, X_te_cat_f,
        n_classes
    )
    save_submissions(blend_pred, xgb_pred, cat_pred, le, test_ids)

    print("\n" + "=" * 60)
    print("  Classification Pipeline Complete ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
