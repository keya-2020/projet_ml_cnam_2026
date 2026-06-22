"""
Wrappers pour les modèles ML/DL du projet — version optimisée + nouveaux modèles.

Modèles supervisés      : Dummy, LogisticRegression, RandomForest, XGBoost,
                          LightGBM, CatBoost, AdaBoost, MLP.
Modèles non supervisés  : IsolationForest, LocalOutlierFactor (LOF),
                          One-Class SVM, Autoencoder.
Ensemble                : StackingClassifier (base learners décorrélés +
                          CatBoost/AdaBoost optionnels).

Conventions :
- Chaque `train_*` renvoie un modèle entraîné (le tuning de seuil et la
  sauvegarde sont gérés ailleurs : evaluation.py / notebooks).
- Déséquilibre géré par pondération (`scale_pos_weight`, `class_weight`).
- Early stopping réellement activé via un fold de validation temporel
  automatique (cf. `make_validation_split`).
"""

from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    IsolationForest, RandomForestClassifier, StackingClassifier, AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
import xgboost as xgb
import lightgbm as lgb


# --------------------------------------------------------------------------- #
#  Utilitaires internes
# --------------------------------------------------------------------------- #
def _scale_pos_weight(y_train: pd.Series) -> float:
    """Ratio classe majoritaire / minoritaire (≈ 96 sur BAF Base)."""
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    return n_neg / n_pos if n_pos > 0 else 1.0


def make_validation_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_fraction: float = 0.2,
    month_col: str = "month",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Découpe un jeu de validation pour l'early stopping.

    - Si une colonne temporelle (`month`) existe : on isole le(s) dernier(s)
      mois (split TEMPOREL, cohérent avec la dérive du dataset).
    - Sinon : repli sur un holdout stratifié aléatoire.
    """
    if month_col in X.columns:
        months = np.sort(X[month_col].unique())
        n_val_months = max(1, int(round(len(months) * val_fraction)))
        val_months = set(months[-n_val_months:])
        mask_val = X[month_col].isin(val_months)
        if mask_val.any() and (~mask_val).any():
            return X[~mask_val], X[mask_val], y[~mask_val], y[mask_val]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=random_state
    )
    return X_tr, X_val, y_tr, y_val


# --------------------------------------------------------------------------- #
#  Baselines
# --------------------------------------------------------------------------- #
def train_dummy(X_train, y_train, strategy: str = "stratified", random_state: int = 42):
    """Baseline naïf — PR-AUC attendu ≈ taux de fraude (~0.01)."""
    model = DummyClassifier(strategy=strategy, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: str = "balanced",
    max_iter: int = 1000,
    random_state: int = 42,
    scale: bool = True,
) -> Tuple[LogisticRegression, Optional[StandardScaler]]:
    """Régression logistique — baseline ML linéaire. Renvoie (model, scaler)."""
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    model = LogisticRegression(
        class_weight=class_weight, max_iter=max_iter,
        random_state=random_state, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, scaler


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 14,
    min_samples_leaf: int = 20,
    max_features: str = "sqrt",
    class_weight: str = "balanced_subsample",
    random_state: int = 42,
) -> RandomForestClassifier:
    """Random Forest — point de comparaison bagging."""
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features=max_features,
        class_weight=class_weight, random_state=random_state, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------------------------- #
#  Boosting (XGBoost / LightGBM / CatBoost) — early stopping actif
# --------------------------------------------------------------------------- #
def train_xgboost(
    X_train, y_train, X_val=None, y_val=None, auto_val: bool = True,
    params: Optional[Dict[str, Any]] = None, n_estimators: int = 2000,
    learning_rate: float = 0.03, max_depth: int = 6,
    early_stopping_rounds: int = 50, random_state: int = 42,
) -> xgb.XGBClassifier:
    """XGBoost — scale_pos_weight auto, eval_metric=aucpr, early stopping auto."""
    spw = _scale_pos_weight(y_train)
    default_params = dict(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8, gamma=0.0,
        reg_lambda=1.0, reg_alpha=0.0, scale_pos_weight=spw, eval_metric="aucpr",
        tree_method="hist", random_state=random_state, n_jobs=-1,
    )
    if params:
        default_params.update(params)
        default_params.setdefault("scale_pos_weight", spw)
        default_params["eval_metric"] = "aucpr"

    if (X_val is None or y_val is None) and auto_val:
        X_train, X_val, y_train, y_val = make_validation_split(
            X_train, y_train, random_state=random_state)

    model = xgb.XGBClassifier(**default_params)
    if X_val is not None and y_val is not None:
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)
    return model


def train_lightgbm(
    X_train, y_train, X_val=None, y_val=None, auto_val: bool = True,
    params: Optional[Dict[str, Any]] = None, n_estimators: int = 2000,
    learning_rate: float = 0.03, num_leaves: int = 64,
    early_stopping_rounds: int = 50, random_state: int = 42,
) -> lgb.LGBMClassifier:
    """LightGBM — class_weight=balanced, early stopping auto sur PR-AUC."""
    default_params = dict(
        n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
        min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.0, class_weight="balanced", objective="binary",
        random_state=random_state, n_jobs=-1, verbose=-1,
    )
    if params:
        default_params.update(params)
        default_params.setdefault("class_weight", "balanced")

    if (X_val is None or y_val is None) and auto_val:
        X_train, X_val, y_train, y_val = make_validation_split(
            X_train, y_train, random_state=random_state)

    model = lgb.LGBMClassifier(**default_params)
    fit_params: Dict[str, Any] = {}
    if X_val is not None and y_val is not None:
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["eval_metric"] = "average_precision"
        fit_params["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
    model.fit(X_train, y_train, **fit_params)
    return model


def train_catboost(
    X_train, y_train, X_val=None, y_val=None, auto_val: bool = True,
    params: Optional[Dict[str, Any]] = None, iterations: int = 2000,
    learning_rate: float = 0.03, depth: int = 6,
    early_stopping_rounds: int = 50, random_state: int = 42,
):
    """
    CatBoost — 3e booster, ajoute de la diversité au stacking.

    Optimisations vs script initial :
    - `scale_pos_weight` calculé automatiquement (au lieu de rien → le modèle
      ignorait le déséquilibre 1:96).
    - `eval_metric="PRAUC"` + early stopping via fold de validation temporel
      automatique (au lieu de 200 itérations fixes).
    - `allow_writing_files=False` pour ne pas polluer le disque (catboost_info).

    CatBoost est importé paresseusement : `pip install catboost` requis.
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("CatBoost requis : pip install catboost") from exc

    spw = _scale_pos_weight(y_train)
    default_params = dict(
        iterations=iterations, learning_rate=learning_rate, depth=depth,
        loss_function="Logloss", eval_metric="PRAUC", scale_pos_weight=spw,
        random_state=random_state, verbose=0, allow_writing_files=False,
    )
    if params:
        default_params.update(params)
        default_params.setdefault("scale_pos_weight", spw)

    if (X_val is None or y_val is None) and auto_val:
        X_train, X_val, y_train, y_val = make_validation_split(
            X_train, y_train, random_state=random_state)

    model = CatBoostClassifier(**default_params)
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=(X_val, y_val),
                  early_stopping_rounds=early_stopping_rounds, verbose=0)
    else:
        model.fit(X_train, y_train, verbose=0)
    return model


def train_adaboost(
    X_train, y_train, n_estimators: int = 300, learning_rate: float = 0.05,
    base_max_depth: int = 3, random_state: int = 42,
) -> AdaBoostClassifier:
    """
    AdaBoost — inclus pour comparaison, mais peu adapté au déséquilibre extrême.

    Optimisations vs script initial :
    - Apprenant de base = arbre `class_weight="balanced"` (AdaBoost n'a pas de
      gestion native du déséquilibre ; sans cela il ignore quasiment la classe
      fraude). C'est la seule façon propre d'injecter la pondération.
    - Pas de paramètre `algorithm` (retiré dans scikit-learn ≥ 1.6).

    Avertissement : AdaBoost est sensible au bruit et aux outliers ; sur BAF on
    s'attend à une PR-AUC inférieure aux gradient boosters. À traiter comme un
    point de comparaison plus que comme un candidat sérieux.
    """
    base = DecisionTreeClassifier(
        max_depth=base_max_depth, class_weight="balanced", random_state=random_state)
    model = AdaBoostClassifier(
        estimator=base, n_estimators=n_estimators,
        learning_rate=learning_rate, random_state=random_state)
    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------------------------- #
#  Modèles non supervisés (anomalie) — NE PAS empiler comme classifieurs
# --------------------------------------------------------------------------- #
def train_isolation_forest(
    X_train, contamination: float = 0.011, n_estimators: int = 200,
    random_state: int = 42,
) -> IsolationForest:
    """
    Isolation Forest — détection d'anomalies sans labels.
    Score d'anomalie = -model.decision_function(X)  (plus élevé = plus suspect).
    """
    model = IsolationForest(
        contamination=contamination, n_estimators=n_estimators,
        random_state=random_state, n_jobs=-1)
    model.fit(X_train)
    return model


def train_lof(
    X_train, n_neighbors: int = 20, contamination: float = 0.011,
    novelty: bool = True, n_jobs: int = -1,
) -> LocalOutlierFactor:
    """
    Local Outlier Factor — densité locale relative aux voisins.

    `novelty=True` permet d'appeler `.predict` / `.decision_function` sur des
    données NON vues (test), à condition d'avoir entraîné sur des données
    propres. Convention de score identique à Isolation Forest :
        y_scores = -model.decision_function(X_test)   # plus élevé = plus suspect
        y_pred   = np.where(model.predict(X_test) == -1, 1, 0)

    Performance : LOF interroge les voisins de chaque point ; sur ~800k lignes
    × ~40 features l'inférence peut être lente. Envisager un sous-échantillon
    d'entraînement représentatif, ou réduire `n_neighbors`.
    """
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination,
        novelty=novelty, n_jobs=n_jobs)
    model.fit(X_train)
    return model


def train_oneclass_svm(
    X_train, nu: float = 0.05, method: str = "sgd", gamma="scale",
    n_components: int = 300, max_train_size: int = 50000, random_state: int = 42,
) -> Pipeline:
    """
    One-Class SVM — frontière englobant la classe "normale".

    ATTENTION À L'ÉCHELLE : un One-Class SVM à noyau RBF a un coût ~O(n²)–O(n³)
    et est INFAISABLE sur ~800k lignes. Deux modes sont donc proposés :

    - method="sgd"  (défaut) : approximation de noyau (Nyström) + SGDOneClassSVM,
      qui passe à l'échelle linéairement. RECOMMANDÉ sur le dataset complet.
    - method="rbf"            : OneClassSVM RBF exact, entraîné sur un
      SOUS-ÉCHANTILLON de `max_train_size` lignes (sinon trop lent).

    Renvoie un Pipeline (scaling inclus). Scoring :
        y_scores = -pipe.decision_function(X_test)    # plus élevé = plus suspect
        y_pred   = np.where(pipe.predict(X_test) == -1, 1, 0)
    """
    if method == "sgd":
        from sklearn.linear_model import SGDOneClassSVM
        from sklearn.kernel_approximation import Nystroem
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("feature_map", Nystroem(n_components=n_components, random_state=random_state)),
            ("ocsvm", SGDOneClassSVM(nu=nu, random_state=random_state)),
        ])
        pipe.fit(X_train)
        return pipe

    elif method == "rbf":
        X = X_train
        if len(X_train) > max_train_size:
            idx = np.random.RandomState(random_state).choice(
                len(X_train), max_train_size, replace=False)
            X = X_train.iloc[idx] if hasattr(X_train, "iloc") else X_train[idx]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ocsvm", OneClassSVM(nu=nu, kernel="rbf", gamma=gamma)),
        ])
        pipe.fit(X)
        return pipe

    raise ValueError(f"method inconnu : {method}. Choisir 'sgd' ou 'rbf'.")


def anomaly_score(model, X: pd.DataFrame) -> np.ndarray:
    """
    Score d'anomalie homogène (plus élevé = plus suspect) pour les modèles non
    supervisés à `decision_function` (IsolationForest, LOF novelty, One-Class SVM).

    Sert à transformer un détecteur d'anomalies en FEATURE pour le stacking
    hybride (cf. `add_anomaly_features`).
    """
    return -model.decision_function(X)


def add_anomaly_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, detectors: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ajoute les scores d'anomalie comme nouvelles colonnes (stacking HYBRIDE).

    C'est la bonne façon d'exploiter LOF / One-Class SVM / Isolation Forest dans
    un ensemble supervisé : leurs scores deviennent des features lues par les
    boosters et le méta-modèle, au lieu d'être empilés comme classifieurs (ce
    qui n'aurait pas de sens, cf. note dans train_stacking_ensemble).

    Parameters
    ----------
    detectors : dict {nom: modèle non supervisé déjà entraîné sur X_train}
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    for name, det in detectors.items():
        X_train[f"anomaly_{name}"] = anomaly_score(det, X_train)
        X_test[f"anomaly_{name}"] = anomaly_score(det, X_test)
    return X_train, X_test


# --------------------------------------------------------------------------- #
#  Tuning Optuna
# --------------------------------------------------------------------------- #
def tune_lightgbm(X_train, y_train, n_trials: int = 50, random_state: int = 42) -> Dict[str, Any]:
    """Recherche Optuna maximisant la PR-AUC sur validation temporelle (LightGBM)."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError("Optuna requis : pip install optuna") from exc

    X_tr, X_val, y_tr, y_val = make_validation_split(X_train, y_train, random_state=random_state)

    def objective(trial):
        params = dict(
            n_estimators=3000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 16, 256),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 300),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            class_weight="balanced", objective="binary", subsample_freq=1,
            random_state=random_state, n_jobs=-1, verbose=-1,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgboost(X_train, y_train, n_trials: int = 50, random_state: int = 42) -> Dict[str, Any]:
    """Recherche Optuna maximisant la PR-AUC sur validation temporelle (XGBoost)."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError("Optuna requis : pip install optuna") from exc

    X_tr, X_val, y_tr, y_val = make_validation_split(X_train, y_train, random_state=random_state)
    spw = _scale_pos_weight(y_train)

    def objective(trial):
        params = dict(
            n_estimators=3000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist",
            random_state=random_state, n_jobs=-1, early_stopping_rounds=50,
        )
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# --------------------------------------------------------------------------- #
#  Stacking — base learners décorrélés + CatBoost/AdaBoost optionnels
# --------------------------------------------------------------------------- #
def train_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    xgb_params: Optional[Dict[str, Any]] = None,
    lgbm_params: Optional[Dict[str, Any]] = None,
    catboost_params: Optional[Dict[str, Any]] = None,
    include_catboost: bool = True,
    include_adaboost: bool = False,
    cv=None,
    passthrough: bool = False,
    random_state: int = 42,
) -> StackingClassifier:
    """
    Stacking : XGBoost + LightGBM + LogisticRegression (+ CatBoost / AdaBoost
    optionnels) → méta-modèle LogisticRegression.

    Choix de conception :
    - Base learners à PLEINE PUISSANCE (et non bridés).
    - LogisticRegression comme base learner DÉCORRÉLÉ (la diversité fait le gain).
    - CatBoost ajouté par défaut : 3e booster, implémentation différente
      (ordered boosting) → apporte un peu de diversité supplémentaire.
    - AdaBoost désactivé par défaut : peu performant sur déséquilibre extrême,
      il dilue souvent plus qu'il n'aide.
    - CV TEMPORELLE (TimeSeriesSplit) si colonne `month` présente.

    NOTE IMPORTANTE — modèles non supervisés (LOF, One-Class SVM,
    Isolation Forest) : ils ne sont PAS empilés ici. Un StackingClassifier
    combine des classifieurs à `predict_proba` ; or ces détecteurs renvoient un
    score d'anomalie non probabiliste et non supervisé, faiblement prédictif sur
    BAF (PR-AUC ~0.02). Les ajouter comme estimateurs de base injecterait du
    quasi-bruit. La bonne intégration est HYBRIDE : transformer leurs scores en
    features via `add_anomaly_features(X_train, X_test, detectors)` AVANT
    d'appeler ce stacking.
    """
    spw = _scale_pos_weight(y_train)

    # CV temporelle si possible
    if cv is None:
        if "month" in X_train.columns:
            order = X_train["month"].argsort(kind="stable")
            X_train = X_train.iloc[order]
            y_train = y_train.iloc[order]
            cv = TimeSeriesSplit(n_splits=3)
        else:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    # Modèles de base à pleine puissance
    xgb_base = dict(
        n_estimators=500, learning_rate=0.05, max_depth=6, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=spw,
        eval_metric="aucpr", tree_method="hist", random_state=random_state, n_jobs=-1)
    if xgb_params:
        xgb_base.update(xgb_params); xgb_base["scale_pos_weight"] = spw
        xgb_base["eval_metric"] = "aucpr"

    lgbm_base = dict(
        n_estimators=500, learning_rate=0.05, num_leaves=64, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        class_weight="balanced", objective="binary", random_state=random_state,
        n_jobs=-1, verbose=-1)
    if lgbm_params:
        lgbm_base.update(lgbm_params); lgbm_base["class_weight"] = "balanced"

    base_estimators: List[Tuple[str, Any]] = [
        ("xgb", xgb.XGBClassifier(**xgb_base)),
        ("lgbm", lgb.LGBMClassifier(**lgbm_base)),
        ("lr", make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=random_state, n_jobs=-1))),
    ]

    if include_catboost:
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("CatBoost requis (ou include_catboost=False) : "
                              "pip install catboost") from exc
        cat_base = dict(
            iterations=500, learning_rate=0.05, depth=6, loss_function="Logloss",
            scale_pos_weight=spw, random_state=random_state, verbose=0,
            allow_writing_files=False)
        if catboost_params:
            cat_base.update(catboost_params); cat_base["scale_pos_weight"] = spw
        base_estimators.append(("catboost", CatBoostClassifier(**cat_base)))

    if include_adaboost:
        ada = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=3, class_weight="balanced", random_state=random_state),
            n_estimators=300, learning_rate=0.05, random_state=random_state)
        base_estimators.append(("adaboost", ada))

    final_estimator = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=random_state)

    stack = StackingClassifier(
        estimators=base_estimators, final_estimator=final_estimator,
        cv=cv, n_jobs=-1, passthrough=passthrough)
    stack.fit(X_train, y_train)
    return stack


# --------------------------------------------------------------------------- #
#  Calibration
# --------------------------------------------------------------------------- #
def calibrate_model(fitted_model, X_calib, y_calib, method: str = "isotonic"):
    """Recale les probabilités d'un modèle déjà entraîné (cv='prefit')."""
    calibrated = CalibratedClassifierCV(fitted_model, method=method, cv="prefit")
    calibrated.fit(X_calib, y_calib)
    return calibrated


# --------------------------------------------------------------------------- #
#  Deep Learning
# --------------------------------------------------------------------------- #
def build_autoencoder(input_dim: int, encoding_dim: int = 16):
    """Autoencoder pour détection d'anomalies (entraîné sur les transactions normales)."""
    try:
        from tensorflow import keras
    except ImportError:
        raise ImportError("TensorFlow requis pour l'Autoencoder. pip install tensorflow")

    encoder = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(encoding_dim, activation="relu"),
    ], name="encoder")
    decoder = keras.Sequential([
        keras.layers.Input(shape=(encoding_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(input_dim, activation="linear"),
    ], name="decoder")
    autoencoder = keras.Sequential([encoder, decoder], name="autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def build_mlp(input_dim: int, dropout: float = 0.3, learning_rate: float = 1e-3):
    """Réseau dense supervisé — représentant du DL tabulaire (comparaison)."""
    try:
        from tensorflow import keras
    except ImportError:
        raise ImportError("TensorFlow requis pour le MLP. pip install tensorflow")

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="fraud_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.AUC(curve="PR", name="pr_auc")])
    return model
