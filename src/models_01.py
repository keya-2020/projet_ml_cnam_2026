
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
import xgboost as xgb
import lightgbm as lgb


#  Utilitaires internes

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

    - Si une colonne temporelle (`month`) existe : on isole le ou les derniers
      mois (split TEMPOREL, cohérent avec la dérive du dataset).
    - Sinon : repli sur un holdout stratifié aléatoire.

    Returns
    -------
    X_tr, X_val, y_tr, y_val
    """
    if month_col in X.columns:
        months = np.sort(X[month_col].unique())
        # On garde le dernier mois en validation (au moins 1 mois).
        n_val_months = max(1, int(round(len(months) * val_fraction)))
        val_months = set(months[-n_val_months:])
        mask_val = X[month_col].isin(val_months)
        if mask_val.any() and (~mask_val).any():
            return X[~mask_val], X[mask_val], y[~mask_val], y[mask_val]

    # Repli : holdout stratifié.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=random_state
    )
    return X_tr, X_val, y_tr, y_val


#  Baselines

def train_dummy(X_train, y_train, strategy: str = "stratified", random_state: int = 42):
    """
    Baseline naïf — référence absolue à battre.

    PR-AUC attendu ≈ taux de fraude (~0.01). Tout modèle ML doit faire
    significativement mieux.
    """
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
    """
    Régression logistique — baseline ML linéaire.

    Sert de référence pour mesurer l'apport des modèles non linéaires.
    Nécessite une normalisation des features (StandardScaler).

    Returns
    -------
    model, scaler : tuple
        Le scaler doit être appliqué au X_test avant prédiction.
    """
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, scaler


def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float = 0.011,
    n_estimators: int = 200,
    random_state: int = 42,
) -> IsolationForest:
    """
    Isolation Forest — détection d'anomalies sans labels.

    Le paramètre `contamination` doit refléter le taux de fraude attendu
    (~1.1% sur BAF Base).

    Notes
    -----
    Pour la prédiction, utiliser `.decision_function(X_test)` — un score
    élevé = normal, score bas = anormal. Inverser le signe pour avoir
    "plus élevé = plus suspect".
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


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
    """
    Random Forest - modèle d'ensemble par bagging.

    Sert de point de comparaison pour les méthodes de boosting (XGBoost, LightGBM).
    Réglages légèrement renforcés vs version initiale : arbres un peu plus
    profonds, feuilles minimales pour limiter le sur-apprentissage sur la classe
    majoritaire, et `balanced_subsample` (rééquilibrage par bootstrap) mieux
    adapté au déséquilibre extrême.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


#  Boosting - avec early stopping réellement activé

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    auto_val: bool = True,
    params: Optional[Dict[str, Any]] = None,
    n_estimators: int = 2000,
    learning_rate: float = 0.03,
    max_depth: int = 6,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """
    XGBoost — modèle supervisé principal du projet.

    Optimisations :
    - `scale_pos_weight` calculé automatiquement (ratio neg/pos).
    - `eval_metric="aucpr"` (PR-AUC), métrique de référence du projet.
    - Early stopping TOUJOURS actif : si X_val n'est pas fourni et `auto_val`,
      un fold de validation temporel est découpé automatiquement. On peut donc
      mettre `n_estimators` élevé sans risque — le modèle s'arrête seul.
    - Régularisation et sous-échantillonnage ajoutés (subsample, colsample,
      min_child_weight, gamma, reg_lambda).

    Parameters
    ----------
    params : dict, optionnel
        Hyperparamètres issus d'un tuning Optuna. S'ils sont fournis, ils
        surchargent les valeurs par défaut.
    auto_val : bool
        Si True et qu'aucun X_val n'est fourni, découpe un fold de validation
        temporel pour l'early stopping.
    """
    spw = _scale_pos_weight(y_train)

    default_params = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    if params:
        default_params.update(params)
        # On garde toujours scale_pos_weight et la métrique cohérents.
        default_params.setdefault("scale_pos_weight", spw)
        default_params["eval_metric"] = "aucpr"

    # Découpe automatique d'un fold de validation si nécessaire.
    if (X_val is None or y_val is None) and auto_val:
        X_train, X_val, y_train, y_val = make_validation_split(
            X_train, y_train, random_state=random_state
        )

    model = xgb.XGBClassifier(**default_params)

    if X_val is not None and y_val is not None:
        model.set_params(early_stopping_rounds=early_stopping_rounds)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)

    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    auto_val: bool = True,
    params: Optional[Dict[str, Any]] = None,
    n_estimators: int = 2000,
    learning_rate: float = 0.03,
    num_leaves: int = 64,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
) -> lgb.LGBMClassifier:
    """
    LightGBM — alternative à XGBoost, souvent plus rapide.

    Mêmes optimisations que `train_xgboost` : early stopping toujours actif via
    un fold de validation temporel automatique, régularisation et
    sous-échantillonnage ajoutés. Complémentaire à XGBoost pour le stacking.
    """
    default_params = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        class_weight="balanced",
        objective="binary",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    if params:
        default_params.update(params)
        default_params.setdefault("class_weight", "balanced")

    if (X_val is None or y_val is None) and auto_val:
        X_train, X_val, y_train, y_val = make_validation_split(
            X_train, y_train, random_state=random_state
        )

    model = lgb.LGBMClassifier(**default_params)

    fit_params: Dict[str, Any] = {}
    if X_val is not None and y_val is not None:
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["eval_metric"] = "average_precision"
        fit_params["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

    model.fit(X_train, y_train, **fit_params)
    return model


#  Tuning Optuna — optimise directement la PR-AUC sur validation temporelle

def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Recherche bayésienne (Optuna) des hyperparamètres LightGBM maximisant la
    PR-AUC sur un fold de validation temporel.

    Returns
    -------
    dict : meilleurs hyperparamètres, réutilisables via
           `train_lightgbm(..., params=best_params)`.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError("Optuna requis : pip install optuna") from exc

    X_tr, X_val, y_tr, y_val = make_validation_split(
        X_train, y_train, random_state=random_state
    )

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
            class_weight="balanced",
            objective="binary",
            subsample_freq=1,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Recherche bayésienne (Optuna) des hyperparamètres XGBoost maximisant la
    PR-AUC sur un fold de validation temporel.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError("Optuna requis : pip install optuna") from exc

    X_tr, X_val, y_tr, y_val = make_validation_split(
        X_train, y_train, random_state=random_state
    )
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
            scale_pos_weight=spw,
            eval_metric="aucpr",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
            early_stopping_rounds=50,
        )
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, proba)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


#  Stacking — réparé : base learners à pleine puissance + LR + CV temporelle

def train_stacking_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    xgb_params: Optional[Dict[str, Any]] = None,
    lgbm_params: Optional[Dict[str, Any]] = None,
    cv=None,
    passthrough: bool = False,
    random_state: int = 42,
) -> StackingClassifier:
    """
    Stacking : XGBoost + LightGBM + Logistic Regression → méta-modèle LR.

    Corrections vs version initiale :
    - Les modèles de base sont désormais à PLEINE PUISSANCE (et non bridés à
      200 arbres @ lr 0.1), cohérents avec les modèles autonomes.
    - Ajout d'un VRAI apprenant linéaire décorrélé (LogisticRegression dans un
      pipeline de normalisation) comme base learner — c'est la diversité qui
      fait gagner le stacking, pas l'empilement de deux boosters corrélés.
    - Validation croisée TEMPORELLE (TimeSeriesSplit) si une colonne `month`
      est présente : on évite la fuite d'information "future" des folds
      aléatoires, cohérent avec la dérive du dataset.

    Parameters
    ----------
    xgb_params, lgbm_params : dict, optionnels
        Hyperparamètres issus de `tune_xgboost` / `tune_lightgbm`. Si fournis,
        ils remplacent les valeurs par défaut des modèles de base.
    cv : objet de cross-validation, optionnel
        Par défaut : TimeSeriesSplit(3) si `month` présent (données triées par
        mois), sinon StratifiedKFold(3).
    """
    spw = _scale_pos_weight(y_train)

    # --- CV temporelle si possible -----------------------------------------
    if cv is None:
        if "month" in X_train.columns:
            order = X_train["month"].argsort(kind="stable")
            X_train = X_train.iloc[order]
            y_train = y_train.iloc[order]
            cv = TimeSeriesSplit(n_splits=3)
        else:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    # --- Modèles de base à pleine puissance --------------------------------
    xgb_base = dict(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=spw, eval_metric="aucpr",
        tree_method="hist", random_state=random_state, n_jobs=-1,
    )
    if xgb_params:
        xgb_base.update(xgb_params)
        xgb_base["scale_pos_weight"] = spw
        xgb_base["eval_metric"] = "aucpr"

    lgbm_base = dict(
        n_estimators=500, learning_rate=0.05, num_leaves=64,
        min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, class_weight="balanced",
        objective="binary", random_state=random_state, n_jobs=-1, verbose=-1,
    )
    if lgbm_params:
        lgbm_base.update(lgbm_params)
        lgbm_base["class_weight"] = "balanced"

    base_estimators = [
        ("xgb", xgb.XGBClassifier(**xgb_base)),
        ("lgbm", lgb.LGBMClassifier(**lgbm_base)),
        # Apprenant linéaire décorrélé — apporte la diversité au méta-modèle.
        ("lr", make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000,
                               random_state=random_state, n_jobs=-1),
        )),
    ]

    final_estimator = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=random_state,
    )

    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=-1,
        passthrough=passthrough,
    )
    stack.fit(X_train, y_train)
    return stack


#  Calibration des probabilités (utile après scale_pos_weight ≈ 96)

def calibrate_model(
    fitted_model,
    X_calib: pd.DataFrame,
    y_calib: pd.Series,
    method: str = "isotonic",
) -> CalibratedClassifierCV:
    """
    Calibre les probabilités d'un modèle déjà entraîné sur un jeu de calibration
    séparé (idéalement récent, pour absorber la dérive).

    Un rééquilibrage fort (`scale_pos_weight` ≈ 96 ou `class_weight="balanced"`)
    déforme les probabilités : la calibration les recale, ce qui améliore le
    choix du seuil opérationnel sans changer le classement (donc sans toucher la
    PR-AUC / ROC-AUC).

    Notes
    -----
    `cv="prefit"` suppose que `fitted_model` est déjà entraîné. X_calib/y_calib
    doivent être DISJOINTS des données d'entraînement.
    """
    calibrated = CalibratedClassifierCV(fitted_model, method=method, cv="prefit")
    calibrated.fit(X_calib, y_calib)
    return calibrated


#  Modèles Deep Learning

def build_autoencoder(input_dim: int, encoding_dim: int = 16):
    """
    Construit un Autoencoder pour la détection d'anomalies non supervisée.

    L'Autoencoder est entraîné UNIQUEMENT sur les transactions légitimes.
    Face à une fraude, l'erreur de reconstruction est élevée → signal d'anomalie.

    Parameters
    ----------
    input_dim : int
        Nombre de features en entrée.
    encoding_dim : int, default=16
        Dimension de l'espace latent (compression).

    Returns
    -------
    keras.Model
        Modèle compilé prêt à être entraîné avec .fit(X_normal, X_normal)
    """
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
    """
    Réseau de neurones dense supervisé.

    Sert à comparer le DL aux modèles à base d'arbres sur données tabulaires.
    La littérature 2022+ (Grinsztajn et al., NeurIPS) montre que les arbres
    surpassent généralement le DL dans ce contexte — ce résultat fait
    partie de la discussion.

    Améliorations légères : BatchNormalization pour stabiliser l'entraînement
    sur features hétérogènes, et taux d'apprentissage explicite. Pour gérer le
    déséquilibre, passer `class_weight={0: 1.0, 1: ~96}` à `.fit()` côté notebook.
    """
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
        metrics=[keras.metrics.AUC(name="auc"),
                 keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model