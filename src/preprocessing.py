
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from .data_loader import CATEGORICAL_COLS


def temporal_split(
    df: pd.DataFrame,
    train_months: list = None,
    test_months: list = None,
    target_col: str = "fraud_bool",
    month_col: str = "month",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame BAF complet.
    train_months : list, default=[0, 1, 2, 3, 4, 5]
        Mois utilisés pour l'entraînement.
    test_months : list, default=[6, 7]
        Mois utilisés pour le test.
    target_col : str
        Nom de la colonne cible.
    month_col : str
        Nom de la colonne temporelle.

    Returns
    -------
    X_train, X_test, y_train, y_test 
  
    """
    if train_months is None:
        train_months = [0, 1, 2, 3, 4, 5]
    if test_months is None:
        test_months = [6, 7]

    train_mask = df[month_col].isin(train_months)
    test_mask = df[month_col].isin(test_months)

    cols_to_drop = [target_col, month_col]
    X_train = df.loc[train_mask].drop(columns=cols_to_drop)
    X_test = df.loc[test_mask].drop(columns=cols_to_drop)
    y_train = df.loc[train_mask, target_col]
    y_test = df.loc[test_mask, target_col]

    return X_train, X_test, y_train, y_test


def encode_categoricals(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    method: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode les variables catégorielles.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame      
    categorical_cols : list, optional
        Colonnes à encoder. Par défaut : auto-détection (dtype object).
    method : {default="label" ou "onehot"}
        - "label" : LabelEncoder (suffisant pour modèles à base d'arbres)       

    Returns
    -------
    X_train_enc, X_test_enc : pd.DataFrame

    Notes
    -----
    Pour les modèles à base d'arbres (XGBoost, RandomForest, LightGBM),
    LabelEncoder est suffisant car ils n'imposent pas de relation d'ordre.
    Pour les modèles linéaires (LogisticRegression, MLP), il faut OneHot.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    if categorical_cols is None:
        categorical_cols = X_train.select_dtypes(include="object").columns.tolist()

    if method == "label":
        for col in categorical_cols:
            if col not in X_train.columns:
                continue
            le = LabelEncoder()
            combined = pd.concat([X_train[col], X_test[col]]).astype(str)
            le.fit(combined)
            X_train[col] = le.transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

    elif method == "onehot":
        X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
        # Aligner les colonnes (au cas où une modalité est absente du test)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    else:
        raise ValueError(f"Méthode inconnue : {method}. Choisir 'label' ou 'onehot'.")

    return X_train, X_test


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features dérivées améliorant afin d' améliorer le signal de détection.

    Features créées :
    - velocity_ratio_6h_24h : concentration temporelle 6h vs 24h
    - velocity_ratio_6h_4w : concentration temporelle 6h vs 4 semaines
    - device_ever_fraud : flag binaire (device a déjà fraudé)
    - both_phones_valid : cohérence téléphonique
    - risk_flags_count : compteur de signaux suspects

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame BAF avec les colonnes brutes.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les features supplémentaires.
    """
    df = df.copy()

    # 1. Ratios de vélocité — concentration temporelle de l'activité
    if {"velocity_6h", "velocity_24h"}.issubset(df.columns):
        df["velocity_ratio_6h_24h"] = df["velocity_6h"] / (df["velocity_24h"] + 1)

    if {"velocity_6h", "velocity_4w"}.issubset(df.columns):
        df["velocity_ratio_6h_4w"] = df["velocity_6h"] / (df["velocity_4w"] + 1)

    # 2. Flag binaire : ce device a-t-il déjà fraudé ?
    if "device_fraud_count" in df.columns:
        df["device_ever_fraud"] = (df["device_fraud_count"] > 0).astype(int)

    # 3. Cohérence téléphonique
    if {"phone_home_valid", "phone_mobile_valid"}.issubset(df.columns):
        df["both_phones_valid"] = (
            (df["phone_home_valid"] == 1) & (df["phone_mobile_valid"] == 1)
        ).astype(int)
        df["no_phone_valid"] = (
            (df["phone_home_valid"] == 0) & (df["phone_mobile_valid"] == 0)
        ).astype(int)

    # 4. Score de risque agrégé : combinaison de flags suspects
    risk_components = []
    if "email_is_free" in df.columns:
        risk_components.append((df["email_is_free"] == 1).astype(int))
    if "foreign_request" in df.columns:
        risk_components.append((df["foreign_request"] == 1).astype(int))
    if "phone_home_valid" in df.columns:
        risk_components.append((df["phone_home_valid"] == 0).astype(int))
    if "device_fraud_count" in df.columns:
        risk_components.append((df["device_fraud_count"] > 0).astype(int))

    if risk_components:
        df["risk_flags_count"] = sum(risk_components)

    return df

def drop_constant_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    extra_drop: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Supprime les colonnes à variance nulle (constantes) + une liste optionnelle.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
    extra_drop : list, optional
        Colonnes à retirer en plus, quelle que soit leur variance.
    verbose : bool
        Affiche les colonnes supprimées.

    Returns
    -------
    X_train, X_test, dropped : DataFrames nettoyés + liste des colonnes retirées.
    """
    constant_cols = [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]
    to_drop = sorted(set(constant_cols) | set(extra_drop or []))
    to_drop = [c for c in to_drop if c in X_train.columns]

    if verbose and to_drop:
        print(f"Colonnes supprimées (constantes / forcées) : {to_drop}")

    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])
    return X_train, X_test, to_drop


def add_missing_indicators(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cols: Optional[List[str]] = None,
    min_missing_rate: float = 0.01,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    
    Parameters
    ----------
    X_train, X_test : pd.DataFrame
    cols : list, optional
        Liste explicite de colonnes. Si fournie, elle PRIME sur le seuil
        (les indicateurs sont créés pour ces colonnes, quel que soit leur taux).
    min_missing_rate : float, default=0.01
        Taux de manquant minimal (sur le train) pour créer un indicateur.
        Ignoré si `cols` est fourni.

    Returns
    -------
    X_train, X_test, indicator_cols
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    if cols is None:
        rate = X_train.isna().mean()
        cols = rate[rate >= min_missing_rate].index.tolist()

    indicator_cols = []
    for col in cols:
        if col not in X_train.columns:
            continue
        name = f"{col}_is_missing"
        X_train[name] = X_train[col].isna().astype(int)
        X_test[name] = X_test[col].isna().astype(int)
        indicator_cols.append(name)

    return X_train, X_test, indicator_cols


def impute_missing(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    strategy: str = "sentinel",
    fillna_value: float = -999.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute les valeurs manquantes.

    Parameters
    ----------
    strategy : {"sentinel", "median"}, default="sentinel"
        - "sentinel" : remplace par `fillna_value` (-999). Idéal pour les arbres,
          qui apprennent ce code comme une modalité « manquant ».
        - "median" : médiane ajustée sur le TRAIN, appliquée au test (pas de
          fuite). À privilégier pour les modèles linéaires / MLP, où le sentinelle
          est toxique. À combiner avec `add_missing_indicators`.
    fillna_value : float
        Valeur du sentinelle.

    Returns
    -------
    X_train, X_test imputés.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    if strategy == "sentinel":
        X_train = X_train.fillna(fillna_value)
        X_test = X_test.fillna(fillna_value)

    elif strategy == "median":
        medians = X_train.median(numeric_only=True)
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)
        # Filet de sécurité : colonnes entièrement NaN dans le train.
        X_train = X_train.fillna(fillna_value)
        X_test = X_test.fillna(fillna_value)

    else:
        raise ValueError(f"Stratégie inconnue : {strategy}. Choisir 'sentinel' ou 'median'.")

    return X_train, X_test


def handle_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "smote",
    sampling_strategy: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applique une stratégie de gestion du déséquilibre de classes.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Données d'entraînement (uniquement - ne JAMAIS appliquer au test).
    method : {"smote", "undersample", "none"}, default="smote"
        - "smote" : oversampling synthétique de la classe minoritaire
        - "undersample" : sous-échantillonnage aléatoire de la classe majoritaire
        - "none" : retourne les données inchangées
    sampling_strategy : float, default=0.1
        Ratio cible classe_minoritaire/classe_majoritaire après resampling.
    random_state : int
        Pour la reproductibilité.

    Returns
    -------
    X_resampled, y_resampled : pd.DataFrame, pd.Series

    """
    if method == "none":
        return X_train, y_train

    elif method == "smote":
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method == "undersample":
        sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy, random_state=random_state
        )
    else:
        raise ValueError(f"Méthode inconnue : {method}")

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def preprocess_pipeline(
    df: pd.DataFrame,
    feature_engineering: bool = True,
    encoding_method: str = "label",
    imbalance_method: str = "none",
    fillna_value: float = -999.0,
    drop_constant: bool = True,
    extra_drop: Optional[List[str]] = None,
    add_missing_flags: bool = True,
    impute_strategy: str = "sentinel",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Pipeline complet de preprocessing — appliqué dans le bon ordre.

    Étapes :
    1. Feature engineering (sur l'ensemble pour cohérence)
    2. Split temporel
    3. Encodage des catégorielles (fit sur train, transform sur test)
    4. Suppression des colonnes constantes (détectées sur le train)   
    5. Indicateurs de manquant `{col}_is_missing` (avant imputation) 
    6. Imputation des NaN (sentinelle -999 OU médiane train-fit)     
    7. Gestion du déséquilibre (sur train uniquement)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset BAF chargé via load_baf().
    feature_engineering : bool
        Si True, applique engineer_features().
    encoding_method : {"label", "onehot"}
        Méthode d'encodage des catégorielles.
    imbalance_method : {"smote", "undersample", "none"}
        Stratégie de gestion du déséquilibre.
    fillna_value : float
        Valeur du sentinelle si impute_strategy="sentinel".
    drop_constant : bool, default=True
        Supprime les colonnes à variance nulle (ex. device_fraud_count).
    extra_drop : list, optional
        Colonnes supplémentaires à retirer explicitement.
    add_missing_flags : bool, default=True
        Ajoute les indicateurs de manquant avant imputation.
    impute_strategy : {"sentinel", "median"}, default="sentinel"
        "sentinel" pour les arbres (recommandé), "median" pour linéaire/MLP.

    Returns
    -------
    X_train, X_test, y_train, y_test : ready for modeling
    """
    # 1. Feature engineering
    if feature_engineering:
        df = engineer_features(df)

    # 2. Split temporel
    X_train, X_test, y_train, y_test = temporal_split(df)

    # 3. Encodage
    X_train, X_test = encode_categoricals(X_train, X_test, method=encoding_method)

    # 4. Suppression des colonnes constantes
    if drop_constant:
        X_train, X_test, _ = drop_constant_columns(
            X_train, X_test, extra_drop=extra_drop
        )
    elif extra_drop:
        cols = [c for c in extra_drop if c in X_train.columns]
        X_train = X_train.drop(columns=cols)
        X_test = X_test.drop(columns=[c for c in cols if c in X_test.columns])

    # 5. Indicateurs de manquant (AVANT imputation pour préserver le signal MNAR)
    if add_missing_flags:
        X_train, X_test, _ = add_missing_indicators(X_train, X_test)

    # 6. Imputation des NaN
    X_train, X_test = impute_missing(
        X_train, X_test, strategy=impute_strategy, fillna_value=fillna_value
    )

    # 7. Gestion du déséquilibre (uniquement sur train)
    if imbalance_method != "none":
        X_train, y_train = handle_imbalance(X_train, y_train, method=imbalance_method)

    return X_train, X_test, y_train, y_test