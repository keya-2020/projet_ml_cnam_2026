from pathlib import Path
from typing import List
import warnings

import numpy as np
import pandas as pd


# Variables avec NaN codés en -1 (selon documentation officielle BAF datasheet)
COLS_WITH_NEG_NAN: List[str] = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

# Variable particulière : intended_balcon_amount
# Selon le datasheet officiel BAF : range [-16, 114], TOUTES les négatives sont des NaN
# (pas seulement -1). À traiter séparément avec une logique < 0.
COLS_WITH_ALL_NEG_NAN: List[str] = [
    "intended_balcon_amount",
]

# Variables catégorielles dans BAF
CATEGORICAL_COLS: List[str] = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

# Attributs sensibles pour l'analyse fairness
SENSITIVE_ATTRIBUTES: List[str] = [
    "customer_age",
    "income",
    "employment_status",
]


def load_baf(
    path: str | Path,
    variant: str = "Base",
    fix_missing: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """        
        Parameters
        ----------
        path : str or Path
            Chemin vers le dossier contenant les fichiers CSV, ou directement vers un fichier.
        variant : str, default="Base"
            Variante à charger : "Base", "Variant I", "Variant II", ..., "Variant V".
            Ignoré si `path` pointe vers un fichier spécifique.
        fix_missing : bool, default=True
            Si True, convertit les -1 en NaN sur les colonnes documentées.
        verbose : bool, default=True
            Affiche les informations de chargement.

        Returns
        -------
        pd.DataFrame
            DataFrame contenant les données BAF, avec NaN correctement encodés.
 """
    path = Path(path)

    # Résoudre le chemin du fichier
    if path.is_file():
        file_path = path
    elif path.is_dir():
        file_path = path / f"{variant}.csv"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {file_path}. "                
            )
    else:
        raise FileNotFoundError(f"Chemin inexistant : {path}")

    # Chargement
    if verbose:
        print(f"Chargement de {file_path.name}...")
    df = pd.read_csv(file_path)

    # Correction des valeurs manquantes codées en -1
    if fix_missing:
        df = fix_missing_values(df, verbose=verbose)

    if verbose:
        print(f"Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
        if "fraud_bool" in df.columns:
            fraud_rate = df["fraud_bool"].mean()
            print(f"Taux de fraude : {fraud_rate:.2%}")

    return df


def fix_missing_values(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Convertit les valeurs codées -1 en NaN selon la documentation officielle BAF.

    Deux familles de colonnes :
    1. COLS_WITH_NEG_NAN : seul -1 = NaN (valeurs négatives autres = valeurs valides)
    2. COLS_WITH_ALL_NEG_NAN : TOUTES les valeurs négatives = NaN
       Concerne notamment `intended_balcon_amount` dont la doc précise
       "ranges between [-16, 114] (negatives are missing values)".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame brut chargé depuis un CSV BAF.
    verbose : bool
        Affiche le nombre de NaN détectés par colonne.

    Returns
    -------
    pd.DataFrame
        Copie avec les NaN correctement encodés.
    """
    df = df.copy()
    nan_counts = {}

    # Famille 1 : seul -1 = NaN
    for col in COLS_WITH_NEG_NAN:
        if col not in df.columns:
            warnings.warn(f"Colonne attendue absente : {col}")
            continue
        n_neg = (df[col] == -1).sum()
        if n_neg > 0:
            df[col] = df[col].replace(-1, np.nan)
            nan_counts[col] = n_neg

    # Famille 2 : TOUTES les négatives = NaN
    for col in COLS_WITH_ALL_NEG_NAN:
        if col not in df.columns:
            warnings.warn(f"Colonne attendue absente : {col}")
            continue
        neg_mask = df[col] < 0
        n_neg = neg_mask.sum()
        if n_neg > 0:
            df.loc[neg_mask, col] = np.nan
            nan_counts[col] = n_neg

    if verbose and nan_counts:
        print("\nValeurs manquantes corrigées (codage négatif → NaN) :")
        for col, count in sorted(nan_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(df) * 100
            tag = "(toutes négatives)" if col in COLS_WITH_ALL_NEG_NAN else "(-1 uniquement)"
            print(f"  {col:35s} : {count:>8,} ({pct:.1f}%) {tag}")

    return df


def get_feature_groups(df: pd.DataFrame) -> dict:
    """
    Retourne les features groupées par famille thématique.

    Utile pour les analyses ciblées (ex. EDA par groupe, feature importance par groupe).

    Returns
    -------
    dict
        Dictionnaire {nom_groupe: [colonnes]}
    """
    groups = {
        "target": ["fraud_bool"],
        "temporal": ["month"],
        "sensitive": ["customer_age", "income", "employment_status"],
        "address": [
            "prev_address_months_count",
            "current_address_months_count",
            "housing_status",
            "zip_count_4w",
        ],
        "velocity": ["velocity_6h", "velocity_24h", "velocity_4w"],
        "device": [
            "device_os",
            "device_distinct_emails_8w",
            "device_fraud_count",
            "session_length_in_minutes",
        ],
        "phone_email": [
            "phone_home_valid",
            "phone_mobile_valid",
            "email_is_free",
        ],
        "bank": [
            "bank_months_count",
            "bank_branch_count_8w",
            "has_other_cards",
            "credit_risk_score",
        ],
        "financial": ["proposed_credit_limit", "intended_balcon_amount"],
        "context": [
            "source",
            "payment_type",
            "keep_alive_session",
            "foreign_request",
            "days_since_request",
        ],
    }
    # Filtrer pour ne garder que les colonnes présentes
    return {k: [c for c in v if c in df.columns] for k, v in groups.items()}


def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produit un résumé synthétique du dataset : type, NaN, valeurs uniques.

    Returns
    -------
    pd.DataFrame
        Une ligne par colonne avec : dtype, n_unique, n_missing, pct_missing.
    """
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "n_unique": df.nunique(),
        "n_missing": df.isna().sum(),
        "pct_missing": df.isna().mean() * 100,
    })
    return summary.sort_values("pct_missing", ascending=False)