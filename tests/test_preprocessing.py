"""Tests unitaires pour le module preprocessing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest

from src.preprocessing import (
    temporal_split, encode_categoricals, engineer_features,
    handle_imbalance, preprocess_pipeline
)


@pytest.fixture
def sample_df():
    """Crée un DataFrame de test imitant la structure BAF."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'fraud_bool': np.random.choice([0, 1], n, p=[0.99, 0.01]),
        'month': np.random.choice(range(8), n),
        'velocity_6h': np.random.exponential(50, n),
        'velocity_24h': np.random.exponential(100, n),
        'velocity_4w': np.random.exponential(200, n),
        'device_fraud_count': np.random.choice([0, 1, 2], n, p=[0.95, 0.04, 0.01]),
        'phone_home_valid': np.random.choice([0, 1], n),
        'phone_mobile_valid': np.random.choice([0, 1], n),
        'email_is_free': np.random.choice([0, 1], n),
        'foreign_request': np.random.choice([0, 1], n),
        'payment_type': np.random.choice(['AA', 'AB', 'AC'], n),
        'employment_status': np.random.choice(['CA', 'CB', 'CC'], n),
        'credit_risk_score': np.random.normal(100, 50, n),
    })


def test_temporal_split(sample_df):
    """Vérifie que le split temporel respecte les mois."""
    X_train, X_test, y_train, y_test = temporal_split(sample_df)

    # Aucune colonne target ou month dans X
    assert 'fraud_bool' not in X_train.columns
    assert 'month' not in X_train.columns

    # Bonne séparation
    assert len(X_train) + len(X_test) == len(sample_df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_engineer_features(sample_df):
    """Vérifie que les features dérivées sont créées."""
    df_eng = engineer_features(sample_df)

    expected_features = [
        'velocity_ratio_6h_24h',
        'velocity_ratio_6h_4w',
        'device_ever_fraud',
        'both_phones_valid',
        'no_phone_valid',
        'risk_flags_count',
    ]

    for feat in expected_features:
        assert feat in df_eng.columns, f"Feature manquante : {feat}"

    # Vérifications de cohérence
    assert df_eng['device_ever_fraud'].isin([0, 1]).all()
    assert df_eng['both_phones_valid'].isin([0, 1]).all()
    assert (df_eng['velocity_ratio_6h_24h'] >= 0).all()
    assert (df_eng['risk_flags_count'] >= 0).all()
    assert (df_eng['risk_flags_count'] <= 4).all()


def test_encode_categoricals(sample_df):
    """Vérifie l'encodage des variables catégorielles."""
    X_train, X_test, _, _ = temporal_split(sample_df)
    X_train_enc, X_test_enc = encode_categoricals(X_train, X_test, method='label')

    # Aucune colonne object après encodage
    assert (X_train_enc.dtypes != 'object').all()
    assert (X_test_enc.dtypes != 'object').all()


def test_handle_imbalance_smote(sample_df):
    """Vérifie que SMOTE rééquilibre les classes."""
    X_train, _, y_train, _ = temporal_split(sample_df)
    X_train, _ = encode_categoricals(X_train, X_train, method='label')
    X_train = X_train.fillna(-999)

    X_sm, y_sm = handle_imbalance(X_train, y_train, method='smote',
                                    sampling_strategy=0.5)

    # La classe minoritaire représente au moins 30% après SMOTE
    minority_ratio = (y_sm == 1).sum() / (y_sm == 0).sum()
    assert minority_ratio >= 0.3, f"Ratio insuffisant : {minority_ratio}"


def test_preprocess_pipeline(sample_df):
    """Test du pipeline complet."""
    X_train, X_test, y_train, y_test = preprocess_pipeline(
        sample_df,
        feature_engineering=True,
        encoding_method='label',
        imbalance_method='none'
    )

    assert X_train.shape[1] == X_test.shape[1]
    assert len(X_train) == len(y_train)
    assert (X_train.dtypes != 'object').all()
    assert X_train.isna().sum().sum() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
