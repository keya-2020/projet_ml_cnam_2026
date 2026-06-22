Utilisation du jeu de données initial : Base.csv

Analyse préliminaire et nettoyage des données
Ouvrir et exécuter : Prelim0_eda_baf.ipynb
Ouvrir et exécuter : Prelim0_drift_analysis.ipynb
Ouvrir et exécuter : Prelim0_fairness_analysis.ipynb
Ouvrir et exécuter : Prelim0_data_quality.ipynb

Prétraitement du jeu de données initial
Ouvrir et exécuter : Prelim0_preprocessing.py → Génère le fichier nettoyé data_processed.csv
Ouvrir et exécuter : Prelim0_feature_selection.py → Réalise la sélection de variables et génère data_processed_alternative.csv, contenant un nombre réduit de variables.

Utilisation du nouveau jeu de données pour les tests des modèles
Ouvrir common_ml.py et définir : df_selected = pd.read_csv("data_selected_report.csv")
Par défaut, l'ensemble des expérimentations utilise le jeu de données data_selected_report.csv. Si nécessaire, il peut être remplacé par data_processed.csv.

Tous les résultats sont enregistrés progressivement dans le fichier All_results.csv.

Vérifier que le script common_ml.py fonctionne correctement : python common_ml.py

Exécution des modèles
Exécuter successivement les scripts suivants afin d'obtenir les résultats de chaque modèle :

baseline_logistic.py
train_xgboost.py
train_catboost.py
train_rf.py
train_adaboost.py
train_LightGBM.py
train_isolation_forest.py
train_ocsvm.py
train_lof.py
train_voting.py
train_stacking.py
train_stacking2.py
train_mlp_fraud.py
train_mlp_fraud_rev.py
train_mlp_smote.py
train_autoencoder.py
train_autoencoder_V2.py
train_transformer.py

À l'issue de ces exécutions, l'ensemble des résultats est centralisé dans le fichier All_results.csv.

Analyse et diagnostic des résultats
Les scripts suivants permettent d'effectuer les analyses et diagnostics complémentaires :

Analysis_shap.py
Analysis_lime.py
Analysis_feature_importance.py
Analysis_metrics.py
Analysis_ROC.py
Analysis_fairness.py
Analysis_sensitivity.py
Analysis_conclusion.py