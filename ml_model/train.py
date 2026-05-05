from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import joblib
from datetime import date
import nltk
from models import ModelMetadata

# Custom modules
from ml_model.data_utils import (
    load_dataset,
    inspect_dataset,
    get_features_and_labels,
    preprocess_dataframe,
    split_dataset,
)
from ml_model.vectorizer_utils import (
    create_vectorizer,
    fit_vectorizer,
    transform_text,
    fit_transform_text,
    save_vectorizer,
    load_vectorizer,
)
from ml_model.train_model import (
    build_model,
    predict_proba,
    train_model,
    predict,
    evaluate_model,
    print_classification_report,
    plot_confusion_matrix,
    save_model,
    load_model,
)
from ml_model.save_model_metadeta import save_model_metadata


def main():
    nltk.download("stopwords")

    # Configuration
    TARGET_LABEL = "toxic"
    MAX_FEATURES = 10000
    NGRAM_RANGE = (1, 2)
    MODEL_TYPE = "logistic"
    MODEL_VERSION = "1.0.0"
    DATASET_NAME = "toxic_comments_train.csv"

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "data", "train.csv")
    artifacts_dir = os.path.join(script_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, f"model_{TARGET_LABEL}.joblib")
    vectorizer_path = os.path.join(artifacts_dir, f"vectorizer_{TARGET_LABEL}.joblib")

    # 1 Load dataset
    print("Loading dataset...")
    df = load_dataset(dataset_path)
    if df is None:
        return

    inspect_dataset(df)

    # 2 Preprocess text
    print("Preprocessing text...")
    df = preprocess_dataframe(df)

    # 3 Extract features and labels
    print("Extracting features and labels...")
    X, y = get_features_and_labels(df, label_column=TARGET_LABEL)

    # 4 Vectorize text
    print("Vectorizing text...")
    vectorizer = create_vectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    X_vectorized = fit_transform_text(vectorizer, X)
    print(f"Vectorization complete. Shape: {X_vectorized.shape}\n")

    # 5 Split data (AFTER vectorization to avoid data leakage)
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(X_vectorized, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    # 6 Build and train model
    print("Building and training model...")
    model = build_model(model_type=MODEL_TYPE)
    model = train_model(model, X_train, y_train)
    print("Training complete\n")

    # 7 Evaluate
    print("Evaluating model...")
    y_pred = predict(model, X_test)
    y_proba = predict_proba(model, X_test)

    metrics = evaluate_model(y_test, y_pred, y_proba)

    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    if metrics.get("roc_auc"):
        print(f"📊 ROC-AUC:   {metrics['roc_auc']:.4f}")

    print_classification_report(y_test, y_pred, label_name=TARGET_LABEL)

    # 8 Plot visualizations (optional)
    # plot_confusion_matrix(y_test, y_pred, label_name=TARGET_LABEL,
    #                      save_path=os.path.join(artifacts_dir, f"cm_{TARGET_LABEL}.png"))
    # if y_proba is not None:
    #     plot_roc_curve(y_test, y_proba, label_name=TARGET_LABEL,
    #                   save_path=os.path.join(artifacts_dir, f"roc_{TARGET_LABEL}.png"))

    # 9 Save artifacts
    print("\nSaving model and vectorizer...")
    save_artifacts(model, vectorizer, model_path, vectorizer_path)

    # 10 Save metadata to database
    print("\nSaving model metadata to database...")
    algorithm_name = (
        "Logistic Regression" if MODEL_TYPE == "logistic" else "Random Forest"
    )

    save_model_metadata(
        version=MODEL_VERSION,
        algorithm=algorithm_name,
        metrics=metrics,
        dataset_name=DATASET_NAME,
        training_date=date.today(),
    )


def retrain_with_corrections(corrections):
    """
    Retrain model using approved corrections from database.
    Combines original training data + corrected predictions.
    Auto-increments model version from database.
    """

    nltk.download("stopwords")

    # Configuration
    TARGET_LABEL = "toxic"
    MAX_FEATURES = 10000
    NGRAM_RANGE = (1, 2)
    MODEL_TYPE = "logistic"
    DATASET_NAME = "toxic_comments_train.csv + DB corrections"

    # 1. Get Current Model Version from Database & Increment
    print("Fetching current model version from database...")
    latest_meta = ModelMetadata.query.order_by(ModelMetadata.created_at.desc()).first()

    if latest_meta and latest_meta.version:
        version_parts = latest_meta.version.split(".")
        if len(version_parts) == 3:
            major, minor, patch = map(int, version_parts)
            patch += 1
            MODEL_VERSION = f"{major}.{minor}.{patch}"
        else:
            MODEL_VERSION = "1.0.1"
    else:
        MODEL_VERSION = "1.0.0"

    print(f"New model version: {MODEL_VERSION}\n")

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "data", "train.csv")
    artifacts_dir = os.path.join(script_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, f"model_{TARGET_LABEL}.joblib")
    vectorizer_path = os.path.join(artifacts_dir, f"vectorizer_{TARGET_LABEL}.joblib")

    # 2. Load Original Dataset
    print("Loading original dataset...")
    df_original = load_dataset(dataset_path)
    if df_original is None:
        print("Failed to load original dataset. Aborting retrain.")
        return False

    print(f"Original dataset size: {len(df_original)} rows\n")

    # 3. Add Corrections from Database
    print(f"Incorporating {len(corrections)} approved corrections from database...")

    # Map corrected_label to 1/0
    labels = [1 if c.corrected_label == "toxic" else 0 for c in corrections]
    texts = [c.comment_text for c in corrections]

    df_corrections = pd.DataFrame(
        {
            "comment_text": texts,
            "toxic": labels,
            "severe_toxic": 0,
            "obscene": 0,
            "threat": 0,
            "insult": 0,
            "identity_hate": 0,
        }
    )

    # Combine datasets
    df_combined = pd.concat([df_original, df_corrections], ignore_index=True)
    print(f"Combined dataset size: {len(df_combined)} rows\n")

    # 4. Preprocess Text
    print("Preprocessing text...")
    df_combined = preprocess_dataframe(df_combined)

    # 5. Extract Features and Labels
    print("Extracting features and labels...")
    X, y = get_features_and_labels(df_combined, label_column=TARGET_LABEL)

    # 6. Vectorize Text
    print("Vectorizing text...")
    vectorizer = create_vectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    X_vectorized = fit_transform_text(vectorizer, X)
    print(f"Vectorization complete. Shape: {X_vectorized.shape}\n")

    # 7. Split Data (AFTER Vectorization)
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = split_dataset(X_vectorized, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    # 8. Build and Train Model
    print("Building and training model...")
    model = build_model(model_type=MODEL_TYPE)
    model = train_model(model, X_train, y_train)
    print("Training complete\n")

    # 9. Evaluate
    print("Evaluating model...")
    y_pred = predict(model, X_test)
    y_proba = predict_proba(model, X_test)

    metrics = evaluate_model(y_test, y_pred, y_proba)

    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    if metrics.get("roc_auc"):
        print(f"📊 ROC-AUC:   {metrics['roc_auc']:.4f}")

    print_classification_report(y_test, y_pred, label_name=TARGET_LABEL)

    # 10. Save Artifacts
    print("\nSaving model and vectorizer...")
    save_artifacts(model, vectorizer, model_path, vectorizer_path)

    # 11. Save Metadata to Database (with NEW version)
    print("\nSaving model metadata to database...")
    algorithm_name = (
        "Logistic Regression" if MODEL_TYPE == "logistic" else "Random Forest"
    )

    save_model_metadata(
        version=MODEL_VERSION,
        algorithm=algorithm_name,
        metrics=metrics,
        dataset_name=DATASET_NAME,
        training_date=date.today(),
    )

    print(f"\n✅ Model v{MODEL_VERSION} trained and saved successfully!")
    return True


def save_artifacts(model, vectorizer, model_path, vectorizer_path):
    save_model(model, model_path)
    save_vectorizer(vectorizer, vectorizer_path)


if __name__ == "__main__":
    main()
