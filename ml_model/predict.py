import os
import nltk
from ml_model.data_utils import clean_text
from ml_model.vectorizer_utils import load_vectorizer
from ml_model.train_model import load_model, predict, predict_proba
from typing import Tuple, Any, Optional, Dict

_model_cache: Optional[Any] = None
_vectorizer_cache: Optional[Any] = None
_pipeline_loaded: bool = False


def _get_artifact_paths() -> Tuple[str, str]:
    """Get paths to model and vectorizer artifacts."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(script_dir, "artifacts")

    model_path = os.path.join(artifacts_dir, "model_toxic.joblib")
    vectorizer_path = os.path.join(artifacts_dir, "vectorizer_toxic.joblib")

    return model_path, vectorizer_path


def _load_pipeline() -> Tuple[Any, Any]:
    """
    Load model and vectorizer (called once at startup or first use).
    Returns cached instances on subsequent calls.
    """
    global _model_cache, _vectorizer_cache, _pipeline_loaded

    if _pipeline_loaded and _model_cache is not None and _vectorizer_cache is not None:
        return _model_cache, _vectorizer_cache

    print("Loading prediction pipeline (first time)...")

    model_path, vectorizer_path = _get_artifact_paths()

    _vectorizer_cache = load_vectorizer(vectorizer_path)
    _model_cache = load_model(model_path)

    if _model_cache is None or _vectorizer_cache is None:
        raise RuntimeError("Failed to load prediction pipeline components")

    _pipeline_loaded = True
    print("Pipeline loaded and cached for reuse\n")

    return _model_cache, _vectorizer_cache


def predict_toxicity(text: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Predicts toxicity of a comment using cached model/vectorizer.

    Args:
        text: Input comment string
        threshold: Classification threshold (default: 0.5)

    Returns:
        Dictionary with prediction results
    """
    model, vectorizer = _load_pipeline()

    from ml_model.data_utils import clean_text

    text_clean = clean_text(text)

    text_vec = vectorizer.transform([text_clean])

    prediction = int(model.predict(text_vec)[0])
    probability = float(model.predict_proba(text_vec)[0][1])

    classified_toxic = probability >= threshold
    confidence = float(max(probability, 1 - probability))

    return {
        "input_text": text,
        "cleaned_text": text_clean,
        "is_toxic": bool(classified_toxic),
        "prediction": prediction,
        "toxic_probability": probability,
        "confidence": confidence,
        "threshold_used": threshold,
    }


def predict_comment(text: str) -> Dict[str, Any]:
    """
    Wrapper function for Flask routes.
    Returns prediction result or error dict.
    """
    try:
        result = predict_toxicity(text)
        print(
            f"🔮 Predicted: {result['is_toxic']} | "
            f"P(toxic)={result['toxic_probability']:.3f} | "
            f"Confidence={result['confidence']:.3f}"
        )
        return result
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"error": str(e), "is_toxic": None}


def warmup_pipeline():
    """
    Pre-load the pipeline at app startup (call this in app factory).
    Prevents first-request latency.
    """
    try:
        _load_pipeline()
        print("Prediction pipeline warmed up successfully")
        return True
    except Exception as e:
        print(f"Pipeline warmup failed: {e}")
        return False


def main():
    """Test predictions (for standalone script usage)."""
    nltk.download("stopwords", quiet=True)

    _load_pipeline()

    test_comments = [
        "This is a great article, thanks for sharing!",
        "You're an idiot and should be banned.",
        "I disagree with your point but respect your opinion.",
        "What a stupid waste of time, you moron!",
    ]

    print("🔮 Running toxicity predictions:\n")
    for comment in test_comments:
        result = predict_toxicity(comment)
        status = "TOXIC" if result["is_toxic"] else "✅ Safe"
        print(f"{result['input_text'][:50]}...")
        print(f"{status} | P(toxic) = {result['toxic_probability']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()
