import os
from datetime import datetime, date
from flask import current_app
from extensions import db
from models import ModelMetadata


def save_model_metadata(
    version: str,
    algorithm: str,
    metrics: dict,
    dataset_name: str = "toxic_comments_train.csv",
    training_date: date = None,
):
    """
    Saves model metadata to database.
    Must be called within Flask app context.
    """
    try:
        _ = current_app.config
        in_app_context = True
    except RuntimeError:
        in_app_context = False

    if not in_app_context:
        from app import create_app

        app = create_app()
        ctx = app.app_context()
        ctx.push()
        should_pop = True
    else:
        ctx = None
        should_pop = False

    try:
        existing = ModelMetadata.query.filter_by(version=version).first()
        if existing:
            print(f"Model version '{version}' already exists in database")
            model_meta = existing
        else:
            model_meta = ModelMetadata()

        model_meta.version = version
        model_meta.algorithm = algorithm
        model_meta.accuracy = metrics.get("accuracy", 0.0)
        model_meta.precision_score = metrics.get("precision", 0.0)
        model_meta.recall_score = metrics.get("recall", 0.0)
        model_meta.training_date = training_date or date.today()
        model_meta.dataset_name = dataset_name

        db.session.add(model_meta)
        db.session.commit()

        print(f"Model metadata saved successfully!")
        print(f"Version: {model_meta.version}")
        print(f"Accuracy: {model_meta.accuracy:.4f}")

        return model_meta

    finally:
        if should_pop:
            ctx.pop()


def get_latest_model_metadata():
    """Retrieves the most recent model metadata."""
    try:
        _ = current_app.config
        in_app_context = True
    except RuntimeError:
        in_app_context = False

    if not in_app_context:
        from app import create_app

        app = create_app()
        ctx = app.app_context()
        ctx.push()
        should_pop = True
    else:
        ctx = None
        should_pop = False

    try:
        latest = ModelMetadata.query.order_by(
            ModelMetadata.training_date.desc()
        ).first()
        return latest
    finally:
        if should_pop:
            ctx.pop()
