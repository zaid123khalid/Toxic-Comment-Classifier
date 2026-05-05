from extensions import db
from datetime import datetime
from flask_login import UserMixin


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), default="user", nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    predictions = db.relationship("PredictionLog", backref="user", lazy=True)


class PredictionLog(db.Model):
    __tablename__ = "prediction_logs"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    comment_text = db.Column(db.Text, nullable=False)
    predicted_label = db.Column(db.String(50), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    category_scores = db.relationship("CategoryScore", backref="prediction", lazy=True)


class FlaggedPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    comment_text = db.Column(db.Text, nullable=False)
    original_prediction = db.Column(db.String(20), nullable=False)
    original_confidence = db.Column(db.Float, nullable=False)
    corrected_label = db.Column(db.String(20), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    status = db.Column(db.String(20), default="pending")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    reviewed_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    reviewed_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship(
        "User", foreign_keys=[user_id], backref="flagged_predictions"
    )
    reviewer = db.relationship(
        "User", foreign_keys=[reviewed_by], backref="reviewed_flags"
    )


class CategoryScore(db.Model):
    __tablename__ = "category_scores"

    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(
        db.Integer, db.ForeignKey("prediction_logs.id"), nullable=False
    )

    category_name = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Float, nullable=False)


class ModelMetadata(db.Model):
    __tablename__ = "model_metadata"

    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(50), nullable=False)
    algorithm = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    precision_score = db.Column(db.Float, nullable=True)
    recall_score = db.Column(db.Float, nullable=True)
    training_date = db.Column(db.Date)
    dataset_name = db.Column(db.String(150))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
