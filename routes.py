from datetime import datetime
from sys import flags

from flask import Blueprint, render_template, request, url_for, redirect, flash, abort
from flask_login import login_user, logout_user, current_user, login_required
from extensions import db
from models import FlaggedPrediction, PredictionLog, User, CategoryScore, ModelMetadata
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
from sqlalchemy import func

from ml_model.predict import predict_comment
from ml_model.train import retrain_with_corrections

main = Blueprint("main", __name__)


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "admin":
            abort(403)
        return f(*args, **kwargs)

    return decorated


@main.route("/")
def home():
    return render_template("home.html")


@main.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            if user.role == "admin":
                return redirect(url_for("main.dashboard"))
            return redirect(url_for("main.analyze"))
        flash("Invalid email or password", "danger")
    return render_template("login.html")


@main.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        if User.query.filter_by(email=email).first():
            flash("Email already registered", "danger")
            return redirect(url_for("main.signup"))
        user = User(
            username=request.form["username"],
            email=email,
            password_hash=generate_password_hash(request.form["password"]),
            role="user",
        )
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("main.login"))
    return render_template("signup.html")


@main.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("main.home"))


@main.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    result = None

    if request.method == "POST":
        text = request.form.get("comment")

        prediction_result = predict_comment(text)

        label = "toxic" if prediction_result.get("is_toxic") else "non-toxic"
        confidence = prediction_result.get("confidence", 0.0)

        new_prediction = PredictionLog(
            comment_text=text,
            predicted_label=label,
            confidence_score=confidence,
            user_id=current_user.id,
        )

        db.session.add(new_prediction)
        db.session.commit()

        result = {
            "label": label,
            "confidence": confidence,
            "probability": prediction_result.get("toxic_probability", 0.0),
        }

    return render_template("analyze.html", result=result)


@main.route("/flag-prediction", methods=["POST"])
@login_required
def flag_prediction():
    comment = request.form.get("comment")
    prediction = request.form.get("prediction")
    confidence = request.form.get("confidence")

    try:
        confidence_val = float(confidence) if confidence else 0.0
    except (ValueError, TypeError):
        confidence_val = 0.0

    flagged_prediction = FlaggedPrediction(
        comment_text=comment,
        original_prediction=prediction,
        original_confidence=confidence_val,
        user_id=current_user.id,
    )

    db.session.add(flagged_prediction)
    db.session.commit()

    flash("Prediction flagged for review.", "success")
    return redirect(url_for("main.analyze"))


@main.route("/history")
@login_required
def history():
    predictions = (
        PredictionLog.query.filter_by(user_id=current_user.id)
        .order_by(PredictionLog.created_at.desc())
        .all()
    )
    return render_template("history.html", predictions=predictions)


@main.route("/stats")
@login_required
def stats():
    total = PredictionLog.query.filter_by(user_id=current_user.id).count()
    toxic_count = PredictionLog.query.filter_by(
        user_id=current_user.id, predicted_label="toxic"
    ).count()
    non_toxic_count = total - toxic_count

    return render_template(
        "stats.html", total=total, toxic=toxic_count, non_toxic=non_toxic_count
    )


# ADMIN ROUTES
@main.route("/admin/dashboard")
@login_required
@admin_required
def dashboard():
    total_users = User.query.count()
    total_preds = PredictionLog.query.count()
    toxic_count = PredictionLog.query.filter_by(predicted_label="toxic").count()
    non_toxic_count = total_preds - toxic_count
    toxic_rate = round((toxic_count / total_preds * 100), 1) if total_preds > 0 else 0

    recent = (
        PredictionLog.query.order_by(PredictionLog.created_at.desc()).limit(10).all()
    )

    return render_template(
        "admin/dashboard.html",
        total_users=total_users,
        total_preds=total_preds,
        toxic_count=toxic_count,
        non_toxic_count=non_toxic_count,
        toxic_rate=toxic_rate,
        recent=recent,
    )


@main.route("/admin/predictions")
@login_required
@admin_required
def predictions():
    all_predictions = PredictionLog.query.order_by(
        PredictionLog.created_at.desc()
    ).all()
    return render_template("admin/predictions.html", predictions=all_predictions)


@main.route("/admin/predictions/<int:id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_prediction(id):
    prediction = PredictionLog.query.get_or_404(id)

    CategoryScore.query.filter_by(prediction_id=prediction.id).delete()

    db.session.delete(prediction)
    db.session.commit()
    flash("Prediction deleted.", "success")
    return redirect(url_for("main.predictions"))


@main.route("/admin/flags")
@login_required
@admin_required
def view_flags():
    flags = FlaggedPrediction.query.order_by(FlaggedPrediction.created_at.desc()).all()

    return render_template("admin/flagged_predictions.html", flags=flags)


@main.route("/admin/correct-prediction", methods=["POST"])
@login_required
@admin_required
def correct_prediction():
    flag_id = request.form.get("flag_id")
    corrected_label = request.form.get("corrected_label")
    action = request.form.get("action")

    flagged = FlaggedPrediction.query.get(flag_id)
    if not flagged:
        flash("Flagged prediction not found.", "danger")
        return redirect(url_for("main.view_flags"))

    if action == "approve":
        flagged.corrected_label = corrected_label
        flagged.status = "approved"
        flagged.reviewed_by = current_user.id
        flagged.reviewed_at = datetime.utcnow()

        flash(
            f"✓ Corrected! Added '{flagged.comment_text[:30]}...' to training queue.",
            "success",
        )
    else:
        flagged.status = "rejected"
        flash("✗ Flag rejected.", "info")

    db.session.commit()
    return redirect(url_for("main.view_flags"))


@main.route("/admin/retrain", methods=["POST"])
@login_required
@admin_required
def retrain_model():
    try:
        corrections = FlaggedPrediction.query.filter(
            FlaggedPrediction.status.in_(["approved", "retrained"])
        ).all()

        if not corrections:
            flash("⚠️ No approved corrections to retrain with.", "warning")
            return redirect(url_for("main.view_flags"))

        flash(
            f"🔄 Retraining started with {len(corrections)} corrections... This may take 1-2 minutes.",
            "info",
        )

        success = retrain_with_corrections(corrections)

        if success:
            for c in corrections:
                c.status = "retrained"
            db.session.commit()

            flash(f"✅ Model retrained successfully! Version incremented.", "success")
        else:
            flash("❌ Retraining failed.", "danger")

    except Exception as e:
        db.session.rollback()
        flash(f"❌ Retraining error: {str(e)}", "danger")

    return redirect(url_for("main.view_flags"))


@main.route("/admin/stats")
@login_required
@admin_required
def system_stats():
    total_preds = PredictionLog.query.count()
    toxic_count = PredictionLog.query.filter_by(predicted_label="toxic").count()
    non_toxic_count = total_preds - toxic_count
    toxic_rate = round((toxic_count / total_preds * 100), 1) if total_preds > 0 else 0

    total_users = User.query.count()
    model_info = ModelMetadata.query.order_by(ModelMetadata.id.desc()).first()

    top_users = (
        db.session.query(
            User.username, func.count(PredictionLog.id).label("pred_count")
        )
        .join(PredictionLog)
        .group_by(User.id)
        .order_by(func.count(PredictionLog.id).desc())
        .limit(6)
        .all()
    )

    return render_template(
        "admin/system_stats.html",
        total_preds=total_preds,
        toxic_count=toxic_count,
        non_toxic_count=non_toxic_count,
        toxic_rate=toxic_rate,
        total_users=total_users,
        model_info=model_info,
        top_users=top_users,
    )


@main.route("/admin/model-history")
@login_required
@admin_required
def model_history():
    models = ModelMetadata.query.order_by(ModelMetadata.id.desc()).all()
    return render_template("admin/model_history.html", models=models)


@main.route("/admin/users")
@login_required
@admin_required
def users():
    all_users = User.query.order_by(User.created_at.desc()).all()
    return render_template("admin/users.html", users=all_users)


@main.route("/admin/users/<int:id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_user(id):
    if id == current_user.id:
        flash("You cannot delete your own account.", "warning")
        return redirect(url_for("admin.users"))

    user = User.query.get_or_404(id)

    for pred in user.predictions:
        CategoryScore.query.filter_by(prediction_id=pred.id).delete()
        db.session.delete(pred)

    db.session.delete(user)
    db.session.commit()
    flash(f"User {user.username} deleted.", "success")
    return redirect(url_for("main.users"))
