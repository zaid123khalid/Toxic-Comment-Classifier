from flask import Flask
from config import Config
from extensions import db
from ml_model.predict import warmup_pipeline
import models
from flask_login import LoginManager


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    login_manager = LoginManager()
    login_manager.login_view = "main.login"
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return models.User.query.get(int(user_id))

    db.init_app(app)

    with app.app_context():
        db.create_all()
        warmup_pipeline()

    from routes import main

    app.register_blueprint(main)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
