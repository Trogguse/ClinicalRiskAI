# app package
from flask import Flask
from flask_cors import CORS
from config import Config


def create_app():
    """Flask application factory."""
    app = Flask(
        __name__,
        static_folder='../static',
        static_url_path='/static',
    )
    app.config.from_object(Config)

    # Enable CORS for all routes
    CORS(app)

    # Register blueprint
    from app.routes import api
    app.register_blueprint(api)

    return app
