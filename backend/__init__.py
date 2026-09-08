"""
Flask application factory.
"""

import os
from flask import Flask
from backend.database import db

def create_app(test_config=None):
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates'),
        static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
    )

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_dir = os.path.join(project_root, 'instance')
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, 'churn.db')

    upload_dir = os.path.join(project_root, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'customer-churn-secret-key-2026-prod'),
        SQLALCHEMY_DATABASE_URI=f'sqlite:///{db_path}',
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER=upload_dir,
        MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16 MB max upload
    )

    if test_config:
        app.config.update(test_config)

    # Initialize extensions
    db.init_app(app)

    with app.app_context():
        db.create_all()

    # Register routes
    from backend.routes import main_bp
    app.register_blueprint(main_bp)

    return app
