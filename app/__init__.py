import connexion
import logging
from api.models import db
from app.config import app_config
from app.constant import SERVER, DB_CONNECTION_URL
from api.processor.ml_model import model
current_model = model

# Only displaying critical logs from connexion to reduce noise.
logging.getLogger('connexion.decorators.parameter').setLevel(logging.CRITICAL)


def create_app():
    app = connexion.FlaskApp(__name__, instance_relative_config=True, swagger_ui=False)
    app.app.config.from_object(app_config[SERVER])

    app.app.config['SQLALCHEMY_DATABASE_URI'] = DB_CONNECTION_URL
    app.app.config['JSON_SORT_KEYS'] = False
    app.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.app.config['SQLALCHEMY_POOL_RECYCLE'] = 300
    app.app.config['SQLALCHEMY_POOL_SIZE'] = 5
    app.app.config['SQLALCHEMY_MAX_OVERFLOW'] = 20

    app.add_api('route.yaml')

    # if in development initalize the debugger
    if SERVER == 'development':
        app.debug = True

    db.init_app(app.app)

    return app
