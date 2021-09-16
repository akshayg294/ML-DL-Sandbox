from app import create_app
from flask import request

app = create_app()
flask_app = app.app


@flask_app.before_request
def before_request():
    # This is the url which the GLB hit to verify if the service is live or not.
    if request.path == '/':
        return 'Heath Check Successfull', 200


if __name__ == '__main__':
    app.run(port=8080)
