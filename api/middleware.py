import jwt
from flask import request
from app.constant import SECRET, AUTHORIZATION_TOKEN, SURYALOGIX_TOKEN, RAYCON_TOKEN
import traceback
from api import log_error
def jwt_auth():
    def authenticator(func):
        def wrap(*args, **kwargs):
            try:
                payload = request.get_data(as_text=True) or ''
                data = jwt.decode(payload, SECRET, algorithms=['HS256'])
                request.payload = data
                return func(*args, **kwargs)
            except Exception as e:
                print("authorization failed")
                print(e)
                log_error(traceback.print_exc())
                return "Authorization Failed", 200
        wrap.__name__ = func.__name__
        return wrap
    return authenticator


def static_auth(*logger_type):
    def authenticator(func):
        def wrap(*args, **kwargs):
            if 'AUTHORIZATION' not in request.headers:
                return 'auth token not found', 401

            auth_token = request.headers['AUTHORIZATION']
            auth_token = auth_token.replace('Bearer ', '')

            token = ""
            if not logger_type:
                token = AUTHORIZATION_TOKEN
            elif logger_type and logger_type[0] == 'suryalogix':
                token = SURYALOGIX_TOKEN
            elif logger_type and logger_type[0] == 'raycon':
                token = RAYCON_TOKEN

            if auth_token == token:
                return func(*args, **kwargs)
            else:
                return 'invalid auth token', 401
        wrap.__name__ = func.__name__
        return wrap
    return authenticator
