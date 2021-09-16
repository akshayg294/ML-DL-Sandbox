from raven import Client
# from flask import request
from app.constant import SERVER, LOG_NAME
from google.cloud import tasks_v2beta3
from google.cloud import bigquery
from google.cloud import logging as cloud_logging

sentry_client = Client("https://03372b6bec774e279808c4c281864593@o191981.ingest.sentry.io/5896154")
queue_client = tasks_v2beta3.CloudTasksClient.from_service_account_json('queue_auth.json')
bigquery_client = bigquery.Client.from_service_account_json('bigqueryauth.json')
logging_client = cloud_logging.Client.from_service_account_json('loggingauth.json')
logger = logging_client.logger(LOG_NAME)


def log_error(traceback):
    if SERVER == 'development':
        print(traceback)
        return

    sentry_client.extra = {
        'app': 'log_processor',
        'traceback' : str(traceback)
    }
    sentry_client.environment = SERVER
    sentry_client.captureException()
