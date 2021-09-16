from api.timer import timed
import decimal
import hashlib
from flask import request
from datetime import datetime, timedelta
import calendar
from app.constant import DEFAULT_TIMEZONE
import pandas as pd
import numpy as np
from api.models import LeadProperty, LoggerDB, db
from api import bigquery_client
from app.constant import DUMP_DATASET, DASHBOARD_TABLE_ID, DASHBOARD_DATASET, PROJECT_ID
import logging

logger = logging.getLogger(__name__)
logger.setLevel('INFO')


def encrypt_string(data):
    sha_signature = hashlib.sha256(data.encode()).hexdigest()
    return sha_signature


def generate_request_headers():
    headers = {}
    headers['Accept-Encoding'] = None
    headers['User-Agent'] = None
    headers['Content-Type'] = 'application/json'
    headers['HostApi-Method'] = request.method
    return headers


def convert_timestamp_to_utc_datetime(timestamp):
    return datetime.utcfromtimestamp(timestamp) + timedelta(seconds=DEFAULT_TIMEZONE)


def decimal_default(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    raise TypeError


def convert_utc_datetime_to_timestamp(utc_datetime):
    if not utc_datetime:
        return 0
    return calendar.timegm(utc_datetime.timetuple())

@timed
def get_data_for_disaggregation(start_time, end_time, dataset, table_id, property_id, equipment):
    # start_time = start_time - timedelta(hours=5.5)
    # end_time = end_time - timedelta(hours=5.5)
    DATASET = PROJECT_ID + '.' + dataset + "." + table_id
    dashboard_query = """
                        SELECT
                            *
                        FROM
                            `{0}`
                        WHERE logged_on_local <= '{2}' and logged_on_local >= '{1}'
                        AND
                            property_id = {3}
                        AND
                            device_id = '{4}'
                        ORDER BY
                            logged_on_local
                        """.format(DATASET, start_time, end_time, property_id, str(equipment.device_id))

    logger.info('get_data_for_disaggregation      dashboard_query: \n %s', dashboard_query)
    
    query_job = bigquery_client.query(dashboard_query)
    results = query_job.result().to_dataframe()
    results = results.drop_duplicates(subset=['logged_on_local'])
    return results
