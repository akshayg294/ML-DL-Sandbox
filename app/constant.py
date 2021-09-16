import os

SERVER = os.environ['SERVER'] if 'SERVER' in os.environ else 'development'

AUTHORIZATION_TOKEN = '4mvuxHk5TjzZL7kg8JMwWn2d89j8HQ6ebuxtBpk44xLNu74rpVvqE6whUZemWUqY'
SURYALOGIX_TOKEN = '97ab0c6e05f244a293bf6fb0dc'
RAYCON_TOKEN = 'MXTdcGYj3gytvHImXz2vi8s6mFjAn8Q9OGXjlokpAcG8YXVwVppk1TJA'
SECRET = 'UqcNU3m87QP4rxwM'
PROJECT_ID = 'solar-222307'
REGION = 'asia-south1'
DEFAULT_TIMEZONE = 19800  # in seconds
DEFAULT_QUEUE_WAITING_TIME = 900  # in seconds
TUYA_DEFAULT_QUEUE_WAITING_TIME = 300 # in secondss

if SERVER == 'development':
    DB_USER_NAME = 'app-staging'
    DB_PASSWORD = '8yQsM2mP9R7fMygK'
    DB_HOST = '127.0.0.1'
    DB_NAME = 'dev_residential_solar'
    DB_CONNECTION_URL = 'mysql+pymysql://{0}:{1}@{2}/{3}'.format(DB_USER_NAME, DB_PASSWORD, DB_HOST, DB_NAME)
else:
    DB_USER_NAME = os.environ['DB_USER_NAME']
    DB_PASSWORD = os.environ['DB_PASSWORD']
    DB_HOST = os.environ['DB_HOST']
    DB_NAME = os.environ['DB_NAME']
    DB_CONNECTION_URL = 'mysql+pymysql://{0}:{1}@/{2}?unix_socket=/cloudsql/{3}'.format(DB_USER_NAME, DB_PASSWORD,
                                                                                        DB_NAME, DB_HOST)

if SERVER == 'production':
    QUEUE_NAME = 'mqtt-logger-stream'
    SERVICE = 'mqtt-logger-processer'
    DUMP_DATASET = 'loggers'
    DASHBOARD_DATASET = 'dashboard'
    DASHBOARD_TABLE_ID = 'production'
    LOG_NAME = 'mqttt-logger-prod-api'
    REPORT_DATASET = 'reports'
else:
    QUEUE_NAME = 'dev-mqtt-logger-stream'
    SERVICE = 'dev-mqtt-logger-processer'
    DUMP_DATASET = 'dev_loggers'
    DASHBOARD_DATASET = 'dashboard'
    DASHBOARD_TABLE_ID = 'staging'
    LOG_NAME = 'mqttt-logger-staging-api'
    REPORT_DATASET = 'dev_reports'


appId = 202005158754002
appSecret = '72ca42292902ab6f2d446628b88ed847'
orgId = 4463
MEAN_STEP = 2
FACTOR = 10


SURYALOGIX_BASE_CHANNEL_CONFIG = {
    'INV': {
        'vdc': 1,
        'idc': 2,
        'pdc': 3,
        'vac': 4,
        'iac': 5,
        'pac': 6,
        'p_app': 7,
        'p_reac': 9,
        'pf': 10,
        'etotal': 11,
        'day_gen': 12,
        'temp_int': 13,
        'temp_sink': 14,
        'r_active_power': 15,
        'y_active_power': 16,
        'b_active_power': 17,
    },
    'MFM': {
        'l_l_voltage': 2,
        'l_n_voltage': 3,
        'i_avg_line': 4,
        'avg_pf': 5,
        'pac_total': 6,
        'p_total_app': 7,
        'p_total_reac': 8,
        'e_total': 9,
        'e_import': 10,
        'e_export': 11,
        'r_voltage': 12,
        'y_voltage': 13,
        'b_voltage': 14,
        'r_current': 15,
        'y_current': 16,
        'b_current': 17,
        'r_pf': 18,
        'y_pf': 19,
        'b_pf': 20,
        'r_active_power': 21,
        'y_active_power': 22,
        'b_active_power': 23,
    },
    'GOODWE': {
        'l_l_voltage': 2,
        'l_n_voltage': 3,
        'i_avg_line': 4,
        'avg_pf': 5,
        'pac_total': 6,
        'p_total_app': 7,
        'p_total_reac': 8,
        'e_total': 9,
        'e_import': 10,
        'e_export': 11,
        'r_voltage': 12,
        'y_voltage': 13,
        'b_voltage': 14,
        'r_current': 15,
        'y_current': 16,
        'b_current': 17,
        'r_pf': 18,
        'y_pf': 19,
        'b_pf': 20,
        'r_active_power': 21,
        'y_active_power': 22,
        'b_active_power': 23,
    },
}
