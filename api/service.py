import logging
import jwt
import datetime
from api import queue_client
from google.protobuf import timestamp_pb2
from app.constant import PROJECT_ID, QUEUE_NAME, REGION, SECRET, SERVICE
from api.models import LeadProperty, LeadPropertyEquipment, LeadPropertyTuya, LeadPropertyActive
from api.processor.raycon import RayconProcessor

logging.getLogger().setLevel(logging.INFO)

def add_to_queue(payload, seconds=0, property_identifier=None):
    project = PROJECT_ID
    queue = QUEUE_NAME
    location = REGION
    converted_payload = jwt.encode(payload, SECRET, algorithm='HS256')

    parent = queue_client.queue_path(project, location, queue)
    task = {
        'app_engine_http_request': {  # Specify the type of request.
            'http_method': 'POST',
            'relative_uri': '/pull/log',
            'app_engine_routing': {
                'service': SERVICE
            }
        }
    }

    if property_identifier:
        task['name'] = 'projects/{0}/locations/{1}/queues/{2}/tasks/{3}'.format(project, location, queue, property_identifier+"_log_"+str(int(datetime.datetime.utcnow().timestamp())))

    task['app_engine_http_request']['body'] = converted_payload
    logging.info(task)
    d = datetime.datetime.utcnow() + datetime.timedelta(seconds=seconds)
    timestamp = timestamp_pb2.Timestamp()
    timestamp.FromDatetime(d)
    task['schedule_time'] = timestamp

    # Use the client to build and send the task.
    response = queue_client.create_task(parent, task)

    logging.info('Created task {}'.format(response.name))


def stop_queue(property_id, tuya=False):
    if tuya:
        property_logger_details = LeadPropertyTuya.query.filter_by(lead_property_id=property_id).first()
    else:
        property_logger_details = LeadPropertyActive.query.filter_by(lead_property_id=property_id).first()

    if property_logger_details:
        property_logger_details.is_logging = 0
        property_logger_details.save()


def get_property(property_id):
    lead_property = LeadProperty.query.get(property_id)
    if not lead_property:
        return None
    else:
        return lead_property


def get_disaggregation_processor(property_id):
    property = LeadProperty.query.get(property_id)

    property_equipment_details = LeadPropertyEquipment.query.filter_by(lead_property_id=property_id, supplier_type='raycon').all()
    if not property_equipment_details:
        logging.info('No logger details found for the property {0}'.format(str(property_id)))
        return None

    logger_processor = None
    logger_processor = RayconProcessor(property, property_equipment_details)

    return logger_processor
