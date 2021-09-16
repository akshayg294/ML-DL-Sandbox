import logging
import traceback
from flask import request
from api import log_error
from api.service import add_to_queue
from api.middleware import jwt_auth
from app.constant import DEFAULT_QUEUE_WAITING_TIME
from api.service import get_disaggregation_processor, get_property, stop_queue

logging.getLogger().setLevel(logging.INFO)

@jwt_auth()
def pull_log_data():
    try:

        property_id = request.payload['property_id']

        property = get_property(property_id)
        logging.info(property)
        logging.info(property.identifier)

        disaggregation_processor = get_disaggregation_processor(property_id)
        logging.info('processor        :')
        logging.info(disaggregation_processor)
        if not disaggregation_processor:
            # stop_queue(property_id)
            logging.info('No disaggregation processor unit found for the property {0}'.format(str(property_id)))
            return 'The property has no disaggregation processor unit configured and hence stopping the cron', 200

        disaggregation_processor.process()
        if disaggregation_processor.should_log():
            property_identifier = str(property.identifier).replace('#', '') if property else None
            add_to_queue(request.payload, DEFAULT_QUEUE_WAITING_TIME, property_identifier)

        return 'data pulled successfully', 200
    except Exception as e:
        print(str(e))
        log_error(traceback.print_exc())
        add_to_queue(request.payload, DEFAULT_QUEUE_WAITING_TIME)
        return 'error occurred', 200
