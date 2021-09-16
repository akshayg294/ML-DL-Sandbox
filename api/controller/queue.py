import traceback
from flask import request
from api import log_error
from api.service import add_to_queue, get_property
from api.middleware import static_auth


@static_auth()
def push_to_task_queue():
    try:
        data = request.json
        time = data['time'] if 'time' in data else 0

        property_id = data['property_id'] if 'property_id' in data else None
        property_identifier = None
        if property_id:
            property = get_property(property_id)
            property_identifier = str(property.identifier).replace('#', '') if property else None

        add_to_queue(data, time, property_identifier)

        return 'task pushed to queue', 200
    except Exception as e:
        log_error(traceback.print_exc())
        return str(e), 500
