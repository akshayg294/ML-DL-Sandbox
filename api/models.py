from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.mysql import TINYINT

db = SQLAlchemy()


class BaseModelMixin(object):
    @classmethod
    def save(cls):
        db.session.commit()
        return cls


class LeadProperty(BaseModelMixin, db.Model):
    __tablename__ = 'lead_property'
    id = db.Column(db.Integer, primary_key=True)
    identifier = db.Column(db.String(255), nullable=True)


class LeadPropertyActive(BaseModelMixin, db.Model):
    __tablename__ = 'lead_property_active'
    id = db.Column(db.Integer, primary_key=True)
    installed_on = db.Column(db.DateTime, nullable=True)
    lead_property_id = db.Column(db.Integer, db.ForeignKey('lead_property.id'))
    logger_type = db.Column(db.String(255), nullable=True)
    logger_identifier = db.Column(db.String(255), nullable=True)
    logger_flag = db.Column(TINYINT(), default=0)
    meter_flag = db.Column(TINYINT(), default=0)
    net_meter_flag = db.Column(TINYINT(), default=0)
    reset_factor = db.Column(db.Float, nullable=True)
    consumption_reset_factor = db.Column(db.Float, nullable=True)
    bi_directional_reset_factor = db.Column(db.Float, nullable=True)
    current_max_energy = db.Column(db.Float, nullable=True)
    bi_directional_current_max_ac_energy = db.Column(db.Float, nullable=True)
    consumption_current_max_ac_energy = db.Column(db.Float, nullable=True)
    reset_factored_on = db.Column(db.DateTime, nullable=True)
    bi_directional_reset_factored_on = db.Column(db.DateTime, nullable=True)
    consumption_reset_factored_on = db.Column(db.DateTime, nullable=True)
    solar_to_grid_max_energy = db.Column(db.Float, nullable=True)
    solar_to_home_max_energy = db.Column(db.Float, nullable=True)
    grid_to_home_max_energy = db.Column(db.Float, nullable=True)
    timezone = db.Column(db.Integer, default=None, nullable=True)
    last_logged_on = db.Column(db.DateTime, nullable=True)
    last_processed_on = db.Column(db.DateTime, nullable=True)
    is_logging = db.Column(TINYINT(), default=0)
    is_pulled = db.Column(TINYINT(), default=0)
    api_method = db.Column(db.String(255), nullable=True)
    api_url = db.Column(db.String(255), nullable=True)
    api_header = db.Column(db.Text, nullable=True)
    api_body = db.Column(db.Text, nullable=True)
    is_disabled = db.Column(TINYINT(), default=0)
    is_logging = db.Column(TINYINT(), default=0)

    class Meta:
        db_table = 'lead_property_active'


class LeadPropertyDevice(BaseModelMixin, db.Model):
    __tablename__ = 'lead_property_device'
    id = db.Column(db.Integer, primary_key=True)
    lead_property_id = db.Column(db.Integer, db.ForeignKey('lead_property.id'))
    device_type = db.Column(db.String(255), nullable=True)
    device_name = db.Column(db.String(255), nullable=True)
    device_identifier = db.Column(db.String(255), nullable=True)
    device_provider = db.Column(db.String(255), nullable=True)
    sum_factor = db.Column(db.Float, nullable=True)
    installed_on = db.Column(db.DateTime, nullable=True)
    removed_on = db.Column(db.DateTime, nullable=True)
    is_disabled = db.Column(TINYINT(), default=0)
    is_grid_side = db.Column(TINYINT(), default=0)
    is_home_side = db.Column(TINYINT(), default=0)


class LeadPropertyTuya(BaseModelMixin, db.Model):
    __tablename__ = 'lead_property_tuya'
    id = db.Column(db.Integer, primary_key=True)
    lead_property_id = db.Column(db.Integer, db.ForeignKey('lead_property.id'))
    user_id = db.Column(db.String(255), nullable=True)
    home_id = db.Column(db.String(255), nullable=True)
    is_logging = db.Column(TINYINT(), default=0)
    created_on = db.Column(db.DateTime, nullable=True)
    updated_on = db.Column(db.DateTime, nullable=True)
    is_disabled = db.Column(TINYINT(), default=0)


class GlobalEquipment(BaseModelMixin, db.Model):
    __table__name = 'global_equipment'
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(255), nullable=True)
    name = db.Column(db.String(255), nullable=True)
    type = db.Column(db.String(255), nullable=True)
    description = db.Column(db.Text, nullable=True)
    consumption_category = db.Column(db.String(255), nullable=True)
    mean_power_consumption = db.Column(db.Float, nullable=True, default=0)


class LeadPropertyRoom(BaseModelMixin, db.Model):
    __table__name = 'lead_property_room'
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(255), nullable=True)
    lead_property_id = db.Column(db.Integer, db.ForeignKey('lead_property.id'))
    name = db.Column(db.String(255), nullable=True)
    description = db.Column(db.Text, nullable=True)
    image_url = db.Column(db.String(255), nullable=True)


class LeadPropertyEquipment(BaseModelMixin, db.Model):
    __tablename__ = 'lead_property_equipment'
    id = db.Column(db.Integer, primary_key=True)
    lead_property_tuya_id = db.Column(db.Integer, db.ForeignKey('lead_property_tuya.id'))
    lead_property_id = db.Column(db.Integer, db.ForeignKey('lead_property.id'))
    global_equipment_id = db.Column(db.Integer, db.ForeignKey('global_equipment.id'))
    lead_property_room_id = db.Column(db.Integer, db.ForeignKey('lead_property_room.id'))
    device_id = db.Column(db.String(255), nullable=True)
    last_logged_on = db.Column(db.DateTime, nullable=True)
    timezone = db.Column(db.Integer, nullable=True)
    status = db.Column(db.String(45), nullable=True)
    name = db.Column(db.String(255), nullable=True)
    type = db.Column(db.String(255), nullable=True)
    description = db.Column(db.Text, nullable=True)
    consumption_category = db.Column(db.String(255), nullable=True)
    mean_power_consumption = db.Column(db.Float, nullable=True, default=0)
    phase = db.Column(db.String(255), nullable=True)
    installed_on = db.Column(db.DateTime, nullable=True)
    removed_on = db.Column(db.DateTime, nullable=True)
    created_on = db.Column(db.DateTime, nullable=True)
    updated_on = db.Column(db.DateTime, nullable=True)
    supplier_type = db.Column(db.String(255), nullable=True)
    is_disabled = db.Column(TINYINT(), default=0)
    avg_mains_power = db.Column(db.Float, nullable=True)
    active_equipments = db.Column(db.Text, nullable=True)
    historical_data = db.Column(db.Text, nullable=True)
    is_logging = db.Column(TINYINT(), default=0)


class LeadPropertyOnset(BaseModelMixin, db.Model):
    __table__name = 'lead_property_onset'
    id = db.Column(db.Integer, primary_key=True)
    lead_property_id = db.Column(db.Integer, db.ForeignKey('lead_property.id'))
    onset_url = db.Column(db.String(255), nullable=True)
    last_logged_on = db.Column(db.DateTime, nullable=True)
    installed_on = db.Column(db.DateTime, nullable=True)
    removed_on = db.Column(db.DateTime, nullable=True)
    created_on = db.Column(db.DateTime, nullable=True)
    updated_on = db.Column(db.DateTime, nullable=True)
    is_disabled = db.Column(TINYINT(), default=0)
    timezone = db.Column(db.Integer, nullable=True)
    is_logging = db.Column(TINYINT(), default=0)
    avg_mains_power = db.Column(db.Float, nullable=True)
    active_equipments = db.Column(db.Text, nullable=True)
    historical_data = db.Column(db.Text, nullable=True)


class LeadPropertyAqi(BaseModelMixin, db.Model):
    __table__name = 'lead_property_aqi'
    id = db.Column(db.Integer, primary_key=True)
    lead_property_id = db.Column(db.Integer, db.ForeignKey('lead_property.id'))
    device_id = db.Column(db.Integer, nullable=True)
    created_on = db.Column(db.DateTime, nullable=True)
    updated_on = db.Column(db.DateTime, nullable=True)
    is_disabled = db.Column(TINYINT(), default=0)


class LoggerDB(BaseModelMixin, db.Model):
    __tablename__ = 'logger_db'
    id = db.Column(db.Integer, primary_key=True)
    project = db.Column(db.String(255))
    logger_type = db.Column(db.String(255))
    entity_id = db.Column(db.Integer)
    entity_identifier = db.Column(db.String(255), nullable=True)
    last_received_on = db.Column(db.Integer, nullable=True)
    last_notified_on = db.Column(db.Integer, nullable=True)
    last_notified_to = db.Column(db.Text, nullable=True)
    created_on = db.Column(db.DateTime, nullable=True)
    updated_on = db.Column(db.DateTime, nullable=True)
    is_disabled = db.Column(TINYINT(), default=0)


class DisaggregationTrainingData(BaseModelMixin, db.Model):
    __tablename__ = 'disaggregation_training_data'
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(255), nullable=True)
    device_id = db.Column(db.String(255), nullable=True)
    lead_property_id = db.Column(db.Integer, db.ForeignKey('lead_property.id'))
    activity_type = db.Column(db.String(255), nullable=True)
    equipment_type = db.Column(db.String(255), nullable=True)
    event_time = db.Column(db.Integer, default=0)
    event_type = db.Column(db.String(255), nullable=True)
    created_on = db.Column(db.DateTime, nullable=True)
    updated_on = db.Column(db.DateTime, nullable=True)
    is_disabled = db.Column(TINYINT(), default=0)
