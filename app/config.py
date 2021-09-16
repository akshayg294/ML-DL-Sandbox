class Config(object):
    DEBUG = False
    CSRF_ENABLED = True


class DevelopmentConfig(Config):
    DEBUG = True
    ENV = 'development'


class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    ENV = 'testing'


class StagingConfig(Config):
    DEBUG = True
    ENV = 'staging'


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    ENV = 'production'


app_config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'staging': StagingConfig,
    'production': ProductionConfig,
}
