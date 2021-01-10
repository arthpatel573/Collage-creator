import os

class BaseConfig:
    API_KEY = os.getenv("API_KEY", "")
    APP_ENVIRONMENT = os.getenv("APP_ENVIRONMENT", "")

class DevelopmentConfig(BaseConfig):
    pass

class TestingConfig(BaseConfig):
    TESTING = True

class ProductionConfig(BaseConfig):
    pass
