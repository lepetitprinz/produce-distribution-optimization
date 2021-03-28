
import json
import logging

from logging import config as log_config_setter

from SingletonInstance import SingletonInstance


class LogHandler(SingletonInstance):
    """
    Wrapper for logging Module
    """

    LOG_CONFIG_JSON: str = '../conf/logging.json'

    def __init__(self):

        with open(self.LOG_CONFIG_JSON, 'rt') as f:
            log_config = json.load(f)
        log_config_setter.dictConfig(log_config)

    def get_logger(self, name: str):
        return logging.getLogger(name)
