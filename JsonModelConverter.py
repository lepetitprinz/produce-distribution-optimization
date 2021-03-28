# -*- coding: utf-8 -*-

import json

# import tensorflow
from keras.utils.generic_utils import _GLOBAL_CUSTOM_OBJECTS
from keras.layers import deserialize


# Keras 를 통한 Model 인스턴스 작성 시
# 내부 처리 도중 발생하는 Tensorflow 2.x 버전과의 Incompatibility 문제 해결 위함
# - Model 에 Input Layer 추가 시 placeholder 클래스를 찾지 못하는 문제에 대한 대응
# tensorflow.compat.v1.disable_v2_behavior()


class JsonModelConverter(object):
    """
    Conversion Tool
    model info json -> model in keras 2.1.3
    """

    def __init__(self, json_path: str):
        self.json_path: str = json_path

    def get_dict_from_json(self):
        json_string: str = open(self.json_path, encoding='UTF8').read()
        ret_dict: dict = json.loads(json_string)

        return ret_dict

    def build_model_from_json(self, input_shape: int):
        json_model_dict: dict = self.get_dict_from_json()

        # Input Shape 수정
        json_model_dict['config']['layers'][00]['config']['batch_input_shape'] = [None, input_shape]

        model_class: type = self.get_model_class_by_config(json_model_dict)
        model_info_dict: dict = self.get_model_parameters(json_model_dict)

        model = model_class(**model_info_dict)

        layer_infos: list = self.get_layer_infos(json_model_dict)
        for layer_info in layer_infos:
            layer_class: type = self.get_layer_class_by_config(layer_info)
            layer_info_dict: dict = self.get_layer_parameters(layer_info)
            layer = layer_class(**layer_info_dict)
            model.add(layer)

        return model

    def get_model_class_by_config(self, config: dict, **custom_objects):

        #
        model_classes: dict = self._get_model_classes()    # Register All Keras Model Classes

        # Find Key
        if 'class_name' not in config or 'config' not in config:
            raise ValueError('Improper config format: ' + str(config))

        # Find Class Name
        class_name = config['class_name']
        if custom_objects and class_name in custom_objects:
            cls = custom_objects[class_name]
        elif class_name in _GLOBAL_CUSTOM_OBJECTS:
            cls = _GLOBAL_CUSTOM_OBJECTS[class_name]
        else:
            model_classes = model_classes or {}
            cls = model_classes.get(class_name)
            if cls is None:
                raise ValueError('Unknown ' + 'model' + ': ' + class_name)

        return cls

    def get_layer_class_by_config(self, config: dict, **custom_objects):

        #
        layer_classes: dict = self._get_layer_classes()  # Register All Keras Layer Classes

        # Find Key
        if 'class_name' not in config or 'config' not in config:
            raise ValueError('Improper config format: ' + str(config))

        # Find Class Name
        class_name = config['class_name']
        if custom_objects and class_name in custom_objects:
            cls = custom_objects[class_name]
        elif class_name in _GLOBAL_CUSTOM_OBJECTS:
            cls = _GLOBAL_CUSTOM_OBJECTS[class_name]
        else:
            layer_classes = layer_classes or {}
            cls = layer_classes.get(class_name)
            if cls is None:
                raise ValueError('Unknown ' + 'layer' + ': ' + class_name)

        return cls

    def _get_model_classes(self):
        from keras import models  # Locally Import
        ret_dict: dict = dict()
        for key, value in models.__dict__.items():
            if type(value) is not type:
                continue
            if issubclass(value, models.Model):
                ret_dict.update({key: value})
        return ret_dict

    def _get_layer_classes(self):
        from keras import layers    # Locally Import
        ret_dict: dict = dict()
        for key, value in layers.__dict__.items():
            if type(value) is not type:
                continue
            if issubclass(value, layers.Layer):
                ret_dict.update({key: value})
        return ret_dict

    def get_layer_infos(self, config: dict):

        config: dict = config.get('config')
        if config is None:
            raise KeyError(
                f"In .json Dictionary, Key 'config' Not Found"
            )

        return config.get('layers', [])

    def get_model_parameters(self, config: dict):

        config: dict = config.get('config')
        if config is None:
            raise KeyError(
                f"In .json Dictionary, Key 'config' Not Found"
            )

        return {
            key: value
            for key, value in config.items()
            if type(value) is not list
        }

    def get_layer_parameters(self, config: dict):

        config: dict = config.get('config')
        if config is None:
            raise KeyError(
                f"In .json Dictionary, Key 'config' Not Found"
            )

        return {
            key: value
            for key, value in config.items()
        }
