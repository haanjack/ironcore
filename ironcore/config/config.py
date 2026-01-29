# Copyright (c) 2025-2026 Jaegeun Han
#
# SPDX-License-Identifier: MIT
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the above copyright notice,
# this list of conditions, and the following disclaimer are retained.
#
# Full license text is available at LICENSE file.

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

import yaml


# pylint: disable=no-member
@dataclass
class BaseConfig:
    """configuration"""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """update config"""
        for k, v in kwargs.items():
            if k not in self.__dataclass_fields__:
                raise KeyError(
                    f"{k} is not defined in {self.__dataclass_fields__}")
            if not self._type_checker(self):
                raise TypeError(
                    f"'{k}' data type is not match with defined information: {type(v)} vs {self.__dataclass_fields__[k].type}"
                )

            if isinstance(v, dict):
                # convert as config class type
                v = self.__dataclass_fields__[k].type(**v)
            setattr(self, k, v)
        return self

    @classmethod
    def from_dict(cls, kw_values):
        for k, v in kw_values.items():
            setattr(cls, k, v)
        return cls

    @classmethod
    def _type_checker(cls, var):
        """check if arguments' data type is correct"""
        for field_name, field_type in cls.__annotations__.items():
            input_field_value = getattr(var, field_name)
            if input_field_value is None:
                continue
            # if not isinstance(input_field_value, field_type):
            #     return False
            if isinstance(input_field_value, list):
                # validate config option struction is list
                if all(
                    isinstance(item, field_type.__args__[0].__args__)
                    for item in input_field_value
                    if not isinstance(item, list)
                ):
                    # check each element's data type if they follows the config definition
                    continue
                if all(isinstance(item, list) for item in input_field_value):
                    # check if the given items are defined as list
                    continue
                return False

            if get_origin(field_type) is Union:
                union_args = get_args(field_type)
                matched = False
                for union_arg in union_args:
                    # Check for NoneType
                    if union_arg is type(None) and input_field_value is None:
                        matched = True
                        break
                    # Check for Literal types within Union
                    elif get_origin(union_arg) is Literal and input_field_value in get_args(union_arg):
                        matched = True
                        break
                    # Check for regular types
                    elif isinstance(union_arg, type) and isinstance(input_field_value, union_arg):
                        matched = True
                        break
                if matched:
                    continue
                return False
            if get_origin(field_type) is Literal:
                if input_field_value in get_args(field_type):
                    continue
                return False
            if isinstance(input_field_value, field_type):
                continue
            if hasattr(field_type, "_type_checker"):
                # nested check purpose
                if field_type._type_checker(field_type):
                    continue
                return False
            return False

        return True

    def to_yaml(self, filename: Union[str, Path]) -> None:
        def dataclass_to_dict(obj):
            if isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            if hasattr(obj, "__dataclass_fields__"):
                return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
            return obj

        data_dict = dataclass_to_dict(self)
        with open(filename, "w", encoding="utf-8") as file:
            yaml.dump(data_dict, file, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filename: Union[str, Path]):
        def dict_to_dataclass(data, cls):
            if isinstance(data, dict):
                fieldtypes = {f.name: f.type for f in fields(cls)}
                return cls(
                    **{k: dict_to_dataclass(v, fieldtypes[k]) for k, v in data.items()}
                )
            if isinstance(data, list):
                return [dict_to_dataclass(item, cls.__args__[0]) for item in data]
            return data

        with open(filename, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return dict_to_dataclass(data, cls)
