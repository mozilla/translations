"""
This file contains helpers for converting untyped JSON configurations into typed JSON.

1. Dataclasses require underscores, while our configs uses kebab casing. These helpers
   handle the serialization and deserialization to change underscores and dashes.
   See the KebabDataclass.

2. The pipeline uses two different validation schemes, voluptuous types, and json schema.
   These helpers have convertors for both validation types so the source of truth is
   the type-safe dataclass, but then stored and generated yaml files can still be
   validated.

3. Dataclasses don't validate unions of string literals at runtime. These helpers do.
"""

from dataclasses import fields, is_dataclass
import types
from typing import Optional, Any, Literal, Tuple, Union, cast, get_origin, get_args, Type, TypeVar
import voluptuous
import dataclasses
from abc import ABC

T = TypeVar("T", bound="StricterDataclass")


class StricterDataclass(ABC):
    """
    This is the base class to support serializing/deserializing with custom logic for
    key names. It can be mixed in with kebab casing, and has stricter checks around
    unions of literals.

    Usage:
        @dataclass(kw_only=True)
        class MyConfig(StricterDataclass):
            attribute_name: str

        config_dict = { "attribute_name": "Underscore property" }
        config = MyConfig.from_dict(config_dict)

        assert config.attribute_name == "Underscore property"

        # It can be round tripped.
        assert config.to_dict() == config_dict
    """

    @staticmethod
    def _deserialize(data_type: type, data: Any, key_path: str):
        """
        Recursively deserialize the dataclasses in a stricter fashion. This handles the
        kebab casing requirements for the KebabDataclass. It also validates Unions
        of Literals.
        """

        assert not is_type_optional(
            data_type
        ), "Optional types should not be passed into this function."

        key_path_display = key_path or "<root>"
        if is_dataclass(data_type):
            if not isinstance(data, dict):
                print("Data:", data)
                raise ValueError(f'Expected a dictionary at "{key_path_display}".')

            fields = {field.name: field for field in dataclasses.fields(data_type)}
            required_fields = {
                field.name for field in fields.values() if not is_type_optional(field.type)
            }
            vargs = {}
            for dict_key, dict_value in data.items():
                if not isinstance(dict_key, str):
                    print(dict_key)
                    raise ValueError('Expected a dict key to be a string at "{key_path_display}".')
                # Replace the kebab casing where needed.
                vargs_key = cast(str, dict_key)
                if issubclass(data_type, KebabDataclass):
                    vargs_key = dict_key.replace("-", "_")

                next_key_path = f"{key_path}.{dict_key}" if key_path else dict_key
                if vargs_key not in fields:
                    raise ValueError(
                        f'Unexpected field "{dict_key}" when deserializing "{next_key_path}". See dataclass {data_type.__name__}'
                    )

                field: dataclasses.Field = fields[vargs_key]
                field_type = extract_optional_properties(field.type)[1]

                vargs[vargs_key] = StricterDataclass._deserialize(
                    field_type, dict_value, next_key_path
                )
                required_fields.discard(vargs_key)

            if required_fields:
                raise ValueError(
                    f'Fields missing in {key_path_display}: {", ".join(required_fields)}. See dataclass {data_type.__name__}'
                )

            return data_type(**vargs)

        type_origin = get_origin(data_type)

        if type_origin is list:
            if not isinstance(data, list):
                print(data)
                raise ValueError(f'A list was not provided at "{key_path_display}"')
            list_type = get_args(data_type)[0]
            return [
                StricterDataclass._deserialize(list_type, list_item, f"{key_path}[{index}]")
                for index, list_item in enumerate(data)
            ]

        primitives = {float, int, str, dict, None}

        if type_origin is Union:
            union_items = get_args(data_type)
            literals: Optional[list[str]] = None
            has_literals = False
            is_all_literals = True
            for union_item in union_items:
                if get_origin(union_item) is Literal:
                    has_literals = True
                else:
                    is_all_literals = False

            if has_literals:
                if is_all_literals:
                    literals = [get_args(union_item)[0] for union_item in union_items]
                    if data in literals:
                        return data
                    else:
                        raise ValueError(
                            f'An unexpected value "{data}" was provided at "{key_path_display}", expected it to be one of: {literals}'
                        )
                else:
                    raise TypeError(
                        f'Union contained a mix of literal and non-literal types at "{key_path_display}"'
                    )

            for union_item in union_items:
                if union_item not in primitives:
                    raise TypeError(
                        f'A union contained a non-primitive value "{union_item}" at "{key_path_display}"'
                    )

            return data

        if type_origin not in primitives:
            raise ValueError(
                f'Non-primitive value {type_origin} provided at "{key_path_display}".'
            )

        return data

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        result = StricterDataclass._deserialize(cls, data, "")
        assert isinstance(result, cls)
        return result

    @staticmethod
    def _serialize(value: Any) -> Any:
        if isinstance(value, (float, int, str, dict, type(None))):
            return value

        if isinstance(value, list):
            return [StricterDataclass._serialize(v) for v in value]

        if is_dataclass(value):
            result = {}
            for field in fields(value):
                # Handle the Kebab casing.
                field_name = field.name
                if issubclass(type(value), KebabDataclass):
                    field_name = field_name.replace("_", "-")

                # Serialize the value, but omit it if the value is None.
                serialized_value = StricterDataclass._serialize(getattr(value, field.name))
                if serialized_value is not None:
                    result[field_name] = serialized_value

            return result

        raise ValueError("Unexpected value type")

    def to_dict(self) -> dict:
        return StricterDataclass._serialize(self)


class KebabDataclass(StricterDataclass):
    """
    Convert the dataclass's underscore style names to kebab style casing when serializing
    and the reverse for deserializing.

    Serializing:    "mono_max_sentences_trg" -> "mono-max-sentences-trg"
    De-serializing: "mono_max_sentences_trg" -> "mono-max-sentences-trg"

    Usage:
        @dataclass(kw_only=True)
        class MyConfig(KebabDataclass):
            attribute_name: str

        config_dict = { "attribute-name": "Kebab case example" }
        config = MyConfig.from_dict(config_dict)

        assert config.attribute_name == "Kebab case example"

        # It can be round tripped.
        assert config.to_dict() == config_dict
    """

    pass


def extract_voluptuous_optional_type(
    t: Any,
) -> Tuple:
    """
    Determines if a property is optional or not.
    Optional[T] is an alias for Union[T, None].
    """
    origin = get_origin(t)
    if origin is not Union:
        return voluptuous.Required, t

    args = get_args(t)
    if len(args) != 2:
        return voluptuous.Required, t

    arg1, arg2 = args
    if arg1 is types.NoneType:
        return voluptuous.Optional, arg2

    if arg2 is types.NoneType:
        return voluptuous.Optional, arg1

    return voluptuous.Required, t


def extract_optional_properties(
    t: Any,
) -> tuple[bool, Any]:
    """
    Determines if a property is optional or required. If it is required it adds it
    to the required list.

    The following values are returned:
        Optional[T] returns T
        T return T

    Optional[T] is an alias for Union[T, None].
    """
    origin = get_origin(t)
    if origin is not Union:
        return False, t

    args = get_args(t)
    if len(args) != 2:
        return False, t

    arg1, arg2 = args
    if arg1 is types.NoneType:
        return True, arg2

    if arg2 is types.NoneType:
        return True, arg1

    return False, t


def is_type_optional(t: Any) -> bool:
    return extract_optional_properties(t)[0]


def handle_field_casing(type_value: type, field_name: str) -> str:
    """
    Data classes don't support hyphens, so map them to dashes if they have been configured
    to do so with dataclasses_json.
    """

    if issubclass(type_value, KebabDataclass):
        return field_name.replace("_", "-")

    return field_name


def build_voluptuous_schema(value: type, key_requirement=voluptuous.Required):
    # Is this just a basic data type?
    if value in [str, float, int, bool]:
        if key_requirement == voluptuous.Optional:
            # The validation will fail if the value `null` is passed in for a key rather
            # than just omitting the key. Handle that case here by allowing null.
            return voluptuous.Any(value, None)
        else:
            return value

    # Handle creating the schema from a dataclass. Also handle the kebab casing.
    if is_dataclass(value):
        schema_dict = {}
        for field in fields(value):
            key_wrapper, field_type = extract_voluptuous_optional_type(field.type)
            field_name = handle_field_casing(value, field.name)

            schema_dict[key_wrapper(field_name)] = build_voluptuous_schema(field_type, key_wrapper)
        return schema_dict

    origin = get_origin(value)

    # Convert a dict type to voluptous
    # dict[str, str] => {str: str}
    if origin is dict:
        args = get_args(value)
        key: Any = voluptuous.Any
        dict_value = voluptuous.Any

        if len(args) > 0:
            key = build_voluptuous_schema(args[0])

        if len(args) == 2:
            dict_value = build_voluptuous_schema(args[1])

        return {key: dict_value}

    # Convert a list type to voluptous
    # list[float] => [float]
    # list[int]   => [int]
    # list        => list
    if origin is list:
        args = get_args(origin)
        if not args:
            return list
        return [build_voluptuous_schema(args[0])]

    if origin is Union:
        args = get_args(value)
        enum = []
        for literal_type in args:
            if get_origin(literal_type) is not Literal:
                print(literal_type)
                raise ValueError("Currently only Literals are currently supported in Unions")
            literal_str = get_args(literal_type)[0]
            assert isinstance(literal_str, str), "The literal is a string"
            enum.append(literal_str)

        return voluptuous.In(enum)

    raise Exception("Unknown type when converting from dataclass to voluptuous")


json_schema_type = {str: "string", float: "number", int: "number", bool: "boolean"}


def build_json_schema(type_value: type, is_key_optional=False):
    """
    https://json-schema.org/
    """

    # Is this just a basic data type?
    if type_value in json_schema_type:
        if is_key_optional:
            # The validation will fail if the value `null` is passed in for a key rather
            # than just omitting the key. Handle that case here by allowing null.
            return {"type": [json_schema_type[type_value], "null"]}
        else:
            return {"type": json_schema_type[type_value]}

    # Handle creating the schema from a dataclass. Also handle the kebab casing.
    if is_dataclass(type_value):
        properties: dict[str, Any] = {}
        schema_dict = {"type": "object", "additionalProperties": False, "properties": properties}
        required: list[str] = []
        for field in fields(type_value):
            is_optional, field_type = extract_optional_properties(field.type)
            field_name = handle_field_casing(type_value, field.name)
            if not is_optional:
                required.append(field_name)
            properties[field_name] = build_json_schema(field_type, is_optional)
        if required:
            schema_dict["required"] = required
        return schema_dict

    origin = get_origin(type_value)

    if origin is dict:
        args = get_args(type_value)
        additional_properties = {}
        if len(args) == 2:
            additional_properties = build_json_schema(args[1])

        return {"type": "object", "additionalProperties": additional_properties}

    if origin is list:
        args = get_args(origin)
        if not args:
            return {
                "type": "array",
                "items": {},
            }

        return (
            {
                "type": "array",
                "items": build_json_schema(args[0]),
            },
        )

    # Unions are a bit more complicated to handle. In our case make the assumption that
    # only Literals are allowed as a union, as these can than be checked by the schema
    # as an enum type.
    if origin is Union:
        # Only literal args are supported here.
        args = get_args(type_value)
        enum = []

        for literal_type in args:
            if get_origin(literal_type) is not Literal:
                print(literal_type)
                raise ValueError("Currently only Literals are currently supported in Unions")
            literal_str = get_args(literal_type)[0]
            assert isinstance(literal_str, str), "The literal is a string"
            enum.append(literal_str)

        return {
            "type": "string",
            "enum": enum,
        }

    raise Exception("Unknown type when converting from dataclass to voluptuous")
