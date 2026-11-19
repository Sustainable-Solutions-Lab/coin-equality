"""
Command-line configuration overrides shared by the run scripts.

Overrides have the form `--key.subkey value` and replace the value at that dot-separated
path in the JSON configuration dictionary. The new value is converted to the type of
the value it replaces.
"""

import json


def parse_override_arguments(arguments):
    """
    Parse `--key.subkey value` pairs (the arguments argparse did not consume).

    Parameters
    ----------
    arguments : list of str
        Remaining command-line arguments

    Returns
    -------
    dict
        Maps dot-separated key paths to string values, in command-line order
    """
    overrides = {}
    i = 0
    while i < len(arguments):
        arg = arguments[i]
        if not arg.startswith('--'):
            raise ValueError(f"Unexpected argument '{arg}'. Overrides must start with '--'")
        if i + 1 >= len(arguments) or arguments[i + 1].startswith('--'):
            raise ValueError(f"Override '{arg}' requires a value")
        overrides[arg[2:]] = arguments[i + 1]
        i += 2
    return overrides


def apply_config_override(config_dict, key_path, value):
    """
    Apply a command line override to a nested configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        The configuration dictionary to modify
    key_path : str
        Dot-separated path to the key (e.g., "scalar_parameters.alpha")
    value : str
        String value to set (will be converted to appropriate type)
    """
    keys = key_path.split('.')

    # Navigate to the parent dict
    current = config_dict
    for key in keys[:-1]:
        if key not in current:
            raise KeyError(f"Key path '{key_path}' not found in config (failed at '{key}')")
        current = current[key]

    final_key = keys[-1]
    if final_key not in current:
        raise KeyError(f"Key path '{key_path}' not found in config (final key '{final_key}' not found)")

    # Infer type from existing value
    existing_value = current[final_key]

    try:
        if existing_value is None:
            # If existing is None, try int -> float -> bool -> string
            try:
                converted_value = int(value)
            except ValueError:
                try:
                    converted_value = float(value)
                except ValueError:
                    if value.lower() in ('true', 'false'):
                        converted_value = value.lower() == 'true'
                    else:
                        converted_value = value
        elif isinstance(existing_value, bool):
            # Handle bools before int (since bool is subclass of int in Python)
            converted_value = value.lower() in ('true', '1', 'yes')
        elif isinstance(existing_value, int):
            converted_value = int(value)
        elif isinstance(existing_value, float):
            converted_value = float(value)
        elif isinstance(existing_value, str):
            converted_value = value
        elif isinstance(existing_value, (list, dict)):
            # Try to parse as JSON for lists and dicts
            converted_value = json.loads(value)
        else:
            # Fallback: keep as string
            converted_value = value
    except (ValueError, json.JSONDecodeError) as e:
        raise ValueError(f"Cannot convert '{value}' to type {type(existing_value).__name__} for key '{key_path}': {e}")

    current[final_key] = converted_value


def load_config_dict_with_overrides(config_path, overrides):
    """Read a JSON configuration file and apply the command-line overrides to it."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    for key_path, value in overrides.items():
        apply_config_override(config_dict, key_path, value)
    return config_dict
