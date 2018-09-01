"""General-purpose utilities"""

import re
import os
#from .computedmixin import JSON_STR
import numpy as np
from .errors import DataJointError

class ClassProperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


def user_choice(prompt, choices=("yes", "no"), default=None):
    """
    Prompts the user for confirmation.  The default value, if any, is capitalized.

    :param prompt: Information to display to the user.
    :param choices: an iterable of possible choices.
    :param default: default choice
    :return: the user's choice
    """
    assert default is None or  default in choices
    choice_list = ', '.join((choice.title() if choice == default else choice for choice in choices))
    response = None
    while response not in choices:
        response = input(prompt + ' [' + choice_list + ']: ')
        response = response.lower() if response else default
    return response


def to_camel_case(s):
    """
    Convert names with under score (_) separation into camel case names.

    :param s: string in under_score notation
    :returns: string in CamelCase notation

    Example:

    >>> to_camel_case("table_name") # yields "TableName"

    """

    def to_upper(match):
        return match.group(0)[-1].upper()

    return re.sub('(^|[_\W])+[a-zA-Z]', to_upper, s)


def from_camel_case(s):
    """
    Convert names in camel case into underscore (_) separated names

    :param s: string in CamelCase notation
    :returns: string in under_score notation

    Example:

    >>> from_camel_case("TableName") # yields "table_name"

    """

    def convert(match):
        return ('_' if match.groups()[0] else '') + match.group(0).lower()

    if not re.match(r'[A-Z][a-zA-Z0-9]*', s):
        raise DataJointError(
            'ClassName must be alphanumeric in CamelCase, begin with a capital letter')
    return re.sub(r'(\B[A-Z])|(\b[A-Z])', convert, s)


def safe_write(filename, blob):
    """
    A two-step write.
    :param filename: full path
    :param blob: binary data
    :return: None
    """
    temp_file = filename + '.saving'
    with open(temp_file, 'bw') as f:
        f.write(blob)
    os.rename(temp_file, filename)

def read_json(json_field):
    """read a single json entry after having been fetched

    Returns
    -------
    json_field : dict
        json_field reformatted as a dictionary.
    """
    if isinstance(json_field, str):
        return eval(json_field)
    elif isinstance(json_field, np.recarray):
        json_field = [dict(zip(json_field.dtype.names, values)) for values in json_field]
        if len(json_field) != 1:
            raise DataJointError("json field could not be intepreted")
        return json_field[0]
    else:
        return json_field

def read_jsons(json_fields, column_name='json'):
    """read multiple json entries

    Returns
    -------
    json_fields : numpy.recarray
        recarray with each entry modified to a dictionary.
    """
    return np.recarray([read_json(json_field) for json_field in json_fields], dtype=[(column_name, object)])

def interpret_json(json_dict, json_str):
    """Given one json dictionary interpret it for insertion

    Parameters
    ----------
    json_dict : dict
        the json dictionary to be entered
    json_str : bool, optional
        Indicate if the json is to be passed as a string

    Returns
    -------
    json_field : unknown
        the json_field reformatted for entry into the database
    """
    if json_str:
        return str(json_dict)
    else:
        return json_dict

def interpret_jsons(json_dicts, json_str, column_name='json'):
    """Return json_fields for multiple insertion as a numpy recarray.
    """
    return np.recarray([interpret_json(json_dict, json_str) for json_dict in json_dicts], dtype=[(column_name, object)])
