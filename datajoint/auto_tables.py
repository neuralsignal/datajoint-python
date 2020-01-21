"""Defines auto make class and settings table
Contains AutoComputed and AutoImported tables
"""

import collections
import sys

from .user_tables import UserTable, _base_regexp
from .automake import AutoMake

if sys.version_info[1] < 6:
    dict = collections.OrderedDict

Sequence = (collections.MutableSequence, tuple, set)


class AutoImported(UserTable, AutoMake):
    """
    Inherit from this class if the table's values are imported from external
    data sources.
    """

    _prefix = '#_'
    tier_regexp = r'(?P<autoimported>' + _prefix + _base_regexp + ')'

    def __init__(self, *args, **kwargs):

        self.set_true_definition()
        super().__init__(*args, **kwargs)


class AutoComputed(UserTable, AutoMake):
    """
    Inherit from this class if the table's values are computed from other
    relations in the schema.
    """

    _prefix = '_#'
    tier_regexp = r'(?P<autocomputed>' + _prefix + _base_regexp + ')'

    def __init__(self, *args, **kwargs):

        self.set_true_definition()
        super().__init__(*args, **kwargs)
