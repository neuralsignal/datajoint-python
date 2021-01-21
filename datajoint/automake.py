"""Defines Subclass of Autopopulate called Automake
"""

import re
import pandas
import collections
import warnings
import sys
import numpy as np

from .autopopulate import AutoPopulate
from .expression import AndList
from .utils import ClassProperty
from .errors import DataJointError
from .settings_table import Settingstable

if sys.version_info[1] < 6:
    dict = collections.OrderedDict

Sequence = (collections.MutableSequence, tuple, set)


class AutoMake(AutoPopulate):
    """
    AutoMake is a mixin class for autocomputed and autoimported tables.
    It adds a settings table upstream and make method.
    """

    _settings_table = None
    _settings = None
    _verbose = False

    def make_compatible(self, data):
        """function that can be defined for each class to transform data after
        computation.
        """
        return data

    def populate(self, settings_name, *restrictions, **kwargs):
        """
        rel.populate() calls rel.make(key) for every primary key in self.key_source
        for which there is not already a tuple in rel.
        :param settings_name: name of settings to use for autopopulation from SettingsTable.
        :param restrictions: a list of restrictions each restrict (rel.key_source - target.proj())
        :param suppress_errors: if True, do not terminate execution.
        :param return_exception_objects: return error objects instead of just error messages
        :param reserve_jobs: if true, reserves job to populate in asynchronous fashion
        :param order: "original"|"reverse"|"random"  - the order of execution
        :param display_progress: if True, report progress_bar
        :param limit: if not None, checks at most that many keys
        :param max_calls: if not None, populates at max that many keys
        """

        # TODO dealing with restrictions

        setting_restrict = {self.settings_name: settings_name}
        settings_table = self.settings_table & setting_restrict

        if len(settings_table) != 1:
            raise DataJointError(
                'Setting "{}" does not exist'.format(settings_name)
            )
        settings = (settings_table).fetch1()
        settings['fetch_tables'] = (
            settings['fetch_tables'] & AndList(restrictions)
        )
        self._settings = settings
        self._verbose = kwargs.pop('verbose', False)

        return super().populate(
            setting_restrict,
            self._settings['fetch_tables'].proj(),
            *restrictions,
            **kwargs
        )

    def get_entry(self, key):
        """
        Method used within make to get entry/tuple data
        """
        table = self._settings['fetch_tables'] & key

        if 'fetch1' in self._settings['fetch_method']:
            entry = getattr(
                table,
                self._settings['fetch_method']
            )()

        else:
            if len(table) == 0:
                raise DataJointError(
                    'empty joined table for key {0}'.format(key)
                )

            entry = getattr(
                table,
                self._settings['fetch_method']
            )(format='frame').reset_index().to_dict('list')

            for column, value in entry.items():
                if column in self._settings['parse_unique']:
                    # TODO check if unique?
                    entry[column] = value[0]
        return entry

    def prepare_make(self, entry):
        """
        Method used within make to prepare for computation
        """

        args, kwargs = self._create_kwargs(
            entry,
            self._settings['entry_settings'],
            self._settings['global_settings'],
            self._settings['args'],
            self._settings['kwargs']
        )

        func = self._settings['func']

        return func, args, kwargs

    def make(self, key):
        """automated make method
        """

        if self._verbose:
            print("Start autopopulation for key `{0}`".format(key))

        entry = self.get_entry(key)
        func, args, kwargs = self.prepare_make(entry)
        output = func(*args, **kwargs)
        self.insert_output(key, entry, output)

        # verbosity
        if self._verbose:
            print('Populated entry: {key}'.format(key=key))

    def insert_output(self, key, entry, output):
        """
        Method used within make to insert output into table
        """

        if self._settings['assign_output'] is None:
            output = self.make_compatible(output)
        else:
            output = {self._settings['assign_output']: output}

        if output is None:
            warnings.warn('output of function is None for key {0}'.format(key))
            output = {}

        # Test if dict or dataframe, convert to dataframe if necessary
        if isinstance(output, np.recarray):
            output = pandas.DataFrame(output)

        if (
            self.has_part_tables
            and not isinstance(output, (pandas.DataFrame, dict))
        ):
            raise DataJointError(
                "output must be dataframe or dict for table with part tables."
            )

        elif not self.has_part_tables and not isinstance(output, dict):
            raise DataJointError(
                "ouput must be dict for table without part tables."
            )

        # settings name - add to output
        output[self.settings_name] = self._settings[self.settings_name]

        # add columns that are missing in the output (only primary keys)
        for column in (set(self.primary_key) & set(entry) - set(output)):
            if (
                'fetch1' in self._settings['fetch_method']
                or column in self._settings['parse_unique']
            ):
                output[column] = entry[column]
            else:
                output[column] = entry[column][0]

        # insert into table and part_table
        if self.has_part_tables:
            self.insert1p(output, raise_part_missing=False)
        else:
            self.insert1(output)

    @staticmethod
    def _create_kwargs(
        entry, entry_settings, global_settings,
        settings_args, settings_kwargs
    ):
        """create args and kwargs to pass to function
        """
        args = []
        kwargs = global_settings.copy()
        # substitute entry settings
        for kw, arg in entry_settings.items():
            if isinstance(arg, str):
                kwargs[kw] = entry[arg]
            elif isinstance(arg, Sequence):
                kwargs[kw] = type(arg)(
                    [entry[iarg] for iarg in arg]
                )
            elif isinstance(arg, collections.Mapping):
                kwargs[kw] = {
                    key: entry[iarg] for key, iarg in arg.items()
                }
            else:
                raise DataJointError(
                    "argument in entry settings must be "
                    "str, tuple, or list, but is "
                    "{0} for {1}".format(type(arg), kw)
                )

        if settings_args is not None:
            args.extend(kwargs.pop(settings_args))
        if settings_kwargs is not None:
            kw = kwargs.pop(settings_kwargs)
            kwargs.update(kw)
        #
        return args, kwargs

    @property
    def settings_name(self):
        return self.settings_table().settings_name

    @ClassProperty
    def settings_table(cls):
        """return settings table
        """

        if cls._settings_table is None:
            # dynamically assign settings table

            settings_table_name = cls.name + 'Settings'
            child_table = cls

            class Settings(Settingstable):

                @property
                def definition(self):
                    return super().definition.format(
                        table_name=cls.table_name.strip('_#')
                    )

                @ClassProperty
                def name(cls):
                    return settings_table_name

                @ClassProperty
                def child_table(cls):
                    return child_table

            cls._settings_table = Settings

        return cls._settings_table

    @classmethod
    def set_true_definition(cls):
        """add settings table attribute if not in definition
        """

        settings_table_attribute = '-> {}'.format(cls.settings_table.name)

        if isinstance(cls.definition, property):
            pass
        elif settings_table_attribute not in cls.definition:

            definition = re.split(r'\s*\n\s*', cls.definition.strip())

            in_key_index = None

            for line_index, line in enumerate(definition):
                if line.startswith('---') or line.startswith('___'):
                    in_key_index = line_index
                    break

            if in_key_index is None:
                definition.insert(-1, settings_table_attribute)
            else:
                definition.insert(in_key_index, settings_table_attribute)

            cls.definition = '\n'.join(definition)

        return cls

    def primary_parents(self, columns, restrictions=None):
        """returns joined parent tables excluding settings table.
        Uses columns to select what to project and restrictions for
        each table individually.
        """

        joined_primary_parents = None

        if self.target.full_table_name not in self.connection.dependencies:
            self.connection.dependencies.load()

        for freetable in self.target.parents(primary=True, as_objects=True):

            if freetable.full_table_name == self.settings_table.full_table_name:
                continue

            proj_columns = list(set(freetable.heading.names) & set(columns))
            proj_table = freetable.proj(*proj_columns)
            if restrictions is not None:
                proj_table = proj_table & restrictions

            if joined_primary_parents is None:
                joined_primary_parents = proj_table
            else:
                joined_primary_parents = joined_primary_parents * proj_table

        return joined_primary_parents
