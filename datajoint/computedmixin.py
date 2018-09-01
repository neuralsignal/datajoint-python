"""
Class for making make_tuples using approach tables
"""

import numpy as np
import pandas as pd
import importlib
from warnings import warn
from .base_relation import join_restrictions, superjoin
from joblib import Parallel, delayed

from .errors import DataJointError

JSON_STR = True
RESTRICTION_TYPES = (
        dict,
        list,
        np.ndarray,
        )

#these variables can be defined in the json field in the approach table
JSON_DEFINABLE = (
        'wrap_columns', 'not_wrap_columns',
        'from_tables', 'restrict_tables',
        'skip_tables', 'constant_restrictions',
        'update_parents', 'use_uberfetch',
        'np_first', 'mmap_mode', 'skip_external',
        'always_deepjoin', 'additional_columns',
        'multi_fetch',
    )

class ComputedMixin:
    """mixin computed class
    wrap_columns : list or tuple
       A list of columns to wrap in array regardless of uniqueness.
       Only necessary, if columns from part tables of upstream tables are used.
       Only issue with sequential insert.
    not_wrap_columns : list or tuple
       A list of columns not to wrap in array and simply choose the first element
       regardless of uniqueness. Only necessary, if restriction return multiple entries
       (i.e. upstream part table involved).
    """
    _multi_fetch = None
    _approach_table = None
    _strategy_table = None
    _approach_restriction = None
    _restrictions = None
    _constant_restrictions = {}
    _joined_table = None
    _global_settings = None
    _entry_settings = None
    _function = None
    verbose = False
    _joined_columns = None
    _np_first = False
    _mmap_mode = None
    _skip_tables = None
    _restrict_tables = None
    _use_uberfetch = False
    _skip_external = False
    _always_deepjoin = False
    _additional_columns = []
    #these variables are only relevant if the restriction returns multiple entries
    _wrap_columns = None #columns to wrap in array if upstream has part tables
    _not_wrap_columns = None #columns to not wrap in array and just choose the first element
    _from_tables = None
    _update_parents = False #update parent primary tables - work around computed part
    _parallel = None
    _parse_parallel = False

    @property
    def use_uberfetch(self):
        return self._use_uberfetch
    @staticmethod
    def _transform_output(output):
        return output

    def autopopulate(
            self, approach=None, restrictions=None,
            sequential=True,
            rejoin=True, verbose=False, skip_part_tables=False,
            parallel=None, n_jobs=None, parse_parallel=False,
            **kwargs
        ):
        """Autopopulate takes an approach and restrictions to
        the data to be fetched and inserts it into the table.
        It will also recompute the joined table

        Parameters
        ----------
        approach : str or iterable
            The approach(es) to use to calculate entries for the table.
        restrictions : dict, list of dicts, np.recarray
            The restrictions to apply to the joined table before computation.
        sequential : bool
            Whether to insert data sequentially or simultaneously. At the moment,
            only sequential insert is supported.
        rejoin : bool
            Whether to deepjoin the tables again. Necessary, if new restriction or
            approach is defined.
        parallel : joblib.Parallel
            For parallel processing of entry insertion.
        parse_parallel : bool
            If True, will parse parallel to function, instead of parallelizing entry insertion.
            Requires that function takes an argument called parallel.
        kwargs : dict
            Passed to populate.
        """
        self.skip_part_tables = skip_part_tables
        self._parallel = parallel
        self._parse_parallel = parse_parallel
        self.verbose = verbose
        self._redefine(approach, restrictions, rejoin)
        self.load_settings1()
        #
        if not parse_parallel and parallel is not None or n_jobs is not None:
            raise NotImplementedError()
            if n_jobs is not None:
                parallel = Parallel(n_jobs=n_jobs)
            primary_keys = self.joined_table.proj().fetch(as_dict=True)
            parallel(delayed(self._insert_sequential)(k) for k in primary_keys)
        elif sequential:
            key = join_restrictions(self.restrictions, self.approach_restriction)
            self.populate(key, **kwargs)
        else:
            raise NotImplementedError("not sequential insertion")

    def _redefine(self, approach, restrictions, rejoin):
        """helper function to redefine essential variables.
        """
        if rejoin:
            self._joined_table = None
        #
        if approach is None:
            pass
        elif isinstance(approach, str):
            self._approach_restriction = {
                self.approach_table_primary_key : approach
            }
        elif hasattr(approach, '__iter__'):
            self._approach_restriction = [
                {self.approach_table_primary_key : iapproach}
                for iapproach in approach
            ]
        else:
            raise DataJointError("approach must be str or iterable.")
        #
        if restrictions is not None:
            self._restrictions = restrictions

    def vcompute(
            self, approach=None, restrictions=None,
            rejoin=True, exclude_calculated=True
        ):
        """Simultaneous calculation of joined table.
        Simply returns output, but does not insert.
        """
        self._redefine(approach, restrictions, rejoin)
        #define joined table
        if exclude_calculated:
            joined_table = self.joined_table - self
        else:
            joined_table = self.joined_table
        input_table = pd.DataFrame(joined_table.fetch(mmap_mode=self.mmap_mode, skip_external=self._skip_external))
        kwargs = self._create_kwargs(input_table, self.entry_settings, self.global_settings)
        #
        vectorized_function = np.vectorize(self.function, otypes=[object])
        #
        output = vectorized_function(**kwargs)
        #
        return output

    def _make_tuples(self, key):
        """
        """
        self._insert_sequential(key)

    @staticmethod
    def _create_kwargs(dictionary, entry_settings, global_settings):
        """create kwargs to pass to function

        Parameters
        ----------
        dictionary : dict or pandas.DataFrame

        Returns
        -------
        kwargs : dict
        """
        if isinstance(dictionary, pd.DataFrame):
            dictionary = dictionary.to_dict('list')
        elif not isinstance(dictionary, dict):
            raise DataJointError("dictionary must be of type dict or pandas DataFrame")
        kwargs = global_settings
        #
        for kw, arg in entry_settings.items():
            if isinstance(arg, str):
                kwargs[kw] = dictionary[arg]
            elif isinstance(arg, tuple):
                kwargs[kw] = tuple(dictionary[iarg] for iarg in arg)
            elif isinstance(arg, list):
                kwargs[kw] = [dictionary[iarg] for iarg in arg]
            else:
                raise DataJointError(
                    "argument in entry settings must be "
                    "str, tuple, or list, but is "
                    f"{type(arg)} for {kw}"
                )
        #
        return kwargs

    def _insert_sequential(self, key):
        """Fetch data and insert into table
        """
        if self.verbose:
            print(f"multifetching: {self.multi_fetch}")
        if not self.multi_fetch:
            if self.use_uberfetch:
                entry = (self.joined_table & key).uberfetch1(mmap_mode=self.mmap_mode, skip_external=self._skip_external)
            else:
                entry = (self.joined_table & key).fetch1(mmap_mode=self.mmap_mode, skip_external=self._skip_external)
        else:
            #multi_fetching is necessary when fetching upstream part_tables
            restricted_table = self.joined_table & key

            if len(restricted_table) == 0:
                raise DataJointError(f'{self.full_table_name}: empty joined table for key {key}.')
            if self.use_uberfetch:
                entry = restricted_table.uberfetch(
                        mmap_mode=self.mmap_mode, skip_external=self._skip_external
                        ).to_dict('list')
            else:
                entry = pd.DataFrame(
                        restricted_table.fetch(mmap_mode=self.mmap_mode, skip_external=self._skip_external)
                        ).to_dict('list')
            if self.verbose:
                #reformat entry
                warn("using fetch instead of fetch1, consider specifying which columns to wrap")
            for column, value in entry.items():
                if column in self.wrap_columns:
                    entry[column] = np.array(value)
                elif column in self.not_wrap_columns:
                    entry[column] = value[0]
                else:
                    try:
                        if np.unique(value, axis=0).shape[0] == 1:
                            #errors may appear if only one element exists for fetch
                            #need to make explicit with wrap_columns
                            entry[column] = value[0]
                        else:
                            entry[column] = np.array(value)
                    except TypeError:
                        if np.array([np.array_equiv(value[0], v) for v in value[1:]]).all():
                            entry[column] = value[0]
                        else:
                            entry[column] = np.array(value)
            if self.verbose:
                print('adjusting multiple entry is done!')
        #
        if self.verbose:
            print(f'Attempting to compute and insert: {key}')

        kwargs = self._create_kwargs(
                entry, self.entry_settings, self.global_settings
                )
        if self.parse_parallel:
            kwargs['parallel'] = self.parallel

        #TODO add option of executing script - transform output
        output = self.function(**kwargs)
        #transform output to be the right type
        #dictionary for single and dataframe for multiple
        output = self.transform_output(output)

        #if output is None skip it
        if output is None:
            if self.verbose:
                print(f'Output returned None for key {key}')
            return output

        #Test if dict or dataframe
        if self.has_part_tables and not isinstance(output, (pd.DataFrame, dict)):
            raise DataJointError("output must be dataframe or dict for table with part tables.")
        elif not self.has_part_tables and not isinstance(output, dict):
            raise DataJointError("ouput must be dict for table without part tables.")

        #add column in entry table but not in output yet.
        #add approach table restrictions
        if self.approach_table_primary_key in key:
            output[self.approach_table_primary_key] = key[self.approach_table_primary_key]
        elif isinstance(self.approach_restriction, dict):
            output[self.approach_table_primary_key] = self.approach_restriction[self.approach_table_primary_key]
        else:
            raise DataJointError('approach restriction must be dict or primary key in self.')
        #
        for column in (set(self.heading) & set(entry) - set(output)):
            if entry[column] is None or pd.isnull(entry[column]):
                continue
            output[column] = entry[column]
        #
        if self.update_parents:
            if self.verbose:
                warn('update parents has not been tested.')
            if isinstance(output, pd.DataFrame):
                parents_update = output.iloc[0]
            else:
                parents_update = pd.Series(output)
            for parent_table in self.parent_tables(only_primary=True):
                try:
                    parent_table = parent_table()
                except:
                    pass
                update_columns = set(parent_table.heading) & set(parents_update.index) - set(self.heading)
                if update_columns:
                    (parent_table & key).update(parents_update[list(update_columns)].dropna().to_dict())
        #
        if self.has_part_tables and not self.skip_part_tables:
            self.insert1p(output, np_first=self._np_first)
        else:
            output = pd.Series(output)
            columns = set(self.heading) & set(output.index)
            if not self.update_parents:
                assert len(columns) == len(output.index), 'not all output columns recognized.'
            self.insert1(output[list(columns)].dropna().to_dict(), np_first=self._np_first)
        #
        if self.verbose:
            print(f"Inserted {key} for table {self.full_table_name}")

    def load_settings1(self):
        """Load settings for one approach in the approach table.

        Defines
        -------
        function : callable
            Function to be called for computation
        global_settings : dict
            Global settings for entries to be computed
        entry_settings : dict
            Entry-specific settings determined by upstream column names
        """
        #TODO load settings multiple approaches
        #import module
        settings_dict = (
            (self.approach_table & self.approach_restriction)
            * self.strategy_table.proj('package', 'function')
        ).fetch1()
        #define module
        module = importlib.import_module(settings_dict['package'])
        self._function = getattr(module, settings_dict['function'])
        #retrieve specific settings
        if settings_dict['json'] is None:
            json_defined = {}
        elif JSON_STR:
            try:
                json_defined = eval(settings_dict['json'])
            except SyntaxError:
                raise DataJointError('json field cannot be evaluated in approach table.')
        else:
            json_defined = settings_dict['json']
        #see if key in json_definable
        for key, json_field in json_defined.items():
            if key in JSON_DEFINABLE:
                if key == 'constant_restrictions':
                    json_field = join_restrictions(json_field, self.constant_restrictions)
                if key == 'additional_columns':
                    json_field = json_field + self._additional_columns
                setattr(self, '_'+key, json_field)
            else:
                raise DataJointError(f'key {key} is not json definable for approach table.')

        #required if not in json definable settings
        if self._multi_fetch is None:
            self._multi_fetch = False

        if JSON_STR:
            try:
                if settings_dict['global_settings'] is None:
                    self._global_settings = {}
                else:
                    self._global_settings = eval(settings_dict['global_settings'])
                if settings_dict['entry_settings'] is None:
                    self._entry_settings = {}
                else:
                    self._entry_settings = eval(settings_dict['entry_settings'])
            except SyntaxError:
                raise DataJointError('settings columns in approach table cannot be evaluated.')
        else:
            self._global_settings = settings_dict['global_settings']
            self._entry_settings = settings_dict['entry_settings']
        #
        #check if settings are right type
        if not isinstance(self._global_settings, dict):
            raise DataJointError("Settings must be of type dict")
        if not isinstance(self._entry_settings, dict):
            raise DataJointError("Settings must be of type dict")
        if not hasattr(self._function, '__call__'):
            raise DataJointError("Function must be callable")

    @property
    def always_deepjoin(self):
        return self._always_deepjoin

    @property
    def additional_columns(self):
        c = self._additional_columns
        if self.restrict_tables is not None:
            for table in self.restrict_tables:
                c.extend(list(table.heading.primary_key))
        if isinstance(self.restrictions, list):
            c.extend(list(self.restrictions[0].keys()))
        elif isinstance(self.restrictions, dict):
            c.extend(list(self.restrictions.keys()))
        else:
            raise DataJointError('restriction not dict or list')
        c.extend(self.joined_columns)
        return np.unique(c).tolist()

    @property
    def parallel(self):
        if self._parallel is None:
            return None
        elif not isinstance(self._parallel, Parallel):
            raise DataJointError(f'parallel not joblib.Parallel: {type(self._parallel)}')
        return self._parallel

    @property
    def multi_fetch(self):
        if self._multi_fetch is None:
            self.load_settings1()
        return self._multi_fetch

    @property
    def parse_parallel(self):
        if self.parallel is None:
            return False
        return self._parse_parallel

    @property
    def mmap_mode(self):
        if self._mmap_mode in [None, 'r']:
            return self._mmap_mode
        else:
            raise DataJointError('mmap mode must be "r" or None')

    @property
    def update_parents(self):
        if not isinstance(self._update_parents, bool):
            raise DataJointError('update parents must be boolean')
        return self._update_parents

    @property
    def wrap_columns(self):
        if self._wrap_columns is None:
            return []
        return self._wrap_columns.copy()
    @property
    def not_wrap_columns(self):
        if self._not_wrap_columns is None:
            return []
        return self._not_wrap_columns.copy()
    @property
    def from_tables(self):
        if self._from_tables is None:
            return None
        elif isinstance(self._from_tables, (dict, list)):
            return self._from_tables.copy()
    @property
    def transform_output(self):
        if not hasattr(self._transform_output, '__call__'):
            raise DataJointError("transform output must be a function")
        return self._transform_output
    #calls load_settings1 if None
    @property
    def global_settings(self):
        if self._global_settings is None:
            self.load_settings1()
        return self._global_settings.copy()
    @property
    def entry_settings(self):
        if self._entry_settings is None:
            self.load_settings1()
        return self._entry_settings.copy()
    @property
    def function(self):
        if self._function is None:
            self.load_settings1()
        return self._function
    @property
    def joined_columns(self):
        if self._joined_columns is None:
            columns = []
            for column in self.entry_settings.values():
                if isinstance(column, str):
                    columns.append(column)
                elif isinstance(column, (list, tuple)):
                    columns.extend(column)
                else:
                    raise DataJointError(
                        f"Value in entry settings wrong type : {type(column)}"
                    )
            self._joined_columns = np.unique(columns).tolist()
        return self._joined_columns.copy()

    @property
    def skip_tables(self):
        if self._skip_tables is None:
            return [self.approach_table.full_table_name]
        elif isinstance(self._skip_tables, list):
            return self._skip_tables + [self.approach_table.full_table_name]
        else:
            raise DataJointError('skip tables must be list')

    @property
    def restrict_tables(self):
        if self._restrict_tables is None:
            return None
        elif isinstance(self._restrict_tables, list):
            if self.always_deepjoin:
                return self._restrict_tables.copy()
            else:
                restrict_tables = []

                for table_name in self._restrict_tables:
                    table = self.get_table_class(table_name)
                    restrict_tables.append(table)

                return restrict_tables
        else:
            raise DataJointError('restrict tables must be list')
    #
    @property
    def joined_table(self):
        if self._joined_table is None:
            if self.restrict_tables is None or self.always_deepjoin:
                self._joined_table = self.deepjoin(
                        self.joined_columns, self.restrictions,
                        skip_self=True,
                        skip_tables=self.skip_tables,
                        restrict_tables=self.restrict_tables,
                        restrictions_to_columns=True,
                        from_tables=self.from_tables,
                        skip_endswith=('approach', 'strategy') #skip all strategy and approach tables
                    )
            else:
                self._joined_table = superjoin(
                        self.restrict_tables,
                        columns=self.additional_columns, #TODO fix issues here
                        restrictions=self.restrictions
                        )
        return self._joined_table

    @property
    def constant_restrictions(self):
        if isinstance(self._constant_restrictions, RESTRICTION_TYPES):
            pass
        else:
            raise DataJointError(
                "approach restriction is wrong "
                f"type: {type(self._approach_restriction)}"
            )
        return self._constant_restrictions.copy()

    @property
    def restrictions(self):
        if self._restrictions is None:
            return self.constant_restrictions
        elif isinstance(self._restrictions, RESTRICTION_TYPES):
            return join_restrictions(self._restrictions, self.constant_restrictions)

    @property
    def approach_restriction(self):
        if isinstance(self._approach_restriction, RESTRICTION_TYPES):
            pass
        else:
            raise DataJointError(
                "approach restriction is wrong "
                f"type: {type(self._approach_restriction)}"
            )
        return self._approach_restriction.copy()

    @property
    def approach_table(self):
        if self._approach_table is not None:
            return self._approach_table
        else:
            #fetch parent tables
            parent_tables = self.parent_tables()
            #loop through parent tables
            for parent_table in parent_tables:
                table_name = parent_table.full_table_name.replace('`', '')
                #will not check for multiple approach table names
                if table_name.endswith('approach'):
                    self._approach_table = parent_table
                    return self._approach_table
            raise DataJointError("No approach table found")

    @property
    def approach_table_primary_key(self):
        primary_key = self.approach_table.heading.primary_key
        if len(primary_key) != 1:
            raise DataJointError("approach table may only have one primary key")
        return primary_key[0]

    @property
    def strategy_table(self):
        if self._strategy_table is None:
            #fetch strategy table from approach table
            self._strategy_table = self.get_table_class(
                self.approach_table.full_table_name.replace('approach', 'strategy')
            )
        return self._strategy_table
