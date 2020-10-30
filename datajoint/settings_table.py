"""Defines settings table class for automaker
"""

import importlib
import inspect
import collections
import warnings
import sys
import pickle
import os
from pathlib import Path
import json
import numpy as np
import pandas

try:
    import cloudpickle
    pickle.dumps = cloudpickle.dumps
except ImportError:
    pass

from .table import FreeTable
from .user_tables import UserTable, _base_regexp
from .expression import QueryExpression
from .utils import ClassProperty, from_camel_case, safe_write
from .errors import DataJointError
from .settings import config, prefix_to_role


if sys.version_info[1] < 6:
    dict = collections.OrderedDict

Sequence = (collections.MutableSequence, tuple, set)
# which keys to keep in conda list return
CONDA_KEEP = ['name', 'version']
# major python version
PYTHON_VERSION = (sys.version_info[0], sys.version_info[1])


class Settingstable(UserTable):
    """settings table class
    """

    _prefix = r'##'
    tier_regexp = r'(?P<settingstable>' + _prefix + _base_regexp + ')'

    @property
    def definition(self):
        return """
        {table_name}_populate_settings : varchar(63)
        ---
        description = null : varchar(4000) # any string to describe setting
        func : longblob # two-tuple of strings (module, function) or callable
        global_settings : longblob # dictionary
        entry_settings : longblob # dictionary
        fetch_method = 'fetch1' : enum('fetch', 'fetch1', 'farfetch', 'farfetch1')
        fetch_tables = null : longblob # dictionary of dict(table_name: projection)
        assign_output = null : longblob # overrides make_compatible to simply assign the output to a column
        restrictions = null : longblob # dictionary or list of restrictions
        parse_unique = null : longblob # list of unique entries for fetch
        created = CURRENT_TIMESTAMP : timestamp
        """

    @ClassProperty
    def child_table(cls):
        raise NotImplementedError('child table attribute for settings table.')

    @property
    def settings_name(self):
        return self.heading.primary_key[0]

    @staticmethod
    def _check_settings(settings, params, args, kwargs):
        """check if global/entry settings key in function
        """

        if settings is None:
            return {}

        assert isinstance(settings, collections.Mapping), \
            'global_settings must be dictionary'

        for param, value in settings.items():
            if param not in params:
                if param == args:
                    assert isinstance(value, Sequence), (
                        'variable positional (*args) must be sequence, '
                        'but is {}'.format(type(value)))
                elif param == kwargs:
                    assert isinstance(value, collections.Mapping), (
                        'keyword positional (**kwargs) must be mapping, '
                        'but is {}'.format(type(value))
                    )
                elif kwargs is not None:
                    warnings.warn(
                        'unknown keyword argument is being used, '
                        'but variable keyword (**kwargs) exists.'
                    )
                else:
                    raise DataJointError(
                        'global argument {} not in function'.format(param)
                    )

        return settings

    @staticmethod
    def _used_params(global_settings, entry_settings):
        return list(global_settings) + list(entry_settings)

    def _get_required_proj(self, entry_settings):
        """
        """

        required_proj = []

        for value in entry_settings.values():
            if isinstance(value, str):
                required_proj.append(value)
            elif isinstance(value, Sequence):
                required_proj.extend(value)
            elif isinstance(value, dict):
                required_proj.extend(self._get_required_proj(value))

        return required_proj

    def _check_fetch_tables(self, fetch_tables, required_proj):
        """check if fetch tables are correct and return formatted fetch tables
        """

        if fetch_tables is None:
            # default will be to use just the primary parent tables,
            # excluding settings table
            parent_tables = self.child_table().primary_parents(required_proj)

            if parent_tables is None and not required_proj:
                pass
            else:
                assert parent_tables is not None, \
                    'no parent tables, but required projections.'

                required_left = (
                    set(required_proj) - set(parent_tables.heading.names)
                )
                assert not required_left, \
                    'parent table do not contain all required projections.'

            return

        if isinstance(fetch_tables, QueryExpression):
            warnings.warn(
                'inserting query expression for fetch tables not tested.'
            )

            required_left = (
                set(required_proj) - set(fetch_tables.heading.names)
            )

            assert not required_left, \
                'query expression does not contains all required projections.'

            # insert_fetch_tables = fetch_tables
            insert_fetch_tables = pickle.dumps(
                fetch_tables, protocol=pickle.HIGHEST_PROTOCOL
            )

        elif isinstance(fetch_tables, collections.Mapping):
            assert len(fetch_tables) != 0, \
                'fetch tables cannot be empty mapping, use None instead.'
            # load graph
            self.connection.dependencies.load()
            nodes = self.connection.dependencies.nodes
            # load descendents
            descendants = self.child_table().descendants()

            # initialize new dictionary
            insert_fetch_tables = {}
            # initialize joined table to test
            test_joined_table = None

            for table, proj in fetch_tables.items():
                # if not string assume if table instance or class
                if not isinstance(table, str):
                    table = table.full_table_name
                # assume if ` in table then it is in proper notation
                if table.startswith('`'):
                    if table not in nodes:
                        raise DataJointError(
                            'table {} not in database.'.format(table)
                        )
                elif '.' not in table:
                    raise DataJointError(
                        'schema not specified or not separated with a period '
                        'in table name {}.'.format(table)
                    )
                else:
                    # assumed to be in camel case here.
                    table_splits = table.split('.')
                    schema = table_splits.pop(0)

                    # join part tables if necessary
                    table_name = '__'.join([
                        from_camel_case(s) for s in table_splits
                    ])

                    q = '`'
                    combiner = lambda u: (
                        q + schema + q + '.'
                        + q + u + table_name + q)

                    # check manual, imported and computed, autocomputed, autoimported
                    for prefix in prefix_to_role:
                        if combiner(prefix) in nodes:
                            table = combiner(prefix)
                            break
                    else:
                        raise DataJointError((
                            'When processing fetch tables, '
                            'table {table_name} in {schema} was not found.'
                        ).format(table_name=table_name, schema=schema))

                free_table = FreeTable(self.connection, table)
                if free_table.full_table_name in descendants:
                    raise DataJointError((
                        'Cannot have descendant tables in joined table. '
                        'Table {table_name} is a descendant table of '
                        '{child_table_name}.'
                    ).format(
                        table_name=free_table.full_table_name,
                        child_table_name=self.child_table.full_table_name
                    ))

                if isinstance(proj, tuple):
                    assert len(proj) == 2, 'projection must be two-tuple.'

                    assert isinstance(proj[0], Sequence), \
                        'first tuple must be sequence'

                    assert isinstance(proj[1], collections.Mapping), \
                        'second tuple must be mapping'

                    try:
                        free_table.proj(*proj[0], **proj[1])
                    except DataJointError as e:
                        raise DataJointError((
                            'Unable to project table {table}; error: {e}'
                            ).format(table=table, e=e))
                elif isinstance(proj, Sequence):
                    proj = (proj, {})
                elif isinstance(proj, collections.Mapping):
                    proj = ([], proj)
                else:
                    raise DataJointError(
                        'projection must be of type two-tuple, sequence, '
                        'or mapping, but is of type {}'.format(type(proj))
                    )

                try:
                    proj_table = free_table.proj(*proj[0], **proj[1])
                except DataJointError as e:
                    raise DataJointError((
                        'Unable to project table {table}; error: {e}'
                        ).format(table=table, e=e))

                if test_joined_table is None:
                    test_joined_table = proj_table
                else:
                    test_joined_table = test_joined_table * proj_table

                insert_fetch_tables[table] = proj

            required_left = (
                set(required_proj) - set(test_joined_table.heading.names)
            )

            assert not required_left, \
                'joined table does not contain all required projections.'

        return insert_fetch_tables

    def _get_joined_table(self, fetch_tables, required_proj, restrictions):
        """convert fetch_tables to joined table and set fetch_tables_attribute
        """

        if fetch_tables is None:
            # default will be to use just the primary parent tables,
            # excluding settings table
            parent_tables = self.child_table().primary_parents(
                required_proj, restrictions
            )
            return parent_tables

        elif isinstance(fetch_tables, QueryExpression):
            if restrictions is None:
                return fetch_tables
            else:
                return fetch_tables & restrictions

        elif isinstance(fetch_tables, bytes):
            fetch_tables = pickle.loads(fetch_tables)
            if restrictions is None:
                return fetch_tables
            else:
                return fetch_tables & restrictions

        else:
            # load graph
            self.connection.dependencies.load()
            nodes = self.connection.dependencies.nodes

            joined_table = None

            for table, proj in fetch_tables.items():

                if table not in nodes:
                    raise DataJointError(
                        'previously existing table '
                        '{} has been removed'.format(table)
                    )

                proj_table = FreeTable(
                    self.connection, table
                ).proj(*proj[0], **proj[1])

                if joined_table is None:
                    joined_table = proj_table
                else:
                    joined_table = joined_table * proj_table

            if restrictions is None:
                return joined_table
            else:
                return joined_table & restrictions

    @staticmethod
    def _get_func(func):
        """get function
        """

        def func_from_tuple(func):
            """get function from tuple
            """

            module = func[0]
            func = func[1]

            # use importlib to import module
            try:
                module = importlib.import_module(module)
                return getattr(module, func)
            except Exception as e:
                raise DataJointError(
                    'could not load function: {}'.format(e))

        if isinstance(func, (tuple, list)):
            if len(func) == 4:
                # if tuple is of length four it is considered a class
                # with initialization
                cls = func_from_tuple(func)
                args = func[2]
                kwargs = func[3]
                assert isinstance(args, Sequence)
                assert isinstance(args, collections.Mapping)
                assert hasattr(cls, '__init__')
                func = cls(*args, **kwargs)
            elif len(func) == 2:
                # here it is simply considered a function
                if isinstance(func[0], bytes):
                    # in this case it is a python file that was saved
                    if config['tmp_folder'] is None:
                        tmp_folder = os.path.abspath('.')
                    else:
                        tmp_folder = os.path.abspath(config['tmp_folder'])

                    filename, content = func[0].split(b'\0', 1)
                    filename = filename.decode()
                    content = content.decode()

                    temporary_path = Path(tmp_folder) / filename

                    safe_write(temporary_path, content)

                    func = (
                        os.path.splitext(filename)[0],
                        func[1]
                    )

                    # add to sys path if necessary
                    if tmp_folder not in sys.path:
                        try:
                            # put after current directory
                            index = sys.path.index('')
                            sys.path.insert(index+1, tmp_folder)
                        except ValueError:
                            sys.path.append(tmp_folder)
                    func = func_from_tuple(func)

                else:
                    # a normal function was saved
                    func = func_from_tuple(func)
            else:
                raise DataJointError(
                    'tuple must have two or four '
                    'elements, it has {}'.format(len(func))
                )
        elif isinstance(func, bytes):
            # loading pickle function
            func = pickle.loads(func)

        return func

    @staticmethod
    def _check_func(func):

        if (
            not hasattr(func, '__call__')
            or inspect.isclass(func)
            or inspect.ismodule(func)
        ):
            raise DataJointError(
                'function must be two/four-tuple or callable function.'
            )

        module = inspect.getmodule(func)

        if module is None:
            raise DataJointError('module for function is None.')
        elif not hasattr(module, '__file__'):
            raise DataJointError(
                'module {} does not have an associated file'.format(module)
            )

    @staticmethod
    def _get_func_status(func):
        """
        get version and package for function
        get git status for function if it exists
        get conda environment list
        """

        # init func status dictionary
        func_status = {}
        func_status['python_version'] = PYTHON_VERSION

        module = _get_package_status(func_status, func)

        _get_conda_status(func_status)

        _get_git_status(func_status, module, func)

        return func_status

    def _check_func_status(self, func_dict):
        """
        check version and package for function
        check git status between inserted and current version
        check anaconda environment list
        """

        func = func_dict['func']
        func_status = self._get_func_status(func)

        new_sha1 = func_status.get('sha1', None)
        new_branch = func_status.get('branch', None)
        new_modified = func_status.get('modified', None)
        new_version = func_status.get('version', None)
        new_package = func_status.get('package', None)
        new_conda_list = func_status.get('conda_list', None)
        new_python_version = func_status.get('python_version', None)

        old_sha1 = func_dict.get('sha1', None)
        old_branch = func_dict.get('branch', None)
        old_modified = func_dict.get('modified', None)
        old_version = func_dict.get('version', None)
        old_package = func_dict.get('package', None)
        old_conda_list = func_dict.get('conda_list', None)
        old_python_version = func_dict.get('python_version', None)

        # check python version
        if (new_python_version is None) or (old_python_version is None):
            if config['automaker_warnings']:
                warnings.warn((
                    'No python version checking for function '
                    '"{func}" available.'
                ).format(func=func))
        elif new_python_version != old_python_version:
            if config['automaker_warnings']:
                warnings.warn((
                    'Old python version "{old_python_version}" '
                    'does not match current python '
                    'version "{new_python_version}"'
                ).format(
                    new_python_version=new_python_version,
                    old_python_version=old_python_version
                ))

        _check_package_status(
            func,
            new_package, old_package,
            new_version, old_version
        )

        _check_conda_status(
            func,
            new_conda_list, old_conda_list
        )

        _check_git_status(
            new_sha1, old_sha1,
            old_branch, new_branch,
            new_modified, old_modified
        )

    @staticmethod
    def _get_func_params(func):
        """get available function parameters
        """

        signature = inspect.signature(func)

        params = {}
        args = None
        kwargs = None
        pos_or_kw = False
        for arg, param in signature.parameters.items():

            if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                pos_or_kw = True

            if param.kind is inspect.Parameter.POSITIONAL_ONLY:
                raise DataJointError(
                    'function cannot have position only argument.'
                    'case for: {}'.format(arg)
                )
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                args = arg
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                kwargs = arg
            elif param.default is inspect._empty:
                params[arg] = None
            else:
                params[arg] = param.default

        if pos_or_kw and args is not None:
            raise DataJointError(
                'function cannot have arguments before '
                'variable positional (*args).'
            )

        return {
            'params': params,
            'args': args,
            'kwargs': kwargs}

    @staticmethod
    def _check_func_params(func_dict, used):
        """check function parameters

        :param func: callable function
        :param params: dictionary of inserted parameters
        :param args: inserted argument that is a variable positional (*args).
        :param kwargs: inserted argument that is the variable keyword (**kwargs).
        :param used: arguments used for function.
        """

        func = func_dict['func']
        params = func_dict.get('params', {})
        args = func_dict.get('args', None)
        kwargs = func_dict.get('kwargs', None)

        signature = inspect.signature(func)

        dictionary = dict(signature.parameters)

        if args is not None and args in used:
            try:
                dictionary.pop(args)
            except KeyError:
                raise DataJointError(
                    'function does not contain the '
                    '"{}" variable argument anymore.'.format(args))
        if kwargs is not None and kwargs in used:
            try:
                dictionary.pop(kwargs)
            except KeyError:
                raise DataJointError(
                    'function does not contain the '
                    '"{}" variable keyword argument anymore.'.format(kwargs))

        for param, default in params.items():
            # check if
            if param not in dictionary and param in used:
                raise DataJointError(
                    'function does not contain the '
                    '"{}" argument anymore'.format(param)
                )

            if (default != dictionary[param].default) and param not in used:
                raise DataJointError(
                    'unused argument '
                    '"{}" has changed its default value'.format(param)
                )

    @staticmethod
    def _get_insert_func(func):
        """get function ready for insertion
        """

        if isinstance(func, (tuple, list)):
            if len(func) == 2:
                # if python file encode it into bytes
                if func[0].endswith('.py'):
                    if not os.path.exists(func[0]):
                        raise DataJointError(
                            'python file {} does not exist.'.format(func[0])
                        )
                    # init path object
                    python_file = Path(func[0])

                    func = ((
                        str.encode(python_file.name)
                        + b'\0'
                        + python_file.read_bytes()
                    ), func[1])
        else:
            # make binary
            func = pickle.dumps(func, protocol=pickle.HIGHEST_PROTOCOL)

        return func

    def _get_func_attr(self, func):
        """convert to function attribute and check function
        """

        attr = {'func': self._get_insert_func(func)}

        func = self._get_func(attr['func'])
        self._check_func(func)
        attr.update(self._get_func_status(func))
        attr.update(self._get_func_params(func))

        return attr

    @staticmethod
    def _check_restrictions(restrictions):
        """check if restrictions are list or dict
        """

        if restrictions is None:
            pass
        elif not isinstance(restrictions, (list, dict, np.recarray, str)):
            raise DataJointError(
                'constant restriction for insertion must be '
                'list, dict, or recarray (or string).'
            )

    @staticmethod
    def _check_parse_unique(parse_unique, required_proj):
        """check parse unique
        """

        if parse_unique is None:
            pass
        elif not isinstance(parse_unique, Sequence):
            raise DataJointError(
                'the parse unique attribute must be a sequence.'
            )
        else:
            left_parse = set(parse_unique) - set(required_proj)

            assert not left_parse, (
                'parse may only contain elements that are attributes '
                'to be fetched. Also contains {left_parse}, and not '
                'just {required_proj}'
            ).format(left_parse=left_parse, required_proj=required_proj)

    def _check_assign_output(self, row):

        assign_output = row.get('assign_output', None)

        if (
            assign_output is not None
            and assign_output
            not in self.child_table.heading.secondary_attributes
        ):
            raise DataJointError((
                '"{}" not in secondary columns of table "{}".'
            ).format(assign_output, self.child_table.full_table_name))

    def insert(self, *args, **kwargs):
        raise NotImplementedError(
            'For a Settingstable class only'
            ' insert1 is implemented.')

    def insert1(self, row, **kwargs):

        # aliases for primary key
        primary_key = self.settings_name
        truth = primary_key in row
        if not truth and 'settings_name' in row:
            row[primary_key] = row.pop('settings_name')
        elif not truth and 'settings' in row:
            row[primary_key] = row.pop('settings')
        elif not truth and 'populate_settings' in row:
            row[primary_key] = row.pop('populate_settings')
        elif not truth:
            raise DataJointError(
                'Need to provide a name for settings; you can use the '
                'following key aliases in your dictionary for insertion: '
                '"settings_name", "settings", and "populate_settings".'
            )

        # remove inferred/default attributes
        row.pop('args', None)
        row.pop('kwargs', None)
        row.pop('created', None)

        row['func'] = self._get_func_attr(row['func'])

        row['global_settings'] = self._check_settings(
            row.get('global_settings', None),
            row['func']['params'],
            row['func']['args'],
            row['func']['kwargs']
        )

        row['entry_settings'] = self._check_settings(
            row.get('entry_settings', None),
            row['func']['params'],
            row['func']['args'],
            row['func']['kwargs']
        )

        # required to be contained in the final joined table
        required_proj = self._get_required_proj(row['entry_settings'])

        row['fetch_tables'] = self._check_fetch_tables(
            row.get('fetch_tables', None), required_proj)

        self._check_restrictions(row.get('restrictions', None))
        self._check_parse_unique(row.get('parse_unique', None), required_proj)
        self._check_assign_output(row)

        # not implemented farfetch
        if 'farfetch' in row.get('fetch_method', 'fetch1'):
            raise NotImplementedError('farfetch method.')

        return super().insert((row,), **kwargs)

    def fetch1(
        self, *args,
        check_function=True, get_joined_table=True,
        **kwargs
    ):

        row = super().fetch1(*args, **kwargs)

        return self._postfetch_processing(
            row, check_function, get_joined_table
        )

    def fetch(self, *args, **kwargs):
        warnings.warn('fetch in Settingstable class does not process each '
                      'tuple, use fetch1 instead.')
        return super().fetch(*args, **kwargs)

    def _postfetch_processing(
        self, row, check_function=True, get_joined_table=True
    ):
        # convert and check function
        row['func']['func'] = self._get_func(row['func']['func'])
        if check_function:
            self._check_func(row['func']['func'])
            self._check_func_status(row['func'])
            self._check_func_params(
                row['func'],
                used=self._used_params(
                    row['global_settings'], row['entry_settings']
                ),
            )
        row['args'] = row['func'].get('args', None)
        row['kwargs'] = row['func'].get('kwargs', None)
        row['func'] = row['func']['func']

        if get_joined_table:
            required_proj = self._get_required_proj(row['entry_settings'])
            # get joined tables / primary parent tables
            row['fetch_tables'] = self._get_joined_table(
                row['fetch_tables'],
                required_proj,
                row['restrictions']
            )

        return row


# -- misc helper functions -- #

def _get_package_status(func_status, func):
    """
    get package status (used by Settingstable._get_func_status)
    returns module
    """

    # attribute file check for module happend previously
    module = inspect.getmodule(func)

    # check if module is package
    if hasattr(module, '__package__'):
        # get package module
        module_name = getattr(module, '__package__').split('.')[0]
        func_status['package'] = module_name

        # __main__ may be skipped
        try:
            # load package module
            module = importlib.import_module(module_name)
            # possibly redundant
            module = inspect.getmodule(module)
        except ValueError:
            pass

    # check if module has a version
    if hasattr(module, '__version__'):
        func_status['version'] = getattr(module, '__version__')

    return module


def _check_package_status(
    func,
    new_package, old_package,
    new_version, old_version
):
    if not config['automaker_warnings']:
        return
    # check package and package version
    if (new_package is None) or (old_package is None):
        warnings.warn((
            'No package checking for function "{func}" available.'
        ).format(func=func))
    elif new_package != old_package:
        warnings.warn((
            'Old package "{old_package}" does not '
            'match with new package "{new_package}".'
        ).format(old_package=old_package, new_package=new_package))
    elif (new_version is None) or (old_version is None):
        warnings.warn((
            'No version checking for package "{package}" available.'
        ).format(package=old_package))
    elif new_version != old_version:
        warnings.warn((
            'Old version "{old_version}" for package "{package}" '
            'does not match new version "{new_version}"'
        ).format(
            package=old_package,
            old_version=old_version,
            new_version=new_version
        ))


def _get_conda_status(func_status):
    """
    get conda status (used by Settingstable._get_func_status)
    """

    # try importing conda
    try:
        import conda.cli.python_api as conda_api
    except ImportError:
        if config['automaker_warnings']:
            warnings.warn(
                'Did not perform getting conda status: '
                'Conda Python API not installed.'
            )
        return

    try:
        stdout, stderr, return_code = conda_api.run_command(
            conda_api.Commands.LIST, '--json', use_exception_handler=True
        )
    except Exception:
        if config['automaker_warnings']:
            warnings.warn(
                'Could not perform "conda list --json". '
                'No conda status test performed'
            )
        return

    if return_code:
        if config['automaker_warnings']:
            warnings.warn((
                'Could not get conda environment list; Error message:\n{stderr}'
                '\n{error}.'
            ).format(stderr=stderr, error=stdout['error']))
        return

    conda_list = pandas.DataFrame(json.loads(stdout))[CONDA_KEEP]

    func_status['conda_list'] = conda_list.to_dict('list')


def _check_conda_status(
    func,
    new_conda_list, old_conda_list
):
    if not config['automaker_warnings']:
        return
    # check conda list
    if (new_conda_list is None) or (old_conda_list is None):
        warnings.warn((
            'No conda checking for function "{func}" available.'
        ).format(func=func))
    elif new_conda_list != old_conda_list:
        new_conda_list = pandas.DataFrame(new_conda_list)
        old_conda_list = pandas.DataFrame(old_conda_list)
        # only use keys from old conda list
        # i.e. it doesn't care if new packages were installed
        merged_conda_list = pandas.merge(
            new_conda_list, old_conda_list,
            on='name',
            how='right',
            suffixes=('_new', '_old')
        )
        # packages missing in new version
        missing_packages = merged_conda_list['version_new'].isnull()
        if missing_packages.any():
            warnings.warn((
                'These packages, which were present when inserting '
                'this function, are missing in your current anaconda '
                'environment:\n{missing}'
            ).format(
                missing=merged_conda_list.loc[
                    missing_packages, 'name'
                ].tolist()
            ))
        # packages with different versions
        different_version = (
            merged_conda_list['version_old']
            != merged_conda_list['version_new']
        )
        if different_version.any():
            merged_conda_list = merged_conda_list[different_version]
            warnings.warn((
                'These packages have different versions from the time '
                'this function was inserted into the Settingstable:'
                '\n{different}'
            ).format(
                different=merged_conda_list
            ))


def _get_git_status(func_status, module, func):
    """
    get git status (used by Settingstable._get_func_status)
    """

    # try importing git module for git checking
    try:
        import git
    except ImportError:
        if config['automaker_warnings']:
            warnings.warn(
                'Did not perform getting git status: '
                'Git Python API not installed.')
        return

    # get directory name of module
    dir_name = os.path.dirname(inspect.getabsfile(module))
    module_path = dir_name

    while True:
        # set git path
        git_path = os.path.join(dir_name, '.git')
        # set git repo if path exists
        if os.path.exists(git_path):
            repo = git.Repo(git_path)
            sha1, branch = repo.head.commit.name_rev.split()
            # check if files were modified
            modified = (repo.git.status().find('modified') > 0)
            if modified and config['automaker_warnings']:
                warnings.warn(
                    'You have uncommited changes. '
                    'Consider committing before running populate.'
                )

            func_status.update({
                'sha1': sha1,
                'branch': branch,
                'modified': modified,
            })

            break

        parts = os.path.split(dir_name)

        if dir_name in parts:
            # only throw a warning if not package and versioning exists
            if (
                'package' not in func_status
                and 'version' not in func_status
                and config['automaker_warnings']
            ):
                warnings.warn((
                    'No git directory was found for module in {path} for '
                    'function {func}. '
                    'Implementation of git version control recommended.'
                ).format(path=module_path, func=func))

            break

        dir_name = parts[0]


def _check_git_status(
    new_sha1, old_sha1,
    old_branch, new_branch,
    new_modified, old_modified
):
    if not config['automaker_warnings']:
        return
    # check git commit
    if (new_sha1 is None) or (old_sha1 is None):
        return
    if old_branch != new_branch:
        warnings.warn((
            'Working in new branch {new_branch}. '
            'Old branch was {old_branch}'
        ).format(new_branch=new_branch, old_branch=old_branch))
    if new_sha1 != old_sha1:
        warnings.warn(
            'Git commits have occured since insertion.'
        )
    elif new_modified and not old_modified:
        warnings.warn(
            'Files have been modified since insertion.'
        )
