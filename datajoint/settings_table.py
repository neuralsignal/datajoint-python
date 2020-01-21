"""Defines settings table class for automaker
"""

import importlib
import inspect
import collections
import warnings
import sys
import os
import numpy as np

from .table import FreeTable
from .user_tables import UserTable, _base_regexp
from .expression import QueryExpression
from .utils import ClassProperty, from_camel_case
from .errors import DataJointError

if sys.version_info[1] < 6:
    dict = collections.OrderedDict

Sequence = (collections.MutableSequence, tuple, set)


class Settingstable(UserTable):
    """settings table class
    """

    _prefix = r'##'
    tier_regexp = r'(?P<settingstable>' + _prefix + _base_regexp + ')'

    definition = """
    settings_name : varchar(63)
    ---
    description = null : varchar(4000) # any string to describe setting
    func : longblob # two-tuple of strings (module, function) or callable
    global_settings : longblob # dictionary
    entry_settings : longblob # dictionary
    fetch_method = 'fetch1' : enum('fetch', 'fetch1', 'farfetch', 'farfetch1')
    fetch_tables = null : longblob # dictionary of dict(table_name: projection)
    restrictions = null : longblob # dictionary or list of restrictions
    parse_unique = null : longblob # list of unique entries for fetch
    created = CURRENT_TIMESTAMP : timestamp
    """

    @ClassProperty
    def child_table(cls):
        raise NotADirectoryError('child table attribute for settings table.')

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

    @staticmethod
    def _required_proj(entry_settings):
        """
        """

        required_proj = []

        for value in entry_settings.values():
            if isinstance(value, str):
                required_proj.append(value)
            elif isinstance(value, Sequence):
                required_proj.extend(value)

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

            insert_fetch_tables = fetch_tables

        elif isinstance(fetch_tables, collections.Mapping):
            assert len(fetch_tables) != 0, \
                'fetch tables cannot be empty mapping, use None instead.'
            # load graph
            self.connection.dependencies.load()
            nodes = self.connection.dependencies.nodes

            # initialize new dictionary
            insert_fetch_tables = {}
            # initialize joined table to test
            test_joined_table = None

            for table, proj in fetch_tables.items():
                # assume if ` in table then it is in proper notation
                if table.startswith('`'):
                    if table not in nodes:
                        raise DataJointError(
                            'table {} not in database.'.format(table)
                        )
                elif '.' not in table:
                    raise DataJointError(
                        'schema not specified or separated with a period '
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
                    if combiner('') in nodes:
                        table = combiner('')
                    elif combiner('_') in nodes:
                        table = combiner('')
                    elif combiner('__') in nodes:
                        table = combiner('__')
                    elif combiner('_#') in nodes:
                        table = combiner('_#')
                    elif combiner('#_') in nodes:
                        table = combiner('#_')
                    elif combiner('#') in nodes:
                        table = combiner('#')
                    else:
                        raise DataJointError((
                            'When processing fetch tables, '
                            'table {table_name} in {schema} was not found.'
                        ).format(table_name=table_name, schema=schema))

                free_table = FreeTable(self.connection, table)
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
                        'or mapping, but is'.format(type(proj))
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
                func = getattr(module, func)
            except Exception as e:
                raise DataJointError(
                    'could not load function: {}'.format(e))

        if isinstance(func, tuple):
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
                func = func_from_tuple(func)
            else:
                raise DataJointError(
                    'tuple must have two or four '
                    'elements, it has {}'.format(len(func))
                )

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
    def _get_git_status(func):
        """get git status for function if it exists
        """

        # attribute file check for module happend previously
        module = inspect.getmodule(func)
        # init git status dictionary
        git_status = {}

        # check if module is package
        if hasattr(module, '__package__'):
            # get package module
            module = getattr(module, '__package__').split('.')[0]
            git_status['package'] = module
            # load package module
            module = importlib.import_module(module)
            # possibly redundant
            module = inspect.getmodule(module)

        # check if module has a version
        if hasattr(module, '__version__'):
            git_status['version'] = getattr(module, '__version__')

        # try importing git module for git checking
        try:
            import git
        except (ImportError, ModuleNotFoundError):
            warnings.warn(
                'Did not perform getting git status: '
                'Git Python API not installed')
            return git_status

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
                if modified:
                    warnings.warn(
                        'You have uncommited changes. '
                        'Consider committing before running populate.'
                    )

                git_status.update({
                    'sha1': sha1,
                    'branch': branch,
                    'modified': modified,
                })

                break

            parts = os.path.split(dir_name)

            if dir_name in parts:
                # only throw a warning if not package and versioning exists
                if 'package' not in git_status and 'version' not in git_status:
                    warnings.warn((
                        'No git directory was found for module in {path} for '
                        'function {func}. '
                        'Implementation of git version control recommended.'
                    ).format(path=module_path, func=func))

                break

            dir_name = parts[0]

        return git_status

    def _check_git_status(self, func_dict):
        """check git status between inserted and current version
        """

        func = func_dict['func']
        git_status = self._get_git_status(func)

        old_sha1 = git_status.get('sha1', None)
        old_branch = git_status.get('branch', None)
        old_modified = git_status.get('modified', None)
        old_version = git_status.get('version', None)
        old_package = git_status.get('package', None)

        new_sha1 = func_dict.get('sha1', None)
        new_branch = func_dict.get('branch', None)
        new_modified = func_dict.get('modified', None)
        new_version = func_dict.get('version', None)
        new_package = func_dict.get('package', None)

        # check package and package version
        if (new_package is None) or (old_package is None):
            warnings.warn((
                'no package checking for function "{func}" available.'
            ).format(func=func))
        elif new_package != old_package:
            warnings.warn((
                'old package "{old_package}" does not '
                'match with new package "{new_package}".'
            ).format(old_package=old_package, new_package=new_package))
        elif (new_version is None) or (old_version is None):
            warnings.warn((
                'no version checking for package "{package}" available.'
            ).format(package=old_package))
        elif new_version != old_version:
            warnings.warn((
                'old version "{old_version}" for package "{package}" '
                'does not match new version "{new_version}"'
            ).format(
                package=old_package,
                old_version=old_version,
                new_version=new_version
            ))

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

    def _get_func_attr(self, func):
        """convert to function attribute and set self.func
        """

        attr = {'func': func}

        func = self._get_func(func)
        self._check_func(func)
        attr.update(self._get_git_status(func))
        attr.update(self._get_func_params(func))

        return attr

    @staticmethod
    def _check_restrictions(restrictions):
        """check if restrictions are list or dict
        """

        if restrictions is None:
            pass
        elif not isinstance(restrictions, (list, dict, np.recarray)):
            raise DataJointError(
                'constant restriction for insertion must be '
                'list, dict, or recarray.'
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

    def insert1(self, row, **kwargs):

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
        required_proj = self._required_proj(row['entry_settings'])

        row['fetch_tables'] = self._check_fetch_tables(
            row.get('fetch_tables', None), required_proj)

        self._check_restrictions(row.get('restrictions', None))
        self._check_parse_unique(row.get('parse_unique', None), required_proj)

        # not implemented farfetch
        if 'farfetch' in row.get('fetch_method', 'fetch1'):
            raise NotImplementedError('farfetch method.')

        return super().insert1(row, **kwargs)

    def fetch1(self, *args, check_function=True, **kwargs):

        row = super().fetch1(*args, **kwargs)

        # convert and check function
        row['func']['func'] = self._get_func(row['func']['func'])
        if check_function:
            self._check_func(row['func']['func'])
            self._check_git_status(row['func'])
            self._check_func_params(
                row['func'],
                used=self._used_params(
                    row['global_settings'], row['entry_settings']
                ),
            )
        row['args'] = row['func'].get('args', None)
        row['kwargs'] = row['func'].get('kwargs', None)
        row['func'] = row['func']['func']

        required_proj = self._required_proj(row['entry_settings'])
        # get joined tables / primary parent tables
        row['fetch_tables'] = self._get_joined_table(
            row['fetch_tables'],
            required_proj,
            row['restrictions']
        )

        return row
