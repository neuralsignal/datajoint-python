import collections
from collections import OrderedDict
import itertools
import inspect
import platform
import numpy as np
import pandas as pd
import pymysql
import logging
import warnings
from pymysql import OperationalError, InternalError, IntegrityError
from . import config, DataJointError
from .declare import declare, add_columns, find_special_attributes
from .relational_operand import RelationalOperand, Subquery
from .blob import pack
from .utils import user_choice, to_camel_case
from .heading import Heading
from .settings import server_error_codes
from . import __version__ as version

logger = logging.getLogger(__name__)


class BaseRelation(RelationalOperand):
    """
    BaseRelation is an abstract class that represents a base relation, i.e. a table in the database.
    To make it a concrete class, override the abstract properties specifying the connection,
    table name, database, context, and definition.
    A Relation implements insert and delete methods in addition to inherited relational operators.
    """
    _heading = None
    _context = None
    database = None
    _log_ = None
    _external_table = None

    # -------------- required by RelationalOperand ----------------- #
    @property
    def schema_module(self):
        if not hasattr(self, '_schema_module'):
            return None
        return self._schema_module

    _special_attributes = None
    @property
    def special_attributes(self):
        if self._special_attributes is None:
            self._special_attributes = find_special_attributes(
                self.full_table_name, self.definition, self._context
            )
        return self._special_attributes.copy()


    @property
    def heading(self):
        """
        Returns the table heading. If the table is not declared, attempts to declare it and return heading.
        :return: table heading
        """
        if self._heading is None:
            self._heading = Heading()  # instance-level heading
        if not self._heading:  # lazy loading of heading
            self._heading.init_from_database(
                self.connection, self.database, self.table_name,
                self.special_attributes
            )
        return self._heading

    @property
    def context(self):
        return self._context

    def declare(self):
        """
        Use self.definition to declare the table in the database
        """
        try:
            sql, uses_external = declare(self.full_table_name, self.definition, self._context)
            if uses_external:
                # trigger the creation of the external hash lookup for the current schema
                external_table = self.connection.schemas[self.database].external_table
                sql = sql.format(external_table=external_table.full_table_name)
            self.connection.query(sql)
        except pymysql.OperationalError as error:
            # skip if no create privilege
            if error.args[0] == server_error_codes['command denied']:
                logger.warning(error.args[1])
            else:
                raise
        else:
            self._log('Declared ' + self.full_table_name)

    def add_columns(self):
        """
        """
        sql, uses_external = add_columns(self.full_table_name, self.definition, self.heading, self._context)
        if uses_external:
            external_table = self.connection.schemas[self.database].external_table
            sql = sql.format(external_table=external_table.full_table_name)
        self.connection.query(sql)


    @property
    def from_clause(self):
        """
        :return: the FROM clause of SQL SELECT statements.
        """
        return self.full_table_name

    def get_select_fields(self, select_fields=None):
        """
        :return: the selected attributes from the SQL SELECT statement.
        """
        return '*' if select_fields is None else self.heading.project(select_fields).as_sql

    def parents(self, primary=None):
        """
        :param primary: if None, then all parents are returned. If True, then only foreign keys composed of
            primary key attributes are considered.  If False, the only foreign keys including at least one non-primary
            attribute are considered.
        :return: dict of tables referenced with self's foreign keys
        """
        return self.connection.dependencies.parents(self.full_table_name, primary)

    def children(self, primary=None):
        """
        :param primary: if None, then all parents are returned. If True, then only foreign keys composed of
            primary key attributes are considered.  If False, the only foreign keys including at least one non-primary
            attribute are considered.
        :return: dict of tables with foreign keys referencing self
        """
        return self.connection.dependencies.children(self.full_table_name, primary)

    @property
    def is_declared(self):
        """
        :return: True is the table is declared in the database
        """
        return self.connection.query(
            'SHOW TABLES in `{database}` LIKE "{table_name}"'.format(
                database=self.database, table_name=self.table_name)).rowcount > 0

    @property
    def full_table_name(self):
        """
        :return: full table name in the database
        """
        return r"`{0:s}`.`{1:s}`".format(self.database, self.table_name)

    @property
    def _log(self):
        if self._log_ is None:
            self._log_ = Log(self.connection, database=self.database)
        return self._log_

    @property
    def external_table(self):
        if self._external_table is None:
            self._external_table = self.connection.schemas[self.database].external_table
        return self._external_table

    def insertp(self, rows, **kwargs):
        """insert data into self and its part tables
        :param rows: an iterable of pandas DataFrames or dicts
        """
        #TODO make insertp automatic if only pandas DataFrame is passed
        for row in rows:
            self.insert1p(row, **kwargs)

    def insert1p(self, rows, **kwargs):
        """insert data into self and its part tables.
        Does not check extra fields.
        :param rows: a pandas DataFrame or dict.
        """
        if isinstance(rows, pd.DataFrame):
            #does not test if master input is unique
            columns = set(self.heading) & set(rows.columns)
            master_input = rows[list(columns)].iloc[0].dropna().to_dict()
            self.insert1(master_input, **kwargs)
            for part_table in self.part_tables():
                part_columns = set(part_table.heading) & set(rows.columns)
                part_input = rows[list(part_columns)]
                if part_input.isnull().values.any():
                    raise NotImplementedError('insert to part tables with nan values')
                part_input = part_input.to_dict('records')
                part_table.insert(part_input, **kwargs)
        elif isinstance(rows, dict):
            rows = pd.Series(rows)
            columns = set(self.heading) & set(rows.index)
            master_input = rows[list(columns)].dropna().to_dict()
            self.insert1(master_input, **kwargs)
            for part_table in self.part_tables():
                part_columns = set(part_table.heading) & set(rows.index)
                part_input = rows[list(part_columns)].dropna().to_dict()
                part_table.insert1(part_input, **kwargs)
        else:
            raise DataJointError(f"Rows must pandas dataframe or dict not {type(rows)}")

    def insert1(self, row, **kwargs):
        """
        Insert one data record or one Mapping (like a dict).
        :param row: a numpy record, a dict-like object, or an ordered sequence to be inserted as one row.
        For kwargs, see insert()
        """
        self.insert((row,), **kwargs)

    def insert(self, rows, replace=False, skip_duplicates=False, ignore_extra_fields=False, ignore_errors=False):
        """
        Insert a collection of rows.

        :param rows: An iterable where an element is a numpy record, a dict-like object, or an ordered sequence.
            rows may also be another relation with the same heading.
        :param replace: If True, replaces the existing tuple.
        :param skip_duplicates: If True, silently skip duplicate inserts.
        :param ignore_extra_fields: If False, fields that are not in the heading raise error.

        Example::
        >>> relation.insert([
        >>>     dict(subject_id=7, species="mouse", date_of_birth="2014-09-01"),
        >>>     dict(subject_id=8, species="mouse", date_of_birth="2014-09-02")])
        """

        if ignore_errors:
            warnings.warn('Use of `ignore_errors` in `insert` and `insert1` is deprecated. Use try...except... '
                          'to explicitly handle any errors', stacklevel=2)

        heading = self.heading
        if isinstance(rows, RelationalOperand):
            # insert from select
            if not ignore_extra_fields:
                try:
                    raise DataJointError(
                        "Attribute %s not found.  To ignore extra attributes in insert, set ignore_extra_fields=True." %
                        next(name for name in rows.heading if name not in heading))
                except StopIteration:
                    pass
            fields = list(name for name in heading if name in rows.heading)
            query = '{command} INTO {table} ({fields}) {select}{duplicate}'.format(
                command='REPLACE' if replace else 'INSERT',
                fields='`' + '`,`'.join(fields) + '`',
                table=self.full_table_name,
                select=rows.make_sql(select_fields=fields),
                duplicate=(' ON DUPLICATE KEY UPDATE `{pk}`=`{pk}`'.format(pk=self.primary_key[0])
                           if skip_duplicates else ''))
            self.connection.query(query)
            return

        if heading.attributes is None:
            logger.warning('Could not access table {table}'.format(table=self.full_table_name))
            return

        field_list = None  # ensures that all rows have the same attributes in the same order as the first row.

        def make_row_to_insert(row):
            """
            :param row:  A tuple to insert
            :return: a dict with fields 'names', 'placeholders', 'values'
            """

            def make_placeholder(name, value):
                """
                For a given attribute `name` with `value`, return its processed value or value placeholder
                as a string to be included in the query and the value, if any, to be submitted for
                processing by mysql API.
                :param name:
                :param value:
                """
                if ignore_extra_fields and name not in heading:
                    return None
                if heading[name].is_external:
                    if value is None:
                        placeholder, value = 'NULL', None
                    else:
                        placeholder, value = '%s', self.external_table.put(heading[name].type, value)
                elif heading[name].is_blob:
                    if value is None:
                        placeholder, value = 'NULL', None
                    else:
                        placeholder, value = '%s', pack(value)
                elif heading[name].is_jsonstring:
                    if value is None:
                        placeholder, value = 'NULL', None
                    elif isinstance(value, str):
                        placeholder = '%s'
                        try:
                            eval_value = eval(value)
                        except:
                            raise DataJointError(f'Cannot evaluate jsonstring {value}')
                        if not isinstance(eval_value, dict):
                            raise DataJointError(f'jsonstring not dict, but {type(eval_value)}')
                        if set(eval_value) & set(heading):
                            raise DataJointError('jsonstring contains values in heading.')
                    elif isinstance(value, dict):
                        placeholder = '%s'
                        if set(value) & set(heading):
                            raise DataJointError('jsonstring contains values in heading.')
                        value = str(value)
                    else:
                        raise DataJointError(f'jsonstring attribute wrong type {type(value)}')
                elif heading[name].is_liststring:
                    if value is None:
                        placeholder, value = 'NULL', None
                    elif isinstance(value, str):
                        placeholder = '%s'
                        try:
                            eval_value = eval(value)
                        except:
                            raise DataJointError(f'Cannot evaluate jsonstring {value}')
                        if not isinstance(eval_value, list):
                            raise DataJointError(f'liststring not dict, but {type(eval_value)}')
                    elif isinstance(value, list):
                        placeholder = '%s'
                        value = str(value)
                    else:
                        raise DataJointError(f'liststring attribute wrong type {type(value)}')
                elif heading[name].numeric:
                    if value is None or value == '' or np.isnan(np.float(value)):  # nans are turned into NULLs
                        placeholder, value = 'NULL', None
                    else:
                        placeholder, value = '%s', (str(int(value) if isinstance(value, bool) else value))
                else:
                    if value is None or value == '':
                        placeholder, value = 'NULL', None
                    else:
                        placeholder = '%s'
                return name, placeholder, value

            def check_fields(fields):
                """
                Validates that all items in `fields` are valid attributes in the heading
                :param fields: field names of a tuple
                """
                if field_list is None:
                    if not ignore_extra_fields:
                        for field in fields:
                            if field not in heading:
                                raise KeyError(u'`{0:s}` is not in the table heading'.format(field))
                elif set(field_list) != set(fields).intersection(heading.names):
                    raise DataJointError('Attempt to insert rows with different fields')

            if isinstance(row, np.void):  # np.array
                check_fields(row.dtype.fields)
                attributes = [make_placeholder(name, row[name])
                              for name in heading if name in row.dtype.fields]
            elif isinstance(row, collections.abc.Mapping):  # dict-based
                check_fields(row)
                attributes = [make_placeholder(name, row[name]) for name in heading if name in row]
            else:  # positional
                try:
                    if len(row) != len(heading):
                        raise DataJointError(
                            'Invalid insert argument. Incorrect number of attributes: '
                            '{given} given; {expected} expected'.format(
                                given=len(row), expected=len(heading)))
                except TypeError:
                    raise DataJointError('Datatype %s cannot be inserted' % type(row))
                else:
                    attributes = [make_placeholder(name, value) for name, value in zip(heading, row)]
            if ignore_extra_fields:
                attributes = [a for a in attributes if a is not None]

            assert len(attributes), 'Empty tuple'
            row_to_insert = dict(zip(('names', 'placeholders', 'values'), zip(*attributes)))
            nonlocal field_list
            if field_list is None:
                # first row sets the composition of the field list
                field_list = row_to_insert['names']
            else:
                #  reorder attributes in row_to_insert to match field_list
                order = list(row_to_insert['names'].index(field) for field in field_list)
                row_to_insert['names'] = list(row_to_insert['names'][i] for i in order)
                row_to_insert['placeholders'] = list(row_to_insert['placeholders'][i] for i in order)
                row_to_insert['values'] = list(row_to_insert['values'][i] for i in order)

            return row_to_insert

        rows = list(make_row_to_insert(row) for row in rows)
        if rows:
            try:
                query = "{command} INTO {destination}(`{fields}`) VALUES {placeholders}{duplicate}".format(
                    command='REPLACE' if replace else 'INSERT',
                    destination=self.from_clause,
                    fields='`,`'.join(field_list),
                    placeholders=','.join('(' + ','.join(row['placeholders']) + ')' for row in rows),
                    duplicate=(' ON DUPLICATE KEY UPDATE `{pk}`=`{pk}`'.format(pk=self.primary_key[0])
                               if skip_duplicates else ''))
                self.connection.query(query, args=list(
                    itertools.chain.from_iterable((v for v in r['values'] if v is not None) for r in rows)))
            except (OperationalError, InternalError, IntegrityError) as err:
                if err.args[0] == server_error_codes['command denied']:
                    raise DataJointError('Command denied:  %s' % err.args[1]) from None
                elif err.args[0] == server_error_codes['unknown column']:
                    # args[1] -> Unknown column 'extra' in 'field list'
                    raise DataJointError(
                        '{} : To ignore extra fields, set ignore_extra_fields=True in insert.'.format(err.args[1])) from None
                elif err.args[0] == server_error_codes['duplicate entry']:
                    raise DataJointError(
                        '{} : To ignore duplicate entries, set skip_duplicates=True in insert.'.format(err.args[1])) from None
                else:
                    raise

    def delete_quick(self, get_count=False):
        """
        Deletes the table without cascading and without user prompt.
        If this table has populated dependent tables, this will fail.
        """
        query = 'DELETE FROM ' + self.full_table_name + self.where_clause
        self.connection.query(query)
        count = self.connection.query("SELECT ROW_COUNT()").fetchone()[0] if get_count else None
        self._log(query[:255])
        return count

    def delete(self, verbose=True):
        """
        Deletes the contents of the table and its dependent tables, recursively.
        User is prompted for confirmation if config['safemode'] is set to True.
        """
        already_in_transaction = self.connection.in_transaction
        safe = config['safemode']
        if already_in_transaction and safe:
            raise DataJointError('Cannot delete within a transaction in safemode. '
                                 'Set dj.config["safemode"] = False or complete the ongoing transaction first.')
        graph = self.connection.dependencies
        graph.load()
        delete_list = collections.OrderedDict()
        for table in graph.descendants(self.full_table_name):
            if not table.isdigit():
                delete_list[table] = FreeRelation(self.connection, table)
            else:
                raise DataJointError('Cascading deletes across renamed foreign keys is not supported.  See issue #300.')
                parent, edge = next(iter(graph.parents(table).items()))
                delete_list[table] = FreeRelation(self.connection, parent).proj(
                    **{new_name: old_name
                       for new_name, old_name in edge['attr_map'].items() if new_name != old_name})

        # construct restrictions for each relation
        restrict_by_me = set()
        restrictions = collections.defaultdict(list)
        # restrict by self
        if self.restriction:
            restrict_by_me.add(self.full_table_name)
            restrictions[self.full_table_name].append(self.restriction)  # copy own restrictions
        # restrict by renamed nodes
        restrict_by_me.update(table for table in delete_list if table.isdigit())  # restrict by all renamed nodes
        # restrict by tables restricted by a non-primary semijoin
        for table in delete_list:
            restrict_by_me.update(graph.children(table, primary=False))   # restrict by any non-primary dependents

        # compile restriction lists
        for table, rel in delete_list.items():
            for dep in graph.children(table):
                if table in restrict_by_me:
                    restrictions[dep].append(rel)   # if restrict by me, then restrict by the entire relation
                else:
                    restrictions[dep].extend(restrictions[table])   # or re-apply the same restrictions

        # apply restrictions
        for name, r in delete_list.items():
            if restrictions[name]:  # do not restrict by an empty list
                r.restrict([r.proj() if isinstance(r, RelationalOperand) else r
                            for r in restrictions[name]])
        if safe:
            print('About to delete:')

        if not already_in_transaction:
            self.connection.start_transaction()
        total = 0
        try:
            for r in reversed(list(delete_list.values())):
                count = r.delete_quick(get_count=True)
                total += count
                if (verbose or safe) and count:
                    print('{table}: {count} items'.format(table=r.full_table_name, count=count))
        except:
            # Delete failed, perhaps due to insufficient privileges. Cancel transaction.
            if not already_in_transaction:
                self.connection.cancel_transaction()
            raise
        else:
            assert not (already_in_transaction and safe)
            if not total:
                print('Nothing to delete')
                if not already_in_transaction:
                    self.connection.cancel_transaction()
            else:
                if already_in_transaction:
                    if verbose:
                        print('The delete is pending within the ongoing transaction.')
                else:
                    if not safe or user_choice("Proceed?", default='no') == 'yes':
                        self.connection.commit_transaction()
                        if verbose or safe:
                            print('Committed.')
                    else:
                        self.connection.cancel_transaction()
                        if verbose or safe:
                            print('Cancelled deletes.')

    def drop_quick(self):
        """
        Drops the table associated with this relation without cascading and without user prompt.
        If the table has any dependent table(s), this call will fail with an error.
        """
        if self.is_declared:
            query = 'DROP TABLE %s' % self.full_table_name
            self.connection.query(query)
            logger.info("Dropped table %s" % self.full_table_name)
            self._log(query[:255])
        else:
            logger.info("Nothing to drop: table %s is not declared" % self.full_table_name)

    def drop(self):
        """
        Drop the table and all tables that reference it, recursively.
        User is prompted for confirmation if config['safemode'] is set to True.
        """
        if self.restriction:
            raise DataJointError('A relation with an applied restriction condition cannot be dropped.'
                                 ' Call drop() on the unrestricted BaseRelation.')
        self.connection.dependencies.load()
        do_drop = True
        tables = [table for table in self.connection.dependencies.descendants(self.full_table_name)
                  if not table.isdigit()]
        if config['safemode']:
            for table in tables:
                print(table, '(%d tuples)' % len(FreeRelation(self.connection, table)))
            do_drop = user_choice("Proceed?", default='no') == 'yes'
        if do_drop:
            for table in reversed(tables):
                FreeRelation(self.connection, table).drop_quick()
            print('Tables dropped.  Restart kernel.')

    @property
    def size_on_disk(self):
        """
        :return: size of data and indices in bytes on the storage device
        """
        ret = self.connection.query(
            'SHOW TABLE STATUS FROM `{database}` WHERE NAME="{table}"'.format(
                database=self.database, table=self.table_name), as_dict=True).fetchone()
        return ret['Data_length'] + ret['Index_length']

    def show_definition(self):
        logger.warning('show_definition is deprecated.  Use describe instead.')
        return self.describe()

    def describe(self, printout=True):
        """
        :return:  the definition string for the relation using DataJoint DDL.
            This does not yet work for aliased foreign keys.
        """
        if self.full_table_name not in self.connection.dependencies:
            self.connection.dependencies.load()
        parents = self.parents()
        in_key = True
        definition = '# ' + self.heading.table_info['comment'] + '\n'
        attributes_thus_far = set()
        attributes_declared = set()
        for attr in self.heading.attributes.values():
            if in_key and not attr.in_key:
                definition += '---\n'
                in_key = False
            attributes_thus_far.add(attr.name)
            do_include = True
            for parent_name, fk_props in list(parents.items()):  # need list() to force a copy
                if attr.name in fk_props['attr_map']:
                    do_include = False
                    if attributes_thus_far.issuperset(fk_props['attr_map']):
                        # simple foreign key
                        parents.pop(parent_name)
                        if not parent_name.isdigit():
                            definition += '-> {class_name}\n'.format(
                                class_name=lookup_class_name(parent_name, self.context) or parent_name)
                        else:
                            # aliased foreign key
                            parent_name = list(self.connection.dependencies.in_edges(parent_name))[0][0]
                            lst = [(attr, ref) for attr, ref in fk_props['attr_map'].items() if ref != attr]
                            definition += '({attr_list}) -> {class_name}{ref_list}\n'.format(
                                attr_list=','.join(r[0] for r in lst),
                                class_name=lookup_class_name(parent_name, self.context) or parent_name,
                                ref_list=('' if len(attributes_thus_far) - len(attributes_declared) == 1
                                          else '(%s)' % ','.join(r[1] for r in lst)))
                            attributes_declared.update(fk_props['attr_map'])
            if do_include:
                attributes_declared.add(attr.name)
                name = attr.name.lstrip('_')  # for external
                definition += '%-20s : %-28s # %s\n' % (
                    name if attr.default is None else '%s=%s' % (name, attr.default),
                    '%s%s' % (attr.type, ' auto_increment' if attr.autoincrement else ''), attr.comment)
        if printout:
            print(definition)
        return definition

    def _update(self, attrname, value=None):
        """
            Updates a field in an existing tuple. This is not a datajoyous operation and should not be used
            routinely. Relational database maintain referential integrity on the level of a tuple. Therefore,
            the UPDATE operator can violate referential integrity. The datajoyous way to update information is
            to delete the entire tuple and insert the entire update tuple.

            Safety constraints:
               1. self must be restricted to exactly one tuple
               2. the update attribute must not be in primary key

            Example

            >>> (v2p.Mice() & key).update('mouse_dob',   '2011-01-01')
            >>> (v2p.Mice() & key).update( 'lens')   # set the value to NULL

        """
        if len(self) != 1:
            raise DataJointError('Update is only allowed on one tuple at a time')
        if attrname not in self.heading:
            raise DataJointError('Invalid attribute name')
        if attrname in self.heading.primary_key:
            raise DataJointError('Cannot update a key value.')

        attr = self.heading[attrname]

        if attr.is_external:
            placeholder, value = '%s', self.external_table.put(attr.type, value)
        elif attr.is_blob:
            value = pack(value)
            placeholder = '%s'
        elif attr.numeric:
            if value is None or np.isnan(np.float(value)):  # nans are turned into NULLs
                placeholder = 'NULL'
                value = None
            else:
                placeholder = '%s'
                value = str(int(value) if isinstance(value, bool) else value)
        else:
            placeholder = '%s'
        command = "UPDATE {full_table_name} SET `{attrname}`={placeholder} {where_clause}".format(
            full_table_name=self.from_clause,
            attrname=attrname,
            placeholder=placeholder,
            where_clause=self.where_clause)
        self.connection.query(command, args=(value, ) if value is not None else ())

    def update(self, update_dict):
        """update using a dictionary
        """
        for attr, value in update_dict.items():
            self._update(attr, value)

    @property
    def classname(self):
        return self.__class__.__name__

    def get_table_class(self, full_table_name):
        """get table class from arbitrary table name
        """
        self_class = self.classname
        name = full_table_name.replace('`', '').split('.')
        if '__' in name[1]:
            if name[1].startswith('__') and not '__' in name[1][2:]:
                part_name = None
                master_name = None
            else:
                part_name = to_camel_case(name[1].split('__')[-1])
                master_name = to_camel_case(name[1].split('__')[-2])
        else:
            part_name = None
            master_name = None
        #
        db = name[0]
        class_name = to_camel_case(name[1])
        c = self.context
        if db in c:
            #in other schema
            if master_name is None:
                return getattr(c[db], class_name)
            else:
                return getattr(getattr(c[db], master_name), part_name)
        elif class_name in c:
            #within the same schema
            return c[class_name]
        elif master_name in c and master_name is not None:
            return getattr(c[master_name], part_name)
        elif hasattr(self, class_name.replace(self_class, '')):
            return getattr(self, class_name.replace(self_class, ''))
        elif 'schema_' + db in c:
            if class_name in c['schema_'+db].__dict__['context']:
                return c['schema_'+db].__dict__['context'][class_name]
            elif master_name in c['schema_'+db].__dict__['context']:
                return getattr(c['schema_'+db].__dict__['context'][master_name], part_name)
        elif self.schema_module is None:
            raise DataJointError(f'{full_table_name} not in context of {self.full_table_name}')
        elif hasattr(self.schema_module, db):
            db_class = getattr(self.schema_module, db)
            if master_name is None:
                return getattr(db_class, class_name)
            else:
                return getattr(getattr(db_class, master_name), part_name)
        #
        raise DataJointError(f'{full_table_name} not in context of {self.full_table_name}')

    def dependents(self, skip_aliased=False, graph=None, as_free=True):
        """Return dependents of table
        """
        if graph is None:
            graph = self.connection.dependencies
            graph.load()
        table_dict = OrderedDict()
        for table in graph.descendants(self.full_table_name):
            if not table.isdigit() and table not in table_dict:
                if as_free:
                    table_dict[table] = FreeRelation(self.connection, table)
                else:
                    table_dict[table] = self.get_table_class(table)
            elif not skip_aliased and table.isdigit():
                child, edge = next(iter(graph.children(table).items()))
                if as_free:
                    table_ = FreeRelation(self.connection, child)
                else:
                    table_ = self.get_table_class(child)
                rename_proj = {
                    old_name : new_name
                    for new_name, old_name in edge['attr_map'].items()
                    if new_name != old_name
                }
                to_proj = set(table_.heading) - set(rename_proj.values())
                table_dict[table] = table_.proj(*to_proj, **rename_proj)
        return table_dict

    _part_tables = None
    def part_tables(self, graph=None):
        """return part table classes as list.
        It will not work with aliased part tables.
        """
        if self._part_tables is not None:
            return self._part_tables.copy()
        if graph is None:
            graph = self.connection.dependencies
            graph.load()
        table_dict = graph.children(self.full_table_name)
        self_table_name = self.full_table_name.replace('`', '')
        part_tables_list = []
        for other_table_name, info_dict in table_dict.items():
            other_table_name = other_table_name.replace('`', '')
            if other_table_name == self_table_name:
                continue
            elif self_table_name in other_table_name:
                part_name = other_table_name.replace(self_table_name, '')
                part_tables_list.append(getattr(self, to_camel_case(part_name)))
        self._part_tables = part_tables_list
        return part_tables_list

    _parent_tables = None
    _parent_tables_settings = None
    _to_rename = None
    def parent_tables(self, graph=None, only_primary=False, skip_aliased=False, no_rename=False):
        """return parent tables as list. If no_rename is True then return list of dictionaries
        of to rename columns as well.
        """
        parent_tables_settings = dict(
                only_primary=only_primary,
                skip_aliased=skip_aliased,
                no_rename=no_rename
            )
        if self._parent_tables is not None and (self._parent_tables_settings == parent_tables_settings):
            if no_rename:
                return self._parent_tables.copy(), self._to_rename.copy()
            else:
                return self._parent_tables.copy()
        self._parent_tables_settings = parent_tables_settings
        if graph is None:
            graph = self.connection.dependencies
            graph.load()
        to_rename_list = []
        parents_tables_list = []
        parents_dict = graph.parents(self.full_table_name)
        for table_name, info_dict in parents_dict.items():
            if only_primary and not info_dict['primary']:
                continue
            elif skip_aliased and info_dict['aliased']:
                continue
            elif info_dict['aliased']:
                parent, edge = next(iter(graph.parents(table_name).items()))
                table_ = self.get_table_class(parent)
                rename_proj = {
                    new_name : old_name
                    for new_name, old_name in edge['attr_map'].items()
                    if new_name != old_name
                }
                to_proj = set(table_.heading) - set(rename_proj.values())
                if no_rename:
                    parents_tables_list.append(table_)
                    to_rename_list.append(rename_proj)
                else:
                    parents_tables_list.append(table_.proj(*to_proj, **rename_proj))
                    to_rename_list.append({})
            else:
                parents_tables_list.append(self.get_table_class(table_name))
                to_rename_list.append({})
        self._parent_tables = parents_tables_list
        self._to_rename = to_rename_list
        if no_rename:
            return parents_tables_list, to_rename_list
        else:
            return parents_tables_list

    _child_tables = None
    _child_tables_settings = None
    def child_tables(self, graph=None, only_primary=True, exclude_tables=[]):
        """return child tables as list, excluding part tables, and aliased child tables.
        """
        child_tables_settings = dict(
                only_primary=only_primary,
                exclude_tables=exclude_tables
                )
        if not isinstance(exclude_tables, (str, tuple, list)):
            raise DataJointError('exclude tables must be of type tuple or list.')
        if self._child_tables is not None and (self._child_tables_settings == child_tables_settings):
            return self._child_tables.copy()
        self._child_tables_settings = child_tables_settings
        if graph is None:
            graph = self.connection.dependencies
            graph.load()
        child_tables_list = []
        children_dict = graph.children(self.full_table_name)
        self_table_name = self.full_table_name.replace('`', '')
        for other_table_name, info_dict in children_dict.items():
            full_table_name = other_table_name
            other_table_name = other_table_name.replace('`', '')
            if full_table_name in exclude_tables:
                continue
            if other_table_name == self_table_name:
                continue
            elif self_table_name in other_table_name:
                continue
            elif info_dict['aliased']:
                continue
            elif only_primary and not info_dict['primary']:
                continue
            elif other_table_name in exclude_tables:
                continue
            else:
                child_table = self.get_table_class(full_table_name)
                child_tables_list.append(child_table)
        self._child_tables = child_tables_list
        return child_tables_list

    _has_dependents = None
    _has_part_tables = None
    @property
    def has_dependents(self):
        """
        """
        if self._has_dependents is None:
            self._has_dependents = bool(self.children())
        return self._has_dependents

    @property
    def has_part_tables(self):
        """
        """
        if self._has_part_tables is None:
            self._has_part_tables = bool(self.part_tables())
        return self._has_part_tables

    def deepjoin(
        self, columns, restrictions={}, check_parts=True,
        graph=None, only_primary=False, skip_aliased=False,
        skip_self=False, check_parts_self=True, check_children=True,
        skip_tables=[], restrict_tables=None, from_tables=None
    ):
        """go back and deepjoin ancestors/upstream tables according
        to columns past. Does not work with part tables.

        Parameters
        ----------
        columns : list
            list of columns to look for
        restrictions : dict or list of dicts
            restrictions to apply to each table (i.e. &)
        check_parts : bool
            check if a table has a part table, if so add
            the necessary part table columns to the joined table
        graph : dj.Connection
            connection to the database. If None, it will use self
        only_primary : bool
            Whether to only look at primary keys upstream of the
            database hierarchy.
        skip_aliased : bool
            Whether to skip aliased table columns (i.e. where the
            name has been changed).
        skip_self : bool
            Whether to ignore the existence of columns in self. Will also
            check part tables of self if check_parts_self and not skip_self.
        check_children : bool
            Whether to check children table -- not self.
        skip_tables : list
            List of tables to skip (use full table name or class name).
            Will also skip the parents of this table.
        restrict_tables : list
            List of tables to join. If None, then no restriction
        from_tables : list or dict
            List of tables (full table name or class name) from which
            to take the columns (i.e. same length as columns variable).
            If dictionary mapping: columns -> table.

        Returns
        -------
        joined_table : dj.Relation
            datajoint relation with all the desired columns and
            primary keys joined together.
        """
        #TODO restriction columns should also be checked for
        def remove_column_helper(columns, proj_set):
            for iproj in proj_set:
                columns.remove(iproj)

        def create_proj_set(table, columns, from_tables, to_rename={}, table_name=None, full_table_name=None):
            #
            if table_name is None:
                table_name = to_camel_case(table.table_name)
            if full_table_name is None:
                full_table_name = table.full_table_name
            #
            proj_set = (
                (set(columns) & set(table.heading) - set(to_rename.values()))
                | (set(columns) & set(to_rename))
            )
            if from_tables is None:
                for iproj in proj_set:
                    columns.remove(iproj)
            elif isinstance(from_tables, list):
                for column, from_table in zip(columns, from_tables):
                    if table_name == from_table:
                        pass
                    elif full_table_name == from_table:
                        pass
                    else:
                        proj_set.discard(column)
                for iproj in proj_set:
                    raise NotImplementedError('from tables as list')
            elif isinstance(from_tables, dict):
                for column in (proj_set & set(from_tables)):
                    if table_name == from_tables[column]:
                        del(from_tables[column])
                    elif full_table_name == from_tables[column]:
                        del(from_tables[column])
                    else:
                        proj_set.remove(column)
                for iproj in proj_set:
                    columns.remove(iproj)
            return proj_set
        #
        def part_table_helper(
            parent_join, columns, restrictions, part_table, renames,
            graph=graph, only_primary=only_primary, skip_aliased=skip_aliased,
            from_tables=None, restrict_tables=restrict_tables,
            skip_tables=skip_tables
        ):
            #check if part_table is initialized
            try:
                part_table = part_table()
            except:
                pass
            #handling renamed foreign keys with part tables
            #non master parents not properly tested
            non_master_parents = [
                parent_table
                for parent_table in part_table.parent_tables(
                    graph=graph, only_primary=only_primary,
                    skip_aliased=skip_aliased
                )
                if isinstance(parent_table, Subquery)
            ]
            #
            part_to_rename = {
                new_name : old_name
                for new_name, old_name in renames.items()
                if old_name in (set(part_table.heading) & set(renames.values()))
            }
            part_to_proj = create_proj_set(part_table, columns, from_tables, part_to_rename)
            #
            # non master tables should be renamed already, as subquery
            proj_non_masters = []
            for non_master_parent in non_master_parents:
                #do not include nullables
                non_master_to_proj = set(columns) & set(non_master_parent.heading) - set(part_table.heading.nullables)
                if non_master_to_proj:
                    warnings.warn("non master parents to part table not tested.")
                    remove_column_helper(columns, non_master_to_proj)
                    #don't join unnecessary non master tables
                    non_master_parent = (non_master_parent & restrictions).proj(*non_master_to_proj)
                    part_to_proj |= (
                        set(non_master_parent.heading.primary_key) & set(part_table.heading)
                    )
                    proj_non_masters.append(non_master_parent)
            #
            #only if there is something to proj
            if part_to_proj:
                #Does not work with rename parts!
                part_table = (part_table & restrictions).proj(*part_to_proj, **part_to_rename)
                parent_join = parent_join * part_table
            else:
                #skip proj non masters if nothing to project - redundant
                return parent_join
            #
            for proj_non_master in proj_non_masters:
                parent_join = parent_join * proj_non_master
            return parent_join

        if graph is None:
            graph = self.connection.dependencies
            graph.load()
        if from_tables is not None:
            if isinstance(from_tables, list):
                assert len(from_tables) == len(columns)
            elif isinstance(from_tables, dict):
                if set(from_tables) - set(columns):
                    raise DataJointError('from_tables mapping contains keys not in columns variable.')
            else:
                raise DataJointError('from_tables must be dict or list')
        parents, rename_list = self.parent_tables(
            graph=graph, only_primary=only_primary,
            skip_aliased=skip_aliased, no_rename=True
        ) # also renames it appropriately
        if check_parts_self:
            part_tables = self.part_tables(graph=graph)
            for part_table in part_tables:
                try:
                    part_table = part_table()
                except:
                    pass
                part_parents, part_rename_list = part_table.parent_tables(
                    graph=graph, only_primary=only_primary,
                    skip_aliased=skip_aliased, no_rename=True
                )
                for part_parent, part_renames in zip(part_parents, part_rename_list):
                    if (part_parent not in parents) and (part_parent != self.__class__):
                        parents.append(part_parent)
                        rename_list.append(part_renames)
        #stop massive recursion by remove columns
        to_proj = set()
        if not skip_self:
            to_proj = create_proj_set(self, columns, from_tables)
        parent_joins = []
        for parent, renames in zip(parents, rename_list):
            #initialize, if not yet
            try:
                parent = parent()
            except:
                pass
            #
            if parent.full_table_name in skip_tables:
                continue
            elif parent.classname in skip_tables:
                continue
            if restrict_tables is not None:
                if parent.full_table_name in restrict_tables:
                    pass
                elif parent.classname in restrict_tables:
                    pass
                else:
                    continue
            parent_to_proj = set(parent.heading) - set(renames.values())
            #
            if check_parts:
                part_tables = parent.part_tables(graph=graph)
            else:
                part_tables = []
            if check_children:
                child_tables = parent.child_tables(graph=graph, exclude_tables=[self.full_table_name])
            else:
                child_tables = []
            #if renamed parent do not look deeper
            #if no columns left do not look deeper
            if not columns:
                break
            elif renames:
                parent_table_name = parent.classname
                parent_full_table_name = parent.full_table_name
                parent = parent.proj(*parent_to_proj, **renames)
                parent_to_proj = create_proj_set(
                    parent, columns, from_tables,
                    table_name=parent_table_name,
                    full_table_name=parent_full_table_name
                )
                parent_join = (parent & restrictions).proj(*parent_to_proj)
            else:
                parent_join = parent.deepjoin(
                    columns, restrictions,
                    check_parts=check_parts,
                    graph=graph, only_primary=only_primary,
                    skip_aliased=skip_aliased,
                    check_parts_self=False,
                    check_children=check_children,
                    skip_tables=skip_tables,
                    skip_self=False,
                    restrict_tables=restrict_tables
                )

            for part_table in part_tables:
                if part_table.full_table_name in skip_tables:
                    continue
                elif to_camel_case(part_table.table_name) in skip_tables:
                    continue
                if restrict_tables is not None:
                    if part_table.full_table_name in restrict_tables:
                        pass
                    elif to_camel_case(part_table.table_name) in skip_tables:
                        pass
                    else:
                        continue
                #
                parent_join = part_table_helper(
                    parent_join, columns, restrictions, part_table, renames
                )
            #
            for child_table in child_tables:
                if child_table.full_table_name in skip_tables:
                    continue
                elif to_camel_case(child_table.table_name) in skip_tables:
                    continue
                if restrict_tables is not None:
                    if child_table.full_table_name in restrict_tables:
                        pass
                    elif to_camel_case(child_table.table_name) in restrict_tables:
                        pass
                    else:
                        continue
                child_to_rename = {
                    new_name : old_name
                    for new_name, old_name in renames.items()
                    if old_name in (set(child_table.heading) & set(renames.values()))
                }
                child_to_proj = create_proj_set(child_table, columns, from_tables, child_to_rename)
                if child_to_proj:
                    #does not work with renamed names in child!
                    child_table = (child_table & restrictions).proj(*child_to_proj, **child_to_rename)
                    parent_join = parent_join * child_table
            #only join a parent if its primary key are not nullable in self
            #and are not in to_proj (if it is required projection you do
            #not want to include NULL)
            add_to_proj = (set(parent.heading.primary_key) & set(self.heading))
            dont_proj = add_to_proj & set(self.heading.nullables) - to_proj
            if dont_proj:
                continue
            elif not only_primary:
                to_proj |= add_to_proj
            #add to parents to be joined
            parent_joins.append(parent_join)
            #
        #add part tables of first self
        if check_parts_self:
            part_tables = self.part_tables(graph=graph)
        else:
            part_tables = []
        #
        if not parent_joins and not skip_self:
            #
            parent_join = (self & restrictions).proj(*to_proj)
            #join part tables
            for part_table in part_tables:
                parent_join = part_table_helper(
                    parent_join, columns, restrictions, part_table,
                    renames={}, from_table=from_tables
                )
            #
            return parent_join
        elif not parent_joins and skip_self:
            raise DataJointError('Empty deepjoin.')
        #
        for n, parent_join in enumerate(parent_joins):
            if n == 0:
                joined_table = parent_join
            else:
                joined_table = joined_table * parent_join
        if not skip_self:
            joined_table = (self & restrictions).proj(*to_proj) * joined_table
            #join part tables
            for part_table in part_tables:
                joined_table = part_table_helper(
                    joined_table, columns, restrictions, part_table,
                    renames={}, from_tables=from_tables
                )
        #
        return joined_table

    def deepfetch(self, *args, **kwargs):
        """far/deepfetch columns from upstream
        using deepjoin.
        """
        return self.deepjoin(*args, **kwargs).fetch()

    def fetchparts(self, pivot=False, **kwargs):
        """fetch a table with its part tables
        or just fetch the table if no part tables exist

        Returns
        -------
        fetched_table : numpy.recarray/list of OrderedDicts
        """
        part_tables = self.part_tables()
        if self.part_tables():
            if pivot:
                if len(part_tables) > 1:
                    raise DataJointError('pivot method with multiple part tables not implemented.')
                part_table = part_tables[0]
                joined_table = []
                primary_keys = self.heading.primary_key
                self_columns = set(self.heading)
                for entry in self:
                    key = {
                        primary_key : entry[primary_key]
                        for primary_key in primary_keys
                    }
                    part_dataframe = pd.DataFrame((part_table & key).fetch())
                    part_columns = set(part_dataframe.columns) - self_columns
                    part_dict = part_dataframe[list(part_columns)].to_dict('list')
                    entry.update(part_dict)
                    joined_table.append(entry)
                return joined_table
            else:
                joined_table = self
                for part_table in part_tables:
                    joined_table = joined_table * part_table
                return joined_table.fetch(**kwargs)
        else:
            return self.fetch(**kwargs)

def lookup_class_name(name, context, depth=3):
    """
    given a table name in the form `database`.`table_name`, find its class in the context.
    :param name: `database`.`table_name`
    :param context: dictionary representing the namespace
    :param depth: search depth into imported modules, helps avoid infinite recursion.
    :return: class name found in the context or None if not found
    """
    # breadth-first search
    nodes = [dict(context=context, context_name='', depth=depth)]
    while nodes:
        node = nodes.pop(0)
        for member_name, member in node['context'].items():
            if inspect.isclass(member) and issubclass(member, BaseRelation):
                if member.full_table_name == name:   # found it!
                    return '.'.join([node['context_name'],  member_name]).lstrip('.')
                try:  # look for part tables
                    parts = member._ordered_class_members
                except AttributeError:
                    pass  # not a UserRelation -- cannot have part tables.
                else:
                    for part in (getattr(member, p) for p in parts if p[0].isupper() and hasattr(member, p)):
                        if inspect.isclass(part) and issubclass(part, BaseRelation) and part.full_table_name == name:
                            return '.'.join([node['context_name'], member_name, part.__name__]).lstrip('.')
            elif node['depth'] > 0 and inspect.ismodule(member) and member.__name__ != 'datajoint':
                try:
                    nodes.append(
                        dict(context=dict(inspect.getmembers(member)),
                             context_name=node['context_name'] + '.' + member_name,
                             depth=node['depth']-1))
                except ImportError:
                    pass  # could not import, so do not attempt
    return None


class FreeRelation(BaseRelation):
    """
    A base relation without a dedicated class. Each instance is associated with a table
    specified by full_table_name.
    :param arg:  a dj.Connection or a dj.FreeRelation
    """

    def __init__(self, arg, full_table_name=None):
        super().__init__()
        if isinstance(arg, FreeRelation):
            # copy constructor
            self.database = arg.database
            self._table_name = arg._table_name
            self._connection = arg._connection
        else:
            self.database, self._table_name = (s.strip('`') for s in full_table_name.split('.'))
            self._connection = arg

    def __repr__(self):
        return "FreeRelation(`%s`.`%s`)" % (self.database, self._table_name)

    @property
    def table_name(self):
        """
        :return: the table name in the database
        """
        return self._table_name


class Log(BaseRelation):
    """
    The log table for each database.
    Instances are callable.  Calls log the time and identifying information along with the event.
    """

    def __init__(self, arg, database=None):
        super().__init__()

        if isinstance(arg, Log):
            # copy constructor
            self.database = arg.database
            self._connection = arg._connection
            self._definition = arg._definition
            self._user = arg._user
            return

        self.database = database
        self._connection = arg
        self._definition = """    # event logging table for `{database}`
        timestamp = CURRENT_TIMESTAMP : timestamp
        ---
        version  :varchar(12)   # datajoint version
        user     :varchar(255)  # user@host
        host=""  :varchar(255)  # system hostname
        event="" :varchar(255)  # custom message
        """.format(database=database)

        if not self.is_declared:
            self.declare()
        self._user = self.connection.get_user()

    @property
    def definition(self):
        return self._definition

    @property
    def table_name(self):
        return '~log'

    def __call__(self, event):
        try:
            self.insert1(dict(
                user=self._user,
                version=version + 'py',
                host=platform.uname().node,
                event=event), skip_duplicates=True, ignore_extra_fields=True)
        except DataJointError:
            logger.info('could not log event in table ~log')

    def delete(self):
        """bypass interactive prompts and cascading dependencies"""
        self.delete_quick()

    def drop(self):
        """bypass interactive prompts and cascading dependencies"""
        self.drop_quick()
