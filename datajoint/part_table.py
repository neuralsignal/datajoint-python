"""Part table class
"""

from .utils import from_camel_case, ClassProperty
from .errors import DataJointError
from .user_tables import UserTable, Manual, Lookup, Imported, Computed, \
    _base_regexp
from .auto_tables import AutoComputed, AutoImported


class Part(UserTable):
    """
    Inherit from this class if the table's values are details of an entry in another relation
    and if this table is populated by this relation. For example, the entries inheriting from
    dj.Part could be single entries of a matrix, while the parent table refers to the entire matrix.
    Part relations are implemented as classes inside classes.
    """

    _connection = None
    _master = None

    tier_regexp = r'(?P<master>' + '|'.join(
        [c.tier_regexp for c in (Manual, Lookup, Imported, Computed, AutoComputed, AutoImported)]
    ) + r'){1,1}' + '__' + r'(?P<part>' + _base_regexp + ')'

    @ClassProperty
    def connection(cls):
        return cls._connection

    @ClassProperty
    def full_table_name(cls):
        return None if cls.database is None or cls.table_name is None else r"`{0:s}`.`{1:s}`".format(
            cls.database, cls.table_name)

    @ClassProperty
    def master(cls):
        return cls._master

    @ClassProperty
    def table_name(cls):
        return None if cls.master is None else cls.master.table_name + '__' + from_camel_case(cls.name)

    def delete(self, force=False):
        """
        unless force is True, prohibits direct deletes from parts.
        """
        if force:
            super().delete()
        else:
            raise DataJointError('Cannot delete from a Part directly.  Delete from master instead')

    def drop(self, force=False):
        """
        unless force is True, prohibits direct deletes from parts.
        """
        if force:
            super().drop()
        else:
            raise DataJointError('Cannot drop a Part directly.  Delete from master instead')
