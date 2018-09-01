"""base mixin class to add functionality to datajoint
"""

from .standard_attributes import COMMON_COLUMN_NAMES, SUPER_COLUMN_NAMES, JSON
from warnings import simplefilter

simplefilter('once')

class BaseMixin:
    """Base mixin class to add to datajoint classes
    """
    _schema_module = None

    def sproj(self, *args, **kwargs):
        """super projection - project all columns except if they are
        in the super column names : {}

        Parameters
        ----------
        args : tuple
            tuple of super_column_names to add to the projection,
            if desired.
        kwargs : dict
            dictionary of mapping to rename columns; see datajoint
            proj function.
        """.format(SUPER_COLUMN_NAMES)
        columns = self.heading.names
        columns = set(columns) - SUPER_COLUMN_NAMES
        columns = columns | set(args)
        return self.proj(*columns, **kwargs)

    def jproj(self, **kwargs):
        """json projection - project all columns except the json column named 'json'
        """
        columns = set(self.heading.names) - set([JSON])
        return self.proj(*columns, **kwargs)

    def cproj(self, *args,  **kwargs):
        """common name projection - project all columns except if they are
        in the super column names : {}
        """.format(COMMON_COLUMN_NAMES)
        columns = self.heading.names
        columns = set(columns) - COMMON_COLUMN_NAMES
        columns = columns | set(args)
        return self.proj(*columns, **kwargs)
