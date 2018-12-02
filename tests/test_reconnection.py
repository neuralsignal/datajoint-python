"""
Collection of test cases to test connection module.
"""

from nose.tools import assert_true, assert_false, assert_equal, raises
import datajoint as dj
import numpy as np
from datajoint import DataJointError
from . import CONN_INFO, PREFIX



class TestReconnect:
    """
    test reconnection
    """

    @classmethod
    def setup_class(cls):
        cls.conn = dj.conn(reset=True, **CONN_INFO)

    def test_close(self):
        assert_true(self.conn.is_connected, "Connection should be alive")
        self.conn.close()
        assert_false(self.conn.is_connected, "Connection should now be closed")


    def test_reconnect(self):
        assert_true(self.conn.is_connected, "Connection should be alive")
        self.conn.close()
        self.conn.query('SHOW DATABASES;', reconnect=True).fetchall()
        assert_true(self.conn.is_connected, "Connection should be alive")


    @raises(DataJointError)
    def reconnect_throws_error_in_transaction(self):
        assert_true(self.conn.is_connected, "Connection should be alive")
        self.conn.close()
        with self.conn.transaction:
            self.conn.query('SHOW DATABASES;', reconnect=True).fetchall()
