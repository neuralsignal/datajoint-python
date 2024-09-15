from nose.tools import assert_true, assert_false, assert_equal, assert_list_equal, raises
import datajoint as dj
from pymysql.err import OperationalError
import numpy as np
'''
TODO: Use conn_info defined in init of the tests folder to initialize config
'''
class TestAutoImportCompute:
    @staticmethod
    def test_autoimported():
        dj.config['database.host'] = '127.0.0.1'
        dj.config['database.user'] = 'root'
        dj.config['database.password'] = 'simple'
        dj.config['enable_python_native_blobs'] = True
        dj.config['enable_python_pickle_blobs'] = True
        dj.config['enable_automakers'] = True
        dj.config['tmp_folder'] = '.'
        schema = dj.schema('tutorial')
        dj.conn()
        @schema
        class Multi2(dj.AutoImported):
            definition = """
            id : smallint
            ---
            help : longblob
            help2 : longblob
            help3 : longblob
            help4 = null : longblob
            """
        try:
            Multi2.insert([{'id':9, 'help':'12341234', 'help2':12354, 'help3':'asdfksajdfljk', 'help4':'asdf'}])
        except Exception as e:
            assert_true((isinstance(e,dj.errors.DataJointError)))
            assert_true(np.shape(Multi2().fetch())[0]==0)
        if(np.shape(Multi2.settings_table.fetch())[0]<1):
            Multi2.settings_table.insert1({'settings_name':'std1', 'func':np.std,'global_settings':{'keepdims':False}})
            assert_true(np.shape(Multi2().fetch())[0]>0)
    @staticmethod
    def test_autocomputed():
        dj.config['database.host'] = '127.0.0.1'
        dj.config['database.user'] = 'root'
        dj.config['database.password'] = 'simple'
        dj.config['enable_python_native_blobs'] = True
        dj.config['enable_python_pickle_blobs'] = True
        dj.config['enable_automakers'] = True
        dj.config['tmp_folder'] = '.'
        schema = dj.schema('tutorial')
        dj.conn()
        @schema
        class Multi3(dj.AutoComputed):
            definition = """
            id : smallint
            ---
            help : longblob
            help2 : longblob
            help3 : longblob
            help4 = null : longblob
            """
        try:
            Multi3.insert([{'id':9, 'help':'12341234', 'help2':12354, 'help3':'asdfksajdfljk', 'help4':'asdf'}])
        except Exception as e:
            assert_true((isinstance(e,dj.errors.DataJointError)))
            assert_true(np.shape(Multi3().fetch())[0]==0)
        if(np.shape(Multi3.settings_table.fetch())[0]<1):
            Multi3.settings_table.insert1({'settings_name':'std1', 'func':np.std,'global_settings':{'keepdims':False}})
        assert_true(np.shape(Multi3().fetch())[0]==0)

#driver code
TestAutoImportCompute.test_autoimported()
TestAutoImportCompute.test_autocomputed()
