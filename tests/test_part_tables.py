from nose.tools import assert_true, assert_false, assert_equal, assert_list_equal, raises
import datajoint as dj
from pymysql.err import OperationalError
import numpy as np


'''
TODO: Use conn_info defined in init of the tests folder to initialize config
'''

class TestPartTables:

    @staticmethod
    def test_save_updates():
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
        class Multi(dj.Manual):
            definition = """
            id : smallint
            ---
            help : longblob
            help2 : longblob
            help3 : longblob
            help4 = null : longblob
            """
        try:
            Multi.insert([
            {'id':9, 'help':'12341234', 'help2':12354, 'help3':'asdfksajdfljk', 'help4':'asdf'},
            {'id':10, 'help':'12341234', 'help2':12354, 'help3':'asdfksajdfljk', 'help4':'asdf'},
            {'id':11, 'help':'12341234', 'help2':12354, 'help3':'asdfksajdfljk', 'help4':'asdf'},
            {'id':12, 'help':'12341234', 'help2':12354, 'help3':'asdfksajdfljk', 'help4':'asdf'}])
        except:
            pass
        finally:
            (Multi & {'id':9}).save_updates({'help':213, 'help2':'asdfkjasdlfkj', 'help4': None})
            a=Multi().fetch()==np.array([( 9, 213, 'asdfkjasdlfkj', 'asdfksajdfljk', None),(10, '12341234', 12354, 'asdfksajdfljk', 'asdf'),(11, '12341234', 12354, 'asdfksajdfljk', 'asdf'),(12, '12341234', 12354, 'asdfksajdfljk', 'asdf')],dtype=[('id', '<i8'), ('help', 'O'), ('help2', 'O'), ('help3', 'O'), ('help4', 'O')])
            assert_true(np.all(a==True))
            #print(a)
    @staticmethod
    def test_settings_table():
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
        class ManualEntry(dj.Manual):
            definition = """
            id : smallint
            ---
            help : longblob
            """
        @schema
        class ImportedEntry(dj.AutoImported):
            definition = """
            -> ManualEntry
            ---
            data = null : longblob
            """

            def make_compatible(self, data):

                return {'data':data}
        try:
            ImportedEntry.settings_table.insert1({'settings_name':'mean1','func':np.mean, 'global_settings':{'keepdims':True},'entry_settings':{'a':'help'}})
            ImportedEntry.settings_table.insert1({'settings_name':'mean2', 'func':np.mean, 'global_settings':{'keepdims':False},'entry_settings':{'a':'help'}})
            ImportedEntry.settings_table.insert1({'settings_name':'std1', 'func':np.std, 'global_settings':{'keepdims':False},'entry_settings':{'a':'help'}})
        except:
            pass
        finally:
            df=(ImportedEntry).settings_table.fetch(format='frame')
            assert_true(list(df.index)==['mean1', 'mean2', 'std1'])
    '''
    TODO: add illegal insert to upstream tables
    '''
    @staticmethod
    def test_settings_table():
        pass

'''
driver code will remove once nosetests certification gets upgraded
https://github.com/datajoint/datajoint-python/issues?q=is%3Aissue+nosetests+is%3Aclosed
'''


TestPartTables.test_save_updates()
TestPartTables.test_settings_table()
