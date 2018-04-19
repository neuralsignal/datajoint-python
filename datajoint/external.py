import os
import numpy as np
from tqdm import tqdm
from . import config, DataJointError
from .hash import long_hash
from .blob import pack, unpack
from .base_relation import BaseRelation
from .declare import STORE_HASH_LENGTH, HASH_DATA_TYPE
from . import s3
from .utils import safe_write
from numpy.lib.format import open_memmap
from warnings import warn

class ExternalTable(BaseRelation):
    """
    The table tracking externally stored objects.
    Declare as ExternalTable(connection, database)
    """
    def __init__(self, arg, database=None):
        if isinstance(arg, ExternalTable):
            super().__init__(arg)
            # copy constructor
            self.database = arg.database
            self._connection = arg._connection
            return
        super().__init__()
        self.database = database
        self._connection = arg
        if not self.is_declared:
            self.declare()

    @property
    def definition(self):
        return """
        # external storage tracking
        hash  : {hash_data_type}  # the hash of stored object + store name
        ---
        size      :bigint unsigned   # size of object in bytes
        timestamp=CURRENT_TIMESTAMP  :timestamp   # automatic timestamp
        """.format(hash_data_type=HASH_DATA_TYPE)

    @property
    def table_name(self):
        return '~external'

    def put(self, store, obj, np_first=False):
        """
        put an object in external store
        """
        spec = self._get_store_spec(store)
        is_memmap = False
        if isinstance(obj, np.memmap):
            if store[len('external-'):]:
                raise DataJointError('numpy save only works with `external` folder.')
            blob = obj.tostring()
            blob_hash = long_hash(blob) + '.npy'
            is_memmap = True
        elif np_first:
            if isinstance(obj, np.ndarray):
                if store[len('external-'):]:
                    raise DataJointError('numpy save only works with `external` folder.')
                blob = obj.tostring()
                blob_hash = long_hash(blob) + '.npy'
            else:
                blob = pack(obj)
                blob_hash = long_hash(blob) + store[len('external-'):]
        else:
            try:
                blob = pack(obj)
                blob_hash = long_hash(blob) + store[len('external-'):]
            except DataJointError as e:
                if store[len('external-'):]:
                    raise e
                if isinstance(obj, np.ndarray):
                    blob = obj.tostring()
                    blob_hash = long_hash(blob) + '.npy'
                else:
                    raise e
        ###
        if spec['protocol'] == 'file':
            folder = os.path.join(spec['location'], self.database)
            if not os.path.exists(folder):
                os.mkdir(folder)
            full_path = os.path.join(folder, blob_hash)
            if not os.path.isfile(full_path):
                if full_path.endswith('.npy'):
                    if is_memmap:
                        #memmap object will not be flushed beforehand
                        #TODO test memmapping
                        warn('memmapping object not tested.')
                        new_obj = open_memmap(full_path, mode='w+', dtype=obj.dtype, shape=obj.shape)
                        new_obj[:] = obj[:]
                        new_obj.flush()
                        del(new_obj)
                        del(obj)
                    else:
                        try:
                            np.save(full_path, obj, allow_pickle=False)
                        except ValueError:
                            np.save(full_path, obj, allow_pickle=True)
                else:
                    try:
                        safe_write(full_path, blob)
                    except FileNotFoundError:
                        os.makedirs(folder)
                        safe_write(full_path, blob)
        elif spec['protocol'] == 's3':
            s3.Folder(database=self.database, **spec).put(blob_hash, blob)
        else:
            raise DataJointError('Unknown external storage protocol {protocol} for {store}'.format(
                store=store, protocol=spec['protocol']))

        # insert tracking info
        self.connection.query(
            "INSERT INTO {tab} (hash, size) VALUES ('{hash}', {size}) "
            "ON DUPLICATE KEY UPDATE timestamp=CURRENT_TIMESTAMP".format(
                tab=self.full_table_name,
                hash=blob_hash,
                size=len(blob)))
        return blob_hash

    def get(self, blob_hash, mmap_mode=None):
        """
        get an object from external store.
        Does not need to check whether it's in the table.
        """
        if blob_hash is None:
            return None
        store = blob_hash[STORE_HASH_LENGTH:]
        if store == '.npy':
            np_store = True
            store = 'external'
        else:
            np_store = False
            store = 'external' + ('-' if store else '') + store

        cache_folder = config.get('cache', None)

        blob = None
        if cache_folder and not np_store:
            try:
                with open(os.path.join(cache_folder, blob_hash), 'rb') as f:
                    blob = f.read()
            except FileNotFoundError:
                pass
        elif cache_folder and np_store:
            try:
                return np.load(os.path.join(cache_folder, blob_hash), mmap_mode=mmap_mode)
            except FileNotFoundError:
                pass

        if blob is None:
            spec = self._get_store_spec(store)
            if spec['protocol'] == 'file':
                full_path = os.path.join(spec['location'], self.database, blob_hash)
                if np_store:
                    obj = np.load(full_path, mmap_mode=mmap_mode)
                    ###TODO untested cache folder saving for numpy arrays
                    if cache_folder:
                        if not os.path.exists(cache_folder):
                            os.makedirs(cache_folder)
                        try:
                            np.save(os.path.join(cache_folder, blob_hash), obj, allow_pickle=False)
                        except ValueError:
                            np.save(os.path.join(cache_folder, blob_hash), obj, allow_pickle=True)
                    ###
                    return obj
                try:
                    with open(full_path, 'rb') as f:
                        blob = f.read()
                except FileNotFoundError:
                    raise DataJointError('Lost access to external blob %s.' % full_path) from None
            elif spec['protocol'] == 's3':
                try:
                    blob = s3.Folder(database=self.database, **spec).get(blob_hash)
                except TypeError:
                    raise DataJointError('External store {store} configuration is incomplete.'.format(store=store))
            else:
                raise DataJointError('Unknown external storage protocol "%s"' % spec['protocol'])

            if cache_folder:
                if not os.path.exists(cache_folder):
                    os.makedirs(cache_folder)
                safe_write(os.path.join(cache_folder, blob_hash), blob)

        return unpack(blob)

    @property
    def references(self):
        """
        :return: generator of referencing table names and their referencing columns
        """
        return self.connection.query("""
        SELECT concat('`', table_schema, '`.`', table_name, '`') as referencing_table, column_name
        FROM information_schema.key_column_usage
        WHERE referenced_table_name="{tab}" and referenced_table_schema="{db}"
        """.format(tab=self.table_name, db=self.database), as_dict=True)

    def delete(self):
        return self.delete_quick()

    def delete_quick(self):
        raise DataJointError('The external table does not support delete. Please use delete_garbage instead.')

    def drop(self):
        """drop the table"""
        self.drop_quick()

    def drop_quick(self):
        """drop the external table -- works only when it's empty"""
        if self:
            raise DataJointError('Cannot drop a non-empty external table. Please use delete_garabge to clear it.')
        self.drop_quick()

    def delete_garbage(self):
        """
        Delete items that are no longer referenced.
        This operation is safe to perform at any time.
        """
        self.connection.query(
            "DELETE FROM `{db}`.`{tab}` WHERE ".format(tab=self.table_name, db=self.database) +
            " AND ".join(
                'hash NOT IN (SELECT {column_name} FROM {referencing_table})'.format(**ref)
                for ref in self.references) or "TRUE")
        print('Deleted %d items' % self.connection.query("SELECT ROW_COUNT()").fetchone()[0])

    def clean_store(self, store, display_progress=True):
        """
        Clean unused data in an external storage repository from unused blobs.
        This must be performed after delete_garbage during low-usage periods to reduce risks of data loss.
        """
        spec = self._get_store_spec(store)
        progress = tqdm if display_progress else lambda x: x
        if spec['protocol'] == 'file':
            folder = os.path.join(spec['location'], self.database)
            delete_list = set(os.listdir(folder)).difference(self.fetch('hash'))
            print('Deleting %d unused items from %s' % (len(delete_list), folder), flush=True)
            for f in progress(delete_list):
                os.remove(os.path.join(folder, f))
        elif spec['protocol'] == 's3':
            try:
                s3.Folder(database=self.database, **spec).clean(self.fetch('hash'))
            except TypeError:
                raise DataJointError('External store {store} configuration is incomplete.'.format(store=store))

    @staticmethod
    def _get_store_spec(store):
        try:
            spec = config[store]
        except KeyError:
            raise DataJointError('Storage {store} is requested but not configured'.format(store=store)) from None
        if 'protocol' not in spec:
            raise DataJointError('Storage {store} config is missing the protocol field'.format(store=store))
        if spec['protocol'] not in {'file', 's3'}:
            raise DataJointError(
                'Unknown external storage protocol "{protocol}" in "{store}"'.format(store=store, **spec))
        return spec
