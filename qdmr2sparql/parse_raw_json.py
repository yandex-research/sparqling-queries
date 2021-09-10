import os, sys
import json
import sqlite3
import traceback
import argparse
import re
from process_sql import get_sql


script_path = os.path.dirname(os.path.abspath(__file__))
spider_root = os.path.abspath(os.path.join(script_path, "..", "data"))

sql_path = os.path.join(spider_root, "spider", "dev.json")
db_dir = os.path.join(spider_root, "spider", "database")
output_file = os.path.join(spider_root, "spider", "dev.json")
table_file = os.path.join(spider_root, "spider", "tables.json")


def handle_exception(e, verbose=False):
    '''Get all information about exception.
    '''
    exc_type, exc_val, exc_tb = sys.exc_info()
    format_exc = traceback.format_exception(exc_type, exc_val, exc_tb)
    full_message = ' '.join(format_exc)
    file_name = ''
    module_name = ''
    line_num = -1
    # find file name and line number
    for exc in format_exc:
        if exc.find('File') >= 0 and exc[exc.find('File') + 5] == '\"': # 5 = len('File ')
            exc = exc.split('\n')[0] # first line contains all info 
            if exc.find('/python') >= 0:
                continue
            cur_file_name = exc.split('\"')[1].split('/')[-1] # "abs_path/file_name"
            file_name = cur_file_name
            module_name = exc[exc.find(', in '):].split(' ')[-1] # "in module_name"
            line_num = re.findall('line \d+', exc)[0].split(' ')[-1] # "line num_line"

    assert int(line_num) >= 0, 'unknown line number'
    assert file_name, 'unknown file name'

    if verbose:
        print('Error info:')
        print('message:', str(e)) 
        print('type:', exc_type.__name__)
        print('file:', file_name) 
        print('module_name:', module_name)
        print('line_number:', line_num)
        #print(full_message)

    return {'message': str(e), 'type': exc_type.__name__, 'file': file_name, 
            'module_name': module_name, 'line_number': line_num}


class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema, table):
        self._schema = schema
        self._table = table
        self._idMap = self._map(self._schema, self._table)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema, table):
        column_names_original = table['column_names_original']
        table_names_original = table['table_names_original']
        #print 'column_names_original: ', column_names_original
        #print 'table_names_original: ', table_names_original
        for i, (tab_id, col) in enumerate(column_names_original):
            if tab_id == -1:
                idMap = {'*': i}
            else:
                key = table_names_original[tab_id].lower()
                val = col.lower()
                idMap[key + "." + val] = i

        for i, tab in enumerate(table_names_original):
            key = tab.lower()
            idMap[key] = i

        return idMap
    

def get_schemas_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    db_names = [db['db_id'] for db in data]

    tables = {}
    schemas = {}
    for db in data:
        db_id = db['db_id']
        schema = {} #{'table': [col.lower, ..., ]} * -> __all__
        column_names_original = db['column_names_original']
        table_names_original = db['table_names_original']
        tables[db_id] = {'column_names_original': column_names_original, 'table_names_original': table_names_original}
        for i, tabn in enumerate(table_names_original):
            table = str(tabn.lower())
            cols = [str(col.lower()) for td, col in column_names_original if td == i]
            schema[table] = cols
        schemas[db_id] = schema

    return schemas, db_names, tables



schemas, db_names, tables = get_schemas_from_json(table_file)

with open(sql_path) as inf:
    sql_data = json.load(inf)

sql_data_new = []
for data in sql_data:
    try:
        db_id = data["db_id"]
        schema = schemas[db_id]
        table = tables[db_id]
        schema = Schema(schema, table)
        sql = data["query"]
        sql_label = get_sql(schema, sql)
        data["sql"] = sql_label
        sql_data_new.append(data)
    except Exception as e:
        print("db_id: ", db_id)
        print("sql: ", sql)

        error_details = handle_exception(e, verbose=False)
        print(f"ERROR: {error_details['type']}:{error_details['message']}, file: {error_details['file']}, line {error_details['line_number']}")


with open(output_file, 'wt') as out:
    json.dump(sql_data_new, out, indent=4, separators=(',', ': '))