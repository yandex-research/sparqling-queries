import os
import copy
import json
import ast
import sqlite3
import re
import glob
import attr
import urllib

import rdflib
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import XSD
from dateutil import parser as date_parser

from qdmr2sparql.process_sql import get_sql as parse_sql_spider
from qdmr2sparql.process_sql import AGG_OPS, SchemaFromSpider, parsed_sql_has_superlative


def prepare_dict_for_json(d):
    if isinstance(d,dict):
        d_for_json = {}
        for key, value in d.items():
            d_for_json[str(key)] = prepare_dict_for_json(value)
        return d_for_json
    elif isinstance(d, list):
        d_for_json = []
        for value in d:
            d_for_json.append(prepare_dict_for_json(value))
        return d_for_json
    else:
        return str(d)


def assert_check_grounding_save_load(all_grounding, check_grounding):
    for k, v in check_grounding.items():
        assert k in all_grounding, f"{k} not found in initial grounding"
        try:
            assert v == all_grounding[k], f"{k}: {v} not equal to {all_grounding[k]}"
        except Exception as e:
            if isinstance(v, dict) and isinstance(all_grounding[k], dict):
                check_grnd_indices = sorted(list(v.keys()), key=lambda x: str(x))
                ref_grnd_indices = sorted(list(all_grounding[k].keys()), key=lambda x: str(x))
                assert check_grnd_indices == ref_grnd_indices, f"{k}: Lists of grounding indices do not match: {check_grnd_indices} and {ref_grnd_indices}"
                for grnd_idx in check_grnd_indices:
                    for a, b in zip(v[grnd_idx], all_grounding[k][grnd_idx]):
                        assert a == b, f"{k}: {grnd_idx}: {a} do not match to {b}"
            else:
                raise e
    for k in all_grounding:
        assert k in check_grounding, f"{k} not found in save-load grounding"


def save_grounding_to_file(output_path, all_grounding):
    grounding_for_json = prepare_dict_for_json(all_grounding)
    with open(output_path, "w", encoding="utf8") as outfile:
        json.dump(grounding_for_json, outfile, indent=4, ensure_ascii=False)


def load_grounding_list_from_file(grounding_path, data=None):
    """This function allows to read lists of complete groundings:
        grounding[example_name] = {"GROUNDINGS": [grnds], "MESSAGES": [messages], "ERRORS": [errors]}
    """
    if data is None:
        with open(grounding_path) as json_file:
            data = json.load(json_file)
    grounding = {}
    for example_name, example_dict in data.items():
        grounding[example_name] = {}
        if "ERRORS" in example_dict:
            # no grounding available - had error on the previous stage
            grounding[example_name] = example_dict
            continue

        if "GROUNDINGS" in example_dict:
            grnd_list = example_dict["GROUNDINGS"]
            grnd_list_new = []
            for grnd_dict in grnd_list:
                grnd_dict_new = {}
                for grnd_index, grnd in grnd_dict.items():
                    if grnd_index in ["distinct", "MESSAGES", "ERRORS"]:
                        grnd_dict_new[grnd_index] = grnd
                        continue

                    if grnd_index != "ESSENTIAL_GROUNDINGS":
                        grnd_index = GroundingIndex.from_str(grnd_index)
                        if grnd:
                            grnd = GroundingKey.from_str(grnd)
                        else:
                            grnd = ""
                        grnd_dict_new[grnd_index] = grnd
                    else:
                        grnd_dict_new[grnd_index] = [GroundingKey.from_str(g) for g in grnd]

                grnd_list_new.append(grnd_dict_new)
            grounding[example_name]["GROUNDINGS"] = grnd_list_new

        if "MESSAGES" in example_dict:
            grounding[example_name]["MESSAGES"] = example_dict["MESSAGES"]

    return grounding


def load_grounding_from_file(grounding_path, data=None):
    """This function allows to read groundings where each argument has a list of options
        grnd[GroundingIndex object] = GroundingKey object
        grounding[example_name] = grnd
    """
    if data is None:
        with open(grounding_path) as json_file:
            data = json.load(json_file)
    grounding = {}
    for example_name, grnd_dict in data.items():
        if "ERRORS" in grnd_dict:
            # no grounding available - had error on the previous stage
            grounding[example_name] = grnd_dict
            continue

        grnd_dict_new = {}
        for grnd_index, list_of_grnds in grnd_dict.items():
            if grnd_index in ["distinct", "MESSAGES", "ERRORS"]:
                grnd_dict_new[grnd_index] = list_of_grnds
                continue

            if grnd_index != "ESSENTIAL_GROUNDINGS":
                grnd_index = GroundingIndex.from_str(grnd_index)

            grnds_new = []
            for grnd in list_of_grnds:
                if grnd:
                    grnd = GroundingKey.from_str(grnd)
                    grnds_new.append(grnd)
                else:
                    grnds_new.append("")
            grnd_dict_new[grnd_index] = grnds_new
        grounding[example_name] = grnd_dict_new
    return grounding


def parse_date_str(s, default_date=date_parser.parse("0001-01-01 00:00:00")):
    try:
        date_object = date_parser.parse(str(s), fuzzy=True, default=default_date)
        return date_object.strftime("%Y-%m-%dT%H:%M:%S")
    except:
        return str(s)


class GroundingIndex():
    delimiter_in_str = "|"

    def __init__(self, op_index, arg_index, key_str):
        self.op_index = op_index
        self.arg_index = arg_index
        self.key_str = key_str

    def __hash__(self):
        return hash((self.op_index, self.arg_index, self.key_str))

    def __eq__(self, othr):
        return (isinstance(othr, type(self))
                and (self.op_index, self.arg_index, self.key_str) == (othr.op_index, othr.arg_index, othr.key_str))

    def __repr__(self):
        return f"{self.op_index}{self.delimiter_in_str}{self.arg_index}{self.delimiter_in_str}{self.key_str}"

    @staticmethod
    def from_str(s):
        delimiter = GroundingIndex.delimiter_in_str
        i_del = s.find(delimiter)
        op_index = int(s[:i_del])
        s = s[i_del+1:]
        i_del = s.find(delimiter)
        arg_index = int(s[:i_del])
        key_str = s[i_del+1:]
        return GroundingIndex(op_index, arg_index, key_str)


class GroundingKey():
    delimiter_in_str = "|"

    def __init__(self, grounding_type, keys):
        self.type = grounding_type
        self.keys = tuple(keys)

    @staticmethod
    def make_table_grounding(tbl_name):
        return GroundingKey("tbl", (tbl_name,))

    @staticmethod
    def make_column_grounding(tbl_name, col_name):
        return GroundingKey("col", (tbl_name, col_name))

    @staticmethod
    def make_value_grounding(tbl_name, col_name, value):
        return GroundingKey("val", (tbl_name, col_name, value))

    @staticmethod
    def make_comparative_grounding(comparator, value, column_grounding=None):
        value = str(value)
        if column_grounding is None:
            return GroundingKey("comparative", (comparator, value))
        else:
            assert column_grounding.iscol() or column_grounding.isref(), "Third argument of comparative should be the column or reference grounding"
            return GroundingKey("comparative", (comparator, value, column_grounding))

    @staticmethod
    def make_sortdir_grounding(ascending=True):
        return GroundingKey("sortdir", ("ascending" if ascending else "descending",))

    @staticmethod
    def make_reference_grounding(ref):
        assert QdmrInstance.is_good_qdmr_ref(ref), f"Cannot parse {ref} into QDMR ref"
        return GroundingKey("reference", (ref, ))

    @staticmethod
    def make_text_grounding(s):
        return GroundingKey("str", (s, ))

    def istbl(self):
        return self.type == "tbl"

    def iscol(self):
        return self.type == "col"

    def isval(self):
        return self.type == "val"

    def iscomp(self):
        return self.type == "comparative"

    def issortdir(self):
        return self.type == "sortdir"

    def isref(self):
        return self.type == "reference"

    def isstr(self):
        return self.type == "str"

    def get_tbl_name(self):
        assert self.type in ["tbl", "col", "val"], f"Do not have table name for {self}"
        return self.keys[0]

    def get_col_name(self):
        assert self.type in ["col", "val"], f"Do not have column name for {self}"
        return self.keys[1]

    def get_val(self):
        assert self.type in ["val", "comparative"], f"Do not have value for {self}"
        if self.type == "val":
            return self.keys[2]
        else:
            return self.keys[1]

    def __hash__(self):
        return hash((self.type, self.keys))

    def __eq__(self, othr):
        return (isinstance(othr, type(self))
                and (self.type, self.keys) == (othr.type, othr.keys))

    def __repr__(self):
        return f"{self.type}{self.delimiter_in_str}{self.delimiter_in_str.join(str(k) for k in self.keys)}"

    @staticmethod
    def from_str(s):
        delimiter=GroundingKey.delimiter_in_str
        keys = []
        i_del = s.find(delimiter)
        while i_del != -1:
            part = s[:i_del]
            keys.append(part)
            s = s[i_del+1:]
            i_del = s.find(delimiter)
        keys.append(s)

        grounding_type = keys.pop(0)

        if grounding_type == "comparative" and len(keys) > 2:
            comm_grounding = keys[2:]
            comm_grounding = GroundingKey(comm_grounding[0], comm_grounding[1:])
            keys = keys[:2] + [comm_grounding]

        return GroundingKey(grounding_type, keys)


class DatabaseSchema():
    def __init__(self, table_json=None,
                       db_id=None,
                       table_names=None,
                       column_names=None,
                       primary_keys=None,
                       foreign_keys=None,
                       type_for_column_for_table=None,
                       table_data=None,
                       column_key_in_table=None,
                       column_used_with_keys=None):

        if table_json is not None:
            db_id, table_names, column_names, primary_keys, foreign_keys, type_for_column_for_table = \
                self.parse_sql_database(table_json)

        self.db_id = db_id
        self.table_names = table_names
        self.column_names = column_names
        self.primary_keys = primary_keys
        self.type_for_column_for_table = type_for_column_for_table

        self.foreign_keys = []
        self.foreign_key_tgt_for_src = {}
        self.foreign_keys_src_for_tgt = {}
        self.add_foreign_keys(foreign_keys)

        self.data_loaded_tests = False
        if table_data is None:
            self.data_loaded = False
        else:
            self.data_loaded = True
            self.update_data_to_self(table_data, table_names, column_names, type_for_column_for_table)
            self.column_key_in_table = column_key_in_table
            self.column_used_with_keys = column_used_with_keys
            self.table_data = table_data

    def add_foreign_keys(self, foreign_keys):
        for for_key in foreign_keys:
            if for_key in self.foreign_keys:
                continue
            self.foreign_keys.append(for_key)
            src_key = GroundingKey.make_column_grounding(for_key["table_src"], for_key["col_src"])
            tgt_key = GroundingKey.make_column_grounding(for_key["table_tgt"], for_key["col_tgt"])
            self.foreign_key_tgt_for_src[src_key] = tgt_key
            if tgt_key in self.foreign_keys_src_for_tgt:
                self.foreign_keys_src_for_tgt[tgt_key].append(src_key)
            else:
                self.foreign_keys_src_for_tgt[tgt_key] = [src_key]

    @staticmethod
    def parse_sql_database(table_json):
        # table_json.keys()
        # dict_keys(['column_names',
        #            'column_names_original',
        #            'column_types',
        #            'db_id',
        #            'foreign_keys',
        #            'primary_keys',
        #            'table_names',
        # table_json.keys()
        # dict_keys(['column_names',
        #            'column_names_original',
        #            'column_types',
        #            'db_id',
        #            'foreign_keys',
        #            'primary_keys',
        #            'table_names',
        #            'table_names_original'])
        db_id = table_json["db_id"]
        table_names = table_json["table_names_original"]
        spider_column_names = table_json["column_names_original"]
        spider_column_types = table_json["column_types"]

        assert len(spider_column_types) == len(spider_column_names)
        assert tuple(spider_column_names[0]) == (-1, "*")

        column_names = {tbl : [] for tbl in table_names}
        type_for_column_for_table = {tbl : {} for tbl in table_names}
        # spider_column_names[0] corresponds to *
        for (table_indx, col_name), col_type in zip(spider_column_names[1:], spider_column_types[1:]):
            tbl_name = table_names[table_indx]
            column_names[tbl_name].append(col_name)
            type_for_column_for_table[tbl_name][col_name] = col_type

        # setup primary keys: assume that there are no composite keys
        primary_keys = {}
        for i_col in table_json["primary_keys"]:
            i_table, col_name = spider_column_names[i_col]
            tbl_name = table_names[i_table]
            primary_keys[tbl_name] = col_name

        # collect all the foreign keys
        foreign_keys = []
        for src_key, tgt_key in table_json["foreign_keys"]:
            i_table_src, col_name_src = spider_column_names[src_key]
            i_table_tgt, col_name_tgt = spider_column_names[tgt_key]
            key_data = {}

            table_src = table_names[i_table_src]
            key_data["table_src"] = table_src
            key_data["col_src"] = col_name_src

            table_tgt = table_names[i_table_tgt]
            key_data["table_tgt"] = table_tgt
            key_data["col_tgt"] = col_name_tgt
            foreign_keys.append(key_data)

        return db_id, table_names, column_names, primary_keys, foreign_keys, type_for_column_for_table

    @staticmethod
    def get_db_folder(db_path, db_id):
        return os.path.join(db_path, db_id)

    def load_table_data(self, db_path):
        if self.data_loaded:
            return

        db_id = self.db_id
        self.db_path = db_path
        print(f"Loading {db_id} from {db_path}")
        self.sqlite_file = os.path.join(self.get_db_folder(db_path, db_id), f"{db_id}.sqlite")

        tbl_data, table_names, col_names, type_for_column_for_table, foreign_keys_from_data =\
            self.load_data_from_sqlite(self.sqlite_file)
        self.add_foreign_keys(foreign_keys_from_data)
        self.type_for_column_for_table = type_for_column_for_table
        self.table_data = tbl_data

        self.update_self_to_loaded_data(tbl_data, table_names, col_names, type_for_column_for_table, foreign_keys_from_data)
        self.data_loaded = True

    @staticmethod
    def load_data_from_sqlite(sqlite_file):
        conn = sqlite3.connect(sqlite_file)
        conn.text_factory = lambda b: b.decode(errors = 'ignore')
        c = conn.cursor()

        # get all the tables in the database
        query_get_all_tables = "SELECT name FROM sqlite_master WHERE type='table'"
        c.execute(query_get_all_tables)
        table_names = c.fetchall()
        table_names = [t[0] for t in table_names]

        # get data from all the tables
        tbl_data = {}
        col_names = {}
        foreign_keys_from_data = []
        type_for_column_for_table = {}
        for tbl_name in table_names:
            # get columns
            c.execute(f"PRAGMA table_info({tbl_name})")
            col_data = c.fetchall()
            col_names[tbl_name] = [t[1] for t in col_data]
            type_for_column_for_table[tbl_name] = {}
            for col_name, t in zip(col_names[tbl_name], col_data):
                type_for_column_for_table[tbl_name][col_name] = t[2]
                if "char" in type_for_column_for_table[tbl_name][col_name].lower():
                    type_for_column_for_table[tbl_name][col_name] = "TEXT"
                if "text" == type_for_column_for_table[tbl_name][col_name].lower():
                    type_for_column_for_table[tbl_name][col_name] = "TEXT"
                if any(t in type_for_column_for_table[tbl_name][col_name].lower() for t in ["number", "int", "bit"]):
                    type_for_column_for_table[tbl_name][col_name] = "INTEGER"
                if any(t in type_for_column_for_table[tbl_name][col_name].lower() for t in ["float", "real", "numeric", "decimal"]):
                    type_for_column_for_table[tbl_name][col_name] = "FLOAT"

        # load foreign keys from the database
        for tbl_name in table_names:
            c.execute(f"PRAGMA foreign_key_list({tbl_name});")
            foreign_keys = c.fetchall()
            for key in foreign_keys:
                col_src = key[3]
                col_tgt = key[4]
                table_src = tbl_name
                table_tgt = key[2]
                if col_src not in col_names[table_src]\
                   or col_tgt not in col_names[table_tgt]:
                    # some times there are keys to non-existent columns - just ignore those
                    continue
                foreign_keys_from_data.append({'col_src': col_src, 'col_tgt': col_tgt, 'table_src': table_src, 'table_tgt': table_tgt})

        # get content
        for tbl_name in table_names:
            c.execute(f"SELECT * FROM {tbl_name}")
            tbl_data[tbl_name] = c.fetchall()

        conn.close()

        return tbl_data, table_names, col_names, type_for_column_for_table, foreign_keys_from_data

    def load_test_table_data(self, db_path):
        if self.data_loaded_tests:
            return

        self.load_table_data(db_path)

        paths = sorted(glob.glob(os.path.join(self.get_db_folder(db_path, self.db_id), "*.sqlite")))
        self.test_schemas = []
        for p in paths:
            if os.path.abspath(p) == os.path.abspath(self.sqlite_file):
                continue
            print(f"Loading test database {p}")
            tbl_data, table_names, col_names, type_for_column_for_table, foreign_keys_from_data =\
                self.load_data_from_sqlite(p)
            self.update_data_to_self(tbl_data, table_names, col_names, type_for_column_for_table)

            test_schema = DatabaseSchema(None, self.db_id, table_names, col_names, self.primary_keys, self.foreign_keys, type_for_column_for_table, tbl_data, self.column_key_in_table, self.column_used_with_keys)
            test_schema.sqlite_file = p
            self.test_schemas.append(test_schema)
        self.data_loaded_tests = True

    def execute_sql_query(self, sql_query):
        assert self.data_loaded, "Data should be loaded into schema to enable SQL execution"

        # # print(f"Loading {db_id} from {db_path}")
        # sqlite_file = os.path.join(self.db_path, self.db_id, f"{self.db_id}.sqlite")
        # schema_file = os.path.join(self.db_path, self.db_id, "schema.sql")
        conn = sqlite3.connect(self.sqlite_file)
        conn.text_factory = lambda b: b.decode(errors = 'ignore')

        # print("Executing SQL", query)
        c = conn.cursor()
        c.execute(sql_query)
        result = c.fetchall()
        # print("SQL result:", result)
        # Command to get names of the output
        # # output_col_names = list(map(lambda x: x[0], c.description))
        conn.close()
        return result

    def update_self_to_loaded_data(self, tbl_data, table_names, col_names, type_for_column_for_table, foreign_keys_from_data):
        # check the consistency of the data from the database with the data from the schema
        assert sorted(self.table_names) == sorted(table_names),\
               f"Table names loaded from database file do not match the ones in the schema file: {table_names} and {self.table_names}"
        for t in table_names:
            assert self.column_names[t] == col_names[t], f"Column names of {t} from the database are not consistent with the ones from the schema: {col_names[t]} and {self.column_names[t]}"

        # setup primary keys: assume that there are no composite keys
        for tbl_name in table_names:
            if tbl_name not in self.primary_keys:
                continue
            col_name = self.primary_keys[tbl_name]
            # check if is an actual primary key:
            if not self.check_column_is_key(tbl_name, col_name, self.column_names[tbl_name], self.table_data[tbl_name]):
                # the specified primary key has duplicated entries - is not a valid primary key
                del self.primary_keys[tbl_name]

        # check that all the tables have primary keys, otherwise add a new column - key
        for tbl_name in table_names:
            if tbl_name not in self.primary_keys:
                # could not find any primary key - try to create one
                cols = self.column_names[tbl_name]
                key_name = f"{tbl_name}_id"
                if key_name in cols:
                    key_name_base = key_name
                    i = 0
                    while key_name in cols:
                        key_name = f"{key_name_base}_{i}"
                        i += 1

                self.column_names[tbl_name] = [key_name] + self.column_names[tbl_name]
                self.primary_keys[tbl_name] = key_name
                self.type_for_column_for_table[tbl_name][key_name] = "INTEGER"

                # add a key to all rows
                old_content = self.table_data[tbl_name]
                self.table_data[tbl_name] = []
                for i, row in enumerate(old_content):
                    self.table_data[tbl_name].append(tuple([i] + list(row)))

                assert tbl_name in self.primary_keys, f"Could now find a primary key in database {self.db_id} table {tbl_name}, cols: {cols}"

        # check what columns are as as sources or targets of foreign keys
        self.column_key_in_table = {}
        self.column_used_with_keys = {}
        # initialize keys with primary keys
        for tbl_name, col_name in self.primary_keys.items():
            self.column_key_in_table[tbl_name] = [col_name]
            self.column_used_with_keys[tbl_name] = [col_name]

        for key_data in self.foreign_keys:
            table_src = key_data["table_src"]
            col_name_src = key_data["col_src"]
            table_tgt = key_data["table_tgt"]
            col_name_tgt = key_data["col_tgt"]

            if col_name_tgt not in self.column_used_with_keys[table_tgt]:
                self.column_used_with_keys[table_tgt].append(col_name_tgt)

                if self.check_column_is_key(table_tgt, col_name_tgt, self.column_names[table_tgt], self.table_data[table_tgt]):
                    # verify that that column is a key
                    self.column_key_in_table[table_tgt].append(col_name_tgt)

            if col_name_src not in self.column_used_with_keys[table_src]:
                self.column_used_with_keys[table_src].append(col_name_src)

    @staticmethod
    def check_column_is_key(tbl_name, col_name, column_names, table_content):
        i_col = column_names.index(col_name)
        col_data = [row[i_col] for row in table_content]
        return len(set(col_data)) == len(col_data)

    def update_data_to_self(self, tbl_data, table_names, col_names, type_for_column_for_table):
        # check the consistency of the data from the database with the data from the schema
        assert sorted(self.table_names) == sorted(table_names),\
               f"Table names loaded from database file do not match the ones in the schema file: {table_names} and {self.table_names}"
        for tbl_name in table_names:
            if len(self.column_names[tbl_name]) == len(col_names[tbl_name]):
                assert self.column_names[tbl_name] == col_names[tbl_name],\
                    f"Column names of {tbl_name} from the database are not consistent with the ones from the schema: {col_names[tbl_name]} and {self.column_names[tbl_name]}"
                assert self.type_for_column_for_table[tbl_name] == type_for_column_for_table[tbl_name],\
                    f"Column types of {tbl_name} from the database are not consistent with the ones from the schema: {type_for_column_for_table[tbl_name]} and {self.type_for_column_for_table[tbl_name]}"
                assert self.check_column_is_key(tbl_name, self.primary_keys[tbl_name], col_names[tbl_name], tbl_data[tbl_name]),\
                    f"Primary key {self.primary_keys[tbl_name]} of {tbl_name} is not a key"
            else:
                # extra primary key was added automatically
                assert self.column_names[tbl_name][1:] == col_names[tbl_name], f"Column names of {tbl_name} from the database are not consistent with the ones from the schema: {col_names[tbl_name]} and {self.column_names[tbl_name]}"

                assert tbl_name in self.primary_keys, f"Do not have primary key for table {tbl_name}"
                key_name = self.primary_keys[tbl_name]
                assert key_name == self.column_names[tbl_name][0], f"Primary key mismatch in table {tbl_name}"
                col_names[tbl_name] = [key_name] + col_names[tbl_name]
                type_for_column_for_table[tbl_name][key_name] = self.type_for_column_for_table[tbl_name][key_name]
                assert self.type_for_column_for_table[tbl_name] == type_for_column_for_table[tbl_name],\
                    f"Column types of {tbl_name} from the database are not consistent with the ones from the schema: {type_for_column_for_table[tbl_name]} and {self.type_for_column_for_table[tbl_name]}"

                # add a key to all rows
                old_content = tbl_data[tbl_name]
                tbl_data[tbl_name] = []
                for i, row in enumerate(old_content):
                    tbl_data[tbl_name].append(tuple([i] + list(row)))

        for key_data in self.foreign_keys:
            table_src = key_data["table_src"]
            col_name_src = key_data["col_src"]
            table_tgt = key_data["table_tgt"]
            col_name_tgt = key_data["col_tgt"]
            assert self.check_column_is_key(table_tgt, col_name_tgt, col_names[table_tgt], tbl_data[table_tgt]), f"Target of a foreign key {table_src}{col_name_src}->{table_tgt}:{col_name_tgt} is not a key"


def urlencode(string):
    """Could not find a proper function for the job;
    For now just removing some bad symbols like <space>
    """
    symbols = [s for s in string if s not in [" ", "\""]]
    return "".join(symbols)


class RdfGraph():
    def __init__(self, schema):
        self.g = Graph()
        self.db_id = schema.db_id
        self.E = Namespace(self.db_id + "/edges/")
        self.K = Namespace(self.db_id + "/keys/")
        self.relation_prefix = "arc"
        self.key_prefix = "key"
        self.g.bind(self.relation_prefix, self.E)
        self.g.bind(self.key_prefix, self.K)
        self.type_for_column_for_table = copy.deepcopy(schema.type_for_column_for_table)

        self.original_values_for_rdf_key = {}
        self.datum_str_for_uri = {}
        self.col_and_table_names_for_uri = {}

        self.build_rdf_lib_graph(schema)


    def query(self, sparql_query):
        return self.g.query(sparql_query)

    def get_uri_str(self, tbl_name, col_name, value):
        uri = self.encode_value_to_uri(tbl_name, col_name, value)
        datum_str = self.datum_str_for_uri[uri]

        return f"{self.key_prefix}:{self.column_value_str(tbl_name, col_name, datum_str)}"

    def encode_value_to_uri(self, tbl_name, col_name, datum):
        datum_str_orig = str(datum)
        data_type = self.type_for_column_for_table[tbl_name][col_name]
        if isinstance(datum, int):
            datum_str = f"{datum:016d}"
        elif data_type.lower() in ["date", "datetime", "timestamp"]:
            datum_str = parse_date_str(datum_str_orig)
        else:
            datum_str = datum_str_orig
            
        datum_str = urlencode(datum_str)

        def uri_from_datum_str(datum_str):
            return self.K[self.column_value_str(tbl_name, col_name, datum_str)]

        datum_str_final = datum_str
        uri = uri_from_datum_str(datum_str_final)

        i = 0
        while uri in self.original_values_for_rdf_key and\
             self.original_values_for_rdf_key[uri] != datum_str_orig and i < 100:
            delimiter = "_"
            datum_str_final = datum_str + delimiter + str(i)
            uri = uri_from_datum_str(datum_str_final)
            i += 1
        assert i < 100, f"Something went wrong, to many trials to get a new URIref for '{datum_str_orig}' in RDF graph"

        self.original_values_for_rdf_key[uri] = datum_str_orig
        self.datum_str_for_uri[uri] = datum_str_final
        return uri

    def column_link_name(self, tbl_name, col_name):
        table_uri = urlencode(str(tbl_name))
        column_uri = urlencode(str(col_name))

        def get_uri_str(tbl_name, col_name):
            return f"{tbl_name}:{col_name}"

        # we need this code for the rare case of collisions after urlencode
        table_final, column_final = table_uri, column_uri
        uri = get_uri_str(table_final, column_final)

        i = 0
        while uri in self.col_and_table_names_for_uri and self.col_and_table_names_for_uri[uri] != (str(tbl_name), str(col_name)) and i < 100:
            delimiter = "_"
            table_final = table_uri + delimiter + str(i)
            column_final = column_uri + delimiter + str(i)
            uri = get_uri_str(table_final, column_final)
            i += 1
        assert i < 100, f"Something went wrong, to many trials to get a new URIref for '{(str(tbl_name), str(col_name))}' in RDF graph"

        self.col_and_table_names_for_uri[uri] = (str(tbl_name), str(col_name))
        return uri

    def column_value_str(self, tbl_name, col_name, value_str):
        return f"{self.column_link_name(tbl_name, col_name)}:{value_str}"

    def foreign_link_name(self, tbl_name, col_name, table_tgt, col_name_tgt):
        return f"{self.column_link_name(tbl_name, col_name)}:{self.column_link_name(table_tgt, col_name_tgt)}"

    def sparql_link_ref(self, tbl_name, col_name):
        return f"{self.relation_prefix}:{self.column_link_name(tbl_name, col_name)}"

    def sparql_foreign_link(self, tbl_name, col_name, table_tgt, col_name_tgt):
        return f"{self.relation_prefix}:{self.foreign_link_name(tbl_name, col_name, table_tgt, col_name_tgt)}"

    def add_datum_key(self, tbl_name, pr_key_col_name, pr_key_value, col_name, col_value):
        uri_str_source = self.encode_value_to_uri(tbl_name, pr_key_col_name, pr_key_value)
        uri_str_target = self.encode_value_to_uri(tbl_name, col_name, col_value)
        self.g.add((uri_str_source,
                    self.E[self.column_link_name(tbl_name, col_name)],
                    uri_str_target
                  ))

    def add_datum_literal(self, tbl_name, pr_key_col_name, pr_key_value, col_name, col_value):
        data_type = self.type_for_column_for_table[tbl_name][col_name]
        orig_value_for_cache = col_value
        data_types_for_rdf = {"int": XSD.integer,
                              "integer": XSD.integer,
                              "year": XSD.integer,
                              "bool": XSD.boolean,
                              "boolean": XSD.boolean,
                              "float": XSD.double,
                              "real": XSD.double,
                              "double": XSD.double,
                              "text": XSD.string,
                              "": XSD.string,
                              "date": XSD.dateTime,
                              "datetime": XSD.dateTime,
                              "timestamp": XSD.dateTime,
                             }
        if data_type.lower() in ["date", "datetime", "timestamp"]:
            col_value = parse_date_str(col_value)
        if data_type.lower() in ["bool", "boolean"]:
            if col_value == "F":
                col_value = False
            elif col_value == "T":
                col_value = True
        if data_type.lower() in data_types_for_rdf:
            val = Literal(col_value, datatype=data_types_for_rdf[data_type.lower()])
        else:
            raise RuntimeError(f"Unknown data type {data_type} for {col_value} in {tbl_name}:{col_name}")

        data_tuple = (tbl_name, col_name, str(val))
        self.original_values_for_rdf_key[data_tuple] = orig_value_for_cache

        uri_str_source = self.encode_value_to_uri(tbl_name, pr_key_col_name, pr_key_value)
        self.g.add((uri_str_source,
                    self.E[self.column_link_name(tbl_name, col_name)],
                    val
                  ))

    def add_foreign_key(self, tbl_name, col_name, col_value,
                              table_tgt, col_name_tgt, col_value_tgt):
        uri_str_source = self.encode_value_to_uri(tbl_name, col_name, col_value)
        uri_str_target = self.encode_value_to_uri(table_tgt, col_name_tgt, col_value_tgt)
        self.g.add((uri_str_source,
                    self.E[self.foreign_link_name(tbl_name, col_name, table_tgt, col_name_tgt)],
                    uri_str_target
                  ))

    def build_rdf_lib_graph(self, schema):
        assert schema.data_loaded, "Need to load data into schema before creating the RDF graph"

        self.type_for_column_for_table = self.detect_type_changes(schema, self.type_for_column_for_table)

        for tbl_name in schema.table_names:
            columns = schema.column_names[tbl_name]
            data = schema.table_data[tbl_name]
            pr_key = schema.primary_keys[tbl_name]
            key_index = columns.index(pr_key)
            foreign_key_indices = {}
            for i_f, f in enumerate(schema.foreign_keys):
                if f["table_src"] == tbl_name:
                    foreign_key_indices[columns.index(f["col_src"])] = i_f

            for row in data:
                for i_col, col_data in enumerate(row):
                    if col_data is not None and not (isinstance(col_data, str) and col_data.lower() in ["null", "nil", "none"]):
                        # add links inside the table: for keys add nodes, for anything else add constants
                        if columns[i_col] in schema.column_used_with_keys[tbl_name]:
                            self.add_datum_key(tbl_name, columns[key_index], row[key_index], columns[i_col], col_data)
                        else:
                            self.add_datum_literal(tbl_name, columns[key_index], row[key_index], columns[i_col], col_data)

                        # add links to other tables for foreign keys
                        if i_col in foreign_key_indices:
                            _key = schema.foreign_keys[foreign_key_indices[i_col]]
                            table_tgt = _key["table_tgt"]
                            col_tgt = _key["col_tgt"]
                            type_target = schema.type_for_column_for_table[table_tgt][col_tgt]
                            tgt_data = col_data
                            if type_target.lower() in ["int", "integer"]:
                                try:
                                    tgt_data = int(col_data)
                                except:
                                    print(f"WARNING: error in loading database into RDF: could not convert '{col_data}' to int (in column {col_tgt} of table {table_tgt})")
                                    tgt_data = str(col_data)
                            elif type_target.lower() in ["float", "real", "double"]:
                                try:
                                    tgt_data = float(col_data)
                                except:
                                    print(f"WARNING: error in loading database into RDF: could not convert '{col_data}' to float (in column {col_tgt} of table {table_tgt})")
                                    tgt_data = str(col_data)
                            elif type_target.lower() in ["text", "date", "datetime", "timestamp"]:
                                tgt_data = str(col_data)
                            else:
                                raise RuntimeError(f"Do not know how to use type {type_target} in a foreign key")

                            self.add_foreign_key(tbl_name, columns[i_col], col_data,
                                                 table_tgt, col_tgt, tgt_data)

    @classmethod
    def detect_type_changes(cls, schema, type_for_column_for_table):
        assert schema.data_loaded, "Need to load data into schema before scanning for types"

        type_for_column_for_table = copy.deepcopy(type_for_column_for_table)

        for tbl_name in schema.table_names:
            columns = schema.column_names[tbl_name]
            data = schema.table_data[tbl_name]

            column_has_all_floats = {col: True for col in columns}

            for row in data:
                for i_col, col_data in enumerate(row):
                    if col_data is not None and not (isinstance(col_data, str) and col_data.lower() in ["null", "nil", "none"]):
                        try: # round float
                            col_data = float(col_data)
                        except ValueError:
                            column_has_all_floats[columns[i_col]] = False

            for col in columns:
                if column_has_all_floats[col] and type_for_column_for_table[tbl_name][col].lower() == "text":
                    # we try to convert text types to numbers
                    type_for_column_for_table[tbl_name][col] = "FLOAT"
                elif not column_has_all_floats[col] and type_for_column_for_table[tbl_name][col].lower() in ["integer", "float"]:
                    # if columns with numeric type have non-numeric columns we convert them to text
                    type_for_column_for_table[tbl_name][col] = "TEXT"

        return type_for_column_for_table


class QdmrInstance():
    def __init__(self, ops, args):
        self.ops = ops
        self.args = args

    def __len__(self):
        return len(self.ops)

    def __getitem__(self, key):
        if isinstance(key, str):
            indx = self.ref_to_index(key)
        else:
            indx = int(key)
        return self.ops[indx], self.args[indx]

    def step_to_str(self, i_step):
        op, args = self[i_step]
        arg_str = ', '.join("'" + a + "'" for a in args)
        return f"{op.upper()}[{arg_str}]"

    def __repr__(self):
        lines = []
        for i in range(len(self)):
            lines.append(f"{self.index_to_ref(i)}: {self.step_to_str(i)}")
        return "\n".join(lines)

    @staticmethod
    def index_to_ref(i):
        return f"#{i+1}"

    @staticmethod
    def ref_to_index(ref, max_index=None):
        assert isinstance(ref, str), f"QDMR step ref should be of type str, but have {ref} of type {type(ref)}"
        assert ref[0] == "#", f"QDMR reference should start with #, but have {ref}"
        index = int(ref[1:])
        assert index >= 1, f"Index should be >= 1, but have {ref}"
        index = index - 1
        if max_index is not None:
            assert index < max_index, f"QDMR positional ref should be <= {max_index}, but have {ref}"
        return index

    @staticmethod
    def is_good_qdmr_ref(ref, max_index=None):
        try:
            QdmrInstance.ref_to_index(ref, max_index=max_index)
            ok = True
        except:
            ok = False
        return ok

    @staticmethod
    def find_qdmr_refs_in_str(s, return_positions=False):
        ref_template = "#[0-9]+"
        if not return_positions:
            list_of_refs = re.findall(ref_template, s)
            return list_of_refs
        else:
            list_of_refs = []
            spans = []
            for entry in re.finditer(ref_template, s):
                span = (entry.span()[0], entry.span()[1])
                spans.append(span)
                list_of_refs.append(entry.group(0))
            return list_of_refs, spans


    @staticmethod
    def parse_break_program(qdmr_program_str, qdmr_ops_str=None):
        args = []
        ops = []

        def convert_str_to_list(x):
            x = ast.literal_eval(x)
            return [n.strip() for n in x]

        for step in convert_str_to_list(qdmr_program_str):
            idx_args_start = step.find('[')
            step_args = convert_str_to_list(step[idx_args_start:step.find(']') + 1])
            step_args = [arg.replace('\'', '').replace('\"', '') for arg in step_args]
            args.append(step_args)
            ops.append(step[:idx_args_start].lower())
        assert len(args) == len(ops), f"Inconsistent QDMR: {args}, {ops}"

        if qdmr_ops_str is not None:
            qdmr_ops = convert_str_to_list(qdmr_ops_str)
            assert len(ops) == len(qdmr_ops), f"Inconsistent number of steps in QDMR: {qdmr_program_str}, {qdmr_ops_str}"
            for op1, op2 in zip(ops, qdmr_ops):
                assert op1.lower() == op2.lower(), f"Inconsistent QDMR op: {op1} and {op2}"

        return ops, args

    def get_strs_for_saving(self):
        op_str = "[" + ", ".join("'" + q.lower() + "'" for q in self.ops) + "]"

        program = []
        for i_step in range(len(self)):
            step = self.step_to_str(i_step)
            program.append(step)
        qdmr_str = "[" + ", ".join("\"" + step + "\"" for step in program) + "]"

        return qdmr_str, op_str

@attr.s
class QueryToRdf:
    query = attr.ib()
    output_cols = attr.ib()
    sorting_info = attr.ib(default=None)
    query_has_superlative = attr.ib(default=False)

@attr.s
class QueryResult:
    source_type = attr.ib()
    output_cols = attr.ib()
    data = attr.ib()
    sorting_info = attr.ib(default=None)
    maxmin_via_limit = attr.ib(default=False)
    limit = attr.ib(default=None)
    query_has_superlative = attr.ib(default=False)

    @classmethod
    def convert_to_python_type(cls, col_value, schema_type):
        data_types_for_rdf = {"int": "int",
                              "integer": "int",
                              "year": "int",
                              "bool": "bool",
                              "boolean": "bool",
                              "float": "float",
                              "real": "float",
                              "double": "float",
                              "text": "str",
                              "": "str",
                              "date": "datetime",
                              "datetime": "datetime",
                              "timestamp": "datetime",
                             }
        assert schema_type.lower() in data_types_for_rdf, f"Can't parse datatype {schema_type}, have these types: {list(data_types_for_rdf.keys())}"
        data_type = data_types_for_rdf[schema_type.lower()]
        if data_type == "datetime":
            col_value = parse_date_str(col_value)
        elif data_type == "bool":
            if col_value == "F":
                col_value = False
            elif col_value == "T":
                col_value = True
            else:
                col_value = bool(col_value)
        elif data_type == "float" or data_type == "int":
            # convert int to float anyway to make comparisons easier later
            try:
                col_value = float(col_value)
                col_value = round(col_value, 3)
            except ValueError:
                col_value = str(col_value)    
        elif data_type == "str":
            col_value = str(col_value)
        else:
            raise RuntimeError(f"Unknown data type {schema_type} for {col_value}")
        return col_value

    @classmethod
    def update_cols_by_following_foreign_links(cls, cols, schema):
        output_cols = []
        for col in cols:
            col_grnd = col.grounding_column
            visited = {col_grnd : 0}
            looped = False
            while col_grnd in schema.foreign_key_tgt_for_src:
                col_grnd = schema.foreign_key_tgt_for_src[col_grnd]
                if col_grnd not in visited:
                    visited[col_grnd] = len(visited)
                else:
                    looped = True
                    break
            if looped:
                col_grnd = min(list(visited.keys()), key=str)

            col.grounding_column = col_grnd
            output_cols.append(col)

        return output_cols

    @classmethod
    def run_sparql_query_in_virtuoso(cls, query, graph_name,
                                     virtuoso_server="http://localhost:8890/sparql/",
                                     rdf_arc_prefix="arc"):

        query_virtuoso = f"prefix {rdf_arc_prefix}: <{graph_name}/edges/>" + "\n" +\
                         f"prefix key: <{graph_name}/keys/>" + "\n" + query
        params={
            "default-graph": graph_name,
            "query": query_virtuoso,
            "debug": "off",
            "timeout": "",
            "format": "application/json", # format "text/turtle" is supposed to provide better numeric accuracy but does not preserve ordering of the output rows
            "save": "display",
            "fname": ""
        }
        querypart = urllib.parse.urlencode(params).encode("utf-8")

        try:
            response = urllib.request.urlopen(virtuoso_server, querypart).read()
        except urllib.error.HTTPError as e:
            error_message = e.read().decode("utf-8")
            raise RuntimeError(f"Failed to run SPARQL query on {virtuoso_server} (HTTPError {e.code}): {error_message}")

        response_json = json.loads(response)
        output = []
        if "results" in response_json and "bindings" in response_json["results"]:
            data = response_json["results"]["bindings"]
            list_of_col_vars = None
            for data_row in data:
                if list_of_col_vars is None:
                    list_of_col_vars = list(data_row.keys())
                assert len(data_row) == len(list_of_col_vars),\
                    f"Have number of columns {len(data_row)} inconsistent with {len(list_of_col_vars)}: {list(data_row.keys())} (need {list_of_col_vars})"
                
                row = []
                for col in list_of_col_vars:
                    assert col in data_row, f"Could not find {col} in {data_row}"
                    item = data_row[col]
                    assert "value" in item, f"Could not find the key 'value' in {item}"
                    if "type" in item and item["type"].lower() == "uri":
                        item = rdflib.URIRef(item["value"])
                    else:
                        item = item["value"]
                    row.append(item)
                output.append(row)

        return output

    @classmethod
    def execute_query_to_rdf(cls, query, rdf_graph, schema, virtuoso_server=None):
        """ Executes SPARQL query and produces results to be compared with other SQL/SPARQL results
        Args:
            query [QueryToRdf] - SPARQL query with meta-information
            rdf_graph [rdflib.Graph] - database stored in the RDF format
            schema [DatabaseSchema] - loaded SQL database
            virtuoso_server [str] - address of the Virtuoso SPARQL HTTP service (like "http://localhost:8890/sparql/")
                default: None - will use the slow execution of rdflib
        """
        assert isinstance(query, QueryToRdf), f"In QueryResult.execute_query_to_rdf, query should be of type QueryToRdf but have {type(query)}"
        assert hasattr(query, "output_cols") and query.output_cols is not None and len(query.output_cols) >= 1

        if virtuoso_server:
            qres = cls.run_sparql_query_in_virtuoso(query.query, graph_name=rdf_graph.db_id,
                                                    virtuoso_server=virtuoso_server,
                                                    rdf_arc_prefix=rdf_graph.relation_prefix)
        else:
            qres = rdf_graph.query(query.query)

        num_output_cols = len(query.output_cols)
        num_extra_cols = len(query.sorting_info["sorting_cols"])\
             if query.sorting_info is not None and "sorting_cols" in query.sorting_info else 0
        num_output_cols_with_sort_vars = num_output_cols + num_extra_cols


        def parse_element(r, col):
            if type(r) == rdflib.URIRef:
                elem = rdf_graph.original_values_for_rdf_key[r]
            else:
                data_tuple = (col.grounding_column.get_tbl_name(),
                                col.grounding_column.get_col_name(),
                                str(r))
                if data_tuple in rdf_graph.original_values_for_rdf_key:
                    elem = rdf_graph.original_values_for_rdf_key[data_tuple]
                else:
                    elem = str(r)

            col_type = schema.type_for_column_for_table[col.grounding_column.get_tbl_name()][col.grounding_column.get_col_name()]
            if col.aggregator == "count":
                col_type = "FLOAT" # converting counts to float

            elem = cls.convert_to_python_type(elem, col_type)
            return elem

        # extract results
        data = []
        for row in qres:
            res_row = []

            # add output cols
            for i_r in range(num_output_cols):
                r = row[i_r]
                col = query.output_cols[i_r]
                res_row.append(parse_element(r, col))

            # add sort cols
            if num_extra_cols > 0:
                for col_dict in query.sorting_info["sorting_cols"]:
                    r = row[col_dict["idx"]]
                    col = col_dict["col"]
                    res_row.append(parse_element(r, col))

            assert len(res_row) == num_output_cols_with_sort_vars, f"Expecting {num_output_cols_with_sort_vars} items in a row but have {len(res_row)}: {res_row}"
            data.append(tuple(res_row))

        # fix sorting if one of the sorting datatype was changed
        if query.sorting_info and "sorting_cols" in query.sorting_info and "sorted" in query.sorting_info and query.sorting_info["sorted"]:
            sorting_cols = [col_dict["col"] for col_dict in query.sorting_info["sorting_cols"]]
            sorting_col_changed_type = any([schema.type_for_column_for_table[col.grounding_column.get_tbl_name()][col.grounding_column.get_col_name()]\
                != rdf_graph.type_for_column_for_table[col.grounding_column.get_tbl_name()][col.grounding_column.get_col_name()]\
                 for col in sorting_cols])
            if sorting_col_changed_type:
                # need to resort data w.r.t. correct types
                sorting_col_range = range(num_output_cols, num_output_cols_with_sort_vars)
                data = sorted(data, key=lambda row: tuple(row[i_] for i_ in sorting_col_range),
                    reverse=True if query.sorting_info["sorted"]\
                        and "sort_direction" in query.sorting_info\
                        and "desc" == query.sorting_info["sort_direction"].lower() else False)

        output_cols = cls.update_cols_by_following_foreign_links(query.output_cols, schema)
        result = QueryResult(source_type="RDF",
                             output_cols=output_cols,
                             data=data,
                             sorting_info=query.sorting_info)

        # check for query_has_superlative for the WeakArgMax comparison mode
        result.query_has_superlative = query.query_has_superlative
        return result

    @classmethod
    def execute_query_sql(cls, query, schema):
        """ Executes SPARQL query and produces results to be compared with other SQL/SPARQL results
        Args:
            query [str] - SQL query
            schema [DatabaseSchema] - loaded SQL database
        """
        data = schema.execute_sql_query(query)

        # SPIDER parsing: https://github.com/taoyds/spider/blob/4d065ee5afe5bb6fc8e73e28d371b7fccef0d6ef/process_sql.py
        column_names = schema.column_names
        schema_for_parse = SchemaFromSpider.build_from_schema(column_names)
        sql_query_parsed_from_spider = parse_sql_spider(schema_for_parse, query)

        is_outer_distinct = sql_query_parsed_from_spider["select"][0]
        output_cols = sql_query_parsed_from_spider["select"][1]

        result_cols = []
        for output_col in output_cols:
            agg_id_outer = output_col[0]
            val_unit = output_col[1]
            unit_op = val_unit[0]
            col_ops = val_unit[1:]
            assert col_ops[1] is None
            col_op = col_ops[0]
            agg_id = col_op[0]
            id_code = col_op[1]
            is_distinct = col_op[2]

            # select aggregator id that is non-none if any
            agg_id = agg_id if AGG_OPS[agg_id] != "none" else agg_id_outer

            tbl_name, col_name = SchemaFromSpider.parse_id_code(id_code, column_names)
            grnd_col = GroundingKey.make_column_grounding(tbl_name, col_name) if tbl_name != "*" and col_name != "*" else None
            output_col = OutputColumnId.from_grounding(grnd_col, schema=schema)
            output_col = OutputColumnId.add_aggregator(output_col, AGG_OPS[agg_id])
            result_cols.append(output_col)

        # check if there is any explicit '*' in the output: add all columns to the output list for each '*'
        stars_found = [col.grounding_column is None and col.aggregator == "none" for col in result_cols]
        if sum(stars_found) >= 1:
            star_indices = [i for i, x in enumerate(stars_found) if x]
            tbl_names_used = schema_for_parse.get_tbl_names_from_tbl_units(sql_query_parsed_from_spider, schema)
            output_cols_for_star = []
            for tbl_name in tbl_names_used:
                for col_name in schema.column_names[tbl_name]:
                    grnd_col = GroundingKey.make_column_grounding(tbl_name, col_name)
                    output_cols_for_star.append(OutputColumnId.from_grounding(grnd_col, schema=schema))

            result_cols_final = []
            start_idx = 0
            for star_index in star_indices:
                result_cols_final.extend(result_cols[start_idx:star_index])
                result_cols_final.extend(output_cols_for_star)
                start_idx = star_index + 1
            if start_idx < len(result_cols):
                 result_cols_final.extend(result_cols[start_idx:])

            result_cols = result_cols_final

        # convert data to the correct type
        data_python_types = []
        for datum in data:
            row = []
            for i, col in enumerate(result_cols):
                elem = datum[i]
                if col.grounding_column is None:
                    assert col.aggregator == "count", f"Can't only process count aggregator with *, but have {col.aggregator}"
                    elem = float(elem) # converting all numeric types to float
                    row.append(elem)
                    continue
                col_type = schema.type_for_column_for_table[col.grounding_column.get_tbl_name()][col.grounding_column.get_col_name()]
                if col.aggregator == "count":
                    col_type = "FLOAT" # converting counts to float

                elem = cls.convert_to_python_type(elem, col_type)
                row.append(elem)
            data_python_types.append(tuple(row))

        result_cols = cls.update_cols_by_following_foreign_links(result_cols, schema)
        result = QueryResult(source_type="SQL",
                             output_cols=result_cols,
                             data=data_python_types)
        if sql_query_parsed_from_spider.get("limit") and sql_query_parsed_from_spider.get("orderBy"):
            result.limit = sql_query_parsed_from_spider["limit"]
            result.maxmin_via_limit = (result.limit == 1)

        if result.maxmin_via_limit or parsed_sql_has_superlative(sql_query_parsed_from_spider, schema):
            result.query_has_superlative = True

        return result

    @classmethod
    def match_column_names(cls, cols_a, cols_b, require_column_order=False):
        matching_b_for_a = {}
        cols_b_used = [False] * len(cols_b)
        cols_a_matched = [False] * len(cols_a)

        # Try full matches first
        for i_col_a, col_a in enumerate(cols_a):
            for i_col_b, col_b in enumerate(cols_b):
                if require_column_order and i_col_b != i_col_a:
                    continue
                if col_a == col_b:
                    cols_a_matched[i_col_a] = True
                    cols_b_used[i_col_b] = True
                    matching_b_for_a[i_col_a] = i_col_b

        # Try matching to star
        for i_col_a, col_a in enumerate(cols_a):
            if cols_a_matched[i_col_a]:
                continue
            for i_col_b, col_b in enumerate(cols_b):
                if require_column_order and i_col_b != i_col_a:
                    continue
                if cols_b_used[i_col_b]:
                    continue
                if col_b.aggregator == col_a.aggregator and (col_b.grounding_column is None or col_a.grounding_column is None):
                    cols_a_matched[i_col_a] = True
                    cols_b_used[i_col_b] = True
                    matching_b_for_a[i_col_a] = i_col_b

        # Trying to match same column or same aggregator
        for i_col_a, col_a in enumerate(cols_a):
            if cols_a_matched[i_col_a]:
                continue
            for i_col_b, col_b in enumerate(cols_b):
                if require_column_order and i_col_b != i_col_a:
                    continue
                if cols_b_used[i_col_b]:
                    continue
                if col_b.aggregator == col_a.aggregator or  (col_b.grounding_column == col_a.grounding_column):
                    cols_a_matched[i_col_a] = True
                    cols_b_used[i_col_b] = True
                    matching_b_for_a[i_col_a] = i_col_b

        # Matching anything that is still unmatched
        # Makes some sense only if not requiring column order - otherwise anything matches anything
        if not require_column_order:
            for i_col_a, col_a in enumerate(cols_a):
                if cols_a_matched[i_col_a]:
                    continue
                for i_col_b, col_b in enumerate(cols_b):
                    if cols_b_used[i_col_b]:
                        continue
                    cols_a_matched[i_col_a] = True
                    cols_b_used[i_col_b] = True
                    matching_b_for_a[i_col_a] = i_col_b

        matched = all(cols_a_matched) and all(cols_b_used)

        return matched, matching_b_for_a

    def is_equal_to(self, other,
                    require_column_order=False,
                    require_row_order=False,
                    weak_mode_argmax=False,
                    return_message=False,
                    schema=None):
        assert isinstance(self, QueryResult) and isinstance(other, QueryResult),\
            f"both self and other have to be of type QueryResult but have {type(self)} and {type(other)}"

        if (self.source_type == "SQL" and other.source_type == "RDF")\
            or (self.source_type == "SQL" and other.source_type == "SQL" and self.maxmin_via_limit and not other.maxmin_via_limit):
            return other.is_equal_to(self,
                                      require_column_order=require_column_order,
                                      require_row_order=require_row_order,
                                      weak_mode_argmax=weak_mode_argmax,
                                      return_message=return_message,
                                      schema=schema)

        if self.sorting_info is None and other.sorting_info is not None and self.source_type == "RDF" and other.source_type == "RDF":
            return other.is_equal_to(self,
                                      require_column_order=require_column_order,
                                      require_row_order=require_row_order,
                                      weak_mode_argmax=weak_mode_argmax,
                                      return_message=return_message,
                                      schema=schema)

        def out(flag, message):
            if return_message:
                return flag, message
            else:
                return flag

        # check for the number of columns
        if len(self.output_cols) != len(other.output_cols):
            return out(False, f"Different number of columns: {len(self.output_cols)} and {len(other.output_cols)}")
        num_cols = len(self.output_cols)

        # match columns
        num_rows = len(self.data)
        if not require_column_order or num_rows == 0:
            # matchs columns for empty outputs as well to avoid some spurious matches
            matched, matching_other_for_self = self.match_column_names(self.output_cols, other.output_cols, require_column_order=require_column_order)
            if not matched:
                return out(False, f"Could not match {self.output_cols} and {other.output_cols} with require_column_order={require_column_order}")
        else:
            matching_other_for_self = list(range(num_cols))

        # reorder data
        other_data = []
        for row in other.data:
            new_row = [row[matching_other_for_self[i_col]] for i_col in range(num_cols)]
            other_data.append(tuple(new_row))
        self_data = copy.deepcopy(self.data)

        def compare_values(a, b, float_relative_precision=1e-3):
            # try floating point comparison
            try:
                a_float = float(a)
                b_float = float(b)
                diff = abs(a_float - b_float)
                max_abs = max(abs(a_float), abs(b_float))
                if max_abs < float_relative_precision:
                    return diff < float_relative_precision
                else:
                    return diff / max_abs < float_relative_precision
            except:
                return a == b

        if weak_mode_argmax and other.maxmin_via_limit:
            # special case for the pattern "ORDER BY ... DESC LIMIT 1" in SQL that implements argmax
            assert other.limit == 1, f"Something went wrong: have other.limit={other.limit} but len(other_data)={len(other_data)}"

            if self.limit is not None and self.limit != other.limit:
                return out(False, f"Different limits: {other.limit} and {self.limit}")
            
            if not self.query_has_superlative:
                # In the WeakArgmaxMode, the other query should have a superlative op otherwise fallback to the reqular result checking  
                pass
            else:
                if len(other_data) == 1:
                    other_row = other_data[0]
                    for i_row, A in enumerate(self_data):
                        row_found = True
                        for a, b in zip(A, other_row):
                            if not compare_values(a, b):
                                row_found = False
                                break
                        if row_found:
                            return out(True, f"OK - special mode for for orderBy ... limit 1")
                    return out(False, f"Could not find row {other_row} in {self_data}")
                else:
                    assert len(other_data) == 0, f"Something is wrong, len(other_data) should be 0 but have {len(other_data)}"
                    if len(self_data) == 0:
                        return out(True, "Equal")
                    else:
                        return out(False, f"other.data is of len {len(other_data)} but self.data is of len {len(self_data)}")

        # check for the number of rows
        if len(self.data) != len(other.data):
            return out(False, f"Different number of rows: {len(self.data)} and {len(other.data)}")

        if not require_row_order:
            # sort rows if do not need order
            self_data = sorted(self_data)
            other_data = sorted(other_data)

            # compare elements
            for i_row, (A, B) in enumerate(zip(self_data, other_data)):
                for i_step, (a, b) in enumerate(zip(A, B)):
                    if b == "NULL" and other.output_cols[i_step].aggregator != "none":
                        # special case for aggregators over empty sets: SQL produces "NULL", but SPARQL produces other things
                        if a not in ["NULL", "None", 0.0, 0]:
                            return out(False, f"Row {i_row}, item {i_step} should be the same in the answers: {A}, {B}")
                    elif not compare_values(a, b):
                        return out(False, f"Row {i_row}, item {i_step} should be the same in the answers: {A}, {B}")
        elif self.sorting_info is None:
            # compare elements directly
            for i_row, (A, B) in enumerate(zip(self_data, other_data)):
                for i_step, (a, b) in enumerate(zip(A, B)):
                    if not compare_values(a, b):
                        return out(False, f"Row {i_row}, item {i_step} should be the same in the answers: {A}, {B}")
        else:
            # do ordered comparison based on the groups of sorted args
            data_indices = list(range(num_cols))
            sorting_indices = list(range(num_cols, num_cols + len(self.sorting_info["sorting_cols"])))

            # loop over rows with processig equal groups
            i_group_started = 0
            i_row = 0
            while i_row < num_rows:
                if i_row + 1 < num_rows: # not the last item
                    group_continues = tuple(self_data[i_row + 1][i_] for i_ in sorting_indices) == \
                         tuple(self_data[i_group_started][i_] for i_ in sorting_indices)
                else:
                    group_continues = False

                if not group_continues:
                    # processing group i_group_start : i_row + 1
                    other_items_used = {}
                    for i_item_self in range(i_group_started, i_row + 1):
                        item_found = False
                        for i_item_other in range(i_group_started, i_row + 1):
                            if i_item_other in other_items_used:
                                continue

                            lhs_tuple = tuple(self_data[i_item_self][i_] for i_ in data_indices)
                            rhs_tuple = tuple(other_data[i_item_other][i_] for i_ in data_indices)
                            item_match = all(compare_values(a, b) for a, b in zip(lhs_tuple, rhs_tuple)) 
                            if item_match:
                                item_found = True
                                other_items_used[i_item_other] = i_item_self
                                break
                        if not item_found:
                            return out(False, f"Could not match item {tuple(self_data[i_item_self][i_] for i_ in data_indices)} to any of the group {[tuple(d[i_] for i_ in data_indices) for d in other_data[i_group_started : i_row + 1]]}")

                    # starting the next group
                    i_group_started = i_row + 1
                i_row += 1

        return out(True, "Equal")


@attr.s
class OutputColumnId:
    grounding_column = attr.ib()
    aggregator = attr.ib(default="none")

    @classmethod
    def from_grounding(cls, grnd, schema=None):
        if grnd is None:
            return cls(grounding_column=grnd)
        tbl_name = grnd.get_tbl_name()
        if grnd.istbl():
            assert schema is not None, "Need schema to find the pripary key of a table to create its column id"
            col_name = schema.primary_keys[tbl_name]
        else:
            col_name = grnd.get_col_name()

        return cls(grounding_column=GroundingKey.make_column_grounding(tbl_name, col_name))

    @classmethod
    def add_aggregator(cls, column_id, aggregator):
        if aggregator is None:
            aggregator = "none"
        assert aggregator in AGG_OPS, f"Have '{aggregator}' as aggregator but expecting one of these: {AGG_OPS}"
        return attr.evolve(column_id, aggregator=aggregator)
