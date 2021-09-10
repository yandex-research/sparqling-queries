################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import json
import sqlite3
from nltk import word_tokenize
import attr
import copy

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')



class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {'*': "__all__"}
        id = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"
                id += 1

        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        return idMap


def get_schema(db):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """

    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    # fetch table info
    for table in tables:
        cursor.execute("PRAGMA table_info({})".format(table))
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    return schema


def get_schema_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)

    schema = {}
    for entry in data:
        table = str(entry['table'].lower())
        cols = [str(col['column_name'].lower()) for col in entry['col_data']]
        schema[table] = cols

    return schema


def tokenize(string):
    string = str(string)
    string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val

    toks = [word.lower() for word in word_tokenize(string)]
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx-1]
        if pre_tok in prefix:
            toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]

    return toks


def scan_alias(toks):
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    alias = {}
    for idx in as_idxs:
        alias[toks[idx+1]] = toks[idx-1]
    return alias


def get_tables_with_alias(schema, toks):
    tables = scan_alias(toks)
    for key in schema:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key
    return tables


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, column id
    """
    tok = toks[start_idx]
    if tok == "*":
        return start_idx + 1, schema.idMap[tok]

    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        key = tables_with_alias[alias] + "." + col
        return start_idx+1, schema.idMap[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            return start_idx+1, schema.idMap[key]

    assert False, "Error col: {}".format(tok)


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index('none')

    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema):
    """
        :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx+1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif "\"" in toks[idx]:  # token is a string value
        val = toks[idx]
        idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')'\
                and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS and toks[end_idx] not in JOIN_KEYWORDS:
                    end_idx += 1

            idx, val = parse_col_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, default_tables)
            idx = end_idx

    if isBlock:
        assert toks[idx] == ')'
        idx += 1

    return idx, val


def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    conds = []

    while idx < len_:
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        not_op = False
        if toks[idx] == 'not':
            not_op = True
            idx += 1

        assert idx < len_ and toks[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
        op_id = WHERE_OPS.index(toks[idx])
        idx += 1
        val1 = val2 = None
        if op_id == WHERE_OPS.index('between'):  # between..and... special case: dual values
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            assert toks[idx] == 'and'
            idx += 1
            idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
        else:  # normal case: single value
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            val2 = None

        conds.append((not_op, op_id, val_unit, val1, val2))

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
            break

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1  # skip and/or


    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == 'distinct':
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append((agg_id, val_unit))
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','

    return idx, (isDistinct, val_units)


def parse_from(toks, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['sql'], sql))
        else:
            if idx < len_ and toks[idx] == 'join':
                idx += 1  # skip join
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['table_unit'],table_unit))
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
            if len(conds) > 0:
                conds.append('and')
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc' # default type is 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        return idx, int(toks[idx-1])

    return idx, None


def parse_sql(toks, start_idx, tables_with_alias, schema):
    isBlock = False # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(toks, start_idx, tables_with_alias, schema)
    sql['from'] = {'table_units': table_units, 'conds': conds}
    # select clause
    _, select_col_units = parse_select(toks, idx, tables_with_alias, schema, default_tables)
    idx = from_end_idx
    sql['select'] = select_col_units
    # where clause
    idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
    sql['where'] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['groupBy'] = group_col_units
    # having clause
    idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, default_tables)
    sql['having'] = having_conds
    # order by clause
    idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['orderBy'] = order_col_units
    # limit clause
    idx, limit_val = parse_limit(toks, idx)
    sql['limit'] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    return idx, sql


def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def get_sql(schema, query):
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema)

    return sql


def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx


@attr.s
class SchemaFromSpider:
    idMap = attr.ib()
    schema = attr.ib()

    @classmethod
    def build_from_schema(cls, column_names):
        column_names_backup = column_names
        column_names = {}
        for table_name, cols in column_names_backup.items():
            column_names[table_name.lower()] = [col_name.lower() for col_name in cols]

        idMap = cls.build_id_map(column_names)
        schema_spider = cls(idMap=idMap, schema=column_names)
        return schema_spider

    @classmethod
    def build_id_map(cls, column_names):
        idMap = {'*': "__all__"}
        id = 1
        for key, vals in column_names.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"
                id += 1

        for key in column_names:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        return idMap

    @classmethod
    def parse_id_code(cls, id_code, column_names):
        assert id_code[:2] == "__"
        assert id_code[-2:] == "__"
        id_code = id_code[2:-2]
        if id_code == "all":
            tbl, col = "*", "*"
        else:
            if "." in id_code:
                id_code = id_code.split('.')
                assert len(id_code) == 2
                tbl, col = id_code

                for tbl_name in list(column_names.keys()):
                    if tbl_name.lower() == tbl.lower():
                        tbl = tbl_name
                        break

                for col_name in list(column_names[tbl]):
                    if col_name.lower() == col.lower():
                        col = col_name
                        break
            else:
                return id_code, "*"

        return tbl, col

    def find_code(self, code,):
        found_k = None
        for k, v in self.idMap.items():
            if v == code:
                found_k = k
                break

        assert found_k is not None, f"Could not find code {code}"
        return found_k

    def get_tbl_names_from_tbl_units(self, sql_query_parsed_from_spider, schema):
        tbl_names = [t[1] for t in sql_query_parsed_from_spider["from"]["table_units"]] # in the format like "__concerts__"
        # get the lower cased table names
        for i_tbl in range(len(tbl_names)):
            for name, name_hash in self.idMap.items():
                if name_hash == tbl_names[i_tbl]:
                    tbl_names[i_tbl] = name
                    break
        # get the original table names
        for i_tbl in range(len(tbl_names)):
            for name in schema.table_names:
                if name.lower() == tbl_names[i_tbl]:
                    tbl_names[i_tbl] = name
                    break
        return tbl_names


class CreatorSqlFromParse():
    
    def __init__(self,  schema_for_parse, alias_start_idx):
        self.schema = schema_for_parse
        self.table_aliases = {}
        self.alias_start_idx = alias_start_idx

    @classmethod
    def to_str(cls, sql_dict, schema_for_parse, alias_start_idx=0):
        parser = cls(schema_for_parse, alias_start_idx)
        s = parser.sql_to_str(sql_dict)
        return s

    def id_code_to_str(self, code, use_aliases=True):
        tbl, col = self.schema.parse_id_code(code, self.schema.schema)

        if tbl == "*" and col == "*":
            s = "*"
        else:
            if tbl in self.table_aliases and use_aliases:
                tbl = self.table_aliases[tbl]
            if col == "*":
                # table
                s = tbl
            else:
                s = f"{tbl}.{col}"
        return s

    def col_unit_to_str(self, col_unit):
        s = self.id_code_to_str(col_unit[1])
        if col_unit[2]:
            s = "DISTINCT " + s
        if AGG_OPS[col_unit[0]] != "none":
            s = f"{AGG_OPS[col_unit[0]]}({s})"
        return s

    def val_unit_to_str(self, val_unit):
        s = self.col_unit_to_str(val_unit[1])
        if val_unit[2]:
            assert UNIT_OPS[val_unit[0]] != 'none', f"Can't parse col_unit {val_unit} - bad unit_op '{UNIT_OPS[val_unit[0]]}'"
            s += UNIT_OPS[val_unit[0]] + self.col_unit_to_str(val_unit[2])
        return s

    def select_to_str(self, sql_data):
        assert "select" in sql_data, f"Could not find 'select' in {sql_data}"
        data = sql_data["select"]
        
        query = "SELECT "
        if data[0]:
            query += "DISTINCT "

        val_strs = []
        for agg_id, val_unit in data[1]: 
            val_s = self.val_unit_to_str(val_unit)
            if AGG_OPS[agg_id] != "none":
                val_s = f"{AGG_OPS[agg_id]}({val_s})"
            val_strs.append(val_s)
        query = query + ", ".join(val_strs)

        return query

    def condition_to_str(self, cond):
        have_not = cond[0]
        sign = WHERE_OPS[cond[1]]
        lhs = self.val_unit_to_str(cond[2])
        if sign != "between":
            if isinstance(cond[3], dict) and "select" in cond[3]:
                # have subquery
                rhs = CreatorSqlFromParse.to_str(cond[3], self.schema, alias_start_idx=len(self.table_aliases))
                rhs = f"({rhs})"
            else:
                try:
                    rhs = self.col_unit_to_str(cond[3])
                except:
                    try:
                        if int(cond[3]) == float(cond[3]):
                            rhs = str(int(cond[3]))
                        else:
                            rhs = str(float(cond[3]))
                    except:
                        rhs = str(cond[3])
            cond_s = f"{lhs} {sign.upper() if not have_not else 'NOT ' + sign.upper()} {rhs}"
        else:
            cond_s = f"{lhs} BETWEEN {cond[3]} and {cond[4]}" 

        return cond_s

    def condition_list_to_str(self, condition_list):
        if condition_list:
            conditions = []
            i_cond = 0
            while i_cond < len(condition_list):
                if i_cond % 2 == 0:
                    cond_s = self.condition_to_str(condition_list[i_cond])
                    conditions.append(cond_s)
                else:
                    conditions.append(condition_list[i_cond].upper())
                i_cond += 1
            return " ".join(conditions)
        else:
            return ""

    def from_to_str(self, sql_data):
        assert "from" in sql_data, f"Could not find 'from' in {sql_data}"
        data = sql_data["from"]
        
        tbl_units = []
        for i_unit, unit in enumerate(data["table_units"]):
            if unit[0] == "table_unit":
                tbl_name = self.id_code_to_str(unit[1], use_aliases=False)
                alias = f"T{i_unit + self.alias_start_idx}"
                assert tbl_name not in self.table_aliases,\
                    f"Cannot have several aliases for table '{tbl_name}' in the same (self-joins are not supported) beause the Spider format looses this information"
                self.table_aliases[tbl_name] = alias
                unit_s = f"{tbl_name} AS {alias}"
            else:
                # have subquery here
                unit_s = CreatorSqlFromParse.to_str(unit[1], self.schema, alias_start_idx=len(self.table_aliases))
                alias = f"S{i_unit + self.alias_start_idx}"
                tbl_name = alias
                unit_s = f"({unit_s})"
            
            tbl_units.append(unit_s)

        # merge to a single string (tricky and with heuristics)
        full_str = tbl_units[0]
        for i_unit, unit_s in enumerate(tbl_units[1:]):
            full_str += " JOIN " + unit_s
            i_cond = i_unit * 2
            if len(data["conds"]) >=  i_cond + 1: 
                cond_s = self.condition_to_str(data["conds"][i_cond])
                full_str += " ON " + cond_s

        query = " FROM " + full_str
        return query

    def where_to_str(self, sql_data):
        if "where" not in sql_data or not sql_data["where"]:
            return ""

        data = sql_data["where"]
        conditions_str = self.condition_list_to_str(data)

        query = " WHERE " + conditions_str
        return query

    def groupBy_to_str(self, sql_data):
        if "groupBy" not in sql_data or not sql_data["groupBy"]:
            return ""
        
        data = sql_data["groupBy"]
        col_strs = [self.col_unit_to_str(unit) for unit in data]
        
        query = " GROUP BY " + ", ".join(col_strs)
        return query

    def having_to_str(self, sql_data):
        if "having" not in sql_data or not sql_data["having"]:
            return ""

        data = sql_data["having"]
        conditions_str = self.condition_list_to_str(data)

        query = " HAVING " + conditions_str
        return query

    def orderBy_to_str(self, sql_data):
        if "orderBy" not in sql_data or not sql_data["orderBy"]:
            return ""

        data = sql_data["orderBy"]

        direction_str = "ASC" if data[0].lower() == "asc" else "DESC"
        col_strs = [self.val_unit_to_str(val_unit) for val_unit in data[1]]

        query = " ORDER BY " + ", ".join([s + " " + direction_str for s in col_strs])
        return query

    def limit_to_str(self, sql_data):
        if "limit" not in sql_data or not sql_data["limit"]:
            return ""
        data = sql_data["limit"]
        query = " LIMIT " + str(data)
        return query

    def sql_to_str(self, sql_data):
        sql_query_from = self.from_to_str(sql_data) # parse FROM first to get aliases
        sql_query = self.select_to_str(sql_data)
        sql_query += sql_query_from
        sql_query += self.where_to_str(sql_data)
        sql_query += self.groupBy_to_str(sql_data)
        sql_query += self.having_to_str(sql_data)
        sql_query += self.orderBy_to_str(sql_data)
        sql_query += self.limit_to_str(sql_data)
        
        sql_ops = ["intersect", "except", "union"]
        ops_used = [sql_op in sql_data and sql_data[sql_op] is not None for sql_op in sql_ops]
        assert sum(ops_used) <= 1, f"Spider only supports at most one intersect/union/except, but have {sum(ops_used)}"
        if sum(ops_used) > 0:
            i_op = ops_used.index(True)
            query_part = CreatorSqlFromParse.to_str(sql_data[sql_ops[i_op]], self.schema, alias_start_idx=len(self.table_aliases))
            sql_query += " " + sql_ops[i_op].upper() + " " + query_part

        return sql_query


def replace_orderByLimit1_to_subquery(sql_query, column_names):
    schema_for_parse = SchemaFromSpider.build_from_schema(column_names)
    sql_data = get_sql(schema_for_parse, sql_query)

    try:
        # detect the "order by ... limit 1" construction in sql, trying to replace it with a subquery
        # using information from the parsing of SQL in SPIDER
        assert "limit" in sql_data and sql_data["limit"] == 1 and "orderBy" in sql_data
            
        sort_direction = sql_data["orderBy"][0]
        assert len(sql_data["orderBy"][1]) == 1, f"Only support sorting w.r.t. one argument here"
        sort_val_unit = sql_data["orderBy"][1][0]
        assert sort_val_unit[0] == 0 and sort_val_unit[2] is None, f"Only support sorting w.r.t. one column here"
        id_code_for_sort = sort_val_unit[1][1]
        tbl_name_for_sort, col_name_for_sort = SchemaFromSpider.parse_id_code(id_code_for_sort, column_names)

        # can only do the substitution for simple queries
        banned_sql_ops = ["intersect", "except", "union", "having"]
        for sql_op in banned_sql_ops:
            assert sql_op not in sql_data or not sql_data[sql_op], f"Can't process queries with '{sql_op}'"

        sql_data_mod = copy.deepcopy(sql_data)
        sql_data_subquery = copy.deepcopy(sql_data)

        # changing this:
        #   SELECT T1.cylinders FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T2.Model  =  'volvo' ORDER BY T1.accelerate ASC LIMIT 1
        # to this:
        #   SELECT T0.cylinders FROM cars_data AS T0 JOIN car_names AS T1 ON T0.id = T1.makeid WHERE T1.model = "volvo" AND T0.accelerate = 
        #       (SELECT T1.accelerate FROM cars_data AS T2 JOIN car_names AS T3 ON T2.id = T3.makeid WHERE T3.model = "volvo" ORDER BY T1.accelerate ASC LIMIT 1)   

        # support the group op: changing this:
        #   SELECT YEAR FROM concert GROUP BY YEAR ORDER BY count(*) DESC LIMIT 1
        # to this:
        #   select year from concert group by year having count(*) = (SELECT count(*) from concert group by year order by count(*) DESC limit 1 )

        sql_data_subquery["select"] = (False, [(AGG_OPS.index("none"), val_unit) for val_unit in sql_data_subquery["orderBy"][1]])

        sql_data_mod["orderBy"] = []
        sql_data_mod["limit"] = []

        cond_op = "having" if "groupBy" in sql_data and sql_data["groupBy"] else "where"
        new_condition = (False, WHERE_OPS.index("="), sort_val_unit, sql_data_subquery, None)
        if not sql_data_mod[cond_op]:
            sql_data_mod[cond_op] = [new_condition]
        else:
            sql_data_mod[cond_op].append("and")
            sql_data_mod[cond_op].append(new_condition)

        sql_query_mod = CreatorSqlFromParse.to_str(sql_data_mod, schema_for_parse)
        
        return sql_query_mod
    except:
        return sql_query
