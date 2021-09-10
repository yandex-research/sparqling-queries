import ast
from collections import OrderedDict
from difflib import SequenceMatcher
import re
import sqlite3

import nltk
from word2number import w2n

from qdmr2sparql.query_generator import GroundingKey

SQL_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except', 'distinct')
JOIN_KEYWORDS = ('join', 'on', 'by', 'having') #'as'
WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
COND_OPS = ('and', 'or')
ORDER_OPS = ('desc', 'asc')

def convert_str_to_list(x):
    x = ast.literal_eval(x)
    return [n.strip() for n in x]

def add_to_dict(d, key, val):
    if d.get(key) is None:
        d[key] = [val]
    else:
        if val not in d[key]:
            d[key].append(val)
    return d

def replace_cardinal(arg):
    arg_res = []
    for arg_w in arg.split(' '):
        try:
            num = w2n.word_to_num(arg_w)
            arg_res.append(str(num))
        except:
            arg_res.append(arg_w)
    return ' '.join(arg_res)

def preproc_arg(arg):
    arg = replace_cardinal(arg)
    arg = arg.replace('\'', '').replace('\"', '')
    arg = re.sub('#\d+', '', arg).strip()
    return arg


def clean_qdmr_arg(arg):
    arg = arg.replace('of #REF', '').replace('#REF', '')
    arg = preproc_arg(arg)
    return arg


def qdmr_args(qdmr_ex):
    '''
    Parse qdmr logical form to arguments
    :param qdmr_ex: qdmr logical form (e.g., "["SELECT[\'directors\']")
    :type qdmr_ex: str with list of qdmr steps
    :return: arguments of qdmr steps (e.g., "directors") and their idx in qdmr,
             masks of comparative, superlative and sort steps, comparative '#d' args
    :rtype: dict with str arguments as keys and (index of qdmr step, index of argument) tuples as values,
            dict with lists of boolean vals, dict with idx as keys and '#d' str as keys
    '''
    args_ex = {}
    is_operator = {'is_comparative': [], 'is_superlative': [], 'is_sort': []}
    comparative_ref = {}
    for num_step, ex in enumerate(convert_str_to_list(qdmr_ex)):
        comparative_idx, filter_idx = ex.find('COMPARATIVE'), ex.find('FILTER')
        superlative_idx, sort_idx = ex.find('SUPERLATIVE'), ex.find('SORT')
        is_operator['is_comparative'].append(comparative_idx >= 0 or filter_idx >= 0)
        is_operator['is_superlative'].append(superlative_idx >= 0)
        is_operator['is_sort'].append(sort_idx >= 0)

        all_args = convert_str_to_list(ex[ex.find('['): ex.find(']') + 1])

        all_args = [ex.replace('of #REF', '').replace('#REF', '') for ex in all_args]


        # comparative with '#d'
        if comparative_idx >= 0:
            pattern = re.compile("^#\d+$")
            match = pattern.match(all_args[-1].split(' ')[-1])
            if match is not None:
                comparative_ref[num_step] = match.group(0)

        for num_arg, arg in enumerate(all_args):
            arg = preproc_arg(arg)
            if arg and arg not in ('count', 'sum', 'avg') or is_operator['is_sort'][-1] and num_arg == 1:
                if arg in ('max', 'min') and not is_operator['is_superlative'][-1]:
                    continue
                add_to_dict(args_ex, arg, (num_step, num_arg))
        #for i in re.findall("\'[\w\s]+\'", ex.replace('of #REF', '').replace('#REF', '')):
        #    args_ex.append(i[1:-1].strip())
    return args_ex, is_operator, comparative_ref


def preproc_t_prefix(t_prefix, args):
    '''
    Preprocess aliases in sql query (tx.column_name -> table_name.column_name)
    :param t_prefix: pairs of table names anf their aliases
    :param args: arguments of sql operators
    :type t_prefix: dict with 'tx' keys and 'table_name' values
    :type args: list of str
    :return: arguments of sql operators with replaced aliases to table names
    :rtype: list of str
    '''
    uncased_prefix, cased_prefix = 't\d', 'T\d'
    for i in range(len(args)):
        words = args[i]
        for t_variant in (uncased_prefix, cased_prefix):
            prefix = re.findall(t_variant, words)
            if len(prefix) > 0:
                args[i] = words.replace(prefix[0], t_prefix[prefix[0].lower()])
                break

    return args

def sql_args(sql_ex):
    '''
    Parse sql query to arguments
    :param sql_ex: sql query
    :type sql_ex: str
    :return: arguments of sql operators and mask of table names
    :rtype: list of str, list with 1 for table arguments and 0 for others
    '''
    args_ex = []
    is_table = []

    sql_ex = sql_ex.replace('(', ' ').replace(')', ' ').replace(';', '').replace('"', '').replace('\'', '')
    sql_spl = sql_ex.split()
    prev_word = False
    i = 0
    t_prefix = {}
    while i != len(sql_spl):
        word = sql_spl[i].lower()
        # check sql keyword
        if word in SQL_KEYWORDS or word in JOIN_KEYWORDS or word in WHERE_OPS or\
            word in UNIT_OPS or word in AGG_OPS or word in COND_OPS or word in ORDER_OPS or word == ',':
            prev_word = False
        elif word == 'as':
            # 'table_name as tx'...
            prev_word = False
            if not len(re.findall('t\d', sql_spl[i + 1])) and not len(re.findall('T\d', sql_spl[i + 1])):
                raise ValueError
            t_prefix[sql_spl[i + 1].lower()] = sql_spl[i - 1]
            i += 1
        else:
            # argument
            if sql_spl[i - 1].lower() == 'from' or sql_spl[i - 1].lower() == 'join':
                # table
                is_table.append(1)
            elif not prev_word:
                # not table
                is_table.append(0)
            else:
                pass

            if prev_word:
                args_ex[-1] += ' ' + sql_spl[i]
            else:
                args_ex.append(' '.join(sql_spl[i].split('_')))
            prev_word = True
        i += 1
    return preproc_t_prefix(t_prefix, args_ex), is_table


def similarity_of_words(qdmr_w, sql_w):
    match = SequenceMatcher(None, qdmr_w, sql_w).find_longest_match(0, len(qdmr_w), 0, len(sql_w))
    if match.size > 1:
        p = match.size / len(qdmr_w)
        r = match.size / len(sql_w)
        return 2 * p * r / (p + r)
    else:
        return 0.0


def measure_similarity(qdmr, sql, threshold=0.5):
    '''
    How close qdmr argument is to sql argument in terms of ngrams
    :param qdmr: qdmr argument
    :param sql: sql argument
    :param threshold: threshold for similarity measurement
    :type qdmr: str
    :type sql: str
    Ltype threshold: float in [0...1]
    :return: similar or not
    :rtype: bool
    '''
    qdmr_spl = qdmr.split()
    sql_spl = sql.split()
    res = False
    for qdmr_w in qdmr_spl:
        for sql_w in sql_spl:
            res += similarity_of_words(qdmr_w, sql_w) >= threshold
    return res

def stem_match(qdmr, sql):
    '''
    Whether qdmr argument has the same stem as sql argument or not
    :param qdmr: qdmr argument
    :param sql: sql argument
    :type qdmr: str
    :type sql: str
    :return: flag of stem matching
    :rtype: bool
    '''
    ps = nltk.stem.PorterStemmer()
    sno = nltk.stem.SnowballStemmer('english')
    qdmr_spl = qdmr.split()
    sql_spl = sql.split()
    for qdmr_w in qdmr_spl:
        for sql_w in sql_spl:
            if ps.stem(sql_w) == ps.stem(qdmr_w) or sno.stem(sql_w) == sno.stem(qdmr_w):
                return True
    return False

def get_values(db_id, path_to_db):
    '''
    Get all values from database with the corresponding table and column names
    :param db_id: database id (e.g., 'concert_singer')
    :param path_to_db: path to the folder with databases
    :type db_id: str
    :type path_to_db: str
    :return: values from database
    :rtype: dict = {tables : [[columns], [tuples values]]}
    '''
    conn = sqlite3.connect(path_to_db + '/' + db_id + '/' + db_id + '.sqlite')
    conn.text_factory = lambda b: b.decode(errors = 'ignore')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    values_all = {}
    for t in tables:
        cursor = conn.execute('select * from ' + t[0])
        names = list(map(lambda x: x[0], cursor.description))
        values_all[t[0]] = [names]
        cursor.execute("SELECT * FROM " + t[0])
        vals = cursor.fetchall()
        values_all[t[0]].append(vals)
    return values_all

def find_db_value(qdmr_arg, dict_values, sql_val=None):
    '''
    Find corresponding value (if any) to argument
    :param qdmr_arg: qdmr argument
    :param dict_values: values from database
    :param sql_val: value from SQL query corresponding to qdmr argument or None
    :type qdmr_arg: str
    :type dict_values: dict = {tables : [[columns], [tuples values]]}
    :type sql_val: str or None
    :return: list of all matched values with corresponding table and column names
    :rtype: list of (table, column, value) tuples
    '''
    all_matches = set()
    for table, table_vals in dict_values.items():
        if len(table_vals) < 2:
            # empty table
            continue
        for values in table_vals[1]:
            for i, val in enumerate(values):
                val = str(val)
                val_processed = val.lower()
                val_processed = val_processed.replace('(', '').replace(')', '').replace(';', '').replace('"', '').replace('\'', '')
                val_processed = val_processed.strip()
                if sql_val is None and stem_match(qdmr_arg.lower().strip(), val_processed) > 0:
                    # qdmr_arg not from SQL, find similar value
                    all_matches.add((table, table_vals[0][i], val))
                elif sql_val is not None and val_processed == sql_val.lower().strip():
                    # qdmr_arg is from SQL, find th same value
                    all_matches.add((table, table_vals[0][i], val))

    return list(all_matches)

def get_sql_comparative(sql_ex, val):
    '''
    Get type of comparative op corresponding to given value (e.g., '>')
    :param sql_ex: sql query
    :param val: value
    :type sql_ex: str
    :type val: str
    :return: comparison operator if exists
    :rtype: op or None
    '''
    sql_ex = sql_ex.replace('(', ' ').replace(')', ' ').replace(';', '').replace('"', '').replace('\'', '')
    sql_spl = sql_ex.lower().split()

    for val in val.lower().split(' '):
        for i in range(len(sql_spl)):
            word = sql_spl[i]
            if word == val:
                return sql_spl[i - 1]
    # print(val)
    #raise RuntimeError
    return None

def value_to_grounding(grounding, grounding_index, dict_values, sql_val, comparative_op=None,
                        column_comparative=None, not_used_sql_args=None, qdmr_db_value=True):
    '''
    Add argument and value from database or from sql query to grounding
    :param grounding: existing grounding
    :param qdmr_arg: qdmr argument
    :param dict_values: values from database
    :param sql_val: value from SQL query corresponding to qdmr argument
    :param comparative_op: comparison operator or None, default=None
    :param column_comparative: grounding to column that contains value or None
    :param not_used_sql_args: set of sql args that still not used
    :param qdmr_db_value: if true, search qdmr value in db, else - search sql value
    :type grounding: dict with qdmr arguments as keys and GroundingKey as values
    :type qdmr_arg: str
    :type dict_values: dict = {tables : [[columns], [tuples values]]}
    :type comparative_op: GroundingKey or None
    :type column_comparative: str or None
    :type sql_val: str
    :type not_used_sql_args: set of str
    :type qdmr_db_value: bool
    :return: updated grounding
    :rtype: dict
    '''
    db_value = grounding_index.key_str if qdmr_db_value else sql_val
    all_matches = find_db_value(db_value, dict_values, sql_val)
    is_found = False

    if len(all_matches) > 0:
        # value from database
        for tab, col, val in all_matches:
            #if val != sql_val:
            #    print(val, sql_val)

            # workaround for flight_2 db values and spider_dev_388
            if val.strip() == sql_val or val == sql_val.lower() or val.lower() == sql_val:
                val = sql_val

            if comparative_op is not None:
                if column_comparative is None:
                    add_to_dict(grounding, grounding_index,
                                GroundingKey("comparative", (comparative_op, val, GroundingKey.make_column_grounding(tab, col))))
                    if not_used_sql_args:
                        not_used_sql_args.discard(sql_val)
                    is_found = True
                elif tab == column_comparative.get_tbl_name() and col == column_comparative.get_col_name():
                    add_to_dict(grounding, grounding_index,
                                GroundingKey("comparative", (comparative_op, val, column_comparative)))
                    if not_used_sql_args:
                        not_used_sql_args.discard(sql_val)
                    is_found = True
                    break
            else:
                add_to_dict(grounding, grounding_index, GroundingKey.make_value_grounding(tab, col, val))
                if not_used_sql_args:
                    not_used_sql_args.discard(sql_val)
                is_found = True
    if qdmr_db_value and (len(all_matches) == 0 or not is_found):
        # value not from database
        if comparative_op is not None:
            if column_comparative is None:
                add_to_dict(grounding, grounding_index, GroundingKey("comparative", (comparative_op, sql_val)))
                if not_used_sql_args:
                    not_used_sql_args.discard(sql_val)
            else:
                add_to_dict(grounding, grounding_index,
                        GroundingKey("comparative", (comparative_op, sql_val, column_comparative)))
                if not_used_sql_args:
                    not_used_sql_args.discard(sql_val)
                is_found = True
        else:
            add_to_dict(grounding, grounding_index, GroundingKey.make_value_grounding(None, None, sql_val))

    #assert column_comparative is None or is_found
    return grounding

def match_sql_db_names(scheme, dict_values):
    '''
    Match table and column names used in SQL query and real DB
    :param scheme: database scheme
    :param dict_values: values from database
    :type scheme: list of [idx, table name, column name, PK or FK] lists
    :type dict_values: dict = {tables : [[columns], [tuples values]]}
    :return: dict with matched table names, dict with matched column names
    :rtype: dict = {sql_tab_name: db_tab_name}, dict  = {sql_col_name: db_col_name}
    '''
    table_names = dict()
    col_names = dict()
    idx_tab = 0
    idx_col = 0
    for tab, el in dict_values.items():
        table_names[idx_tab] = tab
        for col in el[0]:
            col_names[idx_col] = col
            idx_col += 1
        idx_tab += 1

    prev_tab = scheme[1][1]
    idx_tab = 0
    db_name = table_names.pop(0)
    table_names[prev_tab] = db_name  # replace {0: db_tab_name} with {sql_tab_name: db_tab_name}
    table_names[db_name.lower().replace('_', ' ')] = db_name
    for el in scheme[1:]:
        idx = el[0]
        tab = el[1]
        col = el[2]
        if tab != prev_tab:
            prev_tab = tab
            idx_tab += 1
            db_name = table_names.pop(idx_tab)
            table_names[prev_tab] = db_name
            table_names[db_name.lower().replace('_', ' ')] = db_name

        db_name = col_names.pop(idx - 1)
        col_names[col] = db_name
        col_names[db_name.lower().replace('_', ' ')] = db_name

    return table_names, col_names

def postprocess_db_names(values, scheme, dict_values):
    '''
    Replace table and column names from SQL query to the original ones from db
    :param values: all possible grounding of one qdmr arg
    :param scheme: database scheme
    :param dict_values: values from database
    :type values: list of GroundingKey
    :type scheme: list of [idx, table name, column name, PK or FK] lists
    :type dict_values: dict = {tables : [[columns], [tuples values]]}
    :return: all possible grounding of one qdmr arg with preprocessed table and column names
    :rtype: list of GroundingKey
    '''
    table_names, col_names = match_sql_db_names(scheme, dict_values)
    new_values = []
    for gk in values:
        if gk is not None:
            if gk.type == 'tbl':
                gk = GroundingKey.make_table_grounding(table_names[gk.keys[0]])
            elif gk.type == 'col':
                gk = GroundingKey.make_column_grounding(table_names[gk.keys[0]], col_names[gk.keys[1]])
        new_values.append(gk)
    return new_values

def postprocess_grounding(grounding, qdmr_parsed_args, scheme, dict_values, origin_db_names):
    '''
    Store only unique grounding, replace table and column names with db versions if origin_db_names=True,
    convert all classes to dicts for json dump
    :param grounding: grounding between qdmr and sql arguments
    :param qdmr_parsed_args:  arguments of qdmr steps (e.g., "directors") and their idx in qdmr
    :param scheme: database scheme
    :param dict_values:  values from database
    :param origin_db_names: if true use original table and column names from db
    :type grounding: dict with qdmr arguments as keys and GroundingKey as values
    :type qdmr_parsed_args: dict with str arguments as keys and (index of qdmr step, index of argument) tuples as values
    :type scheme: list of [idx, table name, column name, PK or FK] lists
    :type dict_values: dict = {tables : [[columns], [tuples values]]}
    :type origin_db_names: bool
    :return: postprecessed grounding between qdmr and sql arguments
    :rtype: dict with qdmr arguments as keys and (dict of argument idx, list of all possible grounding dicts) as values
    '''
    processed_grounding = OrderedDict()
    for key, val in grounding.items():
        val = list(set(val)) # make set
        if origin_db_names: # postproc names if origin_db_names=True
            val = postprocess_db_names(val, scheme, dict_values)

        # convert for json
        idx_all = qdmr_parsed_args[key]
        for num_step, num_arg in idx_all:
            idx_in_qdmr = {'num_step': num_step, 'num_arg': num_arg}
            converted_val = []
            for v in val:
                if v is not None:
                    if v.type == 'comparative' and len(v.keys) == 3:
                        v = GroundingKey("comparative", (v.keys[0], v.keys[1], v.keys[2].__dict__))
                    converted_val.append(v.__dict__)
                else:
                    converted_val.append(v)
            add_to_dict(processed_grounding, key, (idx_in_qdmr, converted_val))
    return processed_grounding

def correct_sql_names(sql_name, scheme, use_preproc=True):
    ''' Unification of SQL arg names. Use preprocessed versions from spider or original from db.
    '''
    for el in scheme:
        tab_name, col_name, orig_tab_name, orig_col_name = el[1:5]
        if sql_name.lower() == tab_name or sql_name.lower() == orig_tab_name.lower():
            if use_preproc:
                return tab_name
            else:
                return orig_tab_name.replace('_', ' ')
        elif sql_name.lower() == col_name or sql_name.lower() == orig_col_name.lower():
            if use_preproc:
                return col_name
            else:
                return orig_col_name.replace('_', ' ')

    return sql_name

def get_column_comparartive(sql_parsed_args, scheme, idx_val_sql):
    i = 0
    for sql_arg in sql_parsed_args[:idx_val_sql][::-1]:
        sql_arg = sql_arg.split('.')
        if len(sql_arg) == 1:
            sql_arg_tab = None
            sql_arg_col = sql_arg[-1].lower()
        else:
            assert len(sql_arg) == 2
            sql_arg_tab, sql_arg_col = sql_arg
            sql_arg_tab = sql_arg_tab.lower().replace('_', ' ')
            sql_arg_col = sql_arg_col.lower().replace('_', ' ')
        for el in scheme[1:]:
            tab_name = el[1].lower().replace('_', ' ')
            col_name = el[2].lower().replace('_', ' ')
            orig_tab_name = el[3].lower().replace('_', ' ')
            orig_col_name = el[4].lower().replace('_', ' ')
            if sql_arg_col == col_name or sql_arg_col == orig_col_name:
                if sql_arg_tab is not None:
                    if sql_arg_tab == tab_name or sql_arg_tab == orig_tab_name:
                        return GroundingKey.make_column_grounding(el[3], el[4])
                else:
                    return GroundingKey.make_column_grounding(el[3], el[4])
        i += 1
        assert i < 3, print(sql_arg, idx_val_sql)
