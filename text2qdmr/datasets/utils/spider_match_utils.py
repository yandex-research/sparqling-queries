import re
import string

import nltk.corpus

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNKS = set(a for a in string.punctuation)


# schema linking, similar to IRNet
def compute_schema_linking(question, column, table, value=None):
    def partial_match(x_list, y_list):
        x_str = " ".join(x_list)
        y_str = " ".join(y_list)
        if x_str in STOPWORDS or x_str in PUNKS:
            return False
        if re.match(rf"\b{re.escape(x_str)}\b", y_str):
            assert x_str in y_str
            return True
        else:
            return False

    def exact_match(x_list, y_list):
        x_str = " ".join(x_list)
        y_str = " ".join(y_list)
        if x_str == y_str:
            return True
        else:
            return False

    q_col_match = dict()
    q_tab_match = dict()
    q_val_match = dict()

    col_id2list = dict()
    for col_id, col_item in enumerate(column):
        if col_id == 0:
            continue
        col_id2list[col_id] = col_item

    tab_id2list = dict()
    for tab_id, tab_item in enumerate(table):
        tab_id2list[tab_id] = tab_item
    
    if value:
        val_id2list = dict()
        for val_id, val_item in enumerate(value):
            val_id2list[val_id] = val_item
    else:
        val_id2list = {}

    # 5-gram
    n = 5
    while n > 0:
        for i in range(len(question) - n + 1):
            n_gram_list = question[i:i + n]
            n_gram = " ".join(n_gram_list)
            if len(n_gram.strip()) == 0:
                continue
            # exact match case
            for col_id in col_id2list:
                if exact_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        q_col_match[f"{q_id},{col_id}"] = "CEM"
            for tab_id in tab_id2list:
                if exact_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        q_tab_match[f"{q_id},{tab_id}"] = "TEM"
            for val_id in val_id2list:
                if exact_match(n_gram_list, val_id2list[val_id]):
                    for q_id in range(i, i + n):
                        q_val_match[f"{q_id},{val_id}"] = "VEM"

            # partial match case
            for col_id in col_id2list:
                if partial_match(n_gram_list, col_id2list[col_id]):
                    for q_id in range(i, i + n):
                        if f"{q_id},{col_id}" not in q_col_match:
                            q_col_match[f"{q_id},{col_id}"] = "CPM"
            for tab_id in tab_id2list:
                if partial_match(n_gram_list, tab_id2list[tab_id]):
                    for q_id in range(i, i + n):
                        if f"{q_id},{tab_id}" not in q_tab_match:
                            q_tab_match[f"{q_id},{tab_id}"] = "TPM"
            for val_id in val_id2list:
                if partial_match(n_gram_list, val_id2list[val_id]):
                    for q_id in range(i, i + n):
                        if f"{q_id},{val_id}" not in q_val_match:
                            q_val_match[f"{q_id},{val_id}"] = "VPM"
        n -= 1
    return {"q_col_match": q_col_match, "q_tab_match": q_tab_match, "q_val_match": q_val_match, "col_val_match": {}}


def compute_cell_value_linking(tokens, schema, get_values=False):
    def isnumber(word):
        try:
            float(word)
            return True
        except:
            return False

    def db_word_match(word, column, table, db_conn, is_number=False):
        cursor = db_conn.cursor()

        if get_values is False:
            p_str = f"select {column} from {table} where {column} like '{word} %' or {column} like '% {word}' or " \
                f"{column} like '% {word} %' or {column} like '{word}'"
        elif is_number:
            p_str = f"select distinct {column} from {table} where {column} = {word}"
        else:
            p_str = f"select distinct {column} from {table} where {column} like '%{word}%'"
        try:
            cursor.execute(p_str)
            p_res = cursor.fetchall()
            if len(p_res) == 0:
                return False
            else:
                return p_res
        except Exception as e:
            return False

    num_date_match = {}
    cell_match = {}
    values = []

    for q_id, word in enumerate(tokens):
        if len(word.strip()) == 0:
            continue
        if word in STOPWORDS or word in PUNKS:
            continue

        num_flag = isnumber(word)

        CELL_MATCH_FLAG = "CELLMATCH"

        for col_id, column in enumerate(schema.columns):
            if col_id == 0:
                assert column.orig_name == "*"
                continue

            # word is number 
            if num_flag:
                # if column.type in ["number", "time"] and not get_values:  # TODO fine-grained date
                #     num_date_match[f"{q_id},{col_id}"] = column.type.upper()
                ret = db_word_match(word, column.orig_name, column.table.orig_name, schema.connection, is_number=True)
                if ret:
                    num_date_match[f"{q_id},{col_id}"] = column.type.upper()
                    if get_values:
                        values.append((ret, column.orig_name, column.table.orig_name, q_id))
            else:
                ret = db_word_match(word, column.orig_name, column.table.orig_name, schema.connection)
                if ret:
                    # print(word, ret)
                    cell_match[f"{q_id},{col_id}"] = CELL_MATCH_FLAG
                    if get_values:
                        values.append((ret, column.orig_name, column.table.orig_name, q_id))

    cv_link = {"num_date_match": num_date_match, "cell_match": cell_match}
    if not get_values:
        return cv_link
    else:
        return cv_link, values
