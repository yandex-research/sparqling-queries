import attr
from difflib import SequenceMatcher
import re
import nltk

from word2number import w2n
from dateutil import parser as date_parser

from text2qdmr.utils import corenlp
from text2qdmr.datasets.utils.spider_match_utils import STOPWORDS, PUNKS
from qdmr2sparql.structures import GroundingIndex, GroundingKey

quote_symbols = ['\'', '\‘', '\"']
corenlp_quotes = [("`", "'"), ("``", "''")]

value_types = ['text', 'number', 'time', 'unknown'] # TODO: all types should be known


GroundingKey = attr.s(these={
    'type': attr.ib(),
    'keys': attr.ib()
}, init=False, auto_detect=True)(GroundingKey)

@attr.s(frozen=True)
class ValueUnit:
    value = attr.ib()

    orig_value = attr.ib(kw_only=True)
    tokenized_value = attr.ib(default=None, kw_only=True)
    bert_tokens = attr.ib(default=None, kw_only=True)

    value_type = attr.ib(default=None, kw_only=True)
    column = attr.ib(default=None, kw_only=True)
    table = attr.ib(default=None, kw_only=True)

    source = attr.ib(default=None, kw_only=True)
    q_match = attr.ib(default=None, kw_only=True)

    idx = attr.ib(default=None, kw_only=True)

    def __str__(self):
        value = self.value
        if self.value_type == 'number':
            try:
                value = int(value)
            except:
                pass
        return str(value)

def similarity_of_words(qdmr_w, sql_w):
    match = SequenceMatcher(None, qdmr_w, sql_w).find_longest_match(0, len(qdmr_w), 0, len(sql_w))
    if match.size > 1:
        p = match.size / len(qdmr_w)
        r = match.size / len(sql_w)
        return 2 * p * r / (p + r)
    else:
        return 0.0

def generate_substr(values):
    return [' '.join(values[i: j]) for i in range(len(values)) 
                  for j in range(i + 1, len(values) + 1)] 

def to_date(token, default_date=date_parser.parse("0001-01-01 00:00:00")):
    date_object = date_parser.parse(str(token), fuzzy=True, default=default_date)
    return date_object.strftime("%-d %B %Y, %H:%M:%S")

def transform_to_type(value, col_type):
    if isinstance(value, str) and (col_type == 'text' or value.lower() == 'null'):
        return value
    elif col_type == 'number':
        try:
            return int(value)
        except:
            return float(value)
    elif col_type == 'time':
        return to_date(value)
    else:
        return value

class ValueExtractor:
    def __init__(self, schemas, extract_value, partition):
        self.schemas = schemas
        self.partition = partition
        self.db_table_column_info = {}
        self.db_data = {}

        self.max_values_from_database = extract_value['max_values_from_database']
        self.matching = extract_value.get('matching', False)

    def get_values(self, qdmr_entry, grounding, db_id, question, break_idx):
        if db_id is not None and self.db_table_column_info.get(db_id) is None:
            # select column types and values from db
            self.sqlite_types(db_id)

        orig_question_tokens = self.tokenize(question)

        values = []
        if self.partition == 'train' and qdmr_entry and grounding:
            # get gt qdmr values
            values += self.get_values_from_qdmr(qdmr_entry, grounding, orig_question_tokens, db_id)

        # get all numbers from question
        values += self.extract_numbers_dates_from_text(orig_question_tokens)
        
        # add question tokens that can be converted to some format
        # here can be duplicates with existing values
        values += self.convert_quotes(question, orig_question_tokens)

        # get values from schema    
        column_data = self.get_column_data(db_id, orig_question_tokens)
        if len(column_data) > self.max_values_from_database:
            column_data = column_data[:self.max_values_from_database] # keep sorting
        values += column_data

        values = sorted(values, key=lambda val_unit: str(val_unit))

        return values, self.db_table_column_info.get(db_id), orig_question_tokens

    @staticmethod
    def extract_numbers_dates_from_text(tokens):

        val_unit_list = []

        def add_number(num, orig, q_start, q_end, match):
            q_val_match = {'idx_question': tuple(range(q_start, q_end)), 'match': match}
            val_unit_list.append(ValueUnit(transform_to_type(num, "number"), value_type="number", 
                                            orig_value=orig, q_match=q_val_match, source="text"))
            return str(num)


        for i in range(len(tokens)):
            for j in range(i+1, len(tokens)):
                s = " ".join(tokens[i:j])

                try:
                    f = float(s)
                    add_number(f, s, i, j, 'VEM')
                except:
                    pass

                try:
                    f = w2n.word_to_num(s)
                    add_number(f, s, i, j, 'VEM')
                except:
                    pass

                try:
                    d = to_date(s)
                    q_val_match = {'idx_question': tuple(range(i, j)), 'match': 'VPM'}
                    val_unit_list.append(ValueUnit(d, value_type="time", orig_value=s, q_match=q_val_match, source="text"))
                except:
                    pass

                s_digit = [d for d in s if d.isdigit()]
                try:
                    f = float(s_digit)
                    add_number(f, s, i, j, 'VPM')
                except:
                    pass

                s_digit = [d for d in s if d.isdigit() or s in ['.', ',']]
                if any(x in s_digit for x in ['.', ',']):
                    try:
                        f = float(s_digit)
                        add_number(f, s, i, j, 'VPM')
                    except:
                        pass

        return val_unit_list

    def sqlite_types(self, db_id):
        self.db_table_column_info[db_id] = {}

        conn = self.schemas[db_id].connection
        for table in self.schemas[db_id].tables:
            table_name = table.orig_name
            self.db_table_column_info[db_id][table_name] = {}
            full_data = {}

            cur = conn.execute("PRAGMA table_info('{}') ".format(table_name))
            for j, col in enumerate(cur.fetchall()):
                column_name = col[1]

                #varchar, '' -> text, int, numeric -> integer, ...
                col_type = col[2].lower()
                if 'char' in col_type or col_type == '' or 'text' in col_type or 'var' in col_type:
                    col_type = 'text'
                elif 'int' in col_type or 'decimal' in col_type or 'number' in col_type\
                or 'id' in col_type or 'boolean' in col_type \
                or 'real' in col_type or 'double' in col_type or 'float' in col_type or 'numeric' in col_type:
                    col_type = 'number'
                elif 'date' in col_type or 'time' in col_type or 'year' in col_type:
                    col_type = 'time'
                else:
                    col_type = 'others'
                self.db_table_column_info[db_id][table_name][column_name] = col_type

            
    def get_column_data(self, db_id, question, threshold=0.5):
        column_data = []
        type_float = {}
        if db_id is None:
            return column_data
        
        def add_to_column_data(results, q_id=None):
            for row in results:
                assert len(row.keys()) == 1, row.keys()
                for column_name in row.keys():
                    orig_value = row[column_name]
                    value_type = self.db_table_column_info[db_id][table_name][column_name]

                    if str(orig_value).strip() == '' or orig_value is None\
                        or str(orig_value).lower() in ["null", "none", "nil"]:
                        continue

                    try: # round float
                        float(orig_value)
                        if type_float:
                            type_float[table_name][column_name].append(True)
                    except ValueError:
                        if type_float:
                            type_float[table_name][column_name].append(False)

                    if q_id:
                        match = 'VEM' if question[q_id] == str(orig_value) else 'VPM'
                        q_val_match = {'idx_question': tuple([q_id]), 'match': match}
                    else:
                        q_val_match = None

                    column_data.append(ValueUnit(orig_value, orig_value=orig_value,value_type=value_type,
                                            column=column_name, table=table_name, 
                                            source='schema', q_match=q_val_match))

        column_data_matches = []
        conn = self.schemas[db_id].connection
        ps = nltk.stem.PorterStemmer()
        sno = nltk.stem.SnowballStemmer('english') 

        if db_id not in self.db_data:
            # get content and real types
            for table_name, columns in self.db_table_column_info[db_id].items():
                type_float[table_name] = {}
                for column_name in columns.keys():
                    type_float[table_name][column_name] = []

                for column_name in columns.keys():
                    cur = conn.execute(f"SELECT DISTINCT \"{column_name}\" FROM {table_name}")
                    table_data = cur.fetchall()

                    # empty table cases
                    if len(table_data) == 0:
                        if not(db_id == 'soccer_1' and table_name == 'sqlite_sequence' \
                            or db_id == 'formula_1' and table_name in ('pitStops', 'lapTimes') \
                            or db_id == 'sakila_1' and table_name in ('film_text', 'language', 'staff', 'store') \
                            or db_id == 'music_2'):
                            print('Empty table {} in db {}'.format(table_name, db_id))
                        continue
                    add_to_column_data(table_data)  
            self.db_data[db_id] = column_data

            # to real type
            for table_name, columns in self.db_table_column_info[db_id].items():
                for column_name, col_type in columns.items():
                    if not type_float[table_name][column_name]:
                        continue

                    is_float = all(type_float[table_name][column_name])
                    if col_type == 'number' and not is_float:
                        self.db_table_column_info[db_id][table_name][column_name] = 'text'
                    elif col_type == 'text' and is_float:
                        self.db_table_column_info[db_id][table_name][column_name] = 'number'
        else:
            column_data = self.db_data[db_id]
        
        for q_id, word in enumerate(question):
            word = word.strip().lower()
            if len(word) == 0:
                continue
            if word in STOPWORDS or word in PUNKS:
                continue
            
            # is number
            try:
                float(word)
                is_number = True
            except:
                is_number = False

            for val_unit in column_data:
                value_type = self.db_table_column_info[db_id][val_unit.table][val_unit.column]
                if is_number:
                    # check exact matching
                    try:
                        value = float(val_unit.orig_value)
                        if float(word) == value:
                            value = transform_to_type(val_unit.orig_value, value_type)
                            val_unit = attr.evolve(val_unit, value=value, value_type=value_type,
                                                    q_match={'idx_question': tuple([q_id]), 'match': 'VEM'})
                            column_data_matches.append((val_unit, 1.0))
                    except:
                        continue
                else:
                    # check stemming and similarity
                    value = str(val_unit.orig_value).strip().lower()
                    match = 'VEM' if word == value else 'VPM'
                    q_val_match = {'idx_question': tuple([q_id]), 'match': match}

                    similarity = similarity_of_words(value, word)

                    if ps.stem(value) == ps.stem(word) or sno.stem(value) == sno.stem(word):
                        value = transform_to_type(val_unit.orig_value, value_type)
                        val_unit = attr.evolve(val_unit, value=value, value_type=value_type, q_match=q_val_match)
                        column_data_matches.append((val_unit, 1.0))
                    elif similarity >= threshold and not is_number: # TODO matching numbers?
                        value = transform_to_type(val_unit.orig_value, value_type)
                        val_unit = attr.evolve(val_unit, value=value, value_type=value_type, q_match=q_val_match)
                        column_data_matches.append((val_unit, similarity))
                    
        column_data_matches = sorted(column_data_matches, key=lambda x: x[1], reverse=True)
        column_data = [item[0] for item in column_data_matches]
        return column_data

    def values_in_grnd(self, grounding_list, db_id):
        grounding = grounding_list[0]
        if not isinstance(grounding, GroundingKey):
            return 

        column, table, value_type = None, None, None
        if grounding.isval():
            orig_value = grounding.get_val()
            column, table = grounding.get_col_name(), grounding.get_tbl_name()
        elif grounding.iscomp():
            op, orig_value = grounding.keys[:2]
            if op in ('max', 'min') or not orig_value or orig_value.find('#') != -1:
                return
            if len(grounding.keys) > 2:
                column, table = grounding.keys[2].get_col_name(), grounding.keys[2].get_tbl_name() 
        elif grounding.type == 'str' and len(grounding.keys[0]) > 0:
            orig_value, value_type = grounding.keys[0], 'text'
            # TODO: try guess if we have a number or a date here
        else:
            return

        if str(orig_value)[0] == '%':
            orig_value = orig_value[1:]
        if str(orig_value)[-1] == '%':
            orig_value = orig_value[:-1]

        if column and table:
            value_type = self.db_table_column_info[db_id][table][column] # find value type
            value = transform_to_type(orig_value, value_type)
        else:
            value = orig_value
            if value_type is None:
                value_type = 'unknown'
                # TODO: try guess if we have a number or a date here
        return ValueUnit(value, orig_value=orig_value, value_type=value_type, column=column, table=table, source='qdmr')
    
    def get_values_from_qdmr(self, qdmr_entry, grounding, question_tokens, db_id, verbose=False):
        values = []
        ps = nltk.stem.PorterStemmer()
        sno = nltk.stem.SnowballStemmer('english') 

        for i, all_args in enumerate(qdmr_entry.args):
            for i_arg, arg in enumerate(all_args):
                grnd_index = GroundingIndex(i, i_arg, arg)
                cur_grounding_list = [grounding.get(grnd_index)]
                cur_value = self.values_in_grnd(cur_grounding_list, db_id)
                if grnd_index in grounding:
                    grounding[grnd_index] = cur_grounding_list[0]
                if cur_grounding_list[0] and cur_value:
                    if self.matching:
                        arg_tokens = self.tokenize(arg)
                        matches = []
                        for arg_tok in arg_tokens:
                            for i, tok in enumerate(question_tokens):
                                if ps.stem(arg_tok) == ps.stem(tok) or sno.stem(arg_tok) == sno.stem(tok):
                                    matches.append(i)

                        if not matches:
                            if verbose:
                                print('not matched', question_tokens, arg_tokens)
                                print()
                        else:
                            tokenized_arg = ' '.join(arg_tokens)
                            tokenized_question = ' '.join(question_tokens)
                            if tokenized_arg in tokenized_question:
                                match_type = 'VEM'
                            else:
                                match_type = 'VPM'
                            cur_value = attr.evolve(cur_value, q_match={'idx_question': matches, 'match': match_type})
                    values.append(cur_value)
        return values
   
    def parse_const_tree(self, tree_str, level=None):
        # split str
        tree_str = tree_str.replace('\n', '').replace('(', '( ').replace(')', ' )')
        tree = tree_str.split(' ')

        stack = []
        substrings = []
        for el in tree:
            if el != ')':
                stack.append(el)
            else:
                last = stack.pop() 
                phrase = []
                while last != '(':
                    phrase.append(last)
                    last = stack.pop() 
                phrase = phrase[:-1] # get rid of tag
                phrase_str = ' '.join(i for i in phrase[::-1]) # convert to str
                if not level or len(phrase) - 1 <= level:
                    substrings.append(phrase_str)
                    #substrings.append(phrase_str.translate(translate_table))  # without punctuation
                stack.append(phrase_str)
        return list(dict.fromkeys(substrings))
   
    def tokenize(self, text):
        ann = corenlp.annotate(text, ['tokenize', 'ssplit'])
        res = [tok.word for sent in ann.sentence for tok in sent.token]
        res[0] = res[0].lower()
        return res

    def token_to_types(self, token):
        results, res_types = [token], ['text']
        try:
            results.append(to_date(token))
            res_types.append('time')
        except:
            pass

        try:
            results.append(int(token))
            res_types.append('number')
        except:
            pass

        try:
            results.append(float(token))
            res_types.append('number')
        except:
            pass

        return results, res_types

    def match_quote_question(self, question_tokens, tok):
        q_idx, fl = None, False

        first_quote = False
        for i, question_tok in enumerate(question_tokens):
            # exect match should be after open qoute symbol
            prev_tok = question_tokens[i - 1]

            if question_tok[0] in quote_symbols and len(question_tok) > 1:
                question_tok = question_tok[1:]
                first_quote = True

            if question_tok == tok:
                if prev_tok == corenlp_quotes[0][0] or prev_tok == corenlp_quotes[1][0] or \
                    prev_tok in quote_symbols or first_quote:
                    assert q_idx is None, (q_idx, tok, question_tok, question_tokens)
                    q_idx = [i]
                    first_quote = False
            elif tok.find(question_tok) == 0 and not fl:
                assert q_idx is None, (fl, q_idx, tok, question_tok, question_tokens)
                q_idx = [i]
                fl = True
                part = len(re.findall(question_tok, tok)[0])
            elif fl:
                if tok.find(question_tok) > 0:
                    assert q_idx[-1] == i - 1, ('1', question_tokens[q_idx[0]: i + 1], tok)
                    if tok[part:].find(question_tok) == 0:
                        q_idx.append(i)
                        part += len(re.findall(question_tok, tok[part:])[0])
                        if part == len(tok):
                            fl = False 
                else:
                    q_idx = None
                    fl = False

        assert q_idx is not None, (question_tokens, tok, fl)

        if len(q_idx) != 1:
            q_idx = range(q_idx[0], q_idx[-1] + 1)
        assert list(q_idx), (question_tokens, tok, q_idx)
        return {'idx_question': tuple(q_idx), 'match': 'VEM'}

    def convert_quotes(self, question, question_tokens):
        tokens_in_quotes = []
        converted_toks = []


        def add_quote(question_str, pattern, open_quote, close_quote):
            quote_tokens = re.findall(open_quote + pattern + close_quote, question_str)
            for tok in quote_tokens:
                tok = tok[len(open_quote):-len(close_quote)].strip()
                tok = tok[:-1] if tok[-1] == '.' else tok
                tokens_in_quotes.append(tok) # exclude quote symbols

        # extract words in quotes from orig sentence
        for quote in quote_symbols:
            add_quote(question, "[\S]+", quote, quote)

        # extract tokens in quotes from tokenized sentence
        tokenized_question = ' '.join(question_tokens)
        for (open_quote, close_quote) in corenlp_quotes:
            add_quote(tokenized_question, "\s*[\S]+\s*", open_quote, close_quote)

        tokens_in_quotes = list(dict.fromkeys(tokens_in_quotes)) # delete duplicates

        # converting tokens in quotes
        for tok in tokens_in_quotes:
            q_val_match = self.match_quote_question(question_tokens, tok)
            results, res_types = self.token_to_types(tok)
            for res, res_type in zip(results, res_types):
                converted_toks.append(ValueUnit(res, value_type=res_type, orig_value=tok, q_match=q_val_match, source="text"))
        return converted_toks