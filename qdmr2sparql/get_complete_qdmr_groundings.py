import os
import argparse

from qdmr2sparql.datasets import DatasetBreak, DatasetSpider
from qdmr2sparql.structures import GroundingKey, GroundingIndex, QdmrInstance
from qdmr2sparql.structures import save_grounding_to_file, load_grounding_from_file, assert_check_grounding_save_load

from qdmr2sparql.utils_qdmr2sparql import handle_exception
from qdmr2sparql.get_qdmr_grounding_from_sql import is_text_match

from qdmr2sparql.grounding_utils import clean_qdmr_arg, measure_similarity, similarity_of_words



SQL_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except', 'distinct')
JOIN_KEYWORDS = ('join', 'on', 'by', 'having') #'as'
WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
COND_OPS = ('and', 'or')
ORDER_OPS = ('desc', 'asc')


def parse_args():
    parser = argparse.ArgumentParser(description='Build grounding between QDMR and SQL.')
    parser.add_argument('--qdmr_path', type=str, help='path to break dataset')
    parser.add_argument('--spider_path', type=str, help='path to spider dataset')
    parser.add_argument('--output_path', type=str, default='grounding.json',help='path to output file with grounding')
    parser.add_argument('--dev', action='store_true', help='if true, use dev, else use train')
    parser.add_argument('--start_spider_idx', type=int, help='index of first spider example')
    parser.add_argument('--end_spider_idx', type=int, help='index of last spider example')
    parser.add_argument('--input_grounding', type=str, default=None, help='grounding to start from')
    parser.add_argument('--not_all_sql', action='store_true', default=False, help='allows not grounded some sql')
    parser.add_argument('--without_sql', action='store_true', default=False, help='ground only scheme, without sql args')
    parser.add_argument('--time_limit', type=int, default=3000, help='time limit in seconds to process one example')
    parser.add_argument('--spider_idx', type=str, help='index of spider example, use only for debugging')
    args = parser.parse_args()
    return args


class SqlDataHarvester():
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

    def __init__(self, sql_data, table_data):
        self.sql_data = sql_data
        self.table_data = table_data
        self.tables = set()
        self.columns = set()
        self.conditions = set()
        self.essential_groundings = set()

    def parse(self):
        self.parse_sql_unit(self.sql_data["sql"])

    def parse_sql_unit(self, unit):
        if unit is None:
            return
        #   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
        self.parse_select_field(unit["select"])
        #   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
        self.parse_from_field(unit["from"])
        #   'where': condition
        self.parse_condition(unit["where"])
        #   'groupBy': [col_unit1, col_unit2, ...]
        if unit["groupBy"]:
            for col_unit in unit["groupBy"]:
                self.parse_col_unit(col_unit)
        #   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
        if unit["orderBy"]:
            for val_unit in unit["orderBy"][1]:
                self.parse_val_unit(val_unit)
            if unit["limit"] == 1:
                comparator = "max" if unit["orderBy"][0].lower() == "desc" else "min"
                self.conditions.add(GroundingKey.make_comparative_grounding(comparator, "None"))
        #   'having': condition
        self.parse_condition(unit["having"])
        #   'limit': None/limit value
        # do not know what to do with this
        #   'intersect': None/sql
        self.parse_sql_unit(unit["intersect"])
        #   'except': None/sql
        self.parse_sql_unit(unit["except"])
        #   'union': None/sql
        self.parse_sql_unit(unit["union"])

    def parse_select_field(self, unit):
        assert unit
        # 'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
        is_distinct_total, out_list = unit
        for out in out_list:
            agg_id, val_unit = out
            self.parse_val_unit(val_unit)
            agg = AGG_OPS[agg_id]
            if agg in ["min", "max"]:
                grnd = GroundingKey.make_comparative_grounding(agg, "None")
                self.conditions.add(grnd)

    def parse_from_field(self, unit):
        assert unit
        # 'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
        for table_type, data in unit["table_units"]:
            if table_type.lower() == "sql":
                self.parse_sql_unit(data)
            elif table_type.lower() == "table_unit":
                table_name = self.table_data["table_names_original"][data]
                self.tables.add(GroundingKey.make_table_grounding(table_name))

        self.parse_condition(unit["conds"])

    def parse_constant(self, val, grnd=None):
        """Parse a constant appearing in SQL; return constant and its type"""
        try:
            self.parse_col_unit(val)
            # return None not to add it to the list
            return None, "col"
        except:
            pass

        try:
            if isinstance(val, float):
                assert val.is_integer()
            val = int(val)
            return val, "int"
        except:
            pass

        try:
            val = float(val)
            return val, "float"
        except:
            pass

        # we can also have sql here
        try:
            self.parse_sql_unit(val)
            # return None not to add it to the list
            return None, "sql"
        except:
            pass

        # if None of the types is good interpret it as a str
        val = str(val)
        # remove quotes if the whol str is in them
        quote_symbols = ["\"", "'"]
        for q in quote_symbols:
            if val[0] == q and val[-1] == q:
                val = val[1:-1]
        return val, "str"


    def parse_condition(self, unit):
        if not unit:
            return
        WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
        for i in range(0, len(unit), 2):
            cond_unit = unit[i]
            # cond_unit: (not_op, op_id, val_unit, val1, val2)
            not_op, op_id, val_unit, val1, val2 = cond_unit

            comparator = WHERE_OPS[op_id]
            assert comparator in ('=', '>', '<', '>=', '<=', '!=', 'like', "in", "between"), f"Have op '{comparator}' in comparison, do not know what to do: {self.sql_data['query']}"

            val_unit = self.parse_val_unit(val_unit)

            assert val_unit[1] is None
            val_unit = val_unit[0]

            vals = []
            vals_type = []
            for val in [val1, val2]:
                val, val_type = self.parse_constant(val, val_unit)
                if not val:
                    continue
                vals.append(val)
                vals_type.append(val_type)

            if len(vals_type) == 2 and vals_type[0] == "col" and vals_type[1] == "col":
                # do no use comparisions used in joins
                continue

            if comparator in ("between"):
                assert len(vals) == 2
                conditions = [GroundingKey.make_comparative_grounding(">=", vals[0], column_grounding=val_unit),
                              GroundingKey.make_comparative_grounding("<=", vals[1], column_grounding=val_unit)]
            else:
                assert len(vals) <= 1
                if not vals:
                    continue
                val = vals[0]
                val_type = vals_type[0]

                conditions = [GroundingKey.make_comparative_grounding(comparator, val, column_grounding=val_unit)]

            for c in conditions:
                self.conditions.add(c)
                if len(c.keys) >= 3 and c.keys[2] is not None:
                    self.essential_groundings.add(c)

    def parse_col_unit(self, unit):
        if not unit:
            return None
        assert isinstance(unit, tuple) or isinstance(unit, list)
        agg_id, col_id, is_distinct = unit

        table_id, col_name = self.table_data["column_names_original"][col_id]
        if table_id != -1:
            table_name = self.table_data["table_names_original"][table_id]
            col_grnd = GroundingKey.make_column_grounding(table_name, col_name)
            self.columns.add(col_grnd)
            self.tables.add(GroundingKey.make_table_grounding(table_name))
            return col_grnd
        else:
            return None

    def parse_val_unit(self, unit):
        if not unit:
            return
        unit_op, col_unit1, col_unit2 = unit
        col_unit1 = self.parse_col_unit(col_unit1)
        col_unit2 = self.parse_col_unit(col_unit2)
        return col_unit1, col_unit2


def match_qdmr_arg_to_groundings(arg, grnds):
    matches = []
    arg = clean_qdmr_arg(arg)

    for grnd in grnds:
        if grnd.iscol() or grnd.istbl() or grnd.isval():
            name = grnd.keys[-1]
            if is_text_match(arg, name):
                matches.append(grnd)
        else:
            raise ValueError(f"Do not know how to match to {grnd}")

    return matches


def match_qdmr_arg_to_values_from_conditions(arg, schema, conditions):
    # Try to ground to values from comparatives
    value_groundings = []
    for c in conditions:
        if len(c.keys) > 2 and not QdmrInstance.is_good_qdmr_ref(c.keys[1]):
            set_to_get_value = c.keys[2]
            if c.keys[0] == "=" and isinstance(set_to_get_value, GroundingKey):
                try:
                    tbl_name = set_to_get_value.get_tbl_name()
                    col_name = set_to_get_value.get_col_name() if set_to_get_value.iscol() else schema.primary_keys[tbl_name]

                    value = clean_qdmr_arg(c.keys[1])

                    matches_to_value = match_qdmr_arg_to_groundings(arg, [GroundingKey.make_value_grounding(tbl_name, col_name, value)])
                    value_groundings.extend(matches_to_value)
                except:
                    pass
    return value_groundings


def ground_select_or_project(i_op, i_arg, qdmr, schema, sql_data, grounding):
    op, args = qdmr[i_op]
    arg = args[i_arg]

    grnds = []

    matches_to_tables = match_qdmr_arg_to_groundings(arg, sql_data["spider_sql_parser"].tables)
    matches_to_columns = match_qdmr_arg_to_groundings(arg, sql_data["spider_sql_parser"].columns)
    matches_to_values =  match_qdmr_arg_to_values_from_conditions(arg, schema, sql_data["spider_sql_parser"].conditions)
    grnds = matches_to_tables + matches_to_columns + matches_to_values

    return grnds


def ground_filter_or_comparative(i_op, i_arg, qdmr, schema, sql_data, grounding):
    op, args = qdmr[i_op]
    arg = args[i_arg]

    grnds = set()
    conditions = sql_data["spider_sql_parser"].conditions

    qdmr_refs = QdmrInstance.find_qdmr_refs_in_str(arg)

    if qdmr_refs:
        # grounding for comparing two qdmr refs
        assert len(qdmr_refs) == 1, f"Do not know what to do with {len(qdmr_refs)} refs in comparison"
        stop_words = ["a", "an", "the", "to", "than"]
        stop_words = [qdmr_refs[0]] + stop_words
        words = arg.split(" ")
        arg_parsed = " ".join([w.strip() for w in words if w.strip() not in stop_words])
        arg_parsed = arg_parsed.strip().lower()
        if arg_parsed in ["is", "equal", "equals", "is equal"]:
            comparator = "="
        elif arg_parsed in ["is higher", "higher", "is larger", "larger"]:
            comparator = ">"
        elif arg_parsed in ["is lower", "lower", "is smaller", "smaller"]:
            comparator = "<"
        else:
            raise ValueError(f"Do not understand comparator from arg '{arg}'")

        grnds.add(GroundingKey.make_comparative_grounding(comparator, qdmr_refs[0]))
        return list(grnds)

    # Try to ground as existing comparison
    for c in conditions:
        value = c.keys[1]
        if not isinstance(value, GroundingKey):
            # here were are interested in a comparison against a value

            values_to_search_in_arg = [str(value)]
            if isinstance(value, str):
                values_to_search_in_arg.append(clean_qdmr_arg(value))
            elif isinstance(value, float):
                try:
                    values_to_search_in_arg.append(str(int(value)))
                except:
                    pass
            else:
                raise NotImplementedError("Unknown value type")

            value_matches = False
            for v in values_to_search_in_arg:
                if v in arg or is_text_match(arg, v):
                    value_matches = True
            if value_matches:
                grnds.add(c)

            if c.keys[0] in ["min", "max"]:
                grnds.add(c)

    if not grnds:
        grnds = set(c for c in conditions if not isinstance(c.keys[1], GroundingKey))

        #     # if can extract dates from both - try matching anyway (even if the dates are different)
        #     try:
        #         date_from_arg = parse_date_from_text(clean_qdmr_arg(arg), fuzzy=True, default=parse_date_from_text("0001-01-01 00:00:00"))
        #         date_from_value = parse_date_from_text(clean_qdmr_arg(str(value)), fuzzy=True, default=parse_date_from_text("0001-01-01 00:00:00"))

        #     except:
        #         # could not extract date
        #         pass


    grnds = list(grnds)
    # filter can also be w.r.t. paths to tables and columns
    matches_to_tables = match_qdmr_arg_to_groundings(arg, sql_data["spider_sql_parser"].tables)
    matches_to_columns = match_qdmr_arg_to_groundings(arg, sql_data["spider_sql_parser"].columns)
    if matches_to_tables:
        grnds.extend(matches_to_tables)
    if matches_to_columns:
        grnds.extend(matches_to_columns)

    return grnds


def ground_sort(i_op, i_arg, qdmr, schema, sql_data, grounding):
    raise NotImplementedError()


def ground_superlative(i_op, i_arg, qdmr, schema, sql_data, grounding):
    op, args = qdmr[i_op]
    arg = args[i_arg]

    assert arg.lower() in ["max", "min"], f"Can't parse {arg} as the first arg of superlative"
    grnds = [GroundingKey.make_comparative_grounding(arg.lower(), "None")]

    return grnds


def ground_group_or_aggregate(i_op, i_arg, qdmr, schema, sql_data, grounding):
    op, args = qdmr[i_op]

    # Try empty grounding or assume that no computation is needed
    grnds = [""]
    index_op = QdmrInstance.ref_to_index(args[1])
    if qdmr.ops[index_op] in ["project", "select"]:
        index_arg = 0
        grnd_index = GroundingIndex(index_op, index_arg, qdmr.args[index_op][index_arg])
        if grnd_index in grounding: # we need to have some grounding
            grnds.extend(grounding[grnd_index])

    return grnds


def process_annotated_linking(qdmr_name, linking_data, schema, table_data, sql_data, verbose=False):
    if verbose:
        print("Using the following linking data:")
    linking_data["groundings"] = []
    for link, tok in zip(linking_data["ant"], linking_data["toks"]):
        linking_data["groundings"].append([])
        if not link:
            continue

        if link["type"] == "tbl":
            tbl_id = link["id"]
            tbl_name = table_data["table_names_original"][tbl_id]
            assert tbl_name in schema.table_names, f"{qdmr_name}: cannot find table {tbl_name} in database {schema.db_id}"
            grnd = GroundingKey.make_table_grounding(tbl_name)
            linking_data["groundings"][-1].append(grnd)
            # print(grnd)
        elif link["type"] == "col":
            col_id = link["id"]
            tbl_id, col_name = table_data["column_names_original"][col_id]
            tbl_name = table_data["table_names_original"][tbl_id]
            assert tbl_name in schema.table_names, f"{qdmr_name}: cannot find table {tbl_name} in database {schema.db_id}"
            assert col_name in schema.column_names[tbl_name], f"{qdmr_name}: cannot find column {col_name} in table {tbl_name} in database {schema.db_id}"
            grnd = GroundingKey.make_column_grounding(tbl_name, col_name)
            linking_data["groundings"][-1].append(grnd)
            # print(grnd)
        elif link["type"] == "val":
            col_id = link["id"]
            tbl_id, col_name = table_data["column_names_original"][col_id]
            tbl_name = table_data["table_names_original"][tbl_id]
            assert tbl_name in schema.table_names, f"{qdmr_name}: cannot find table {tbl_name} in database {schema.db_id}"
            if col_name == "*":
                continue
            assert col_name in schema.column_names[tbl_name], f"{qdmr_name}: cannot find column {col_name} in table {tbl_name} in database {schema.db_id}"

            # extract most similar value from the table database
            conds = list(sql_data["spider_sql_parser"].conditions)
            for c in conds:
                if len(c.keys) < 3:
                    continue
                grnd_col = GroundingKey.make_column_grounding(tbl_name, col_name)
                if c.keys[2] != grnd_col:
                    continue

                val_str = str(c.get_val())
                if similarity_of_words(val_str, tok) > 0.1:
                    if link["op"] == "=":
                        grnd = GroundingKey.make_value_grounding(tbl_name, col_name, c.get_val())
                    else:
                        grnd = GroundingKey.make_comparative_grounding(link["op"], c.get_val(), grnd_col)
                    linking_data["groundings"][-1].append(grnd)

            # print(grnd)
        if verbose:
            print(tok, linking_data["groundings"][-1], link)


def find_annotation_link(i_op, i_arg, qdmr, linking_data, schema, sql_data, table_data, qdmr_name):
    op, args = qdmr[i_op]
    arg = args[i_arg]

    arg_words = arg.split(" ")
    arg_words = [w.lower() for w in arg_words]

    word_sim_th = 0.7
    grnds_match = []

    if op in ["group", "aggregate"]:
        if "avg" in arg_words:
            arg_words += ["average"]
        if "count" in arg_words:
            arg_words += ["number", "total"]
        if "sum" in arg_words:
            arg_words += ["total"]
        if "max" in arg_words:
            arg_words += ["maximum"]
        if "min" in arg_words:
            arg_words += ["minimum"]

    for link, tok, grnds in zip(linking_data["ant"], linking_data["toks"], linking_data["groundings"]):
        if not link:
            continue
        tok = tok.lower()
        word_sim = [measure_similarity(word, tok, threshold=word_sim_th) for word in arg_words]
        if any(word_sim) or tok in arg_words:
            # Found a match
            # print(arg, "|", tok, "|", link)

            for grnd in grnds:
                if grnd not in grnds_match:
                    grnds_match.append(grnd)

    return grnds_match


op_args_to_ground = {}
op_args_to_ground["select"] = [0]
op_args_to_ground["filter"] = [1]
op_args_to_ground["project"] = [0]
op_args_to_ground["union"] = []
op_args_to_ground["intersection"] = []
op_args_to_ground["sort"] = [2]
op_args_to_ground["comparative"] = [2]
op_args_to_ground["superlative"] = [0]
op_args_to_ground["aggregate"] = [0]
op_args_to_ground["group"] = [0]
op_args_to_ground["discard"] = []
op_args_to_ground["arithmetic"] = []
op_args_to_ground["boolean"] = []

op_grounder = {}
op_grounder["select"] = ground_select_or_project
op_grounder["filter"] = ground_filter_or_comparative
op_grounder["project"] = ground_select_or_project
op_grounder["sort"] = ground_sort
op_grounder["comparative"] = ground_filter_or_comparative
op_grounder["superlative"] = ground_superlative
op_grounder["group"] = ground_group_or_aggregate
op_grounder["aggregate"] = ground_group_or_aggregate


def compute_grounding(qdmr, qdmr_name, dataset_spider, partial_grounding=None, verbose=True):
    if partial_grounding is None:
        partial_grounding = {}

    sql_data = dataset_spider.sql_data[qdmr_name]
    db_id = sql_data['db_id']
    schema = dataset_spider.schemas[db_id]
    table_data = dataset_spider.table_data[db_id]
    if qdmr_name in dataset_spider.sql_linking_dict:
        linking_data = dataset_spider.sql_linking_dict[qdmr_name]
        # assert linking_data["question"] == sql_data["question"], f"{qdmr_name}: question of annotation does not match question in SQL linking"
        if linking_data["question"] != sql_data["question"]:
            if verbose:
                print(f"{qdmr_name}: WARNING: question of annotation does not match question in SQL linking")
            linking_data = None
        # assert linking_data["query"] == sql_data["query"], f"{qdmr_name}: query of annotation does not match query in SQL linking"
    else:
        linking_data = None

    sql_data["spider_sql_parser"] = SqlDataHarvester(sql_data, table_data)
    sql_data["spider_sql_parser"].parse()

    if verbose:
        print("db_id:", db_id)
        print("Question:", sql_data["question"])
        print("SQL query:", sql_data['query'])
        print(f"QDMR:\n{qdmr}")
        print(f"Partial groundings: {partial_grounding}")
        print(f"Database schema {db_id}:")
        for tbl_name, cols in schema.column_names.items():
            print(f"{tbl_name}: {cols}")

    if linking_data:
        # schema.load_table_data(os.path.join(dataset_spider.dataset_path, "database"))
        process_annotated_linking(qdmr_name, linking_data, schema, table_data, sql_data, verbose=verbose)
    else:
        if verbose:
            print("No linking data found")

    grounding = {}
    num_args_to_ground = 0
    num_args_were_grounded = 0
    num_could_not_ground = 0
    failed_to_ground = []
    message = []
    for i_op, (op, args) in enumerate(qdmr):
        for i_arg, arg in enumerate(args):
            if i_arg not in op_args_to_ground[op]:
                # this arg should not be grounded
                continue

            num_args_to_ground = num_args_to_ground + 1
            grnd_index = GroundingIndex(i_op, i_arg, arg)

            if grnd_index in partial_grounding:
                # this arg is already grounded
                grounding[grnd_index] = partial_grounding[grnd_index]
                num_args_were_grounded = num_args_were_grounded + 1
            else:
                arg_groundings = op_grounder[op](i_op, i_arg, qdmr, schema, sql_data, grounding)

                if arg_groundings:
                    # found a match
                    grounding[grnd_index] = arg_groundings
                else:
                    num_could_not_ground = num_could_not_ground + 1
                    failed_to_ground.append(grnd_index)

            if linking_data:
                # use the linking data to refine groundings
                grnds_match = find_annotation_link(i_op, i_arg, qdmr, linking_data, schema, sql_data, table_data, qdmr_name)
                if len(grnds_match) > 0:
                    if len(grnds_match) > 1:
                        message.append(f"{qdmr_name}:{grnd_index} have multiple groundings {grnds_match} matching to {grnd_index}")

                    for grnd in grnds_match:
                        if grnd_index not in grounding:
                            grounding[grnd_index] = [grnd]
                            message.append(f"{qdmr_name}:{grnd_index} new grounding {grnd} from linking")
                        elif grnd not in grounding[grnd_index]:
                            grounding[grnd_index].append(grnd)
                            message.append(f"{qdmr_name}:{grnd_index} linked argument {grnd} was not in grounding before, had {grounding[grnd_index]}")

    if "distinct" in partial_grounding:
        grounding["distinct"] = partial_grounding["distinct"]

    grounding["ESSENTIAL_GROUNDINGS"] = list(sql_data["spider_sql_parser"].essential_groundings)

    message.append(f"{os.path.basename(__file__)}: Total {num_args_to_ground} args, before {num_args_were_grounded}, grounded {num_args_to_ground-num_args_were_grounded-num_could_not_ground}, failed {num_could_not_ground}.")
    if num_could_not_ground > 0:
        message.append(f"Failed to ground: {failed_to_ground}.")
    message.append(f"Groundings: {grounding}.")

    message = "\n".join(message)
    if verbose:
        print(message)

    return grounding, message


def main(args):
    print(args)

    split_name = 'dev' if args.dev else 'train'

    dataset_break = DatasetBreak(args.qdmr_path, split_name)
    dataset_spider = DatasetSpider(args.spider_path, split_name)

    if args.input_grounding:
        partial_grounding = load_grounding_from_file(args.input_grounding)
    else:
        partial_grounding = {}


    if args.spider_idx is not None:
        qdmr_name = None
        for name in dataset_break.names:
            idx = dataset_break.get_index_from_name(name)
            if idx == args.spider_idx:
                qdmr_name = name
                break

        assert qdmr_name is not None, "Could find QDMR with index {args.spider_idx}"
        qdmr = dataset_break.qdmrs[qdmr_name]
        qdmr_grounding = partial_grounding[qdmr_name] if qdmr_name in partial_grounding else None

        # debugging
        print()
        print(qdmr_name)

        grounding, _ = compute_grounding(qdmr, qdmr_name, dataset_spider, partial_grounding=qdmr_grounding, verbose=True)
    else:
        all_grounding = {}
        for qdmr_name, qdmr in dataset_break:
            spider_idx = DatasetBreak.get_index_from_name(qdmr_name)
            qdmr_grounding = partial_grounding[qdmr_name] if qdmr_name in partial_grounding else None

            print(qdmr_name, end=" ")
            try:
                grounding, message = compute_grounding(qdmr, qdmr_name, dataset_spider, partial_grounding=qdmr_grounding, verbose=False)
                all_grounding[qdmr_name] = grounding
                message_list = qdmr_grounding["MESSAGES"] if "MESSAGES" in qdmr_grounding else []
                message_list.append(message)
                all_grounding[qdmr_name]["MESSAGES"] = message_list
                print(message)
            except Exception as e:
                error_details = handle_exception(e, verbose=False)
                print(f"ERROR: {error_details['type']}:{error_details['message']}, file: {error_details['file']}, line {error_details['line_number']}")
                all_grounding[qdmr_name] = {"ERRORS" : [error_details]}

        if args.output_path:
            # save data to the new file
            save_grounding_to_file(args.output_path, all_grounding)

            # check correctness of save-load
            check = load_grounding_from_file(args.output_path)
            assert_check_grounding_save_load(all_grounding, check)


if __name__ == "__main__":
    args = parse_args()
    main(args)