import textwrap
import copy
import queue
import attr

from qdmr2sparql.structures import GroundingIndex, GroundingKey, QdmrInstance
from qdmr2sparql.structures import parse_date_str
from qdmr2sparql.structures import QueryToRdf, QueryResult, OutputColumnId

class SchemaWithRdfGraph(object):
    def __init__(self, schema, rdf_graph):
        self.schema = schema
        self.rdf_graph = rdf_graph

    def get_table_grounding_rel(self, ground_key):
        assert ground_key.istbl()
        table_primary_key = self.schema.primary_keys[ground_key.get_tbl_name()]
        return self.rdf_graph.sparql_link_ref(ground_key.get_tbl_name(), table_primary_key)

    def get_col_grounding_rel(self, ground_key):
        assert ground_key.iscol()
        return self.rdf_graph.sparql_link_ref(ground_key.get_tbl_name(), ground_key.get_col_name())

    def get_foreign_rel(self, src, tgt):
        assert src.iscol()
        assert tgt.iscol()
        return self.rdf_graph.sparql_foreign_link(src.get_tbl_name(), src.get_col_name(),
                                                  tgt.get_tbl_name(), tgt.get_col_name())

    def get_key_for_comparison(self, grnd):
        if type(grnd) == GroundingKey:
            assert grnd.isval(), f"When adding value to SPARQL it can only be grounded to a value but have {grnd}"
            if grnd.get_col_name() in self.schema.column_used_with_keys[grnd.get_tbl_name()]:
                # add special string if the requested value is a 
                val = grnd.get_val()
                # try getting the correct type: int if possible
                if self.rdf_graph.type_for_column_for_table[grnd.get_tbl_name()][grnd.get_col_name()].lower() == "integer":
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                
                uri_str = self.rdf_graph.get_uri_str(grnd.get_tbl_name(), grnd.get_col_name(), val)
                return uri_str
            else:
                # use only the value for further processing
                col_type = self.rdf_graph.type_for_column_for_table[grnd.get_tbl_name()][grnd.get_col_name()]
                if col_type.lower() == "text":
                    txt = grnd.get_val()
                    if type(txt) == str and txt[0] == "\"" and txt[-1] == "\"":
                        return str(grnd.get_val()) + "^^xsd:string"
                    else:
                        return "\"" + str(grnd.get_val()) + "\"" + "^^xsd:string"
                elif col_type.lower() in ["number", "int", "integer", "bool", "boolean", "float", "real", "double"]:
                    return str(grnd.get_val())
                elif col_type.lower() in ["date", "datetime"]:
                    value = grnd.get_val()
                    value = parse_date_str(value)
                    return "\"" + str(value)+ "\"" + "^^xsd:dateTime"
                else:
                    raise RuntimeError(f"Unknown column type {col_type} of {grnd}")

        try:
            # check if the value if a number, and then return it converted to string
            float(grnd)
            return str(grnd)
        except ValueError:
            # if not a number than it is a string - need to add quotes
            return "\"" + str(grnd) + "\""


    def find_path(self, source_groundings, target_grounding, get_var_name,
                        var_for_grounding=None):
        if var_for_grounding is None:
            var_for_grounding = {}

        # get starting variables
        def add_start_grounding(grnd):
            if grnd not in var_for_grounding:
                var_for_grounding[grnd] = get_var_name(grnd.keys[1] if len(grnd.keys) > 1 else grnd.keys[0])

        if type(source_groundings) == GroundingKey:
            source_groundings = [source_groundings]

        try:
            for grnd in source_groundings:
                add_start_grounding(grnd)
        except:
            raise RuntimeError(f"Can not start finding paths from grounding {source_groundings}")


        previous_column = {}

        def get_col_from_grounding(grnd):
            if grnd.istbl():
                tbl_name = grnd.get_tbl_name()
                col_name = self.schema.primary_keys[tbl_name]
            elif grnd.iscol() or grnd.isval():
                tbl_name = grnd.get_tbl_name()
                col_name = grnd.get_col_name()
            else:
                raise RuntimeError(f"Path finding should start from grounding to tbl, col or val but have {grnd}")
            return GroundingKey.make_column_grounding(tbl_name, col_name)

        for grnd in source_groundings:
            previous_column[get_col_from_grounding(grnd)] = None
        target_grounding_col = get_col_from_grounding(target_grounding)

        # start BFS over columns
        queue_to_visit = queue.SimpleQueue()
        queue_size = 0
        for k in previous_column:
            queue_to_visit.put(k)
            queue_size += 1

        def add_to_queue(key, prev_col, queue_size):
            if key not in previous_column:
                previous_column[key] = prev_col
                queue_to_visit.put(key)
                queue_size += 1
            return queue_size

        def pop_from_queue(queue_size):
            cur_col_grounding = queue_to_visit.get()
            queue_size -= 1
            return cur_col_grounding, queue_size


        while queue_size > 0:
            cur_col_grounding, queue_size = pop_from_queue(queue_size)

            if cur_col_grounding == target_grounding_col:
                break

            tbl_name = cur_col_grounding.get_tbl_name()
            col_name = cur_col_grounding.get_col_name()

            # if is a primary key - visit all columns of the same table
            if col_name == self.schema.primary_keys[tbl_name]:
                for next_col in self.schema.column_names[tbl_name]:
                    key = GroundingKey.make_column_grounding(tbl_name, next_col)
                    queue_size = add_to_queue(key, cur_col_grounding, queue_size)

            # if is not a primary key - visit the primary key
            if col_name != self.schema.primary_keys[tbl_name]:
                key = GroundingKey.make_column_grounding(tbl_name, self.schema.primary_keys[tbl_name])
                queue_size = add_to_queue(key, cur_col_grounding, queue_size)
            # if is a source of a foreign key - visit target
            if cur_col_grounding in self.schema.foreign_key_tgt_for_src:
                key = self.schema.foreign_key_tgt_for_src[cur_col_grounding]
                queue_size = add_to_queue(key, cur_col_grounding, queue_size)
            # if is a target of a foreign key - visit source
            if cur_col_grounding in self.schema.foreign_keys_src_for_tgt:
                for key in self.schema.foreign_keys_src_for_tgt[cur_col_grounding]:
                    queue_size = add_to_queue(key, cur_col_grounding, queue_size)


        # backward pass - restore path
        if target_grounding_col in previous_column:
            # found the target column

            def update_grounding(col):
                if col.get_col_name() == self.schema.primary_keys[col.get_tbl_name()]:
                    tbl_grounding = GroundingKey.make_table_grounding(col.get_tbl_name())
                    if col not in var_for_grounding:
                        assert tbl_grounding not in var_for_grounding
                        var_name = get_var_name(col.get_tbl_name())
                        var_for_grounding[col] = var_name
                        var_for_grounding[tbl_grounding] = var_name
                    else:
                        assert tbl_grounding in var_for_grounding
                        assert var_for_grounding[col] == var_for_grounding[tbl_grounding]
                else:
                    if col not in var_for_grounding:
                        var_for_grounding[col] = get_var_name(col.get_col_name())


            def add_query_line_in_same_table(pr_key, col):
                rel = self.get_col_grounding_rel(col)

                update_grounding(pr_key)
                prev_var = var_for_grounding[pr_key]

                update_grounding(col)
                cur_var = var_for_grounding[col]

                query_line = f"{prev_var} {rel} {cur_var}."
                return query_line


            def add_query_line_cross_tables(src, tgt):
                rel = self.get_foreign_rel(src, tgt)

                update_grounding(src)
                src_var = var_for_grounding[src]

                update_grounding(tgt)
                tgt_var = var_for_grounding[tgt]

                query_line = f"{src_var} {rel} {tgt_var}."
                return query_line


            query_path = ""
            found_grounded_var = False
            cur_col_grounding = target_grounding_col
            while not found_grounded_var and previous_column[cur_col_grounding] is not None:
                prev_col_grounding = previous_column[cur_col_grounding]

                # is previous step already has a variable - then quit
                found_grounded_var = prev_col_grounding in var_for_grounding

                cur_tbl_name = cur_col_grounding.get_tbl_name()
                cur_col_name = cur_col_grounding.get_col_name()
                prev_tbl_name = prev_col_grounding.get_tbl_name()
                prev_col_name = prev_col_grounding.get_col_name()

                if prev_tbl_name == cur_tbl_name and prev_col_name == self.schema.primary_keys[prev_tbl_name]:
                    # if is a primary key - visit all columns of the same table
                    query_line = add_query_line_in_same_table(prev_col_grounding, cur_col_grounding)
                elif prev_tbl_name == cur_tbl_name and cur_col_name == self.schema.primary_keys[cur_tbl_name]:
                    # if is not a primary key - visit the primary key
                    query_line = add_query_line_in_same_table(cur_col_grounding, prev_col_grounding)
                elif prev_col_grounding in self.schema.foreign_key_tgt_for_src:
                    # src to tgt foreign link
                    query_line = add_query_line_cross_tables(prev_col_grounding, cur_col_grounding)
                elif cur_col_grounding in self.schema.foreign_key_tgt_for_src:
                    # tgt to src foreign link
                    query_line = add_query_line_cross_tables(cur_col_grounding, prev_col_grounding)
                else:
                    raise RuntimeError(f"Could not process connection {prev_col_grounding} --> {cur_col_grounding}")

                # is prev_col_grounding is to the primary key add grounding to the corresponding table
                if prev_col_name == self.schema.primary_keys[prev_tbl_name]:
                    tbl_grounding = GroundingKey.make_table_grounding(prev_tbl_name)
                    if tbl_grounding in var_for_grounding:
                        assert var_for_grounding[tbl_grounding] == var_for_grounding[prev_col_grounding]
                    else:
                        var_for_grounding[tbl_grounding] = var_for_grounding[prev_col_grounding]

                if not query_path:
                    query_path = query_line
                else:
                    query_path = query_line + "\n" + query_path

                cur_col_grounding = prev_col_grounding
        else:
            raise RuntimeError(f"Could not find path from {source_groundings} to {target_grounding_col}")

        return query_path, var_for_grounding


class VariableNamer():
    def __init__(self, max_var_index=100):
        self._var_names_cache = {}
        self.max_var_index = max_var_index

    def get_var_name(self, name="var"):
        name = name.replace("'", "")
        try_name = "?" + str(name)
        if try_name not in self._var_names_cache:
            new_name = try_name
        else:
            new_name = None
            for i in range(1, self.max_var_index):
                try_name = "?" + str(name) + "_" + str(i)
                if try_name not in self._var_names_cache:
                    new_name = try_name
                    break
            if not new_name:
                raise RuntimeError(f"Could not find a variable name for {name}, current names: {list(self._var_names_cache.keys())}")

        self._var_names_cache[new_name] = True
        return new_name


@attr.s
class ContextOutputUnit:
    var = attr.ib()
    grnd = attr.ib()
    output_col = attr.ib()


class LocalContext():
    def __init__(self):
        self.query = ""
        self.var_for_grounding = {}
        self.output_units_for_qdmr_index = {}
        self.var_with_distinct = {}
        self.output_col_for_var_name = {}
        self.sorting_info = None

    def is_empty(self):
        return not self.query

    def swap_var(self, src_var, tgt_var):
        # swap output vars for different qdmr ops
        for qdmr_index in self.output_units_for_qdmr_index:
            for unit in self.output_units_for_qdmr_index[qdmr_index]:
                if unit.var == src_var:
                    unit.var = tgt_var
        # swap vars in contexts
        for gr in self.var_for_grounding:
            if self.var_for_grounding[gr] == src_var:
                self.var_for_grounding[gr] = tgt_var
            # replace var name in query with possible delimiters afterwards
            for delimiter in [" ", ".", ";", ",", ")", "\n"]:
                self.query = self.query.replace(src_var + delimiter, tgt_var + delimiter)
        # swap var in self.var_with_distinct
        if src_var in self.var_with_distinct:
            self.var_with_distinct[tgt_var] = copy.deepcopy(self.var_with_distinct[src_var])
            del self.var_with_distinct[src_var]

    def append_query(self, new_line):
        if self.query:
            if new_line:
                self.query += "\n" + new_line
        else:
            self.query = new_line

    def __repr__(self):
        output = "Local context:\n"
        output += self.query
        output += "\n" + "Grounded vars:"
        for gr in self.var_for_grounding:
            output += "\n" + str(gr) + " : " + self.var_for_grounding[gr]
        output += "\n" + "Have QDMR steps: " + str(list(self.output_units_for_qdmr_index.keys()))
        return output

    def create_table_pattern(self, grounding, schema, get_var_name, qdmr_index=None):
        tbl_name = grounding.get_tbl_name()
        tbl_grounding = GroundingKey.make_table_grounding(tbl_name)
        assert tbl_name in schema.schema.table_names, f"Table grounding {tbl_grounding} should correspond to one of the tables: {schema.schema.table_names}"

        pr_key_grounding = GroundingKey.make_column_grounding(tbl_name, schema.schema.primary_keys[tbl_name])

        if tbl_grounding not in self.var_for_grounding and \
           pr_key_grounding not in self.var_for_grounding:
            # neither table nor its primary key are grounded
            output_var = get_var_name(tbl_name)

            self.var_for_grounding[tbl_grounding] = output_var
            self.var_for_grounding[pr_key_grounding] = output_var

            # get self-reference for the primary key
            relation = schema.get_table_grounding_rel(tbl_grounding)
            query_line = f"{output_var} {relation} {output_var}."
            self.append_query(query_line)
        elif tbl_grounding in self.var_for_grounding and \
             pr_key_grounding not in self.var_for_grounding:
            output_var = self.var_for_grounding[tbl_grounding]
            self.var_for_grounding[pr_key_grounding] = output_var
        elif tbl_grounding not in self.var_for_grounding and \
             pr_key_grounding in self.var_for_grounding:
            output_var = self.var_for_grounding[pr_key_grounding]
            self.var_for_grounding[tbl_grounding] = output_var

        # both table and its primary key should be grounded to the same thing
        assert tbl_grounding in self.var_for_grounding and pr_key_grounding in self.var_for_grounding
        tbl_var = self.var_for_grounding[tbl_grounding]
        col_var = self.var_for_grounding[pr_key_grounding]
        assert tbl_var == col_var, \
            f"Table {tbl_grounding} and its primary key {pr_key_grounding} should be grounded tot he same things but not to {tbl_var} and {col_var}"

        if qdmr_index is not None:
            self.output_units_for_qdmr_index[qdmr_index] = [ContextOutputUnit(var=tbl_var, grnd=tbl_grounding,
                output_col=OutputColumnId.from_grounding(tbl_grounding, schema.schema))]

    def create_column_pattern(self, grounding, schema, get_var_name, qdmr_index=None):
        tbl_name = grounding.get_tbl_name()
        col_name = grounding.get_col_name()
        tbl_grounding = GroundingKey.make_table_grounding(tbl_name)
        col_grounding = GroundingKey.make_column_grounding(tbl_name, col_name)

        assert tbl_name in schema.schema.table_names, f"Column grounding table name {tbl_grounding} should correspond to one of the tables: {schema.schema.table_names}"
        assert col_name in schema.schema.column_names[tbl_name], f"Column grounding column name {col_grounding} should correspond to one of the columns {schema.schema.column_names[tbl_name]} in table {tbl_name}"

        if col_grounding not in self.var_for_grounding:
            self.create_table_pattern(col_grounding, schema, get_var_name)
            if col_grounding not in self.var_for_grounding:
                # if the column is a primary key it should be grounded with a table
                output_var = get_var_name(col_name)
                self.var_for_grounding[col_grounding] = output_var
                tbl_var = self.var_for_grounding[tbl_grounding]
                relation = schema.get_col_grounding_rel(col_grounding)
                query_line = f"{tbl_var} {relation} {output_var}."
                self.append_query(query_line)

        assert col_grounding in self.var_for_grounding
        output_var = self.var_for_grounding[col_grounding]

        if qdmr_index is not None:
            self.output_units_for_qdmr_index[qdmr_index] = [ContextOutputUnit(var=output_var, grnd=col_grounding,
                output_col=OutputColumnId.from_grounding(col_grounding, schema.schema))]

        self.clear_table_self_ref()

    def add_and_fix_grounding(self, target_grounding, output_var, schema):
        if target_grounding.istbl():
            tbl_name = target_grounding.get_tbl_name()
            col_name = schema.schema.primary_keys[tbl_name]
            pr_key_grounding = GroundingKey.make_column_grounding(tbl_name, col_name)

            if pr_key_grounding in self.var_for_grounding or target_grounding in self.var_for_grounding:
                # Tricky case: the context already has a variable at the same grounding
                # in this case keep the old grounding and rename all the vars in the query
                old_var = self.var_for_grounding[pr_key_grounding] if pr_key_grounding in self.var_for_grounding else self.var_for_grounding[target_grounding]
                if old_var != output_var:
                    self.swap_var(output_var, old_var)
                    output_var = old_var

            self.var_for_grounding[pr_key_grounding] = output_var
            self.var_for_grounding[target_grounding] = output_var
        elif target_grounding.iscol() or target_grounding.isval():
            tbl_name = target_grounding.get_tbl_name()
            col_name = target_grounding.get_col_name()
            col_grounding = GroundingKey.make_column_grounding(tbl_name, col_name)

            if col_grounding in self.var_for_grounding:
                # Tricky case: the context already has a variable at the same grounding
                # in this case keep the old grounding and rename all the vars in the query
                old_var = self.var_for_grounding[col_grounding]
                if old_var != output_var:
                    self.swap_var(output_var, old_var)
                    output_var = old_var

            self.var_for_grounding[col_grounding] = output_var
            if schema.schema.primary_keys[tbl_name] == col_name:
                tbl_grounding = GroundingKey.make_table_grounding(tbl_name)
                self.var_for_grounding[tbl_grounding] = output_var
        elif target_grounding.iscomp():
            if len(target_grounding.keys) == 3:
                self.add_and_fix_grounding(target_grounding.keys[2], output_var, schema)
        else:
            raise RuntimeError(f"Do not know what to do with grounding {target_grounding}")

    def add_column(self, col_grnd, schema, get_var_name, source_grnds, force_add=False):
        # if have additional column - compare it with the value
        assert col_grnd.iscol(), "The first arg of add_column should be a column grounding"
        if col_grnd not in self.var_for_grounding or force_add:
            if not force_add:
                var_for_grounding = self.var_for_grounding
            else:
                var_for_grounding = {source_grnd : self.var_for_grounding[source_grnd] for source_grnd in source_grnds}
                for grnd, var in self.var_for_grounding.items():
                    for source_grnd in source_grnds:
                        if var == var_for_grounding[source_grnd]:
                            var_for_grounding[grnd] = var
                            break

            query_path, var_for_grounding = \
                schema.find_path(source_grnds,
                                 col_grnd, get_var_name,
                                 var_for_grounding=var_for_grounding)
            self.append_query(query_path)
        else:
            var_for_grounding = self.var_for_grounding

        for source_grnd in source_grnds:
            if type(source_grnd) == GroundingKey: # can be also a list of GroundingKey
                assert source_grnd in var_for_grounding

        if not force_add:
            self.var_for_grounding = var_for_grounding
            assert col_grnd in self.var_for_grounding
            self.clear_table_self_ref()
            return self.var_for_grounding[col_grnd]
        else:
            return var_for_grounding[col_grnd]

    def clear_table_self_ref(self):
        if not self.query:
            return

        # split query in terms of logical lines: group in {} keep together
        query_parts = []
        bracket_counter = 0
        last_line_start = 0
        if self.query[-1] != "\n":
            query_ = self.query + "\n"
        else:
            query_ = self.query
        for i, sym in enumerate(query_):
            assert bracket_counter >= 0, f"Incorrect bracket sequence in\n{query_}"
            if sym == "{":
                bracket_counter += 1
            elif sym == "}":
                bracket_counter -= 1
            elif sym == "\n" and bracket_counter == 0:
                last_line = query_[last_line_start : i]
                last_line_start = i + 1
                if last_line:
                    query_parts.append(last_line)

        # try to remove separate lines for grounding tables.
        for grnd, var in self.var_for_grounding.items():
            if grnd.istbl():
                # find a line with self-grounding and with a column to that grounding
                self_ref_index = None
                col_ref_index = None
                for i_line, line in enumerate(query_parts):
                    if "{" not in line and "}" not in line:
                        # check only lines without nested queries
                        line_parts = line.split(" ")
                        if len(line_parts) == 3:
                            # should have exactly three components
                            last_part = line_parts[-1][:-1] if line_parts[-1][-1] in [".", ",", ";", ":"] else line_parts[-1]
                            if line_parts[0] == var:
                                if last_part == var:
                                    self_ref_index = i_line
                                else:
                                    col_ref_index = i_line
                            else:
                                if last_part == var:
                                    col_ref_index = i_line
                if self_ref_index is not None and col_ref_index is not None:
                    # add exceptions:
                    if self_ref_index < col_ref_index and "MINUS" in query_parts[self_ref_index:col_ref_index]:
                        # have MINUS inbetween the two positions
                        continue

                    del query_parts[self_ref_index]

        # join the parts back into query
        self.query = "\n".join(query_parts)


class QueryStep():
    indent_block = "  " # indent queries with 2 spaces
    template_full = textwrap.dedent("""\
        SELECT {output_var}
        WHERE
        {{
        {query}
        }}""")
    template_full_with_distinct = textwrap.dedent("""\
        SELECT DISTINCT {output_var}
        WHERE
        {{
        {query}
        }}""")
    template_to_inline = textwrap.dedent("""\
        {{
        {query}
        }}""")
    def __init__(self, creator):
        assert type(creator) == QueryCreator, f"When constructing an op should get QueryCreator but have {type(creator)}"
        self.creator = creator

    def build_step(self, qdmr_index, inline_query, context=None):
        if context is not None and qdmr_index in context.output_units_for_qdmr_index and inline_query:
            return context
        if context is None:
            context = LocalContext()
        if not inline_query:
            assert context.is_empty(), "Observing a request for non-inline query with non-empty context passed, do not know what to do"
        return self.build_step_op(qdmr_index, inline_query, context)

    def build_step_op(self, qdmr_index, inline_query, context):
        raise NotImplementedError("This is an interface method, should be using its subclasses")

    def build_full_query_from_inline(self, query, output_vars, context=None):
        query = textwrap.indent(query, self.indent_block)
        if context is None or not any(output_var in context.var_with_distinct for output_var in output_vars):
            return self.template_full.format(output_var=" ".join(output_vars), query=query)
        else:
            return self.template_full_with_distinct.format(output_var=" ".join(output_vars), query=query)

    def build_inline_query_from_full(self, query):
        query = textwrap.indent(query, self.indent_block)
        return self.template_to_inline.format(query=query)

    def extract_args(self, qdmr_index, target_op, num_args=None):
        op = self.creator.qdmr.ops[qdmr_index]
        assert op.lower() == target_op, f"{type(self).__name__} can execute {target_op} but not {op}"
        args = self.creator.qdmr.args[qdmr_index]
        if num_args is not None:
            assert len(args) == num_args, f"{op} should take {num_args} args, but has {args}"
        return args


class QueryStepAggregate(QueryStep):
    aggregator_str = "({aggregator}({input_var}) AS {output_var})"
    aggregator_str_distinct = "({aggregator}(DISTINCT {input_var}) AS {output_var})"
    template_full = textwrap.dedent("""\
        SELECT {output_str}
        WHERE
        {{
        {query}
        }}""")

    def __init__(self, creator):
        super().__init__(creator)

    @staticmethod
    def parse_aggregator_value(aggregator):
        aggregator = aggregator.replace("'", "")
        aggregator_good_values = ["count", "min", "max", "avg", "sum"]
        assert aggregator.lower() in aggregator_good_values, f"First arg of aggregator should be in {aggregator_good_values}, but have {aggregator}"
        return aggregator

    def build_step_op(self, qdmr_index, inline_query, context):
        grounding = self.creator.grounding
        schema = self.creator.schema
        get_var_name = self.creator.namer.get_var_name

        op = "aggregate"
        args = self.extract_args(qdmr_index, op, num_args=2)

        aggregator = args[0]
        var_to_aggregate = args[1]

        aggregator = self.parse_aggregator_value(aggregator)

        argument_index = QdmrInstance.ref_to_index(var_to_aggregate, qdmr_index)

        context_args = \
            self.creator.construct_set_of_args([argument_index], inline_query=True, context=None)

        # add proper indentation
        query = textwrap.indent(context_args.query, self.indent_block)

        input_vars = [u.var for u in context_args.output_units_for_qdmr_index[argument_index]]
        output_vars = [get_var_name(aggregator) for input_var in input_vars]

        output_strs = []
        for input_var, output_var in zip(input_vars, output_vars):
            with_distinct = input_var in context_args.var_with_distinct
            aggregator_template = self.aggregator_str if not with_distinct else self.aggregator_str_distinct
            aggregator_str = aggregator_template.format(aggregator=aggregator, input_var=input_var, output_var=output_var)
            output_strs.append(aggregator_str)

        query = self.template_full.format(output_str=" ".join(output_strs),
                                          query=query)

        if inline_query:
            query = self.build_inline_query_from_full(query)

        context.append_query(query)
        output_units = []
        for input_var, output_var in zip(input_vars, output_vars):
            output_col = OutputColumnId.add_aggregator(context_args.output_col_for_var_name[input_var], aggregator)
            u = ContextOutputUnit(var=output_var, grnd=None, output_col=output_col)
            context.output_col_for_var_name[output_var] = u
            output_units.append(u)
        context.output_units_for_qdmr_index[qdmr_index] = output_units

        return context


class QueryStepGroup(QueryStep):
    template_full = textwrap.dedent("""\
        SELECT {output_vars}
        WHERE
        {{
        {query}
        }}
        GROUP BY {index_var}""")
    template_group_var = "({aggregator}({target_var}) AS {aggregated_var})"

    def __init__(self, creator):
        super().__init__(creator)

    def build_step_op(self, qdmr_index, inline_query, context, order_of_outputs=None):
        grounding = self.creator.grounding
        schema = self.creator.schema
        get_var_name = self.creator.namer.get_var_name

        op = "group"
        args = self.extract_args(qdmr_index, op, num_args=3)

        aggregator = args[0]
        target_var = args[1]
        index_var = args[2]

        aggregator = QueryStepAggregate.parse_aggregator_value(aggregator)

        target_var_index = QdmrInstance.ref_to_index(target_var, qdmr_index)
        index_var_index = QdmrInstance.ref_to_index(index_var, qdmr_index)

        context_args = \
            self.creator.construct_set_of_args([target_var_index, index_var_index], inline_query=True, context=None)

        # add proper indentation
        query = textwrap.indent(context_args.query, self.indent_block)

        target_vars = [u.var for u in context_args.output_units_for_qdmr_index[target_var_index]]
        output_vars = [get_var_name(aggregator) for target_var in target_vars]

        group_output_vars = []
        for target_var, output_var in zip(target_vars, output_vars):
            group_output_var = self.template_group_var.format(aggregator=aggregator,
                                                              target_var=target_var,
                                                              aggregated_var=output_var)
            group_output_vars.append(group_output_var)

        index_units = context_args.output_units_for_qdmr_index[index_var_index]
        index_vars = [u.var for u in index_units]

        # process information about output columns
        output_col_for_var_name = copy.deepcopy(context_args.output_col_for_var_name)
        for u in index_units:
            output_col_for_var_name[u.var] = u.output_col
        group_output_cols = []
        for output_var, target_var in zip(output_vars, target_vars):
            output_col_group = context_args.output_col_for_var_name[target_var]
            output_col_group = OutputColumnId.add_aggregator(output_col_group, aggregator)
            output_col_for_var_name[output_var] = output_col_group
            group_output_cols.append(output_col_group)

        output_units_group = []
        for output_col_group, output_var in zip(group_output_cols, output_vars):
            u = ContextOutputUnit(var=output_var, grnd=None,
                                output_col=output_col_group)
            output_units_group.append(u)

        if order_of_outputs is None:
            # the default behaviour when called separately
            if inline_query:
                output_vars = index_vars + group_output_vars
                output_units = output_units_group # have correct outputs for further processing
            else:
                output_vars = index_vars + group_output_vars
                output_units = index_units + output_units_group
        else:
            if len(order_of_outputs) == 2 and \
               order_of_outputs[0] == index_var_index and \
               order_of_outputs[1] == qdmr_index:
                # order of outputs as in the default regime
                output_vars = index_vars + group_output_vars
                output_units = index_units + output_units_group
            elif len(order_of_outputs) == 2 and \
               order_of_outputs[0] == index_var_index and \
               order_of_outputs[1] == qdmr_index:
                # order of outputs reverse to the default regime
                output_vars = group_output_vars + index_vars
                output_units = output_units_group + index_units
            else:
                raise RuntimeError(f"GROUP: unrecognized order of outputs: {order_of_outputs}, ops: {', '.join(self.creator.qdmr.ops[i] for i in order_of_outputs)}")

        query = self.template_full.format(output_vars=" ".join(output_vars),
                                          index_var=" ".join(index_vars),
                                          query=query
                                         )

        if inline_query:
            query = self.build_inline_query_from_full(query)

        context.append_query(query)
        for index_var in index_vars:
            context.var_with_distinct[index_var] = True

        context.output_col_for_var_name.update(output_col_for_var_name)
        context.output_units_for_qdmr_index.update(context_args.output_units_for_qdmr_index)
        context.output_units_for_qdmr_index[qdmr_index] = output_units

        for u in (index_units + output_units_group):
            if u.grnd is not None:
                context.add_and_fix_grounding(target_grounding=u.grnd,
                                              output_var=u.var,
                                              schema=schema)

        # # A bug source:
        # gr = context.grounding_for_qdmr_index[index_var_index]
        # context.var_for_grounding[gr] != context_args.var_for_grounding[gr]

        return context


class QueryStepSelect(QueryStep):
    def __init__(self, creator):
        super().__init__(creator)

    def build_step_op(self, qdmr_index, inline_query, context):
        grounding = self.creator.grounding
        schema = self.creator.schema
        get_var_name = self.creator.namer.get_var_name

        op = "select"
        args = self.extract_args(qdmr_index, op, num_args=1)

        select_target = GroundingIndex(qdmr_index, 0, args[0])
        assert select_target in grounding, f"Target for {op} {select_target} should be grounded but have {grounding}"

        target_grounding = grounding[select_target]

        if target_grounding.iscol():
            target_grounding_col = target_grounding
        elif target_grounding.istbl():
            tbl_name = target_grounding.get_tbl_name()
            target_grounding_col = GroundingKey.make_column_grounding(tbl_name, schema.schema.primary_keys[tbl_name])
        elif target_grounding.isval():
            target_grounding_col = GroundingKey.make_column_grounding(target_grounding.get_tbl_name(), target_grounding.get_col_name())
        else:
            raise RuntimeError(f"Do not know what to do with {target_grounding}")

        if context.is_empty() or not list(context.var_for_grounding.keys()):
            if target_grounding.istbl():
                context.create_table_pattern(target_grounding, schema, get_var_name, qdmr_index=qdmr_index)
            elif target_grounding.iscol():
                context.create_column_pattern(target_grounding, schema, get_var_name, qdmr_index=qdmr_index)
                output_var = context.var_for_grounding[target_grounding]
                context.var_with_distinct[output_var] = True
            elif target_grounding.isval():
                context.create_column_pattern(target_grounding_col, schema, get_var_name, qdmr_index=qdmr_index)
                output_var = context.var_for_grounding[target_grounding_col]
                comparison_sparql_str = schema.get_key_for_comparison(target_grounding)
                query_line = QueryStepComparative.make_filter_line(output_var, "=", comparison_sparql_str)
                context.var_for_grounding[target_grounding] = output_var
                context.append_query(query_line)
            else:
                raise RuntimeError(f"Unknown grounding type {target_grounding.type} for {target_grounding}")
        else:
            # effectively do project from selected context
            # create path to the column grounding
            if target_grounding_col in context.var_for_grounding:
                output_var = context.var_for_grounding[target_grounding_col]
            else:
                source_groundings = list(context.var_for_grounding.keys())
                context.add_column(target_grounding_col, schema, get_var_name, source_groundings)
                output_var = context.var_for_grounding[target_grounding_col]

            if target_grounding.isval():
                # was gronding to a value: need to add a filter operation
                comparison_sparql_str = schema.get_key_for_comparison(target_grounding)
                query_line = QueryStepComparative.make_filter_line(context.var_for_grounding[target_grounding_col], "=", comparison_sparql_str)
                context.append_query(query_line)

        output_var = context.var_for_grounding[target_grounding]
        context.output_col_for_var_name[output_var] = OutputColumnId.from_grounding(target_grounding_col)
        context.output_units_for_qdmr_index[qdmr_index] =\
            [ContextOutputUnit(var=output_var, grnd=target_grounding_col,
                output_col=context.output_col_for_var_name[output_var])]

        if qdmr_index in grounding["distinct"]:
            context.var_with_distinct[output_var] = True

        if not inline_query:
            context.query = self.build_full_query_from_inline(query=context.query,
                output_vars=[u.var for u in context.output_units_for_qdmr_index[qdmr_index]],
                context=context)
            for unit in context.output_units_for_qdmr_index[qdmr_index]:
                if unit.grnd is not None:
                    context.add_and_fix_grounding(target_grounding=unit.grnd,
                                                output_var=unit.var,
                                                schema=schema)

        return context


class QueryStepProject(QueryStep):
    def __init__(self, creator):
        super().__init__(creator)

    @classmethod
    def get_target_grounding_col(cls, target_grounding, qdmr_index, context, schema):
        if target_grounding.iscol():
            target_groundings_col = [target_grounding]
        elif target_grounding.istbl():
            tbl_name = target_grounding.get_tbl_name()
            target_groundings_col = [GroundingKey.make_column_grounding(tbl_name, schema.schema.primary_keys[tbl_name])]
        elif target_grounding.isval():
            target_groundings_col = [GroundingKey.make_column_grounding(target_grounding.get_tbl_name(), target_grounding.get_col_name())]
        elif target_grounding.isref():
            ref_index = QdmrInstance.ref_to_index(target_grounding.keys[0], qdmr_index)
            assert ref_index in context.output_units_for_qdmr_index, f"Qdmr step {target_grounding.keys[0]} should be grounded"

            target_groundings_col = []
            for unit in context.output_units_for_qdmr_index[ref_index]:
                grnd = unit.grnd
                assert grnd.istbl() or grnd.iscol() or  grnd.isval()
                target_groundings_col.append(GroundingKey.make_column_grounding(grnd.get_tbl_name(), grnd.get_col_name()))
        else:
            raise RuntimeError(f"Do not know what to do with {target_grounding}")

        return target_groundings_col


    def build_step_op(self, qdmr_index, inline_query, context):
        grounding = self.creator.grounding
        schema = self.creator.schema
        get_var_name = self.creator.namer.get_var_name

        op = "project"
        args = self.extract_args(qdmr_index, op, num_args=2)

        target = GroundingIndex(qdmr_index, 0, args[0])
        source = args[1]

        argument_index = QdmrInstance.ref_to_index(source, qdmr_index)

        target_indices = [argument_index]
        # allow to have reference gronding in project - useful when the group op is changed
        if target in grounding and grounding[target]:
            target_grounding_ = grounding[target]
            if isinstance(target_grounding_, GroundingKey) and target_grounding_.isref():
                extra_index = QdmrInstance.ref_to_index(target_grounding_.keys[0], qdmr_index)
                target_indices = target_indices + [extra_index]

        context = \
            self.creator.construct_set_of_args(target_indices, inline_query=True, context=context)

        source_output_units = context.output_units_for_qdmr_index[argument_index]

        # allow to skip target grounding in project, in this case set the same grounding as the source
        if target in grounding and grounding[target]:
            target_groundings = [grounding[target]]
        else:
            target_groundings = [unit.grnd for unit in source_output_units]

        target_groundings_col = []
        target_groundings_new = []
        for target_grounding in target_groundings:
            target_grounding_col_list = self.get_target_grounding_col(target_grounding, qdmr_index, context, schema)
            target_groundings_col.extend(target_grounding_col_list)
            target_groundings_new.extend([target_grounding] * len(target_grounding_col_list))
        target_groundings = target_groundings_new

        # create path to the column grounding
        for target_grounding_col in target_groundings_col:
            if target_grounding_col not in context.var_for_grounding:
                source_groundings = [u.grnd for u in source_output_units]
                context.add_column(target_grounding_col, schema, get_var_name, source_groundings)

        output_var = context.var_for_grounding[target_grounding_col]

        for target_grounding, target_grounding_col in zip(target_groundings, target_groundings_col):
            if target_grounding.isval():
                # was gronding to a value: need to add a filter operation
                comparison_sparql_str = schema.get_key_for_comparison(target_grounding)
                query_line = QueryStepComparative.make_filter_line(context.var_for_grounding[target_grounding_col], "=", comparison_sparql_str)
                context.append_query(query_line)

        output_units_for_qdmr_index = []
        for target_grounding_col in target_groundings_col:
            output_var = context.var_for_grounding[target_grounding_col]
            context.output_col_for_var_name[output_var] = OutputColumnId.from_grounding(target_grounding_col)
            unit = ContextOutputUnit(var=output_var,
                                     grnd=target_grounding_col,
                                     output_col=context.output_col_for_var_name[output_var])
            output_units_for_qdmr_index.append(unit)

        context.output_units_for_qdmr_index[qdmr_index] = output_units_for_qdmr_index

        if qdmr_index in grounding["distinct"]:
            context.var_with_distinct[output_var] = True

        if not inline_query:
            context.query = self.build_full_query_from_inline(query=context.query,
                output_vars=[u.var for u in context.output_units_for_qdmr_index[qdmr_index]],
                context=context)
            for unit in context.output_units_for_qdmr_index[qdmr_index]:
                if unit.grnd is not None:
                    context.add_and_fix_grounding(target_grounding=unit.grnd,
                                                output_var=unit.var,
                                                schema=schema)

        return context


class QueryStepUnion(QueryStep):
    def __init__(self, creator):
        super().__init__(creator)

    def build_step_op(self, qdmr_index, inline_query, context):
        """There are several variants of the UNION op:
            1) Horizontal union - when outputing multiple related columns
            2) Vertical union - when outputing multiple filters on the same column
            3) Union of aggregators - when outputing multiple aggregators
            4) Union after GROUP
        """
        qdmr = self.creator.qdmr
        grounding = self.creator.grounding
        schema = self.creator.schema
        get_var_name = self.creator.namer.get_var_name

        op = "union"
        args = self.extract_args(qdmr_index, op)

        source_indices = [QdmrInstance.ref_to_index(arg, qdmr_index) for arg in args]

        # check if we have a union of aggregators
        aggregate_in_args = [qdmr.ops[i] == "aggregate" for i in source_indices]
        if all(aggregate_in_args):
            # do union of aggregators
            return self.union_of_aggregates(qdmr_index, inline_query, context, source_indices)

        # check if we have UNION after GROUP
        if any(qdmr.ops[i] == "group" for i in source_indices):
            return self.union_after_group(qdmr_index, inline_query, context, source_indices)

        # use static analisys to check whether union is horizontal or vertical
        # results saved in self.creator.grounding_for_qdmr_index
        col_groundings = []
        for i in source_indices:
            col_groundings_i = []
            for gr in self.creator.grounding_for_qdmr_index[i]:
                if gr is not None:
                    tbl_name = gr.get_tbl_name()
                    if gr.istbl():
                        col_name = schema.schema.primary_keys[tbl_name]
                    else:
                        col_name = gr.get_col_name()
                    col_gr = GroundingKey.make_column_grounding(tbl_name, col_name)
                    col_groundings_i.append(col_gr)
            col_groundings_i = tuple(col_groundings_i)
            col_groundings.append(col_groundings_i)

        is_vertical_union = all(col_gr == col_groundings[0] for col_gr in col_groundings[1:]) and len(col_groundings) > 1
        is_horizontal_union = not is_vertical_union

        if is_horizontal_union:
            context = \
                self.creator.construct_set_of_args(source_indices, inline_query=True, context=context)

            context.output_units_for_qdmr_index[qdmr_index] = sum([context.output_units_for_qdmr_index[i] for i in source_indices], [])

            if not inline_query:
                output_vars = sum([[u.var for u in context.output_units_for_qdmr_index[i]] for i in source_indices], [])
                query = self.template_full.format(output_var=" ".join(output_vars),
                                                  query=textwrap.indent(context.query, self.indent_block),
                                                )
                context.query = query

            for unit in context.output_units_for_qdmr_index[qdmr_index]:
                if unit.grnd is not None:
                    context.add_and_fix_grounding(target_grounding=unit.grnd,
                                                output_var=unit.var,
                                                schema=schema)

            return context
        elif is_vertical_union:
            return self.vertical_union(qdmr_index, inline_query, context, source_indices, col_groundings[0])
        else:
            raise RuntimeError(f"Unrecognized type of union at step {qdmr_index}, args: {qdmr.args[qdmr_index]}")

    def union_of_aggregates(self, qdmr_index, inline_query, context, source_indices):
        aggregate_in_args = [self.creator.qdmr.ops[i] == "aggregate" for i in source_indices]
        assert all(aggregate_in_args), f"UNION of aggregators requires all aggregators as arguments, but have {[self.creator.qdmr.ops[i] for i in source_indices]}"

        grounding = self.creator.grounding
        schema = self.creator.schema
        get_var_name = self.creator.namer.get_var_name

        aggregate_ops = [self.creator.qdmr.args[i][0] for i in source_indices]
        aggregate_ops = [QueryStepAggregate.parse_aggregator_value(op) for op in aggregate_ops]

        aggregate_arguments = [self.creator.qdmr.args[i][1] for i in source_indices]
        aggregate_arguments = [QdmrInstance.ref_to_index(arg, qdmr_index) for arg in aggregate_arguments]
        aggregate_arguments_unique = list(set(aggregate_arguments))

        context_args = \
            self.creator.construct_set_of_args(aggregate_arguments_unique, inline_query=True, context=None)

        output_units = []
        aggregator_vars_list = []
        aggregate_ops_list = []
        output_vars = []
        for index, aggregate_op in zip(aggregate_arguments, aggregate_ops):
            units = context_args.output_units_for_qdmr_index[index]
            for u in units:
                output_var = get_var_name(aggregate_op)
                output_vars.append(output_var)
                aggregator_vars_list.append(u.var)
                aggregate_ops_list.append(aggregate_op)
                output_col = OutputColumnId.add_aggregator(u.output_col, aggregate_op)
                output_units.append(ContextOutputUnit(var=output_var, grnd=None, output_col=output_col))
                context.output_col_for_var_name[output_var] = output_col

        query_output_var_str = " ".join([f"({op}({'DISTINCT ' if var in context_args.var_with_distinct else ''}{var}) AS {out_var})" for op, var, out_var in zip(aggregate_ops_list, aggregator_vars_list, output_vars)])

        query = self.template_full.format(output_var=query_output_var_str,
                                          query=textwrap.indent(context_args.query, self.indent_block))

        if inline_query:
            query = self.build_inline_query_from_full(query)

        context.append_query(query)
        context.output_units_for_qdmr_index[qdmr_index] = output_units

        return context

    def vertical_union(self, qdmr_index, inline_query, context, source_indices, col_grounding):
        context_for_each_arg = \
            [self.creator.construct_set_of_args([arg], inline_query=True, context=None)
                for arg in source_indices]

        i_arg = 0
        query_step = textwrap.indent(context_for_each_arg[i_arg].query, self.indent_block)
        query_template = textwrap.dedent("""\
            {{
            {query_step}
            }}""")
        query = query_template.format(query_step=query_step)

        output_units = context_for_each_arg[i_arg].output_units_for_qdmr_index[source_indices[i_arg]]
        for unit in output_units:
            if unit.var in context_for_each_arg[i_arg].var_with_distinct:
                context.var_with_distinct[unit.var] = True

        for context_step, step_index in zip(context_for_each_arg[1:], source_indices[1:]):
            # all contexts except the first one
            cur_output_units = context_step.output_units_for_qdmr_index[step_index]
            for cur_unit, unit in zip(cur_output_units, output_units):
                context_step.swap_var(cur_unit.var, unit.var)
            query_step = textwrap.indent(context_step.query, self.indent_block)
            query_template = textwrap.dedent("""\
                UNION
                {{
                {query_step}
                }}""")
            query_step = query_template.format(query_step=query_step)
            query += "\n" + query_step

            for cur_unit in output_units:
                if unit.var in context_step.var_with_distinct:
                    context.var_with_distinct[unit.var] = True

        context.append_query(query)
        i_arg = 0
        context.output_col_for_var_name.update(context_for_each_arg[i_arg].output_col_for_var_name)
        context.output_units_for_qdmr_index.update(context_for_each_arg[i_arg].output_units_for_qdmr_index)
        context.output_units_for_qdmr_index[qdmr_index] = context_for_each_arg[i_arg].output_units_for_qdmr_index[source_indices[i_arg]]

        for unit in context.output_units_for_qdmr_index[qdmr_index]:
            if unit.grnd is not None:
                context.var_for_grounding[unit.grnd] = unit.var

                context.add_and_fix_grounding(target_grounding=unit.grnd,
                                                output_var=unit.var,
                                                schema=self.creator.schema)
            context.output_col_for_var_name[unit.var] = OutputColumnId.from_grounding(unit.grnd)

        if not inline_query:
            context.query = self.build_full_query_from_inline(query=context.query,
                output_vars=[u.var for u in context.output_units_for_qdmr_index[qdmr_index]],
                context=context)
            for unit in context.output_units_for_qdmr_index[qdmr_index]:
                context.add_and_fix_grounding(target_grounding=unit.grnd,
                                            output_var=unit.var,
                                            schema=self.creator.schema)

        return context

    def union_after_group(self, qdmr_index, inline_query, context, source_indices):
        qdmr = self.creator.qdmr
        grounding = self.creator.grounding
        schema = self.creator.schema
        get_var_name = self.creator.namer.get_var_name

        assert any(qdmr.ops[i] == "group" for i in source_indices),\
            f"UNION after GROUP cannot handle this. {len(source_indices)} args, ops: {', '.join(qdmr.ops[i] for i in source_indices)}"

        # add simplified pattern for just one group
        if len(source_indices) == 2 and not inline_query:
            try:
                qdmr_index_group = source_indices[[qdmr.ops[i] for i in source_indices].index("group")]
                op = QueryStepGroup(self.creator)
                context = op.build_step_op(qdmr_index_group, inline_query=inline_query, context=context, order_of_outputs=source_indices)
                context.output_units_for_qdmr_index[qdmr_index] = context.output_units_for_qdmr_index[qdmr_index_group]
                return context
            except RuntimeError:
                # could not do a simplified group op
                pass

        # add all the ops - group will be processed first
        context = \
            self.creator.construct_set_of_args(source_indices, inline_query=True, context=context)

        output_units = sum([context.output_units_for_qdmr_index[i] for i in source_indices], [])
        context.output_units_for_qdmr_index[qdmr_index] = output_units

        if not inline_query:
            context.query = self.build_full_query_from_inline(query=context.query, output_vars=[u.var for u in output_units])

        return context


class QueryStepIntersection(QueryStep):
    def __init__(self, creator):
        super().__init__(creator)

    def build_step_op(self, qdmr_index, inline_query, context):
        grounding = self.creator.grounding
        schema = self.creator.schema
        get_var_name = self.creator.namer.get_var_name

        op = "intersection"
        args = self.extract_args(qdmr_index, op)

        source_index = QdmrInstance.ref_to_index(args[0], qdmr_index)
        filter_indices = [QdmrInstance.ref_to_index(arg, qdmr_index) for arg in args[1:]]

        context_src = \
            self.creator.construct_set_of_args([source_index], inline_query=True, context=None)

        output_units = context_src.output_units_for_qdmr_index[source_index]
        context_src.output_units_for_qdmr_index[qdmr_index] = output_units
        # add distinct to output vars anyway, otherwise we see Cartesian products everywhere, do not know how to fix this better
        for u in output_units:
            context_src.var_with_distinct[u.var] = True
        source_groundings = [u.output_col.grounding_column for u in output_units]

        for filter_index in filter_indices:
            context_filter = self.creator.construct_set_of_args([filter_index], inline_query=True, context=None)

            filter_units = context_filter.output_units_for_qdmr_index[filter_index]
            filter_columns = [u.output_col.grounding_column for u in filter_units]
            for filter_column in filter_columns:
                if filter_column not in source_groundings:
                    added_var = context_src.add_column(filter_column, schema, get_var_name, source_groundings, force_add=True)
                else:
                    added_var = context_src.var_for_grounding[filter_column]

                context_filter.swap_var(context_filter.var_for_grounding[filter_column],
                                        added_var)
            context_src.query = context_src.query + "\n" + context_filter.query

            context_src = QueryStepComparative.wrap_into_subquery(self, qdmr_index, context_src,
                                                                  schema, get_var_name, inline_query,
                                                                  output_units, filter_columns)
            context_src.query = self.build_inline_query_from_full(context_src.query)

        context.append_query(context_src.query)
        context.output_units_for_qdmr_index[qdmr_index] = output_units
        for u in output_units:
            if u.grnd is not None:
                context.var_for_grounding[u.grnd] = u.var
                context.add_and_fix_grounding(target_grounding=u.grnd,
                                                output_var=u.var,
                                                schema=self.creator.schema)
            context.var_with_distinct[u.var] = context_src.var_with_distinct[u.var]

        context.output_col_for_var_name = copy.deepcopy(context_src.output_col_for_var_name)

        if not inline_query:
            context.query = self.build_full_query_from_inline(query=context.query, output_vars=[u.var for u in output_units], context=context)
            for u in output_units:
                if u.grnd is not None:
                    context.add_and_fix_grounding(target_grounding=u.grnd,
                                                  output_var=u.var,
                                                  schema=schema)

        return context


class QueryStepDiscard(QueryStep):
    template_inline = textwrap.dedent("""\
        {query}
        MINUS
        {{
        {query_minus}
        }}
        """)
    def __init__(self, creator):
        super().__init__(creator)

    def build_step_op(self, qdmr_index, inline_query, context):
        grounding = self.creator.grounding
        schema = self.creator.schema
        get_var_name = self.creator.namer.get_var_name

        op = "discard"
        args = self.extract_args(qdmr_index, op, num_args=2)

        arg_full = QdmrInstance.ref_to_index(args[0], qdmr_index)
        arg_minus = QdmrInstance.ref_to_index(args[1], qdmr_index)

        context = \
            self.creator.construct_set_of_args([arg_full], inline_query=True, context=context)
        context_minus = \
            self.creator.construct_set_of_args([arg_minus], inline_query=True, context=None)

        output_units = context.output_units_for_qdmr_index[arg_full]
        output_units_minus = context_minus.output_units_for_qdmr_index[arg_minus]
        # swap vars in the minus context
        for u, u_minus in zip(output_units, output_units_minus):
            context_minus.swap_var(u_minus.var, u.var)

        query = context.query
        query_minus = context_minus.query
        query_minus = textwrap.indent(query_minus, self.indent_block)
        query = self.template_inline.format(query=query, query_minus=query_minus)

        context.query = query
        context.output_units_for_qdmr_index[qdmr_index] = output_units

        if not inline_query:
            context.query = self.build_full_query_from_inline(query=query, output_vars=[u.var for u in output_units], context=context)
            for u in output_units:
                if u.grnd is not None:
                    context.add_and_fix_grounding(target_grounding=u.grnd,
                                                output_var=u.var,
                                                schema=schema)

        return context


class QueryStepComparative(QueryStep):
    def __init__(self, creator):
        super().__init__(creator)

    @staticmethod
    def parse_comparator_value(grounding_comparative, good_values=[">", "<", ">=", "<=", "=", "!=", "like"]):
        assert grounding_comparative.iscomp(), f"Comparator should be grounded to a key of type 'comparative' but have {grounding_comparative.type}"
        assert len(grounding_comparative.keys) in [2, 3], f"Key of comparator should be of len 2 or 3 but have {grounding_comparative.keys}"

        comparator = grounding_comparative.keys[0]
        comparison_value = grounding_comparative.keys[1]
        assert comparator in good_values, f"Third arg of comparator should be grounded to {good_values}, but have {comparator}"

        if comparison_value is not None:
            comparison_value = comparison_value.replace("'", "")

        if comparator == "like":
            comparison_value = comparison_value.replace("%", "")

        if len(grounding_comparative.keys) == 3:
            grounding_col = grounding_comparative.keys[2]
        else:
            grounding_col = None

        return comparator, comparison_value, grounding_col

    @staticmethod
    def make_filter_line(filter_var, comparator, comparison_value):
        if comparator != "like":
            return f"FILTER({filter_var} {comparator} {comparison_value})."
        else:
            comparison_value = str(comparison_value).replace("\"", "").replace("^^xsd:string", "")
            return f"FILTER(REGEX(STR({filter_var}), \"(.*{comparison_value}.*)\", \"i\"))."

    def build_step_op(self, qdmr_index, inline_query, context):
        grounding = self.creator.grounding
        schema = self.creator.schema
        get_var_name = self.creator.namer.get_var_name

        op = "comparative"
        args = self.extract_args(qdmr_index, op, num_args=3)

        source_index = QdmrInstance.ref_to_index(args[0], qdmr_index)
        filter_index = QdmrInstance.ref_to_index(args[1], qdmr_index)

        grounding_comparative = GroundingIndex(qdmr_index, 2, args[2])
        if grounding_comparative in grounding and grounding[grounding_comparative]:
            assert grounding_comparative in grounding, f"Comparator {grounding_comparative} should be grounded but have {grounding}"
            target_grounding = grounding[grounding_comparative]
        else:
            grounding_comparative = None
            target_grounding = None

        list_of_indices = [source_index, filter_index]
        try:
            comparator, comparison_value, comparator_col = self.parse_comparator_value(grounding[grounding_comparative])
            comparison_index = QdmrInstance.ref_to_index(comparison_value, qdmr_index)
            list_of_indices.append(comparison_index)
        except:
            comparison_index = None
            comparator_cols = [None]
            comparison_values = [None]

        context_old = context
        context = copy.deepcopy(context_old)
        context = \
            self.creator.construct_set_of_args(list_of_indices, inline_query=True, context=context)

        filter_units = context.output_units_for_qdmr_index[filter_index]

        if target_grounding is not None:
            if target_grounding.iscol():
                comparator_cols = [target_grounding]
                comparison_values = [None]
            elif target_grounding.istbl():
                tbl_name = target_grounding.get_tbl_name()
                comparator_cols = [GroundingKey.make_column_grounding(tbl_name, schema.schema.primary_keys[tbl_name])]
                comparison_values = [None]
            elif target_grounding.isval():
                comparator_cols = [GroundingKey.make_column_grounding(target_grounding.get_tbl_name(), target_grounding.get_col_name())]
                comparator = "="
                comparison_values = [schema.get_key_for_comparison(target_grounding)]
            elif target_grounding.iscomp():
                comparator, comparison_value, comparator_col = self.parse_comparator_value(target_grounding)
                if comparator_col is None:
                    comparator_cols = [u.grnd for u in filter_units]
                else:
                    comparator_cols = [comparator_col]
                if comparison_index is not None:
                    # compare against another variable
                    comparison_values = [u.var for u in context.output_units_for_qdmr_index[comparison_index]]
                else:
                    comparison_values = [comparison_value]
                    # deal with the case when comparator_col contains keys
                    assert len(comparator_cols) == len(comparison_values)
                    comparator_cols_new = []
                    comparison_values_new = []
                    for comparator_col, comparison_value in zip(comparator_cols, comparison_values):
                        if isinstance(comparator_col, GroundingKey) and comparator_col.isref():
                            index_of_grounding = QdmrInstance.ref_to_index(comparator_col.keys[0], qdmr_index)
                            comparator_units = context.output_units_for_qdmr_index[index_of_grounding]
                            for u in comparator_units:
                                val = GroundingKey.make_value_grounding(u.output_col.grounding_column.get_tbl_name(),
                                                                        u.output_col.grounding_column.get_col_name(),
                                                                        comparison_value)
                                val = schema.get_key_for_comparison(val)
                                comparator_cols_new.append(u)
                                comparison_values_new.append(val)
                        else:
                            if isinstance(comparator_col, GroundingKey):
                                comparison_value = GroundingKey.make_value_grounding(comparator_col.get_tbl_name(),
                                                                                     comparator_col.get_col_name(),
                                                                                     comparison_value)
                                comparison_value = schema.get_key_for_comparison(comparison_value)
                            else:
                                # leave comparison value as is
                                pass

                            comparison_values_new.append(comparison_value)
                            comparator_cols_new.append(comparator_col)

                    comparator_cols = comparator_cols_new
                    comparison_values = comparison_values_new
            else:
                raise RuntimeError(f"Do not know what to do with grounding {target_grounding}")
        else:
            target_grounding_col = None

        # Comparison can be between variables in one context (and when the r.h.s. is a constant)
        # or between variables from different contexts (e.g., the r.h.s. is the output of an aggregator)
        # Distinguishing between the two is somewhat tricky and heuristical for now
        if comparison_index is None or\
            any((a.grnd != b.grnd) and (b.grnd is not None) for a,b in zip(context.output_units_for_qdmr_index[filter_index], context.output_units_for_qdmr_index[comparison_index])):

            assert len(comparator_cols) == len(comparison_values), f"This mode only supports comparator_cols and comparison_values of same len but have {comparator_cols} and {comparison_values}"

            query_lines = []
            filter_units = context.output_units_for_qdmr_index[filter_index]
            for i_comp, (comparator_col, comparison_value) in enumerate(zip(comparator_cols, comparison_values)):

                # comparing arguments from different groundings
                if comparator_col is not None and isinstance(comparator_col, GroundingKey):
                    context.add_column(comparator_col, schema, get_var_name, [u.grnd for u in filter_units])
                    filter_var = context.var_for_grounding[comparator_col]
                elif isinstance(comparator_col, ContextOutputUnit):
                    filter_var = comparator_col.var
                    comparator_col = comparator_col.grnd
                else:
                    filter_var = filter_units[i_comp].var

                if comparison_value is not None:
                    query_lines.append(self.make_filter_line(filter_var, comparator, comparison_value))
                else:
                    query_lines.append("")
        else:
            # create filter and comparion args in separate contexts
            context = copy.deepcopy(context_old)
            context = \
                self.creator.construct_set_of_args([source_index, filter_index], inline_query=True, context=context)
            context_comp = \
                self.creator.construct_set_of_args([comparison_index], inline_query=True, context=None)

            filter_units = context.output_units_for_qdmr_index[filter_index]
            comp_units = context_comp.output_units_for_qdmr_index[comparison_index]

            context.append_query(context_comp.query)

            query_lines = []
            for filter_unit, comp_unit in zip(filter_units, comp_units):
                query_line = self.make_filter_line(filter_unit.var, comparator, comp_unit.var)
                query_lines.append(query_line)

        for query_line in query_lines:
            context.append_query(query_line)

        context.output_units_for_qdmr_index[qdmr_index] = context.output_units_for_qdmr_index[source_index]

        for unit in context.output_units_for_qdmr_index[qdmr_index]:
            if qdmr_index in grounding["distinct"]:
                context.var_with_distinct[unit.var] = True

        context = QueryStepComparative.wrap_into_subquery(self, qdmr_index, context,
                                                            schema, get_var_name, inline_query,
                                                            context.output_units_for_qdmr_index[qdmr_index], comparator_cols)
        return context

    @staticmethod
    def wrap_into_subquery(self, qdmr_index, context,
                           schema, get_var_name,
                           inline_query,
                           source_output_units, comparator_cols):
        # two separate cases: filter on the same column or on a different column
        # if having a different column we need to wrap into a subquery otherwise we end up with Cartesian products
        # (the path as a function is not injective)

        if comparator_cols is not None and tuple(u.grnd for u in source_output_units) == tuple(comparator_cols):
            if not inline_query:
                context.query = self.build_full_query_from_inline(query=context.query,
                    output_vars=[u.var for u in source_output_units],
                    context=context)
                for u in source_output_units:
                    if u.grnd is not None:
                        context.add_and_fix_grounding(target_grounding=u.grnd,
                                                    output_var=u.var,
                                                    schema=schema)
            return context
        else:
            # wrap everything in a group by
            flag_not_adding_primary_key = False
            flag_output_var_with_distinct = False
            for unit in reversed(source_output_units):
                tbl_name = unit.grnd.get_tbl_name()
                try:
                    col_name = unit.grnd.get_col_name()
                except:
                    col_name = schema.schema.primary_keys[tbl_name]
                cur_output_var = context.var_for_grounding[unit.grnd]

                cur_output_var_with_distinct = cur_output_var in context.var_with_distinct
                flag_output_var_with_distinct = flag_output_var_with_distinct or cur_output_var_with_distinct
                flag_not_adding_primary_key = flag_not_adding_primary_key or col_name in schema.schema.column_key_in_table[tbl_name] or cur_output_var_with_distinct

            output_units = context.output_units_for_qdmr_index[qdmr_index]
            output_col_for_var_name = context.output_col_for_var_name

            flag_output_extra_key = False
            if flag_not_adding_primary_key:
                query = context.query
                query = textwrap.indent(query, self.indent_block)
                query = QueryStep.template_full_with_distinct.format(output_var=" ".join([u.var for u in output_units]), query=query)
            else:
                # make sure the primary key is added
                tbl_names = [u.grnd.get_tbl_name() for u in output_units if u.grnd is not None]
                # getting the key of some table
                tbl_name = tbl_names[0]
                key_grnd = GroundingKey.make_column_grounding(tbl_name, schema.schema.primary_keys[tbl_name])
                context.add_column(key_grnd, schema, get_var_name, [u.grnd for u in output_units if u.grnd is not None])
                group_var = context.var_for_grounding[key_grnd]
                query = context.query
                query = textwrap.indent(query, self.indent_block)

                if inline_query:
                    to_output = " ".join([u.var for u in output_units] + [group_var])
                    flag_output_extra_key = True
                    output_col_for_var_name[group_var] = OutputColumnId.from_grounding(key_grnd, schema.schema)
                else:
                    to_output = f"DISTINCT {' '.join([u.var for u in output_units])}" if flag_output_var_with_distinct else ' '.join([u.var for u in output_units])

                query = QueryStep.template_full.format(output_var=to_output, query=query)

            if inline_query:
                query = self.build_inline_query_from_full(query)

            context_new = LocalContext()
            context_new.query = query
            context_new.output_col_for_var_name = output_col_for_var_name
            context_new.output_units_for_qdmr_index[qdmr_index] = output_units

            for unit in output_units:
                if unit.var in context.var_with_distinct:
                    context_new.var_with_distinct[unit.var] = context.var_with_distinct[unit.var]
                if unit.grnd is not None:
                    context_new.add_and_fix_grounding(target_grounding=unit.grnd,
                                                        output_var=unit.var,
                                                        schema=schema)

            if flag_output_extra_key:
                context_new.add_and_fix_grounding(target_grounding=GroundingKey.make_table_grounding(tbl_name),
                                                  output_var=group_var,
                                                  schema=schema)

            return context_new


class QueryStepSuperlative(QueryStep):
    template_inline = textwrap.dedent("""\
        {{
          SELECT ({minmax_op}({query_inner_filter_var}) AS {minmax_var})
          WHERE
          {{
        {query_inner}
          }}
        }}
        {query_outer}
        FILTER({query_outer_filter_var} = {minmax_var}).""")

    def __init__(self, creator):
        super().__init__(creator)

    def build_step_op(self, qdmr_index, inline_query, context):
        grounding = self.creator.grounding
        schema = self.creator.schema
        get_var_name = self.creator.namer.get_var_name

        op = "superlative"
        args = self.extract_args(qdmr_index, op, num_args=3)

        source_index = QdmrInstance.ref_to_index(args[1], qdmr_index)
        filter_index = QdmrInstance.ref_to_index(args[2], qdmr_index)

        minmax_op = GroundingIndex(qdmr_index, 0, args[0])
        assert minmax_op in grounding, f"Min/max value {minmax_op} should be grounded but have {grounding}"

        minmax_op, comparison_value, comparator_col = QueryStepComparative.parse_comparator_value(grounding[minmax_op], good_values=["min", "max"])

        # get the context to compute min/max
        context_inner = \
            self.creator.construct_set_of_args([filter_index], inline_query=True, context=None)

        context_inner_output_units = context_inner.output_units_for_qdmr_index[filter_index]
        if comparator_col is not None:
            context_inner.add_column(comparator_col, schema, get_var_name, [u.grnd for u in context_inner_output_units if u.grnd is not None])
            query_inner_filter_vars = [context_inner.var_for_grounding[comparator_col]]
        else:
            query_inner_filter_vars = [u.var for u in context_inner_output_units]

        # get the full context
        context = \
            self.creator.construct_set_of_args([source_index, filter_index], inline_query=True, context=context)

        context_output_units = context.output_units_for_qdmr_index[filter_index]
        if comparator_col is not None:
            context.add_column(comparator_col, schema, get_var_name, [u.grnd for u in context_output_units if u.grnd is not None])
            query_outer_filter_vars = [context.var_for_grounding[comparator_col]]
            output_groundings = [comparator_col]
        else:
            query_outer_filter_vars = [u.var for u in context_output_units]
            output_groundings = [u.grnd for u in context_output_units]

        if self.creator.strict_mode:
            for grnd in output_groundings:
                if isinstance(grnd, GroundingKey):
                    if grnd.istbl() or\
                       grnd.iscol() and schema.schema.primary_keys[grnd.get_tbl_name()] == grnd.get_col_name():
                       # do no take superlative w.r.t. primary keys - it results in bad groundings
                       raise RuntimeError(f"Trying to compute SUPERLATIVE '{minmax_op}' w.r.t. the primary key of table '{grnd.get_tbl_name()}', aborting as this is likely wrong")

        minmax_var = get_var_name(minmax_op)
        query = \
            self.template_inline.format(minmax_op=minmax_op,
                                        query_inner=textwrap.indent(context_inner.query, self.indent_block + self.indent_block),
                                        query_inner_filter_var=" ".join(query_inner_filter_vars),
                                        query_outer=context.query,
                                        query_outer_filter_var=" ".join(query_outer_filter_vars),
                                        minmax_var=minmax_var)

        context.query = query
        context.output_units_for_qdmr_index[qdmr_index] = context.output_units_for_qdmr_index[source_index]

        for u in context.output_units_for_qdmr_index[qdmr_index]:
            if u.grnd is not None:
                context.output_col_for_var_name[u.var] = OutputColumnId.from_grounding(u.grnd, schema.schema)

        context = QueryStepComparative.wrap_into_subquery(self, qdmr_index, context,
                                                          schema, get_var_name, inline_query,
                                                          context.output_units_for_qdmr_index[qdmr_index], [comparator_col])

        return context


class QueryStepSort(QueryStep):
    template_full = textwrap.dedent("""\
        SELECT {output_vars}
        WHERE
        {{
        {query}
        }}
        ORDER BY {sort_var_str}""")

    def __init__(self, creator):
        super().__init__(creator)

    def build_step_op(self, qdmr_index, inline_query, context):
        grounding = self.creator.grounding
        schema = self.creator.schema
        get_var_name = self.creator.namer.get_var_name

        op = "sort"
        args = self.extract_args(qdmr_index, op, num_args=None)
        num_args = len(args)
        assert num_args == 2 or num_args == 3, f"{op} should take 2 or 3 args, but has {num_args}"

        sort_data = args[0]
        sort_arg = args[1]
        if num_args >= 3:
            sort_order = GroundingIndex(qdmr_index, 2, args[2])
        else:
            sort_order = None

        data_argument_index = QdmrInstance.ref_to_index(sort_data, qdmr_index)
        sort_arg_index = QdmrInstance.ref_to_index(sort_arg, qdmr_index)

        if sort_order is not None and sort_order in grounding:
            assert grounding[sort_order].issortdir(), f"Sort order for {op} {sort_order} should be grounded to sortdir but have {grounding}"
            sort_order_name = grounding[sort_order].keys[0]
            sort_direction = "ASC" if sort_order_name.lower() == "ascending" else "DESC"
        else:
            sort_direction = "ASC"

        context_args = \
            self.creator.construct_set_of_args([data_argument_index, sort_arg_index], inline_query=True, context=None)

        output_units = context_args.output_units_for_qdmr_index[data_argument_index]
        output_vars = [u.var for u in output_units]

        # add sorting args so that one could resolve ambigous sort later
        context.sorting_info = {"sorted" : True, "sort_direction" : sort_direction}
        context.sorting_info["sorting_cols"] = []
        sort_vars = [u.var for u in context_args.output_units_for_qdmr_index[sort_arg_index]]

        output_vars_for_output_str = copy.deepcopy(output_vars)
        for u, sort_var in zip(context_args.output_units_for_qdmr_index[sort_arg_index], sort_vars):
            if sort_var not in output_vars_for_output_str:
                # adding the sort var to the output string (to be remove at the postprocessing)
                idx = len(output_vars_for_output_str)
                output_vars_for_output_str.append(sort_var)
            else:
                idx = output_vars_for_output_str.index(sort_var)
            context.sorting_info["sorting_cols"].append({"col": u.output_col, "idx": idx})
        
        # output variables for the query
        output_vars_str = " ".join(output_vars_for_output_str)

        sort_var_str = " ".join(f"{sort_direction}({v})" for v in sort_vars)
        query = self.template_full.format(output_vars=output_vars_str,
                                          query=textwrap.indent(context_args.query, self.indent_block),
                                          sort_var_str=sort_var_str,
                                         )
        if inline_query:
            query = self.build_inline_query_from_full(query)

        context.query = query
        context.output_units_for_qdmr_index[qdmr_index] = output_units
        context.output_col_for_var_name = context_args.output_col_for_var_name

        for u in output_units:
            if u.grnd is not None:
                context.add_and_fix_grounding(target_grounding=u.grnd,
                                                output_var=u.var,
                                                schema=schema)

        return context


class QueryStepArithmetic(QueryStep):
    def __init__(self, creator):
        super().__init__(creator)

    @staticmethod
    def parse_arithmetic_op(operator):
        operator = operator.replace("'", "")
        operator_good_values = {"sum" : "+", "difference" : "-", "division" : "/", "multiplication" : "*"}
        assert operator.lower() in operator_good_values, f"First arg of arithmetic op should be in {operator_good_values}, but have {operator}"
        return operator_good_values[operator.lower()]

    def build_step_op(self, qdmr_index, inline_query, context):

        raise NotImplementedError("ARITHMETIC op is not supported!")

        grounding = self.creator.grounding
        schema = self.creator.schema
        get_var_name = self.creator.namer.get_var_name

        op = "arithmetic"
        args = self.extract_args(qdmr_index, op, num_args=3)

        arg_index_0 = QdmrInstance.ref_to_index(args[1], qdmr_index)
        arg_index_1 = QdmrInstance.ref_to_index(args[2], qdmr_index)

        arithmetic_op_name = args[0].replace("'", "")
        arithmetic_op = self.parse_arithmetic_op(arithmetic_op_name)

        op_vars = {}
        for arg_index in [arg_index_0, arg_index_1]:
            context_arg = \
                self.creator.construct_set_of_args([arg_index], inline_query=False, context=None)
            query_step = self.template_to_inline.format(query=textwrap.indent(context_arg.query, self.indent_block))
            context.append_query(query_step)

            op_vars[arg_index] = context_arg.var_name_for_qdmr_index[arg_index]
            context.var_name_for_qdmr_index[arg_index] = op_vars[arg_index]
            context.grounding_for_qdmr_index[arg_index] = None

        output_var = get_var_name(arithmetic_op_name)
        op_query_line = f"({op_vars[arg_index_0]} {arithmetic_op} {op_vars[arg_index_1]} AS {output_var})"

        if inline_query:
            query_line = f"BIND{op_query_line}."
            context.append_query(query_line)
        else:
            context.query = self.build_full_query_from_inline(context.query, op_query_line)
            context.var_name_for_qdmr_index = {}
            context.grounding_for_qdmr_index = {}

        context.var_name_for_qdmr_index[qdmr_index] = output_var
        context.grounding_for_qdmr_index[qdmr_index] = None

        return context


class QueryCreator():
    def __init__(self, strict_mode=False):
        """Flag strict_mode should be set to True when collecting groundings to avoid some strange cases
        and to False on generation to get some query if possible
        """
        self.namer = VariableNamer()
        self.creator_for_op_name = {}
        self.creator_for_op_name["aggregate"] = QueryStepAggregate
        self.creator_for_op_name["select"] = QueryStepSelect
        self.creator_for_op_name["project"] = QueryStepProject
        self.creator_for_op_name["union"] = QueryStepUnion
        self.creator_for_op_name["intersection"] = QueryStepIntersection
        self.creator_for_op_name["sort"] = QueryStepSort
        self.creator_for_op_name["comparative"] = QueryStepComparative
        self.creator_for_op_name["superlative"] = QueryStepSuperlative
        self.creator_for_op_name["group"] = QueryStepGroup
        self.creator_for_op_name["discard"] = QueryStepDiscard
        self.creator_for_op_name["arithmetic"] = QueryStepArithmetic

        self.strict_mode = strict_mode

    def build_sparql_query(self, qdmr, schema, rdf_graph, grounding):
        # make copies of data
        qdmr = copy.deepcopy(qdmr)
        grounding = copy.deepcopy(grounding)
        schema = copy.deepcopy(schema)

        # clean up the qdmr ops
        self.cleanup_qdmr(qdmr, grounding)

        # check the distinct key word
        self.parse_distinct_grounding(grounding, len(qdmr))

        # check that have only supported ops
        for op, args in qdmr:
            assert op in self.creator_for_op_name, f"Could not find a method for the QDMR op {op}"

        # create static groundings for the steps
        self.grounding_for_qdmr_index = self.build_grounding_qdmr_steps_before_execution(qdmr, grounding)

        # run the very last op
        self.qdmr = qdmr
        self.schema = SchemaWithRdfGraph(schema, rdf_graph)
        self.grounding = grounding

        qdmr_index = len(qdmr) - 1
        context = self.construct_set_of_args([qdmr_index], inline_query=False)
        query = QueryToRdf(query=context.query,
                           output_cols=[u.output_col for u in context.output_units_for_qdmr_index[qdmr_index]],
                           sorting_info=context.sorting_info)
        return query

    def construct_set_of_args(self, qdmr_target_indices, inline_query, context=None):
        if context is None:
            context = LocalContext()

        # CAUTION! ops like filter and comparative can eliminate already computed QDMR entries
        # to mitigate that will have to 1) eliminate only the dependency tree of this vars
        # 2) perform multiple rounds of graph construction to reconstruct what was eleminated
        # expect to see the problem mostly with DISCARD op

        # Group op has a side effect - always add it first
        op_list = [self.qdmr.ops[i] for i in qdmr_target_indices]
        if "group" in op_list:
            non_group_pos = 0
            qdmr_target_indices = copy.deepcopy(qdmr_target_indices)
            # make a loop in case there are multiple entries of the "group" op
            while "group" in op_list[non_group_pos:]:
                i_group = op_list.index("group", non_group_pos)

                # move found "group" to the first position
                op_list = op_list[:non_group_pos] + [op_list[i_group]] + op_list[non_group_pos:i_group] + op_list[i_group+1:]
                qdmr_target_indices = qdmr_target_indices[:non_group_pos] + [qdmr_target_indices[i_group]] + qdmr_target_indices[non_group_pos:i_group] + qdmr_target_indices[i_group+1:]
                non_group_pos += 1

        # add ops one by one
        num_loops = 0
        max_num_loops = 10
        while not all(index in context.output_units_for_qdmr_index for index in qdmr_target_indices) and num_loops < max_num_loops:
            for index in qdmr_target_indices:
                if index not in context.output_units_for_qdmr_index:
                    op = self.creator_for_op_name[self.qdmr.ops[index]](self)
                    context = op.build_step(index, inline_query=inline_query, context=context)
            num_loops += 1
        assert num_loops < max_num_loops, f"Cound not compute target indices {qdmr_target_indices}, got this: {context.output_units_for_qdmr_index}"

        return context

    @classmethod
    def build_grounding_qdmr_steps_before_execution(cls, qdmr, grounding):
        grounding_for_qdmr_index = {}
        for qdmr_index, (op, args) in enumerate(qdmr):
            op = op.lower()
            if op in ["select"]:
                i_arg = 0
                grnd = [grounding[GroundingIndex(qdmr_index, i_arg, args[i_arg])]]
            elif op in ["project"]:
                i_arg = 0
                index = GroundingIndex(qdmr_index, i_arg, args[i_arg])
                if index in grounding:
                    grnd = [grounding[index]]
                else:
                    i_arg = 1
                    source_index = QdmrInstance.ref_to_index(args[i_arg], qdmr_index)
                    grnd = grounding_for_qdmr_index[source_index]
            elif op in ["comparative", "superlative", "intersection", "discard", "sort"]:
                i_arg = 0 if op not in ["superlative"] else 1
                source_index = QdmrInstance.ref_to_index(args[i_arg], qdmr_index)
                grnd = grounding_for_qdmr_index[source_index]
            elif op in ["aggregate", "group"]:
                i_arg = 1
                source_index = QdmrInstance.ref_to_index(args[i_arg], qdmr_index)
                grnd = [None for s in grounding_for_qdmr_index[source_index]]
            elif op in ["union"]:
                source_indices = [QdmrInstance.ref_to_index(arg, qdmr_index) for arg in args]
                source_grnds = [tuple(grounding_for_qdmr_index[i_]) for i_ in source_indices]

                # check if we have a union of aggregators
                aggregate_in_args = [qdmr.ops[i] == "aggregate" for i in source_indices]
                if all(aggregate_in_args):
                    # do union of aggregators
                    grnd = sum([grounding_for_qdmr_index[i_] for i_ in source_indices], [])
                elif any(qdmr.ops[i] == "group" for i in source_indices):
                    # check if we have UNION after GROUP
                    grnd = sum([grounding_for_qdmr_index[i_] for i_ in source_indices], [])
                else:
                    # determine vertical or horizontal union
                    is_vertical_union = all(source_grnds[0] == g for g in source_grnds[1:])
                    if is_vertical_union:
                        grnd = grounding_for_qdmr_index[source_indices[0]]
                    else:
                        grnd = sum([grounding_for_qdmr_index[i_] for i_ in source_indices], [])
            elif op in ["arithmetic"]:
                grnd = [None]
            else:
                raise RuntimeError(f"Have not implemented static grounding for op {op} in {qdmr}, {grounding}")

            grounding_for_qdmr_index[qdmr_index] = grnd

        return grounding_for_qdmr_index

    @classmethod
    def parse_distinct_grounding(cls, out_grounding, num_ops=None):
        if "distinct" in out_grounding:
            out_grounding["distinct"] = [QdmrInstance.ref_to_index(ref, max_index=num_ops) for ref in out_grounding["distinct"]]
        else:
            out_grounding["distinct"] = []

    @classmethod
    def cleanup_qdmr(cls, out_qdmr, out_grounding):
        for i_op in range(len(out_qdmr)):
            # add cleanup rules one by one
            cls.qdmr_cleanup_change_comparative_to_superlative(i_op, out_qdmr, out_grounding)
            cls.qdmr_cleanup_grounded_aggregator(i_op, out_qdmr, out_grounding)
            cls.qdmr_cleanup_change_filter_to_comparative(i_op, out_qdmr, out_grounding)

    @classmethod
    def qdmr_cleanup_change_comparative_to_superlative(cls, i_op, out_qdmr, out_grounding):
        op, args = out_qdmr[i_op]
        if op.lower() == "comparative":
            grnd_key_src = GroundingIndex(i_op, 2, args[2])
            grnd = out_grounding.get(grnd_key_src)
            if grnd and grnd.iscomp() and grnd.keys[0].lower() in ["min", "max"]:
                # need to change the op to superlative
                out_qdmr.ops[i_op] = "superlative"
                assert len(args) == 3, f"COMPARATIVE should have 3 args but have {len(args)}: {args}"
                out_qdmr.args[i_op] = [args[2], args[0], args[1]]
                # move grounding to a different index
                grnd_key_target = GroundingIndex(i_op, 0, out_qdmr.args[i_op][0])
                out_grounding[grnd_key_target] = out_grounding.pop(grnd_key_src)
        elif op.lower() == "filter":
            grnd_key_src = GroundingIndex(i_op, 1, args[1])
            grnd = out_grounding.get(grnd_key_src)
            if grnd and grnd.iscomp() and grnd.keys[0].lower() in ["min", "max"]:
                # need to change the op to superlative
                out_qdmr.ops[i_op] = "superlative"
                assert len(args) == 2, f"FILTER should have 2 args but have {len(args)}: {args}"
                out_qdmr.args[i_op] = [args[1], args[0], args[0]]
                # move grounding to a different index
                grnd_key_target = GroundingIndex(i_op, 0, out_qdmr.args[i_op][0])
                out_grounding[grnd_key_target] = out_grounding.pop(grnd_key_src)

    @classmethod
    def qdmr_cleanup_grounded_aggregator(cls, i_op, out_qdmr, out_grounding):
        op, args = out_qdmr[i_op]
        if op.lower() in ["group", "aggregate"]:
            grnd_index = GroundingIndex(i_op, 0, args[0])
            if grnd_index in out_grounding and out_grounding[grnd_index]:
                # have the op of QDMR grounded - change the op to project then
                out_qdmr.ops[i_op] = "project"
                if op.lower() == "group":
                    assert len(args) == 3, f"GROUP should have 3 args but have {len(args)}: {args}"
                    i_arg_target = 2
                elif  op.lower() == "aggregate":
                    assert len(args) == 2, f"AGGREGATE should have 2 args but have {len(args)}: {args}"
                    i_arg_target = 1
                else:
                    raise RuntimeError(f"Unknown op '{op}'")
                out_qdmr.args[i_op] = [str(grnd_index), args[i_arg_target]]

                # fix grounding
                out_grounding[GroundingIndex(i_op, 0, out_qdmr.args[i_op][0])] = out_grounding[grnd_index]
                del out_grounding[grnd_index]

    @classmethod
    def qdmr_cleanup_change_filter_to_comparative(cls, i_op, out_qdmr, out_grounding):
        op, args = out_qdmr[i_op]
        if op.lower() == "filter":
            grnd_key_src = GroundingIndex(i_op, 1, args[1])

            # need to change the op to comparative
            out_qdmr.ops[i_op] = "comparative"
            assert len(args) == 2, f"FILTER should have 2 args but have {len(args)}: {args}"
            out_qdmr.args[i_op] = [args[0], args[0], args[1]]
            # move grounding to a different index
            grnd_key_target = GroundingIndex(i_op, 2, out_qdmr.args[i_op][2])
            if grnd_key_src in out_grounding:
                out_grounding[grnd_key_target] = out_grounding.pop(grnd_key_src)


def create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding, strict_mode=False):
    creator = QueryCreator(strict_mode=strict_mode)
    query = creator.build_sparql_query(qdmr, schema, rdf_graph, grounding)
    return query
