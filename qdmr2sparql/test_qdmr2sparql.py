import os
import unittest
from timeout_decorator import timeout

import textwrap
from functools import lru_cache

from qdmr2sparql.datasets import QdmrInstance, DatasetBreak, DatasetSpider
from qdmr2sparql.structures import GroundingIndex, GroundingKey, RdfGraph
from qdmr2sparql.structures import QueryResult, QueryToRdf, OutputColumnId

from qdmr2sparql.query_generator import create_sparql_query_from_qdmr


ONE_TEST_TIMEOUT = 120
VIRTUOSO_SPARQL_SERVICE = None


class TestSelect(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_table(self):
        """When selecting full table we return the set or primary keys
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?singer
            WHERE
            {
            ?singer arc:singer:Singer_ID ?singer.
            }""")

        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
                                          output_cols=[OutputColumnId.from_grounding(GroundingKey.make_table_grounding("singer"), schema)])

        qdmr = QdmrInstance(["select"], [["singers"]])
        grounding = {GroundingIndex(0,0,"singers") : GroundingKey.make_table_grounding("singer")}

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


    @timeout(ONE_TEST_TIMEOUT)
    def test_select_column(self):
        """When selecting the column we return the items of that column
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Name
            WHERE
            {
            ?singer arc:singer:Name ?Name.
            }""")

        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
                                          output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Name"))])

        qdmr = QdmrInstance(["select"], [["name"]])
        grounding = {GroundingIndex(0,0,"name") : GroundingKey.make_column_grounding("singer", "Name")}

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_value(self):
        """When selecting the value we return all etries of that value in that column
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?countries
            WHERE
            {
              ?singer arc:singer:Country ?countries.
              FILTER(?countries = "France"^^xsd:string).
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
                              output_cols=[OutputColumnId.from_grounding(GroundingKey.make_value_grounding("singer", "Country", "France"))])

        qdmr = QdmrInstance(["select"], [["France"]])
        grounding = {GroundingIndex(0,0,"France") : GroundingKey.make_value_grounding("singer", "Country", "France")}

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSelectProject(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_table_project_column(self):
        """Select table, project column should return the column
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?countries
            WHERE
            {
              ?singer arc:singer:Country ?countries.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
                                output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Country"))])

        qdmr = QdmrInstance(["select", "project"], [["singers"], ["countries", "#1"]])
        grounding = { GroundingIndex(0,0,"singers") : GroundingKey.make_table_grounding("singer"),
                      GroundingIndex(1,0,"countries") : GroundingKey.make_column_grounding("singer", "Country")
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_column_project_table(self):
        """Select table, project column should return the column
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?singer
            WHERE
            {
              ?singer arc:singer:Country ?countries.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
                                output_cols=[OutputColumnId.from_grounding(GroundingKey.make_table_grounding("singer"), schema)])

        qdmr = QdmrInstance(["select", "project"], [["countries"], ["singers", "#1"]])
        grounding = { GroundingIndex(0,0,"countries") : GroundingKey.make_column_grounding("singer", "Country"),
                      GroundingIndex(1,0,"singers") : GroundingKey.make_table_grounding("singer")
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_column_project_value(self):
        """Select table, project column should return the column
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?countries
            WHERE
            {
              ?singer arc:singer:Name ?Name.
              ?singer arc:singer:Country ?countries.
              FILTER(?countries = "France"^^xsd:string)
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
                        output_cols=[OutputColumnId.from_grounding(GroundingKey.make_value_grounding("singer", "Country", "France"))])

        qdmr = QdmrInstance(["select", "project"], [["names"], ["France", "#1"]])
        grounding = { GroundingIndex(0,0,"names") : GroundingKey.make_column_grounding("singer", "Name"),
                      GroundingIndex(1,0,"France") : GroundingKey.make_value_grounding("singer", "Country", "France"),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_value_project_column(self):
        """Select table, project column should return the column
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Name
            WHERE
            {
              ?singer arc:singer:Name ?Name.
              ?singer arc:singer:Country ?countries.
              FILTER(?countries = "France"^^xsd:string)
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Name"))])

        qdmr = QdmrInstance(["select", "project"], [["France"], ["names", "#1"]])
        grounding = { GroundingIndex(0,0,"France") : GroundingKey.make_value_grounding("singer", "Country", "France"),
                      GroundingIndex(1,0,"names") : GroundingKey.make_column_grounding("singer", "Name")
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestDifferentColumnOrder(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_project_column_order(self):
        """
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Name ?countries
            WHERE
            {
              ?singer arc:singer:Name ?Name.
              ?singer arc:singer:Country ?countries.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
                        output_cols=[
                            OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Name")),
                            OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Country"))
                            ])

        qdmr = QdmrInstance(["select", "project", "union"], [["names"], ["Country", "#1"], ["#2", "#1"]])
        grounding = { GroundingIndex(0,0,"names") : GroundingKey.make_column_grounding("singer", "Name"),
                      GroundingIndex(1,0,"Country") : GroundingKey.make_column_grounding("singer", "Country"),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=False,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSelectFilter(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_column_filter_value(self):
        """Select table, filter values based on a value in another column
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Name
            WHERE
            {
              ?singer arc:singer:Name ?Name.
              ?singer arc:singer:Country ?countries.
              FILTER(?countries = "France"^^xsd:string)
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Name"))])

        qdmr = QdmrInstance(["select", "filter"], [["names"], ["#1", "France"]])
        grounding = { GroundingIndex(0,0,"names") : GroundingKey.make_column_grounding("singer", "Name"),
                      GroundingIndex(1,1,"France") : GroundingKey.make_value_grounding("singer", "Country", "France")
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_filter_with_comparative(self):
        """Select table, filter values based on a value in another column
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Name
            WHERE
            {
              ?singer arc:singer:Name ?Name.
              ?singer arc:singer:Age ?Age.
              FILTER(?Age > 32).
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Name"))])

        qdmr = QdmrInstance(["select", "filter"], [["names"], ["#1", "older than 32"]])
        grounding = {GroundingIndex(0,0,"names") : GroundingKey.make_column_grounding("singer", "Name"),
                     GroundingIndex(1,1,"older than 32"): GroundingKey.make_comparative_grounding(">", "32", GroundingKey.make_column_grounding("singer", "Age")),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_column_filter_with_value_in_another_column(self):
        """Select table, filter values based on a value in another column.
        The argument of comparative contains reference to a new column.
        """
        rdf_graph, schema = get_graph_and_schema("dev", "car_1")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?ID
            WHERE
            {

              ?ID arc:cars_data:Weight ?Weight.
              ?ID arc:cars_data:Year ?Year.
              FILTER(?Year > ?Weight).
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_table_grounding("cars_data"), schema)])

        qdmr = QdmrInstance(["select", "project", "filter"], [["cars"], ["weights", "#1"], ["#1", "years larger than #2"]])
        grounding = {
                        GroundingIndex(0,0,"cars") : GroundingKey.make_table_grounding("cars_data"),
                        GroundingIndex(1,0,"weights") : GroundingKey.make_column_grounding("cars_data", "Weight"),
                        GroundingIndex(2,1,"years larger than #2"): GroundingKey.make_comparative_grounding(">", "#2", GroundingKey.make_column_grounding("cars_data", "Year")),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_column_filter_superlative(self):
        """Select table, filter values based on a value in another column - with superlative
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Name_1
            WHERE
            {
            {
                SELECT (min(?Age) AS ?min)
                WHERE
                {
                    ?singer arc:singer:Age ?Age.
                }
            }
            ?singer_1 arc:singer:Age ?Age_1.
            ?singer_1 arc:singer:Name ?Name_1.
            FILTER(?Age_1 = ?min).
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Name"))])
        correct_sparql_query.query_has_superlative = True

        qdmr = QdmrInstance(["select", "filter"], [["names"], ["#1", "the youngest"]])
        grounding = { GroundingIndex(0,0,"names") : GroundingKey.make_column_grounding("singer", "Name"),
                      GroundingIndex(1,1,"the youngest") : GroundingKey.make_comparative_grounding("min", None, GroundingKey.make_column_grounding("singer", "Age")),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSelectProjectComparative(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_column_project_another_column_compare_value(self):
        """Select table, filter values based on a value in another column based on project-comparative
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Name
            WHERE
            {
              ?singer arc:singer:Name ?Name.
              ?singer arc:singer:Country ?countries.
              FILTER(?countries != "France"^^xsd:string)
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Name"))])

        qdmr = QdmrInstance(["select", "project", "comparative"], [["names"], ["countries", "#1"], ["#1", "#2", "not from France"]])
        grounding = { GroundingIndex(1,0,"countries") : GroundingKey.make_column_grounding("singer", "Country"),
                      GroundingIndex(0,0,"names") : GroundingKey.make_column_grounding("singer", "Name"),
                      GroundingIndex(2,2,"not from France"): GroundingKey.make_comparative_grounding("!=", "France"),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_column_project_compare_with_another_column(self):
        """Select table, filter values based on a value in another column based on project-comparative
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Country
            WHERE
            {
              ?singer arc:singer:Country ?Country.
              ?singer arc:singer:Name ?Name.
              ?singer arc:singer:Age ?Age.
              FILTER(?Age > 32).
            }
            GROUP BY ?Country""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Country"))])

        qdmr = QdmrInstance(["select", "project", "comparative"], [["countries"], ["names", "#1"], ["#1", "#2", "older than 32"]])
        grounding = { GroundingIndex(0,0,"countries") : GroundingKey.make_column_grounding("singer", "Country"),
                      GroundingIndex(1,0,"names") : GroundingKey.make_column_grounding("singer", "Name"),
                      GroundingIndex(2,2,"older than 32"): GroundingKey.make_comparative_grounding(">", "32", GroundingKey.make_column_grounding("singer", "Age")),
                     "distinct": ["#1"],
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_column_project_another_column_compare_value_in_the_third_column(self):
        """Select table, filter values based on a value in another column based on project-comparative.
        The argument of comparative contains QDMR reference.
        """
        rdf_graph, schema = get_graph_and_schema("dev", "car_1")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?ID
            WHERE
            {

              ?ID arc:cars_data:Weight ?Weight.
              ?ID arc:cars_data:Year ?Year.
              FILTER(?Year > ?Weight).
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_table_grounding("cars_data"), schema)])

        qdmr = QdmrInstance(["select", "project", "project", "comparative"],
                            [["cars"], ["years", "#1"], ["weights", "#1"], ["#1", "#2", "larger than #3"]])
        grounding = {
                        GroundingIndex(0,0,"cars") : GroundingKey.make_table_grounding("cars_data"),
                        GroundingIndex(1,0,"years") : GroundingKey.make_column_grounding("cars_data", "Year"),
                        GroundingIndex(2,0,"weights") : GroundingKey.make_column_grounding("cars_data", "Weight"),
                        GroundingIndex(3,2,"larger than #3"): GroundingKey.make_comparative_grounding(">", "#3"),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestDistinct(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_column_distinct(self):
        """When selecting the column we return the items of that column, adding the distinct flag
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT DISTINCT ?countries
            WHERE
            {
            ?singer arc:singer:Country ?countries.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Country"))])

        qdmr = QdmrInstance(["select"],
                            [["countries"]])
        grounding = {GroundingIndex(0,0,"countries") : GroundingKey.make_column_grounding("singer", "Country")}
        grounding["distinct"] = ["#1"]

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_distinct_count(self):
        """Select table, filter values based on a value in another column based on project-comparative
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT (count(DISTINCT ?countries) AS ?count)
            WHERE
            {
            ?singer arc:singer:Country ?countries.
            }""")
        output_col = OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Country"))
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.add_aggregator(output_col, "count")])

        qdmr = QdmrInstance(["select", "aggregate"],
                            [["countries"], ["count", '#1']])
        grounding = {GroundingIndex(0,0,"countries") : GroundingKey.make_column_grounding("singer", "Country")}
        grounding["distinct"] = ["#1"]

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_column_project_distinct(self):
        """When selecting the column we return the items of that column, adding the distinct flag of the project operator with empty grounding
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT DISTINCT ?countries
            WHERE
            {
            ?singer arc:singer:Country ?countries.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Country"))])

        qdmr = QdmrInstance(["select", "project"],
                            [["countries"], ["distinct of", "#1"]])
        grounding = {GroundingIndex(0,0,"countries") : GroundingKey.make_column_grounding("singer", "Country")}
        grounding["distinct"] = ["#2"]

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestEmptySubqueries(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_filter_empty_result(self):
        """Test projecting the empty output of a subquery
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Name
            WHERE
            {
              ?stadiums arc:stadium:Capacity ?Capacity.
              FILTER(?Capacity < 0)
              ?stadiums arc:stadium:Name ?Name.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("stadium", "Name"))])

        qdmr = QdmrInstance(["select", "filter", "project"],
                            [["stadiums"], ["#1", "cap < 0"], ["name", "#2"]])
        grounding = {GroundingIndex(0,0,"stadiums") : GroundingKey.make_table_grounding("stadium"),
                     GroundingIndex(1,1,"cap < 0") : GroundingKey.make_comparative_grounding("<", "0",
                                                             GroundingKey.make_column_grounding("stadium", "Capacity")),
                     GroundingIndex(2,0,"name") : GroundingKey.make_column_grounding("stadium", "Name")}

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestRefGrounding(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_ref_grounding_comparative(self):
        """Test adding ref grounding in the third arg of COMPARATIVE (2nd arg of FILTER)
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Name
            WHERE
            {
              ?stadiums arc:stadium:Capacity ?Capacity.
              FILTER(?Capacity < 5000)
              ?stadiums arc:stadium:Name ?Name.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("stadium", "Name"))])

        qdmr = QdmrInstance(["select", "project", "filter", "project"],
                            [["stadiums"], ["capacity", "#1"], ["#2", "cap < 5000"], ["name", "#3"]])
        grounding = {GroundingIndex(0,0,"stadiums") : GroundingKey.make_table_grounding("stadium"),
                     GroundingIndex(1,0,"capacity") : GroundingKey.make_column_grounding("stadium", "Capacity"),
                     GroundingIndex(2,1,"cap < 5000") : GroundingKey.make_comparative_grounding("<", "5000",
                                                            GroundingKey.make_reference_grounding("#2")),
                     GroundingIndex(3,0,"name") : GroundingKey.make_column_grounding("stadium", "Name")}

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_ref_grounding_group(self):
        """Test adding ref grounding in the third arg of COMPARATIVE (2nd arg of FILTER)
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Capacity
            WHERE
            {
              ?stadiums arc:stadium:Capacity ?Capacity.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("stadium", "Capacity"))])

        qdmr = QdmrInstance(["select", "project", "group"],
                            [["stadiums"], ["capacity", "#1"], ["sum", "#2", "#1"]])
        grounding = {GroundingIndex(0,0,"stadiums") : GroundingKey.make_table_grounding("stadium"),
                     GroundingIndex(1,0,"capacity") : GroundingKey.make_column_grounding("stadium", "Capacity"),
                     GroundingIndex(2,0,"sum") : GroundingKey.make_reference_grounding("#2"),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestNonInjectiveLink(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_project_forward(self):
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?singer_name
            WHERE
            {
              ?pair_id arc:singer_in_concert:Singer_ID ?singer_in_pair_id.
              ?singer_in_pair_id arc:singer_in_concert:Singer_ID:singer:Singer_ID ?s_id.
              ?s_id arc:singer:Name ?singer_name.
              ?pair_id arc:singer_in_concert:concert_ID ?concert_in_pair_id.
              ?concert_in_pair_id arc:singer_in_concert:concert_ID:concert:concert_ID ?c_id.
              ?c_id arc:concert:concert_Name ?concert_name.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Name"))])

        qdmr = QdmrInstance(["select", "project"], [["concert"], ["singer", "#1"]])
        grounding = { GroundingIndex(0,0,"concert") : GroundingKey.make_column_grounding("concert", "concert_Name"),
                      GroundingIndex(1,0,"singer") : GroundingKey.make_column_grounding("singer", "Name"),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_project_forward_distinct(self):
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT DISTINCT ?singer_name
            WHERE
            {
              ?pair_id arc:singer_in_concert:Singer_ID ?singer_in_pair_id.
              ?singer_in_pair_id arc:singer_in_concert:Singer_ID:singer:Singer_ID ?s_id.
              ?s_id arc:singer:Name ?singer_name.
              ?pair_id arc:singer_in_concert:concert_ID ?concert_in_pair_id.
              ?concert_in_pair_id arc:singer_in_concert:concert_ID:concert:concert_ID ?c_id.
              ?c_id arc:concert:concert_Name ?concert_name.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Name"))])

        qdmr = QdmrInstance(["select", "project"], [["concert"], ["singer", "#1"]])
        grounding = { GroundingIndex(0,0,"concert") : GroundingKey.make_column_grounding("concert", "concert_Name"),
                      GroundingIndex(1,0,"singer") : GroundingKey.make_column_grounding("singer", "Name"),
                      "distinct": ["#2"]
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_project_backward(self):
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?concert_name
            WHERE
            {
              ?pair_id arc:singer_in_concert:Singer_ID ?singer_in_pair_id.
              ?singer_in_pair_id arc:singer_in_concert:Singer_ID:singer:Singer_ID ?s_id.
              ?s_id arc:singer:Name ?singer_name.
              ?pair_id arc:singer_in_concert:concert_ID ?concert_in_pair_id.
              ?concert_in_pair_id arc:singer_in_concert:concert_ID:concert:concert_ID ?c_id.
              ?c_id arc:concert:concert_Name ?concert_name.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "concert_Name"))])

        qdmr = QdmrInstance(["select", "project"], [["singer"], ["concert", "#1"]])
        grounding = { GroundingIndex(0,0,"singer") : GroundingKey.make_column_grounding("singer", "Name"),
                      GroundingIndex(1,0,"concert") : GroundingKey.make_column_grounding("concert", "concert_Name"),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_select_with_intersect(self):
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?singer_name
            WHERE
            {
              ?s_id arc:singer:Name ?singer_name.
              {
                SELECT ?singer_name
                WHERE
                {
                  ?pair_id arc:singer_in_concert:Singer_ID ?singer_in_pair_id.
                  ?singer_in_pair_id arc:singer_in_concert:Singer_ID:singer:Singer_ID ?s_id.
                  ?s_id arc:singer:Name ?singer_name.
                  ?pair_id arc:singer_in_concert:concert_ID ?concert_in_pair_id.
                  ?concert_in_pair_id arc:singer_in_concert:concert_ID:concert:concert_ID ?c_id.
                  ?c_id arc:concert:concert_Name ?concert_name.
                  FILTER(?concert_name = "Super bootcamp"^^xsd:string)
                }
              }
              {
                SELECT ?singer_name
                WHERE
                {
                  ?pair_id arc:singer_in_concert:Singer_ID ?singer_in_pair_id.
                  ?singer_in_pair_id arc:singer_in_concert:Singer_ID:singer:Singer_ID ?s_id.
                  ?s_id arc:singer:Name ?singer_name.
                  ?pair_id arc:singer_in_concert:concert_ID ?concert_in_pair_id.
                  ?concert_in_pair_id arc:singer_in_concert:concert_ID:concert:concert_ID ?c_id.
                  ?c_id arc:concert:concert_Name ?concert_name.
                  FILTER(?concert_name = "Week 1"^^xsd:string)
                }
              }
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Name"))])

        qdmr = QdmrInstance(["select", "filter", "filter", "intersection"],
                            [["singer"], ["#1", "concert name Super bootcamp"], ["#1", "concert name Week 1"], ["#1", "#2", "#3"]])
        grounding = { GroundingIndex(0,0,"singer") : GroundingKey.make_column_grounding("singer", "Name"),
                      GroundingIndex(1,2,"concert name Super bootcamp") : GroundingKey.make_comparative_grounding("=", "Super bootcamp", GroundingKey.make_column_grounding("concert", "concert_Name")),
                      GroundingIndex(2,2,"concert name Week 1") : GroundingKey.make_comparative_grounding("=", "Week 1", GroundingKey.make_column_grounding("concert", "concert_Name")),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestUnionAsArg(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_union_of_horizontal_unions(self):
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?c_id ?concert_name ?Theme ?Stadium_ID ?Year
            WHERE
            {
              ?c_id arc:concert:concert_Name ?concert_name.
              ?c_id arc:concert:Theme ?Theme.
              ?c_id arc:concert:Stadium_ID ?Stadium_ID.
              ?c_id arc:concert:Year ?Year.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "concert_ID")),
                         OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "concert_Name")),
                         OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "Theme")),
                         OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "Stadium_ID")),
                         OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "Year"))])

        qdmr = QdmrInstance(["select", "project", "project", "project", "project", "union", "union", "union"],
                            [["concert"],
                            ["concert_Name", "#1"],
                            ["Theme", "#1"],
                            ["Stadium_ID", "#1"],
                            ["Year", "#1"],
                            ["#1", "#2"],
                            ["#3", "#4", "#5"],
                            ["#6", "#7"],
                            ])
        grounding = { GroundingIndex(0,0,"concert") : GroundingKey.make_table_grounding("concert"),
                      GroundingIndex(1,0,"concert_Name") : GroundingKey.make_column_grounding("concert", "concert_Name"),
                      GroundingIndex(2,0,"Theme") : GroundingKey.make_column_grounding("concert", "Theme"),
                      GroundingIndex(3,0,"Stadium_ID") : GroundingKey.make_column_grounding("concert", "Stadium_ID"),
                      GroundingIndex(4,0,"Year") : GroundingKey.make_column_grounding("concert", "Year"),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_union_of_vertical_unions(self):
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?concert_name ?Theme
            WHERE
            {
              {
                ?c_id arc:concert:concert_Name ?concert_name.
                ?c_id arc:concert:Theme ?Theme.
                ?c_id arc:concert:Year ?Year.
                FILTER(?Year = 2014).
              }
              UNION
              {
                ?c_id arc:concert:concert_Name ?concert_name.
                ?c_id arc:concert:Theme ?Theme.
                ?c_id arc:concert:Year ?Year.
                FILTER(?Year = 2015).
              }
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "concert_Name")),
                         OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "Theme"))])

        qdmr = QdmrInstance(["select", "project", "project", "union", "comparative", "comparative", "union"],
                            [["concert"],
                            ["concert_Name", "#1"],
                            ["Theme", "#1"],
                            ["#2", "#3"],
                            ["#4", "#4", "in 2014"],
                            ["#4", "#4", "in 2015"],
                            ["#5", "#6"],
                            ])
        grounding = { GroundingIndex(0,0,"concert") : GroundingKey.make_table_grounding("concert"),
                      GroundingIndex(1,0,"concert_Name") : GroundingKey.make_column_grounding("concert", "concert_Name"),
                      GroundingIndex(2,0,"Theme") : GroundingKey.make_column_grounding("concert", "Theme"),
                      GroundingIndex(4,2,"in 2014") : GroundingKey.make_comparative_grounding("=", "2014", GroundingKey.make_column_grounding("concert", "Year")),
                      GroundingIndex(5,2,"in 2015") : GroundingKey.make_comparative_grounding("=", "2015", GroundingKey.make_column_grounding("concert", "Year")),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_comparative_after_union(self):
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?concert_name ?Theme
            WHERE
            {
              ?c_id arc:concert:concert_Name ?concert_name.
              ?c_id arc:concert:Theme ?Theme.
              ?c_id arc:concert:Year ?Year.
              FILTER(?Year = 2014).
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "concert_Name")),
                         OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "Theme")),
                         ])

        qdmr = QdmrInstance(["select", "project", "union", "comparative"],
                            [["concert_Name"],
                            ["Theme", "#1"],
                            ["#1", "#2"],
                            ["#3", "#3", "in 2014"],
                            ])
        grounding = { GroundingIndex(0,0,"concert_Name") : GroundingKey.make_column_grounding("concert", "concert_Name"),
                      GroundingIndex(1,0,"Theme") : GroundingKey.make_column_grounding("concert", "Theme"),
                      GroundingIndex(3,2,"in 2014") : GroundingKey.make_comparative_grounding("=", "2014", GroundingKey.make_column_grounding("concert", "Year")),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_intersection_after_union(self):
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?concert_name ?Theme
            WHERE
            {
              ?c_id arc:concert:concert_Name ?concert_name.
              ?c_id arc:concert:Theme ?Theme.
              ?c_id arc:concert:Year ?Year.
              FILTER(?Year = 2014).
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "concert_Name")),
                         OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "Theme")),
                         ])

        qdmr = QdmrInstance(["select", "project", "union", "comparative", "intersection"],
                            [["concert_Name"],
                            ["Theme", "#1"],
                            ["#1", "#2"],
                            ["#3", "#3", "in 2014"],
                            ["#3", "#4", "#4"]
                            ])
        grounding = { GroundingIndex(0,0,"concert_Name") : GroundingKey.make_column_grounding("concert", "concert_Name"),
                      GroundingIndex(1,0,"Theme") : GroundingKey.make_column_grounding("concert", "Theme"),
                      GroundingIndex(3,2,"in 2014") : GroundingKey.make_comparative_grounding("=", "2014", GroundingKey.make_column_grounding("concert", "Year")),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSqlWithStar(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_sql_with_star(self):
        """Select table, project column should return the column
        """
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        sql_query = "SELECT * FROM concert join stadium on concert.stadium_id = stadium.Stadium_ID"

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?concert ?concert_Name ?Theme ?Stadium_ID ?Year ?stadium ?Location ?Name ?Capacity ?Highest ?Lowest ?Average
            WHERE
            {
              ?concert arc:concert:concert_ID ?concert.
              ?concert arc:concert:concert_Name ?concert_Name.
              ?concert arc:concert:Theme ?Theme.
              ?concert arc:concert:Stadium_ID ?Stadium_ID.
              ?concert arc:concert:Year ?Year.
              ?Stadium_ID arc:concert:Stadium_ID:stadium:Stadium_ID ?stadium.
              ?stadium arc:stadium:Stadium_ID  ?stadium.
              ?stadium arc:stadium:Location  ?Location.
              ?stadium arc:stadium:Name  ?Name.
              ?stadium arc:stadium:Capacity  ?Capacity.
              ?stadium arc:stadium:Highest  ?Highest.
              ?stadium arc:stadium:Lowest  ?Lowest.
              ?stadium arc:stadium:Average  ?Average.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
                                output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "concert_ID")),
                                             OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "concert_Name")),
                                             OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "Theme")),
                                             OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "Stadium_ID")),
                                             OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "Year")),
                                             OutputColumnId.from_grounding(GroundingKey.make_column_grounding("stadium", "Stadium_ID")),
                                             OutputColumnId.from_grounding(GroundingKey.make_column_grounding("stadium", "Location")),
                                             OutputColumnId.from_grounding(GroundingKey.make_column_grounding("stadium", "Name")),
                                             OutputColumnId.from_grounding(GroundingKey.make_column_grounding("stadium", "Capacity")),
                                             OutputColumnId.from_grounding(GroundingKey.make_column_grounding("stadium", "Highest")),
                                             OutputColumnId.from_grounding(GroundingKey.make_column_grounding("stadium", "Lowest")),
                                             OutputColumnId.from_grounding(GroundingKey.make_column_grounding("stadium", "Average")),
                                            ])

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


        qdmr = QdmrInstance(["select"] +  ["project"] * 11 + ["union"],
                             [["concert"],
                              ["concert_Name", "#1"],
                              ["Theme", "#1"],
                              ["Stadium_ID", "#1"],
                              ["Year", "#1"],
                              ["Stadium_ID", "#1"],
                              ["Location", "#1"],
                              ["Name", "#1"],
                              ["Capacity", "#1"],
                              ["Highest", "#1"],
                              ["Lowest", "#1"],
                              ["Average", "#1"],
                              ["#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8", "#9", "#10", "#11", "#12"],
                             ])
        grounding = { GroundingIndex(0,0,"concert") : GroundingKey.make_table_grounding("concert"),
                      GroundingIndex(1,0,"concert_Name") : GroundingKey.make_column_grounding("concert", "concert_Name"),
                      GroundingIndex(2,0,"Theme") : GroundingKey.make_column_grounding("concert", "Theme"),
                      GroundingIndex(3,0,"Stadium_ID") : GroundingKey.make_column_grounding("concert", "Stadium_ID"),
                      GroundingIndex(4,0,"Year") : GroundingKey.make_column_grounding("concert", "Year"),
                      GroundingIndex(5,0,"Stadium_ID") : GroundingKey.make_table_grounding("stadium"),
                      GroundingIndex(6,0,"Location") : GroundingKey.make_column_grounding("stadium", "Location"),
                      GroundingIndex(7,0,"Name") : GroundingKey.make_column_grounding("stadium", "Name"),
                      GroundingIndex(8,0,"Capacity") : GroundingKey.make_column_grounding("stadium", "Capacity"),
                      GroundingIndex(9,0,"Highest") : GroundingKey.make_column_grounding("stadium", "Highest"),
                      GroundingIndex(10,0,"Lowest") : GroundingKey.make_column_grounding("stadium", "Lowest"),
                      GroundingIndex(11,0,"Average") : GroundingKey.make_column_grounding("stadium", "Average"),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSqlWithNonUniqueArgmax(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_sql_non_unique_agrmax(self):
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")

        sql_query = "SELECT concert_Name, year FROM concert ORDER BY year DESC LIMIT 1 "

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?concert_Name ?Year1
            WHERE
            {
              {
                SELECT (max(?Year) as ?max)  
                {
                  ?c_id arc:concert:Year ?Year.
                }
              }
              ?c_id arc:concert:Year ?Year1.
              FILTER(?Year1 = ?max).
              ?c_id arc:concert:concert_Name ?concert_Name.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "concert_Name")),
                         OutputColumnId.from_grounding(GroundingKey.make_column_grounding("concert", "Year"))])
        correct_sparql_query.query_has_superlative = True

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    weak_mode_argmax=True,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_sql_non_unique_agrmax_no_max_in_output(self):
        rdf_graph, schema = get_graph_and_schema("dev", "concert_singer")
    
        sql_query = "SELECT concert_Name FROM concert ORDER BY year DESC LIMIT 1 "

        qdmr = QdmrInstance(["select", "project", "superlative"],
                            [["concert_Name"],
                            ["Year", "#1"],
                            ["max", "#1", "#2"],
                            ])
        grounding = { GroundingIndex(0,0,"concert_Name") : GroundingKey.make_column_grounding("concert", "concert_Name"),
                      GroundingIndex(1,0,"Year") : GroundingKey.make_column_grounding("concert", "Year"),
                      GroundingIndex(2,0,"max") : GroundingKey.make_comparative_grounding("max", None),
                    }

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    weak_mode_argmax=True,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderWeirdTypes(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_train(self):
        """Test an entry from spider dataset
        """
        split_name = "train"
        db_id = "bike_1"

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)
        sql_query = "SELECT id, zip_code FROM trip where id = 900630 order by zip_code"
        
        correct_sparql_query = textwrap.dedent("""\
            SELECT DISTINCT ?trip ?zip_code
            WHERE
            {
              ?trip arc:trip:zip_code ?zip_code.
              FILTER(?trip = key:trip:id:0000000000900630).
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("trip", "id")),
                         OutputColumnId.from_grounding(GroundingKey.make_column_grounding("trip", "zip_code"))])

        # break_program:
        qdmr = QdmrInstance(["select", "project", "union", "comparative", "sort"],
                            [["trip_id"],
                            ["zip_code", "#1"],
                            ["#1", "#2"],
                            ["#3", "#2", "= 900630"],
                            ["#4", "#2"]
                            ])
        grounding = {}
        grounding[GroundingIndex(0,0,"trip_id")] = GroundingKey.make_table_grounding("trip")
        grounding[GroundingIndex(1,0,"zip_code")] = GroundingKey.make_column_grounding("trip", "zip_code")
        grounding[GroundingIndex(3,2,"= 900630")] = GroundingKey.make_comparative_grounding("=", "900630", GroundingKey.make_column_grounding("trip", "id"))

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=True,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev0(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 0
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)

        correct_sparql_query = textwrap.dedent("""\
            SELECT (COUNT(?singer) AS ?count)
            WHERE
            {
                ?singer arc:singer:Singer_ID ?singer.
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program = ["SELECT['singers']", "AGGREGATE['count', '#1']"]
        grounding = {GroundingIndex(0,0,"singers") : GroundingKey.make_table_grounding("singer")}

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev2(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 2
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)

        qdmr = get_qdmr_from_break(split_name, i_query)

        qdmr.args[-1] = ["#5", "#4", "from oldest to youngest"]
        # break_program = [
        # "SELECT['singers']",
        # "PROJECT['names of #REF', '#1']",
        # "PROJECT["countries of #REF", '#1']",
        # "PROJECT['ages of #REF', '#1']",
        # "UNION['#2', '#3', '#4']",
        # "SORT['#5', '#4', "from oldest to youngest"]"]
        grounding = {}
        grounding[GroundingIndex(0,0,"singers")] = GroundingKey.make_table_grounding("singer")
        grounding[GroundingIndex(1,0,"names of #REF")] = GroundingKey.make_column_grounding("singer", "Name")
        grounding[GroundingIndex(2,0,"countries of #REF")] = GroundingKey.make_column_grounding("singer", "Country")
        grounding[GroundingIndex(3,0,"ages of #REF")] = GroundingKey.make_column_grounding("singer", "Age")
        grounding[GroundingIndex(5,2,"from oldest to youngest")] = GroundingKey.make_sortdir_grounding(ascending=False)

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=True,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev5(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 5
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)

        correct_sparql_query = textwrap.dedent("""\
            SELECT (avg(?Age) as ?avg) (min(?Age) as ?min) (max(?Age) as ?max)
            WHERE
            {
                ?singer arc:singer:Age ?Age.
                ?singer arc:singer:Country ?Country.
                FILTER(?Country = "France"^^xsd:string).
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program =
        # ["SELECT['singers']",
        # "FILTER['#1', 'who are French']",
        # "PROJECT['ages of #REF', '#2']",
        # "AGGREGATE['avg', '#3']",
        # "AGGREGATE['min', '#3']",
        # "AGGREGATE['max', '#3']",
        # "UNION['#4', '#5', '#6']"]
        grounding = {}
        grounding[GroundingIndex(0,0,"singers")] = GroundingKey.make_table_grounding("singer")
        grounding[GroundingIndex(1,1,"who are French")] = GroundingKey.make_value_grounding("singer", "Country", "France")
        grounding[GroundingIndex(2,0,"ages of #REF")] = GroundingKey.make_column_grounding("singer", "Age")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev8(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev_no_distinct(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 8
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Country
            WHERE
            {
                ?singer arc:singer:Age ?Age.
                FILTER(?Age > 20.0).
                ?singer arc:singer:Country ?Country.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Country"))])

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['singers']
        #     PROJECT['ages of #REF', '#1']
        #     COMPARATIVE['#1', '#2', 'is higher than 20']
        #     PROJECT['distinct countries #REF are from', '#3']
        grounding = {}
        grounding[GroundingIndex(0,0,"singers")] = GroundingKey.make_table_grounding("singer")
        grounding[GroundingIndex(1,0,"ages of #REF")] = GroundingKey.make_column_grounding("singer", "Age")
        grounding[GroundingIndex(2,2,"is higher than 20")] = GroundingKey.make_comparative_grounding(">", "20")
        grounding[GroundingIndex(3,0,"distinct countries #REF are from")] = GroundingKey.make_column_grounding("singer", "Country")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 8
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        correct_sparql_query = textwrap.dedent("""\
            SELECT DISTINCT ?Country
            WHERE
            {
                ?singer arc:singer:Age ?Age.
                FILTER(?Age > 20.0).
                ?singer arc:singer:Country ?Country.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("singer", "Country"))])

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['singers']
        #     PROJECT['ages of #REF', '#1']
        #     COMPARATIVE['#1', '#2', 'is higher than 20']
        #     PROJECT['distinct countries #REF are from', '#3']
        grounding = {}
        grounding[GroundingIndex(0,0,"singers")] = GroundingKey.make_table_grounding("singer")
        grounding[GroundingIndex(1,0,"ages of #REF")] = GroundingKey.make_column_grounding("singer", "Age")
        grounding[GroundingIndex(2,2,"is higher than 20")] = GroundingKey.make_comparative_grounding(">", "20")
        grounding[GroundingIndex(3,0,"distinct countries #REF are from")] = GroundingKey.make_column_grounding("singer", "Country")
        grounding["distinct"] = ["#4"]

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev10(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 10
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?country (COUNT(?singer) AS ?count)
            WHERE
            {
                ?singer arc:singer:Country ?country
            }
            GROUP BY ?country
            """)

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['countries']
        #     PROJECT['singers in #REF', '#1']
        #     GROUP['count', '#2', '#1']
        #     UNION['#1', '#3']
        grounding = {}
        grounding[GroundingIndex(0,0,"countries")] = GroundingKey.make_column_grounding("singer", "Country")
        grounding[GroundingIndex(1,0,"singers in #REF")] = GroundingKey.make_table_grounding("singer")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev11(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 11
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?country (COUNT(?singer) AS ?count)
            WHERE
            {
                ?singer arc:singer:Country ?country
            }
            GROUP BY ?country
            """)

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['countries']
        #     PROJECT['singers from #REF', '#1']
        #     GROUP['count', '#2', '#1']
        grounding = {}
        grounding[GroundingIndex(0,0,"countries")] = GroundingKey.make_column_grounding("singer", "Country")
        grounding[GroundingIndex(1,0,"singers from #REF")] = GroundingKey.make_table_grounding("singer")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev14(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 14
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # this query gives empty results, which makes it bad for testing
        # all the capacities: ('4125',), ('2000',), ('10104',), ('4000',), ('3808',), ('3100',), ('3960',), ('11998',), ('52500',)
        # switching 5000 in the query to 3100

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Location ?Name
            WHERE
            {
              ?stadium arc:stadium:Capacity ?Capacity.
              FILTER(?Capacity >= 3100.0).
              FILTER(?Capacity <= 10000.0).
              ?stadium arc:stadium:Location ?Location.
              ?stadium arc:stadium:Name ?Name.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("stadium", "Location")),
                         OutputColumnId.from_grounding(GroundingKey.make_column_grounding("stadium", "Name"))])

        qdmr = get_qdmr_from_break(split_name, i_query)
        qdmr.args[2] = ["#1", "#2", "is at least 3100"]
        # break_program:
        #     SELECT['stadiums']
        #     PROJECT['capacities of #REF', '#1']
        #     COMPARATIVE['#1', '#2', 'is at least 3100',] # original version - 'is at least 5000' - leads to the empty result
        #     COMPARATIVE['#1', '#2', 'is at most 10000']
        #     INTERSECTION['#1', '#3', '#4']
        #     PROJECT['locations of #REF', '#5']
        #     PROJECT['names of #REF', '#5']
        #     UNION['#6', '#7']

        grounding = {}
        grounding[GroundingIndex(0,0,"stadiums")] = GroundingKey.make_table_grounding("stadium")
        grounding[GroundingIndex(1,0,"capacities of #REF")] = GroundingKey.make_column_grounding("stadium", "Capacity")
        grounding[GroundingIndex(2,2,"is at least 3100")] = GroundingKey.make_comparative_grounding(">=", "3100")
        grounding[GroundingIndex(3,2,"is at most 10000")] = GroundingKey.make_comparative_grounding("<=", "10000")
        grounding[GroundingIndex(5,0,"locations of #REF")] = GroundingKey.make_column_grounding("stadium", "Location")
        grounding[GroundingIndex(6,0,"names of #REF")] = GroundingKey.make_column_grounding("stadium", "Name")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev20(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 20
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # CAUTION: this is a not so great test because it selects  all the concerts

        correct_sparql_query = textwrap.dedent("""\
            SELECT (COUNT(?concerts) AS ?count)
            WHERE
            {
                {
                ?concerts arc:concert:Year ?year.
                FILTER(?year = 2014).
                }
                UNION
                {
                ?concerts arc:concert:Year ?year
                FILTER(?year = 2015)
                }
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.add_aggregator(OutputColumnId.from_grounding(GroundingKey.make_table_grounding("concert"), schema), "count")])

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['concerts']
        #     FILTER['#1', 'in 2014']
        #     FILTER['#1', 'in 2015']
        #     UNION['#2', '#3']
        #     AGGREGATE['count', '#4']

        grounding = {}
        grounding[GroundingIndex(0,0,"concerts")] = GroundingKey.make_table_grounding("concert")
        grounding[GroundingIndex(1,1,"in 2014")] = GroundingKey.make_value_grounding("concert", "Year", "2014")
        grounding[GroundingIndex(2,1,"in 2015")] = GroundingKey.make_value_grounding("concert", "Year", "2015")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev24(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 24
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Show the stadium name and capacity with most number of concerts in year 2014 or after.
        # SQL: SELECT T2.name ,  T2.capacity FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id WHERE T1.year  >=  2014 GROUP BY T2.stadium_id ORDER BY count(*) DESC LIMIT 1

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Name ?Capacity
            WHERE
            {
              {
                SELECT DISTINCT ?stadium_3
                WHERE
                {
                  {
                    SELECT (max(?count) AS ?max)
                    WHERE
                    {
                      {
                        SELECT ?stadium_1 (count(?concert) AS ?count)
                        WHERE
                        {
                          {
                            SELECT DISTINCT ?concert
                            WHERE
                            {
                              ?Stadium_ID arc:concert:Stadium_ID:stadium:Stadium_ID ?stadium.
                              ?concert arc:concert:Stadium_ID ?Stadium_ID.
                              ?concert arc:concert:Year ?Year.
                              FILTER(?Year >= 2014).
                            }
                          }
                          ?concert arc:concert:Stadium_ID ?Stadium_ID_1.
                          ?Stadium_ID_1 arc:concert:Stadium_ID:stadium:Stadium_ID ?stadium_1.
                          ?stadium_1 arc:stadium:Stadium_ID ?stadium_1.
                        }
                        GROUP BY ?stadium_1
                      }
                    }
                  }
                  {
                    SELECT ?stadium_3 (count(?concert_1) AS ?count_1)
                    WHERE
                    {
                      {
                        SELECT DISTINCT ?concert_1
                        WHERE
                        {
                          ?Stadium_ID_1 arc:concert:Stadium_ID:stadium:Stadium_ID ?stadium_2.
                          ?concert_1 arc:concert:Stadium_ID ?Stadium_ID_1.
                          ?concert_1 arc:concert:Year ?Year_1.
                          FILTER(?Year_1 >= 2014).
                        }
                      }
                      ?concert_1 arc:concert:Stadium_ID ?Stadium_ID_3.
                      ?Stadium_ID_3 arc:concert:Stadium_ID:stadium:Stadium_ID ?stadium_3.
                      ?stadium_3 arc:stadium:Stadium_ID ?stadium_3.
                    }
                    GROUP BY ?stadium_3
                  }
                  FILTER(?count_1 = ?max).
                }
              }
              ?stadium_3 arc:stadium:Name ?Name.
              ?stadium_3 arc:stadium:Capacity ?Capacity.
            }""")

        # qdmr = get_qdmr_from_break(split_name, i_query)
        qdmr = QdmrInstance(["select", "project", "project", "comparative", "group", "superlative", "project", "project", "union"],
                            [["tbl:​stadium"],
                             ["tbl:​concert", "#1"],
                             ["col:​concert:​Year", "#2"],
                             ["#2", "#3", "comparative:​>=:​2014:​col:​concert:​Year"],
                             ["count", "#4", "#1"],
                             ["comparative:​max:​None", "#1", "#5"],
                             ["col:​stadium:​Name", "#6"],
                             ["col:​stadium:​Capacity", "#6"],
                             ["#7", "#8"],
                            ])
        # break_program:
        # 1. SELECT[tbl:​stadium]
        # 2. PROJECT[tbl:​concert, #1]
        # 3. PROJECT[col:​concert:​Year, #2]
        # 4. COMPARATIVE[#2, #3, comparative:​>=:​2014:​col:​concert:​Year]
        # 5. GROUP[count, #4, #1]
        # 6. SUPERLATIVE[comparative:​max:​None, #1, #5]
        # 7. PROJECT[col:​stadium:​Name, #6]
        # 8. PROJECT[col:​stadium:​Capacity, #6]
        # 9. UNION[#7, #8]

        grounding = {}
        grounding[GroundingIndex(0,0,"tbl:​stadium")] = GroundingKey.make_table_grounding("stadium")
        grounding[GroundingIndex(1,0,"tbl:​concert")] = GroundingKey.make_table_grounding("concert")
        grounding[GroundingIndex(2,0,"col:​concert:​Year")] = GroundingKey.make_column_grounding("concert", "Year")
        grounding[GroundingIndex(3,2,"comparative:​>=:​2014:​col:​concert:​Year")] = GroundingKey.make_comparative_grounding(">=", "2014", GroundingKey.make_column_grounding("concert", "Year"))
        grounding[GroundingIndex(5,0,"comparative:​max:​None")] = GroundingKey.make_comparative_grounding("max", None)
        grounding[GroundingIndex(6,0,"col:​stadium:​Name")] = GroundingKey.make_column_grounding("stadium", "Name")
        grounding[GroundingIndex(7,0,"col:​stadium:​Capacity")] = GroundingKey.make_column_grounding("stadium", "Capacity")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev30(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 30
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Show countries where a singer above age 40 and a singer below 30 are from .
        # SQL: select country from singer where age  >  40 intersect select country from singer where age  <  30

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Country
            WHERE
            {
              {
                SELECT ?Country
                WHERE
                {
                  ?singer arc:singer:Country ?Country.
                  ?singer arc:singer:Age ?Age.
                  FILTER(?Age > 40).
                }
                GROUP BY ?Country
              }
              ?singer_1 arc:singer:Country ?Country.
              ?singer_1 arc:singer:Age ?Age_1.
              FILTER(?Age_1 < 30).
            }
            GROUP BY ?Country""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['countries']
        #     PROJECT['singers from #REF', '#1']
        #     PROJECT['ages of #REF', '#2']
        #     COMPARATIVE['#1', '#3', 'is above 40']
        #     COMPARATIVE['#1', '#3', 'is below 30']
        #     INTERSECTION['#1', '#4', '#5']

        grounding = {}
        grounding[GroundingIndex(0,0,"countries")] = GroundingKey.make_column_grounding("singer", "Country")
        grounding[GroundingIndex(1,0,"singers from #REF")] = GroundingKey.make_table_grounding("singer")
        grounding[GroundingIndex(2,0,"ages of #REF")] = GroundingKey.make_column_grounding("singer", "Age")
        grounding[GroundingIndex(3,2,"is above 40")] = GroundingKey.make_comparative_grounding(">", "40")
        grounding[GroundingIndex(4,2,"is below 30")] = GroundingKey.make_comparative_grounding("<", "30")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev39(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 39
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: what is the name and nation of the singer who have a song having 'Hey' in its name?
        # SQL: SELECT name ,  country FROM singer WHERE song_name LIKE '%Hey%'

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Name ?Country
            WHERE
            {
                ?singer arc:singer:Song_Name ?Song_Name.
                ?singer arc:singer:Name ?Name.
                ?singer arc:singer:Country ?Country.
                FILTER(REGEX(STR(?Song_Name), "(.*hey.*)", "i"))
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['singers']
        #     FILTER['#1', 'who have a song having Hey in its name']
        #     PROJECT['the name of #REF', '#2']
        #     PROJECT['the nation of #REF', '#2']
        #     UNION['#3', '#4']

        grounding = {}
        grounding[GroundingIndex(0,0,"singers")] = GroundingKey.make_table_grounding("singer")
        grounding[GroundingIndex(1,1,"who have a song having Hey in its name")] = GroundingKey.make_comparative_grounding("like", "hey", GroundingKey.make_column_grounding("singer", "Song_Name"))
        grounding[GroundingIndex(2,0,"the name of #REF")] = GroundingKey.make_column_grounding("singer", "Name")
        grounding[GroundingIndex(3,0,"the nation of #REF")] = GroundingKey.make_column_grounding("singer", "Country")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev47(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 47
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: what is the name and nation of the singer who have a song having 'Hey' in its name?
        # SQL is wrong: select weight from pets order by pet_age limit 1

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?weight
            WHERE
            {
              {
                SELECT ?PetType_1 ?Pets_1
                WHERE
                {
                  {
                    SELECT (min(?pet_age) AS ?min)
                    WHERE
                    {
                      ?Pets arc:Pets:PetType ?PetType.
                      FILTER(?PetType = "dog").
                      ?Pets arc:Pets:pet_age ?pet_age.
                    }
                  }
                  ?Pets_1 arc:Pets:PetType ?PetType_1.
                  FILTER(?PetType_1 = "dog").
                  ?Pets_1 arc:Pets:pet_age ?pet_age_1.
                  FILTER(?pet_age_1 = ?min).
                }
                GROUP BY ?Pets_1
              }
              ?Pets_1 arc:Pets:weight ?weight.
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program
        #     SELECT['dogs']
        #     PROJECT['age of #REF', '#1']
        #     COMPARATIVE['#1', '#2', 'is youngest']
        #     PROJECT['weight of #REF', '#3']

        grounding = {}
        grounding[GroundingIndex(0,0,"dogs")] = GroundingKey.make_value_grounding("Pets", "PetType", "dog")
        grounding[GroundingIndex(1,0,"age of #REF")] = GroundingKey.make_column_grounding("Pets", "pet_age")
        grounding[GroundingIndex(2,2,"is youngest")] = GroundingKey.make_comparative_grounding("min", None)
        grounding[GroundingIndex(3,0,"weight of #REF")] = GroundingKey.make_column_grounding("Pets", "weight")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev53(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 53
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Find the number of dog pets that are raised by female students (with sex F).
        # SQL is wrong: 
        #   SELECT count(*) 
        #   FROM student AS T1 JOIN has_pet AS T2 ON T1.stuid  =  T2.stuid JOIN pets AS T3 ON T2.petid  =  T3.petid 
        #   WHERE T1.sex  =  'F' AND T3.pettype  =  'dog'

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program
        # 1: SELECT['students']
        # 2: FILTER['#1', 'that are female']
        # 3: PROJECT['dog pets raised by #REF', '#2']
        # 4: GROUP['count', '#3', '#2']
        # 5: AGGREGATE['sum', '#4']

        grounding = {}
        grounding[GroundingIndex(0,0,"students")] = GroundingKey.make_table_grounding("Student")
        grounding[GroundingIndex(1,1,"that are female")] = GroundingKey.make_comparative_grounding("=", "F", GroundingKey.make_column_grounding("Student", "Sex"))
        grounding[GroundingIndex(2,0,"dog pets raised by #REF")] = GroundingKey.make_value_grounding("Pets", "PetType", "dog")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


# class TestSpiderDev59(unittest.TestCase):

#     @timeout(ONE_TEST_TIMEOUT)
#     def test_spider_dev(self):
#         """Test an entry from spider dataset
#         """
#         split_name = "dev"
#         i_query = 59
#         db_id = get_db_id(split_name, i_query)

#         rdf_graph, schema = get_graph_and_schema(split_name, db_id)

#         sql_query = get_sql_query(split_name, i_query)
#         # question:  Find the first name of students who have both cat and dog pets
#         # SQL:
#         # select t1.fname
#         # from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid
#         # where t3.pettype  =  'cat'
#         # intersect
#         # select t1.fname
#         # from student as t1 join has_pet as t2 on t1.stuid  =  t2.stuid join pets as t3 on t3.petid  =  t2.petid
#         # where t3.pettype  =  'dog'


#         # This SPARQL query looks correct but does not work because the intersection of two filters is the empty set.
#         # What to do with it?

#         correct_sparql_query = textwrap.dedent("""\
#             SELECT ?Fname
#             WHERE
#             {
#               {
#                 SELECT ?Student
#                 WHERE
#                 {
#                   {
#                     SELECT ?Student
#                     WHERE
#                     {
#                       ?StuID arc:Has_Pet:StuID:Student:StuID ?Student.
#                       ?Has_Pet arc:Has_Pet:StuID ?StuID.
#                       ?Has_Pet arc:Has_Pet:PetID:Pets:PetID ?Pets.
#                       ?Pets arc:Pets:PetType ?PetType.
#                       FILTER(?PetType = "cat").
#                     }
#                     GROUP BY ?Student
#                   }
#                   ?StuID_1 arc:Has_Pet:StuID:Student:StuID ?Student.
#                   ?Has_Pet_1 arc:Has_Pet:StuID ?StuID_1.
#                   ?Has_Pet_1 arc:Has_Pet:PetID:Pets:PetID ?Pets_1.
#                   ?Pets_1 arc:Pets:PetType ?PetType_1.
#                   FILTER(?PetType_1 = "dog").
#                 }
#                 GROUP BY ?Student
#               }
#               ?Student arc:Student:Fname ?Fname.
#             }""")

#         qdmr = get_qdmr_from_break(split_name, i_query)
#         # break_program:
#         #     SELECT['students']
#         #     FILTER['#1', 'who have cats']
#         #     FILTER['#1', 'who have dog pets']
#         #     INTERSECTION['#1', '#2', '#3']
#         #     PROJECT['first names of #REF', '#4']

#         grounding = {}
#         grounding[GroundingIndex(0,0,"students")] = GroundingKey.make_table_grounding("Student")
#         grounding[GroundingIndex(1,1,"who have cats")] = GroundingKey.make_comparative_grounding('=', 'cat', GroundingKey.make_column_grounding("Pets", "PetType"))
#         grounding[GroundingIndex(2,1,"who have dog pets")] = GroundingKey.make_comparative_grounding('=', 'dog', GroundingKey.make_column_grounding("Pets", "PetType"))
#         grounding[GroundingIndex(4,0,"first names of #REF")] = GroundingKey.make_column_grounding("Student", "Fname")

#         sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

#         result_correct = QueryResult.execute_query_sql(sql_query, schema)
#         result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
#         equal, message = result.is_equal_to(result_correct,
#                                     require_column_order=True,
#                                     require_row_order=False,
#                                     return_message=True)
#         self.assertTrue(equal, message)


class TestSpiderDev71(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 71
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Find the average and maximum age for each type of pet.
        # SQL is wrong: SELECT pettype ,  avg(pet_age) ,  max(pet_age) FROM pets GROUP BY pettype

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?PetType ?avg ?max
            WHERE
            {
              {
                SELECT ?PetType (avg(?pet_age) AS ?avg)
                WHERE
                {
                  ?Pets arc:Pets:PetType ?PetType.
                  ?Pets arc:Pets:pet_age ?pet_age.
                }
                GROUP BY ?PetType
              }
              {
                SELECT ?PetType (max(?pet_age_1) AS ?max)
                WHERE
                {
                  ?Pets_1 arc:Pets:PetType ?PetType.
                  ?Pets_1 arc:Pets:pet_age ?pet_age_1.
                }
                GROUP BY ?PetType
              }
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program
        # #1: SELECT['pets']
        # #2: PROJECT['types of #REF', '#1']
        # #3: PROJECT['ages of #REF', '#2']
        # #4: GROUP['avg', '#3', '#2']
        # #5: GROUP['max', '#3', '#2']
        # #6: UNION['#2', '#4', '#5']

        grounding = {}
        grounding[GroundingIndex(0,0,"pets")] = GroundingKey.make_table_grounding("Pets")
        grounding[GroundingIndex(1,0,"types of #REF")] = GroundingKey.make_column_grounding("Pets", "PetType")
        grounding[GroundingIndex(2,0,"ages of #REF")] = GroundingKey.make_column_grounding("Pets", "pet_age")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev97(unittest.TestCase):

    # @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 97
        db_id = get_db_id(split_name, i_query)
        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Find the model of the car whose weight is below the average weight.
        # SQL:
        # SELECT T1.model
        # FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id
        # WHERE T2.Weight  <  (SELECT avg(Weight) FROM CARS_DATA)

        # for the correct query, SPARQL generator wants to group by car_names.Model "GROUP BY ?Model",
        # which leads to a different result because car_names.Model is not a key
        # but this example has another error, which I want to debug: spaces in the values with keys

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Model
            WHERE
            {
              {
                SELECT ?car_names
                WHERE
                {
                  ?cars_data arc:cars_data:Id:car_names:MakeId ?car_names.
                  ?cars_data arc:cars_data:Weight ?Weight.
                  {
                    SELECT (avg(?Weight_1) AS ?avg)
                    WHERE
                    {
                      ?cars_data_1 arc:cars_data:Weight ?Weight_1.
                    }
                  }
                  FILTER(?Weight < ?avg).
                }
                GROUP BY ?car_names
              }
              ?car_names arc:car_names:Model ?Model.
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['cars']
        #     PROJECT['models of #REF', '#1']
        #     PROJECT['weights of #REF', '#2']
        #     AGGREGATE['avg', '#3']
        #     COMPARATIVE['#2', '#3', 'is lower than #4']

        grounding = {}
        grounding[GroundingIndex(0,0,"cars")] = GroundingKey.make_table_grounding("car_names")
        grounding[GroundingIndex(1,0,"models of #REF")] = GroundingKey.make_column_grounding("car_names", "Model")
        grounding[GroundingIndex(2,0,"weights of #REF")] = GroundingKey.make_column_grounding("cars_data", "Weight")
        grounding[GroundingIndex(4,2,"is lower than #4")] = GroundingKey.make_comparative_grounding("<", "#4")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev100(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 100
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # SELECT DISTINCT T1.Maker
        # FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker JOIN CAR_NAMES AS T3 ON T2.model  =  T3.model JOIN CARS_DATA AS T4 ON T3.MakeId  =  T4.id
        # WHERE T4.year  =  '1970';

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Maker_2
            WHERE
            {
                {
                    SELECT ?car_makers
                    WHERE
                    {
                        {
                            SELECT ?car_makers
                            WHERE
                            {
                                ?Maker arc:model_list:Maker:car_makers:Id ?car_makers.
                                ?model_list arc:model_list:Maker ?Maker.
                                ?model_list arc:model_list:Model ?Model_1.
                                ?Model arc:car_names:Model:model_list:Model ?Model_1.
                                ?car_names arc:car_names:Model ?Model.
                            }
                            GROUP BY ?car_makers
                        }
                        ?Maker_1 arc:model_list:Maker:car_makers:Id ?car_makers.
                        ?model_list_1 arc:model_list:Maker ?Maker_1.
                        ?model_list_1 arc:model_list:Model ?Model_3.
                        ?Model_2 arc:car_names:Model:model_list:Model ?Model_3.
                        ?car_names_1 arc:car_names:Model ?Model_2.
                        ?cars_data arc:cars_data:Id:car_names:MakeId ?car_names_1.
                        ?cars_data arc:cars_data:Year ?Year.
                        FILTER(?Year = 1970).
                    }
                    GROUP BY ?car_makers
                }
                ?car_makers arc:car_makers:Maker ?Maker_2.
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['different car makers']
        #     FILTER['#1', 'who produced a car']
        #     FILTER['#2', 'in 1970']
        #     PROJECT['the name of #REF', '#3']

        grounding = {}
        grounding[GroundingIndex(0,0,"different car makers")] = GroundingKey.make_table_grounding("car_makers")
        grounding[GroundingIndex(1,1,"who produced a car")] = GroundingKey.make_table_grounding("car_names")
        grounding[GroundingIndex(2,1,"in 1970")] = GroundingKey.make_value_grounding("cars_data", "Year", "1970")
        grounding[GroundingIndex(3,0,"the name of #REF")] = GroundingKey.make_column_grounding("car_makers", "Maker")
        grounding["distinct"] = ["#1"]

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev_change_filter_to_comparative(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 100
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # SELECT DISTINCT T1.Maker
        # FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker JOIN CAR_NAMES AS T3 ON T2.model  =  T3.model JOIN CARS_DATA AS T4 ON T3.MakeId  =  T4.id
        # WHERE T4.year  =  '1970';

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Maker_2
            WHERE
            {
                {
                    SELECT ?car_makers
                    WHERE
                    {
                        {
                            SELECT ?car_makers
                            WHERE
                            {
                                ?Maker arc:model_list:Maker:car_makers:Id ?car_makers.
                                ?model_list arc:model_list:Maker ?Maker.
                                ?model_list arc:model_list:Model ?Model_1.
                                ?Model arc:car_names:Model:model_list:Model ?Model_1.
                                ?car_names arc:car_names:Model ?Model.
                            }
                            GROUP BY ?car_makers
                        }
                        ?Maker_1 arc:model_list:Maker:car_makers:Id ?car_makers.
                        ?model_list_1 arc:model_list:Maker ?Maker_1.
                        ?model_list_1 arc:model_list:Model ?Model_3.
                        ?Model_2 arc:car_names:Model:model_list:Model ?Model_3.
                        ?car_names_1 arc:car_names:Model ?Model_2.
                        ?cars_data arc:cars_data:Id:car_names:MakeId ?car_names_1.
                        ?cars_data arc:cars_data:Year ?Year.
                        FILTER(?Year = 1970).
                    }
                    GROUP BY ?car_makers
                }
                ?car_makers arc:car_makers:Maker ?Maker_2.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("car_makers", "Maker"))])

        qdmr = get_qdmr_from_break(split_name, i_query)
        qdmr.ops[2] = "comparative"
        qdmr.args[2] = ["#2", "#2", "in 1970"]
        # break_program:
        #     SELECT['different car makers']
        #     FILTER['#1', 'who produced a car']
        #     COMPARATIVE['#2', '#2', 'in 1970']
        #     PROJECT['the name of #REF', '#3']

        grounding = {}
        grounding[GroundingIndex(0,0,"different car makers")] = GroundingKey.make_table_grounding("car_makers")
        grounding[GroundingIndex(1,1,"who produced a car")] = GroundingKey.make_table_grounding("car_names")
        grounding[GroundingIndex(2,2,"in 1970")] = GroundingKey.make_comparative_grounding("=", "1970", GroundingKey.make_column_grounding("cars_data", "Year"))
        grounding[GroundingIndex(3,0,"the name of #REF")] = GroundingKey.make_column_grounding("car_makers", "Maker")
        grounding["distinct"] = ["#1"]

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev101(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev_corrected_comparative_to_superlative(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 101
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Find the make and production time of the cars that were produced in the earliest year?
        # sql_query:
        # SELECT T2.Make, T1.Year
        # FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId
        # WHERE T1.Year  = (SELECT min(YEAR) FROM CARS_DATA);'

        correct_sparql_query = textwrap.dedent("""\
                SELECT ?Make ?Year
                WHERE
                {
                    {
                        SELECT (MIN(?years) AS ?min)
                        WHERE
                        {
                            ?cars_data_id arc:cars_data:Year ?years.
                        }
                    }
                    ?Id arc:cars_data:Id:car_names:MakeId ?MakeId.
                    ?MakeId arc:car_names:Make ?Make.
                    ?Id arc:cars_data:Year ?Year.
                    FILTER(?Year = ?min).
                }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # correct Break: substitute compaartive with superlative
        qdmr.ops[2] = "superlative"
        qdmr.args[2] = ["is the earliest", "#1", "#2"]
        # break_program:
        #     SELECT['cars']
        #     PROJECT['years #REF were produced', '#1']
        #     SUPERLATIVE['is the earliest', '#1', '#2'] # original - COMPARATIVE['#1', '#2', 'is the earliest']
        #     PROJECT['make of #REF', '#3']
        #     PROJECT['production time of #REF', '#3']
        #     UNION['#4', '#5']

        grounding = {}
        grounding[GroundingIndex(0,0,"cars")] = GroundingKey.make_table_grounding("cars_data")
        grounding[GroundingIndex(1,0,"years #REF were produced")] = GroundingKey.make_column_grounding("cars_data", "Year")
        grounding[GroundingIndex(2,0,"is the earliest")] = GroundingKey.make_comparative_grounding("min", None)
        grounding[GroundingIndex(3,0,"make of #REF")] = GroundingKey.make_column_grounding("car_names", "Make")
        grounding[GroundingIndex(4,0,"production time of #REF")] = GroundingKey.make_column_grounding("cars_data", "Year")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 101
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Find the make and production time of the cars that were produced in the earliest year?
        # sql_query:
        # SELECT T2.Make, T1.Year
        # FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId
        # WHERE T1.Year  = (SELECT min(YEAR) FROM CARS_DATA);'

        correct_sparql_query = textwrap.dedent("""\
                SELECT ?Make ?Year
                WHERE
                {
                    {
                        SELECT (MIN(?years) AS ?min)
                        WHERE
                        {
                            ?cars_data_id arc:cars_data:Year ?years.
                        }
                    }
                    ?Id arc:cars_data:Id:car_names:MakeId ?MakeId.
                    ?MakeId arc:car_names:Make ?Make.
                    ?Id arc:cars_data:Year ?Year.
                    FILTER(?Year = ?min).
                }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['cars']
        #     PROJECT['years #REF were produced', '#1']
        #     COMPARATIVE['#1', '#2', 'is the earliest']
        #     PROJECT['make of #REF', '#3']
        #     PROJECT['production time of #REF', '#3']
        #     UNION['#4', '#5']

        grounding = {}
        grounding[GroundingIndex(0,0,"cars")] = GroundingKey.make_table_grounding("cars_data")
        grounding[GroundingIndex(1,0,"years #REF were produced")] = GroundingKey.make_column_grounding("cars_data", "Year")
        grounding[GroundingIndex(2,2,"is the earliest")] = GroundingKey.make_comparative_grounding("min", None)
        grounding[GroundingIndex(3,0,"make of #REF")] = GroundingKey.make_column_grounding("car_names", "Make")
        grounding[GroundingIndex(4,0,"production time of #REF")] = GroundingKey.make_column_grounding("cars_data", "Year")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev105(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 105
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['continents']
        #     PROJECT['car makers on #REF', '#1']
        #     GROUP['count', '#2', '#1']
        #     UNION['#1', '#3']

        grounding = {}
        grounding[GroundingIndex(0,0,"continents")] = GroundingKey.make_column_grounding("continents", "Continent")
        grounding[GroundingIndex(1,0,"car makers on #REF")] = GroundingKey.make_table_grounding("car_makers")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev_delete_union_after_group(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 105
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)

        qdmr = get_qdmr_from_break(split_name, i_query)
        # need to decide what to do with union after group by
        del qdmr.ops[3]
        del qdmr.args[3]
        # break_program:
        #     SELECT['continents']
        #     PROJECT['car makers on #REF', '#1']
        #     GROUP['count', '#2', '#1']

        grounding = {}
        grounding[GroundingIndex(0,0,"continents")] = GroundingKey.make_column_grounding("continents", "Continent")
        grounding[GroundingIndex(1,0,"car makers on #REF")] = GroundingKey.make_table_grounding("car_makers")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev108(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 108
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # sql_query:
        # SELECT T2.CountryName
        # FROM CAR_MAKERS AS T1 JOIN COUNTRIES AS T2 ON T1.Country  =  T2.CountryId
        # GROUP BY T1.Country
        # ORDER BY Count(*) DESC LIMIT 1
        # question: What is the name of the country with the most car makers?

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?CountryName
            WHERE
            {
                {
                    SELECT (max(?count) AS ?max)
                    WHERE
                    {
                        {
                            SELECT ?countries (count(?car_makers) AS ?count)
                            WHERE
                            {
                                ?car_makers arc:car_makers:Id ?car_makers.
                                ?car_makers arc:car_makers:Country ?Country.
                                ?Country arc:car_makers:Country:countries:CountryId ?countries.
                            }
                            GROUP BY ?countries
                        }
                    }
                }
                {
                    SELECT ?countries_1 (count(?car_makers_2) AS ?count_1)
                    WHERE
                    {
                        ?car_makers_2 arc:car_makers:Id ?car_makers_2.
                        ?car_makers_2 arc:car_makers:Country ?Country_2.
                        ?Country_2 arc:car_makers:Country:countries:CountryId ?countries_1.
                    }
                    GROUP BY ?countries_1
                }
                FILTER(?count_1 = ?max).
                ?countries_1 arc:countries:CountryName ?CountryName.
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['car makers']
        #     PROJECT['countries of #REF', '#1']
        #     GROUP['count', '#1', '#2']
        #     COMPARATIVE['#2', '#3', 'is the highest']
        #     PROJECT['the name of #REF', '#4']

        grounding = {}
        grounding[GroundingIndex(0,0,"car makers")] = GroundingKey.make_table_grounding("car_makers")
        grounding[GroundingIndex(1,0,"countries of #REF")] = GroundingKey.make_column_grounding("countries", "CountryId")
        grounding[GroundingIndex(3,2,"is the highest")] = GroundingKey.make_comparative_grounding("max", None)
        grounding[GroundingIndex(4,0,"the name of #REF")] = GroundingKey.make_column_grounding("countries", "CountryName")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev110(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 110
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # sql_query:
        # 'SELECT Count(*) ,  T2.FullName ,  T2.id
        # FROM MODEL_LIST AS T1 JOIN CAR_MAKERS AS T2 ON T1.Maker  =  T2.Id
        # GROUP BY T2.id;
        # question: What is the number of car models that are produced by each maker and what is the id and full name of each maker?

        # CAUTION: SQL query has incorrect order of arguments

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?count ?car_makers ?FullName
            WHERE
            {
                {
                    SELECT ?car_makers (count(?model_list) AS ?count)
                    WHERE
                    {
                        ?car_makers arc:car_makers:Id ?car_makers.
                        ?Maker arc:model_list:Maker:car_makers:Id ?car_makers.
                        ?model_list arc:model_list:Maker ?Maker.
                    }
                    GROUP BY ?car_makers
                }
            ?car_makers arc:car_makers:FullName ?FullName.
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.add_aggregator(OutputColumnId.from_grounding(GroundingKey.make_table_grounding("model_list"), schema), "count"),
                         OutputColumnId.from_grounding(GroundingKey.make_column_grounding("car_makers", "Id")),
                         OutputColumnId.from_grounding(GroundingKey.make_column_grounding("car_makers", "FullName"))])

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program
        #     SELECT['car makers']
        #     PROJECT['car models of #REF', '#1']
        #     GROUP['count', '#2', '#1']
        #     PROJECT['ids of #REF', '#1']
        #     PROJECT['full names of #REF', '#1']
        #     UNION['#3', '#4', '#5']

        grounding = {}
        grounding[GroundingIndex(0,0,"car makers")] = GroundingKey.make_table_grounding("car_makers")
        grounding[GroundingIndex(1,0,"car models of #REF")] = GroundingKey.make_table_grounding("model_list")
        grounding[GroundingIndex(3,0,"ids of #REF")] = GroundingKey.make_column_grounding("car_makers", "Id")
        grounding[GroundingIndex(4,0,"full names of #REF")] = GroundingKey.make_column_grounding("car_makers", "FullName")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev124(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 124
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # sql_query:
        # SELECT T1.CountryName ,  T1.CountryId
        # FROM COUNTRIES AS T1 JOIN CAR_MAKERS AS T2 ON T1.CountryId  =  T2.Country
        # GROUP BY T1.CountryId HAVING count(*)  >=  1;
        # question: What are the names and ids of all countries with at least one car maker?

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?CountryName ?countries
            WHERE
            {
                {
                    SELECT ?countries
                    WHERE
                    {
                        ?Country arc:car_makers:Country:countries:CountryId ?countries.
                        ?car_makers arc:car_makers:Country ?Country.
                    }
                    GROUP BY ?countries
                }
                ?countries arc:countries:CountryName ?CountryName.
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['countries']
        #     FILTER['#1', 'with car maker']
        #     PROJECT['names of #REF', '#2']
        #     PROJECT['ids of #REF', '#2']
        #     UNION['#3', '#4']

        grounding = {}
        grounding[GroundingIndex(0,0,"countries")] = GroundingKey.make_table_grounding("countries")
        grounding[GroundingIndex(1,1,"with car maker")] = GroundingKey.make_table_grounding("car_makers")
        grounding[GroundingIndex(2,0,"names of #REF")] = GroundingKey.make_column_grounding("countries", "CountryName")
        grounding[GroundingIndex(3,0,"ids of #REF")] = GroundingKey.make_column_grounding("countries", "CountryId")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev129(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 129
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Which countries in europe have at least 3 car manufacturers?
        # sql_query:
        # SELECT T1.CountryName
        # FROM COUNTRIES AS T1 JOIN CONTINENTS AS T2 ON T1.Continent  =  T2.ContId JOIN CAR_MAKERS AS T3 ON T1.CountryId  =  T3.Country
        # WHERE T2.Continent  =  'europe'
        # GROUP BY T1.CountryName
        # HAVING count(*)  >=  3;

        correct_sparql_query_long = textwrap.dedent("""\
            SELECT ?CountryName
            WHERE
            {
            ?countries arc:countries:CountryName ?CountryName.
            ?countries arc:countries:Continent ?Continent_1.
            ?Continent_1 arc:countries:Continent:continents:ContId ?continents.
            ?continents arc:continents:Continent ?Continent.
            FILTER(?Continent = "europe").
            {
                SELECT ?CountryName (count(?car_makers) AS ?count)
                WHERE
                {
                ?countries_1 arc:countries:CountryName ?CountryName.
                ?countries_1 arc:countries:Continent ?Continent_3.
                ?Continent_3 arc:countries:Continent:continents:ContId ?continents_1.
                ?continents_1 arc:continents:Continent ?Continent_2.
                FILTER(?Continent_2 = "europe").
                ?Country arc:car_makers:Country:countries:CountryId ?countries_1.
                ?car_makers arc:car_makers:Country ?Country.
                }
                GROUP BY ?CountryName
            }
            FILTER(?count >= 3.0).
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['countries']
        #     FILTER['#1', 'in europe']
        #     PROJECT['car manufacturers in #REF', '#2']
        #     GROUP['count', '#3', '#2']
        #     COMPARATIVE['#2', '#4', 'is at least 3']

        grounding = {}
        grounding[GroundingIndex(0,0,"countries")] = GroundingKey.make_column_grounding("countries", "CountryName")
        grounding[GroundingIndex(1,1,"in europe")] = GroundingKey.make_value_grounding("continents", "Continent", "europe")
        grounding[GroundingIndex(2,0,"car manufacturers in #REF")] = GroundingKey.make_table_grounding("car_makers")
        grounding[GroundingIndex(4,2,"is at least 3")] = GroundingKey.make_comparative_grounding(">=", "3")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev132(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev_compare_to_sparql(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 132
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: What is the largest amount of horsepower for the models with 3 cylinders and what make is it ?
        # sql_query:
        # SELECT T2.horsepower ,  T1.Make
        # FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id
        # WHERE T2.cylinders  =  3
        # ORDER BY T2.horsepower DESC LIMIT 1;

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?max_2 ?Make
            WHERE
            {
              {
                SELECT ?car_names_1
                WHERE
                {
                  {
                    SELECT ?car_names_1
                    WHERE
                    {
                      ?cars_data_2 arc:cars_data:Id:car_names:MakeId ?car_names_1.
                      ?cars_data_2 arc:cars_data:Cylinders ?Cylinders_1.
                      FILTER(?Cylinders_1 = 3).
                    }
                    GROUP BY ?car_names_1
                  }
                  ?cars_data_3 arc:cars_data:Id:car_names:MakeId ?car_names_1.
                  ?cars_data_3 arc:cars_data:Horsepower ?Horsepower_1.
                  {
                    SELECT (max(?Horsepower_2) AS ?max_1)
                    WHERE
                    {
                      {
                        SELECT ?car_names_2
                        WHERE
                        {
                          ?cars_data_4 arc:cars_data:Id:car_names:MakeId ?car_names_2.
                          ?cars_data_4 arc:cars_data:Cylinders ?Cylinders_2.
                          FILTER(?Cylinders_2 = 3).
                        }
                        GROUP BY ?car_names_2
                      }
                      ?cars_data_5 arc:cars_data:Id:car_names:MakeId ?car_names_2.
                      ?cars_data_5 arc:cars_data:Horsepower ?Horsepower_2.
                    }
                  }
                  FILTER(?Horsepower_1 = ?max_1).
                }
                GROUP BY ?car_names_1
              }
              ?car_names_1 arc:car_names:Make ?Make.
              {
                SELECT (max(?Horsepower_3) AS ?max_2)
                WHERE
                {
                  {
                    SELECT ?car_names_3
                    WHERE
                    {
                      ?cars_data_6 arc:cars_data:Id:car_names:MakeId ?car_names_3.
                      ?cars_data_6 arc:cars_data:Cylinders ?Cylinders_3.
                      FILTER(?Cylinders_3 = 3).
                    }
                    GROUP BY ?car_names_3
                  }
                  ?cars_data_7 arc:cars_data:Id:car_names:MakeId ?car_names_3.
                  ?cars_data_7 arc:cars_data:Horsepower ?Horsepower_3.
                }
              }
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.add_aggregator(OutputColumnId.from_grounding(GroundingKey.make_column_grounding("cars_data", "Horsepower")), "max"),
                         OutputColumnId.from_grounding(GroundingKey.make_column_grounding("car_names", "Make"))])
        correct_sparql_query.query_has_superlative = True

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #  1: SELECT['models']
        #  2: FILTER['#1', 'with 3 cylinders']
        #  3: PROJECT['horsepowers of #REF', '#2']
        #  4: AGGREGATE['max', '#3']
        #  5: COMPARATIVE['#2', '#3', 'is #4']
        #  6: PROJECT['the make of #REF', '#5']
        #  7: UNION['#4', '#6']

        grounding = {}
        grounding[GroundingIndex(0,0,"models")] = GroundingKey.make_table_grounding("car_names")
        # WRONG grounding:
        # grounding[GroundingIndex(0,0,"models")] = GroundingKey.make_column_grounding("car_names", "Model")
        grounding[GroundingIndex(1,1,"with 3 cylinders")] = GroundingKey.make_comparative_grounding("=", "3", GroundingKey.make_column_grounding("cars_data", "Cylinders"))
        grounding[GroundingIndex(2,0,"horsepowers of #REF")] = GroundingKey.make_column_grounding("cars_data", "Horsepower") # GroundingKey.make_column_grounding("cars_data", "Weight")
        grounding[GroundingIndex(4,2,"is #4")] = GroundingKey.make_comparative_grounding("=", "#4")
        grounding[GroundingIndex(5,0,"the make of #REF")] = GroundingKey.make_column_grounding("car_names", "Make")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev_comparare_to_sql(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 132
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: What is the largest amount of horsepower for the models with 3 cylinders and what make is it ?
        # sql_query:
        # SELECT T2.horsepower ,  T1.Make
        # FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id
        # WHERE T2.cylinders  =  3
        # ORDER BY T2.horsepower DESC LIMIT 1;

        # Substituting this query with "ORDER BY T2.horsepower DESC LIMIT 1" to argmax constructions
        sql_query = textwrap.dedent("""\
            SELECT T2.horsepower ,  T1.Make
            FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id
            WHERE T2.cylinders  =  3
            AND
            T2.horsepower = (
                SELECT max(T2.horsepower)
            FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id
            WHERE T2.cylinders  =  3
            )
            """)

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?max_2 ?Make
            WHERE
            {
              {
                SELECT ?car_names_1
                WHERE
                {
                  {
                    SELECT ?car_names_1
                    WHERE
                    {
                      ?cars_data_2 arc:cars_data:Id:car_names:MakeId ?car_names_1.
                      ?cars_data_2 arc:cars_data:Cylinders ?Cylinders_1.
                      FILTER(?Cylinders_1 = 3).
                    }
                    GROUP BY ?car_names_1
                  }
                  ?cars_data_3 arc:cars_data:Id:car_names:MakeId ?car_names_1.
                  ?cars_data_3 arc:cars_data:Horsepower ?Horsepower_1.
                  {
                    SELECT (max(?Horsepower_2) AS ?max_1)
                    WHERE
                    {
                      {
                        SELECT ?car_names_2
                        WHERE
                        {
                          ?cars_data_4 arc:cars_data:Id:car_names:MakeId ?car_names_2.
                          ?cars_data_4 arc:cars_data:Cylinders ?Cylinders_2.
                          FILTER(?Cylinders_2 = 3).
                        }
                        GROUP BY ?car_names_2
                      }
                      ?cars_data_5 arc:cars_data:Id:car_names:MakeId ?car_names_2.
                      ?cars_data_5 arc:cars_data:Horsepower ?Horsepower_2.
                    }
                  }
                  FILTER(?Horsepower_1 = ?max_1).
                }
                GROUP BY ?car_names_1
              }
              ?car_names_1 arc:car_names:Make ?Make.
              {
                SELECT (max(?Horsepower_3) AS ?max_2)
                WHERE
                {
                  {
                    SELECT ?car_names_3
                    WHERE
                    {
                      ?cars_data_6 arc:cars_data:Id:car_names:MakeId ?car_names_3.
                      ?cars_data_6 arc:cars_data:Cylinders ?Cylinders_3.
                      FILTER(?Cylinders_3 = 3).
                    }
                    GROUP BY ?car_names_3
                  }
                  ?cars_data_7 arc:cars_data:Id:car_names:MakeId ?car_names_3.
                  ?cars_data_7 arc:cars_data:Horsepower ?Horsepower_3.
                }
              }
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.add_aggregator(OutputColumnId.from_grounding(GroundingKey.make_column_grounding("cars_data", "Horsepower")), "max"),
                         OutputColumnId.from_grounding(GroundingKey.make_column_grounding("car_names", "Make"))])
        correct_sparql_query.query_has_superlative = True

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #  1: SELECT['models']
        #  2: FILTER['#1', 'with 3 cylinders']
        #  3: PROJECT['horsepowers of #REF', '#2']
        #  4: AGGREGATE['max', '#3']
        #  5: COMPARATIVE['#2', '#3', 'is #4']
        #  6: PROJECT['the make of #REF', '#5']
        #  7: UNION['#4', '#6']

        grounding = {}
        grounding[GroundingIndex(0,0,"models")] = GroundingKey.make_table_grounding("car_names")
        # WRONG grounding:
        # grounding[GroundingIndex(0,0,"models")] = GroundingKey.make_column_grounding("car_names", "Model")
        grounding[GroundingIndex(1,1,"with 3 cylinders")] = GroundingKey.make_comparative_grounding("=", "3", GroundingKey.make_column_grounding("cars_data", "Cylinders"))
        grounding[GroundingIndex(2,0,"horsepowers of #REF")] = GroundingKey.make_column_grounding("cars_data", "Horsepower") # GroundingKey.make_column_grounding("cars_data", "Weight")
        grounding[GroundingIndex(4,2,"is #4")] = GroundingKey.make_comparative_grounding("=", "#4")
        grounding[GroundingIndex(5,0,"the make of #REF")] = GroundingKey.make_column_grounding("car_names", "Make")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


# class TestSpiderDev135(unittest.TestCase):

#     @timeout(ONE_TEST_TIMEOUT)
#     def test_spider_dev(self):
#         """Test an entry from spider dataset
#         """
#         split_name = "dev"
#         i_query = 135
#         db_id = get_db_id(split_name, i_query)

#         rdf_graph, schema = get_graph_and_schema(split_name, db_id)

#         sql_query = get_sql_query(split_name, i_query)
#         # question: What is the average horsepower of the cars before 1980?
#         # sql_query: SELECT avg(horsepower) FROM CARS_DATA WHERE YEAR  <  1980;

#         # CAUTION! the column to be averaged has "null" in it - SQL substitutes 0 instead of "null"
#         # Is it even a proper NULL or just text?
#         # According to https://www.sqlservercentral.com/articles/gotcha-sql-aggregate-functions-and-null
#         # SQL is supposed to ignore NULL, but smth else happens

#         # correct_sparql_query = textwrap.dedent("""\
#         #     SELECT (avg(?Horsepower) AS ?avg)
#         #     WHERE
#         #     {
#         #     ?cars_data arc:cars_data:Id ?cars_data.
#         #     ?cars_data arc:cars_data:Year ?Year.
#         #     FILTER(?Year < 1980.0).
#         #     ?cars_data arc:cars_data:Horsepower ?Horsepower.
#         #     }""")

#         qdmr = get_qdmr_from_break(split_name, i_query)
#         # break_program:
#         #     SELECT['cars']
#         #     FILTER['#1', 'before 1980']
#         #     PROJECT['horsepower of #REF', '#2']
#         #     AGGREGATE['avg', '#3'

#         grounding = {}
#         grounding[GroundingIndex(0,0,"cars")] = GroundingKey.make_table_grounding("cars_data")
#         grounding[GroundingIndex(1,1,"before 1980")] = GroundingKey.make_comparative_grounding("<", "1980", GroundingKey.make_column_grounding("cars_data", "Year"))
#         grounding[GroundingIndex(2,0,"horsepower of #REF")] = GroundingKey.make_column_grounding("cars_data", "Horsepower")

#         sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

#         result_correct = QueryResult.execute_query_sql(sql_query, schema)
#         result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
#         equal, message = result.is_equal_to(result_correct,
#                                     require_column_order=True,
#                                     require_row_order=False,
#                                     return_message=True)
#         self.assertTrue(equal, message)


class TestSpiderDev138(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 138
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: What is the average edispl for all volvos?
        # sql_query: SELECT avg(T2.edispl) FROM CAR_NAMES AS T1 JOIN CARS_DATA AS T2 ON T1.MakeId  =  T2.Id WHERE T1.Model  =  'volvo';

        correct_sparql_query = textwrap.dedent("""\
            SELECT (avg(?Edispl) AS ?avg)
            WHERE
            {
                ?car_names arc:car_names:Model ?Model.
                FILTER(?Model = key:car_names:Model:volvo).
                ?cars_data arc:cars_data:Id:car_names:MakeId ?car_names.
                ?cars_data arc:cars_data:Edispl ?Edispl.
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['volvos']
        #     PROJECT['edispl of #REF', '#1']
        #     AGGREGATE['avg', '#2']

        grounding = {}
        grounding[GroundingIndex(0,0,"volvos")] = GroundingKey.make_value_grounding("car_names", "Model", "volvo")
        grounding[GroundingIndex(1,0,"edispl of #REF")] = GroundingKey.make_column_grounding("cars_data", "Edispl")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev141(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 141
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Which model has the most version(make) of cars?
        # sql_query:
        # SELECT Model
        # FROM CAR_NAMES
        # GROUP BY Model
        # ORDER BY count(*) DESC LIMIT 1;

        correct_sparql_query_long = textwrap.dedent("""\
            SELECT ?Model_2
            WHERE
            {
              {
                SELECT (max(?count) AS ?max)
                WHERE
                {
                  {
                    SELECT ?Model_1 (count(?Make) AS ?count)
                    WHERE
                    {
                      ?model_list arc:model_list:Model ?Model_1.
                      ?Model arc:car_names:Model:model_list:Model ?Model_1.
                      ?car_names arc:car_names:Model ?Model.
                      ?car_names arc:car_names:Make ?Make.
                    }
                    GROUP BY ?Model_1
                  }
                }
              }
              ?model_list_1 arc:model_list:Model ?Model_2.
              {
                SELECT ?Model_2 (count(?Make_1) AS ?count_1)
                WHERE
                {
                  ?model_list_2 arc:model_list:Model ?Model_2.
                  ?Model_3 arc:car_names:Model:model_list:Model ?Model_2.
                  ?car_names_1 arc:car_names:Model ?Model_3.
                  ?car_names_1 arc:car_names:Make ?Make_1.
                }
                GROUP BY ?Model_2
              }
              FILTER(?count_1 = ?max).
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['cars']
        #     SELECT['models']
        #     PROJECT['version of #REF', '#1']
        #     GROUP['count', '#3', '#2']
        #     SUPERLATIVE['max', '#2', '#4']

        grounding = {}
        grounding[GroundingIndex(0,0,"cars")] = GroundingKey.make_table_grounding("model_list")
        # Have corresponding columns related as foreign key: substitution to simplify
        # grounding[GroundingIndex(1,0,"models")] = GroundingKey.make_column_grounding("model_list", "Model")
        grounding[GroundingIndex(1,0,"models")] = GroundingKey.make_column_grounding("car_names", "Model")
        grounding[GroundingIndex(2,0,"version of #REF")] = GroundingKey.make_column_grounding("car_names", "Make")
        grounding[GroundingIndex(4,0,"max")] = GroundingKey.make_comparative_grounding("max", None)

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev_foreign_key_output_match(self):
        """Test an entry from spider dataset: columns is the output should be matched as foreign keys
        """
        split_name = "dev"
        i_query = 141
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Which model has the most version(make) of cars?
        # sql_query:
        # SELECT Model
        # FROM CAR_NAMES
        # GROUP BY Model
        # ORDER BY count(*) DESC LIMIT 1;

        correct_sparql_query_long = textwrap.dedent("""\
            SELECT ?Model_2
            WHERE
            {
              {
                SELECT (max(?count) AS ?max)
                WHERE
                {
                  {
                    SELECT ?Model_1 (count(?Make) AS ?count)
                    WHERE
                    {
                      ?model_list arc:model_list:Model ?Model_1.
                      ?Model arc:car_names:Model:model_list:Model ?Model_1.
                      ?car_names arc:car_names:Model ?Model.
                      ?car_names arc:car_names:Make ?Make.
                    }
                    GROUP BY ?Model_1
                  }
                }
              }
              ?model_list_1 arc:model_list:Model ?Model_2.
              {
                SELECT ?Model_2 (count(?Make_1) AS ?count_1)
                WHERE
                {
                  ?model_list_2 arc:model_list:Model ?Model_2.
                  ?Model_3 arc:car_names:Model:model_list:Model ?Model_2.
                  ?car_names_1 arc:car_names:Model ?Model_3.
                  ?car_names_1 arc:car_names:Make ?Make_1.
                }
                GROUP BY ?Model_2
              }
              FILTER(?count_1 = ?max).
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['cars']
        #     SELECT['models']
        #     PROJECT['version of #REF', '#1']
        #     GROUP['count', '#3', '#2']
        #     SUPERLATIVE['max', '#2', '#4']

        grounding = {}
        grounding[GroundingIndex(0,0,"cars")] = GroundingKey.make_table_grounding("model_list")
        # Have corresponding columns related as foreign key - need to process this in a special way
        grounding[GroundingIndex(1,0,"models")] = GroundingKey.make_column_grounding("model_list", "Model")
        grounding[GroundingIndex(2,0,"version of #REF")] = GroundingKey.make_column_grounding("car_names", "Make")
        grounding[GroundingIndex(4,0,"max")] = GroundingKey.make_comparative_grounding("max", None)

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev144(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 144
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: What is the number of cars with more than 4 cylinders?
        # sql_query: SELECT count(*) FROM CARS_DATA WHERE Cylinders  >  4

        correct_sparql_query = textwrap.dedent("""\
            SELECT (count(?cars) AS ?count)
            WHERE
            {
              ?cars arc:cars_data:Cylinders ?Cylinders.
              FILTER(?Cylinders > 4).
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.add_aggregator(OutputColumnId.from_grounding(GroundingKey.make_table_grounding("cars_data"), schema), "count")])

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['cars']
        #     PROJECT['cylinders of #REF', '#1']
        #     GROUP['count', '#2', '#1']
        #     COMPARATIVE['#1', '#3', 'is higher than 4']
        #     AGGREGATE['count', '#4']
        #
        # Note: this QDMR does not correspond well to the scheme - column Cylinders actually has the number of cyllinders
        # we are going to fix this using the grounding of the group op

        grounding = {}
        grounding[GroundingIndex(0,0,"cars")] = GroundingKey.make_table_grounding("cars_data")
        grounding[GroundingIndex(1,0,"cylinders of #REF")] = GroundingKey.make_column_grounding("cars_data", "Cylinders")
        grounding[GroundingIndex(2,0,"count")] = GroundingKey.make_column_grounding("cars_data", "Cylinders")
        grounding[GroundingIndex(3,2,"is higher than 4")] = GroundingKey.make_comparative_grounding(">", "4")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev151(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 151
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Which model has the most version(make) of cars?
        # sql_query:
        # SELECT DISTINCT T2.Model
        # FROM CAR_NAMES AS T1 JOIN MODEL_LIST AS T2 ON T1.Model  =  T2.Model JOIN CAR_MAKERS AS T3 ON T2.Maker  =  T3.Id JOIN CARS_DATA AS T4 ON T1.MakeId  =  T4.Id
        # WHERE T3.FullName  =  'General Motors' OR T4.weight  >  3500;
        # question: Which distinctive models are produced by maker with the full name General Motors or weighing more than 3500?

        # CAUTION: SQL is wrong on the current database!

        correct_sparql_query = textwrap.dedent("""\
            SELECT DISTINCT ?Model_2
            WHERE
            {
              {
                SELECT ?Model_2
                WHERE
                {
                  ?model_list_1 arc:model_list:Model ?Model_2.
                  ?model_list_1 arc:model_list:Maker ?Maker_1.
                  ?Maker_1 arc:model_list:Maker:car_makers:Id ?car_makers_1.
                  ?car_makers_1 arc:car_makers:FullName ?FullName_1.
                  FILTER(?FullName_1 = "General Motors").
                }
                GROUP BY ?Model_2
              }
              UNION
              {
                SELECT ?Model_2
                WHERE
                {
                  ?Model_4 arc:car_names:Model:model_list:Model ?Model_2.
                  ?car_names_1 arc:car_names:Model ?Model_4.
                  ?cars_data_1 arc:cars_data:Id:car_names:MakeId ?car_names_1.
                  ?cars_data_1 arc:cars_data:Weight ?Weight_1.
                  FILTER(?Weight_1 > 3500).
                }
                GROUP BY ?Model_2
              }
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("model_list", "Model"))])

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['distinctive models']
        #     FILTER['#1', 'which are produced by maker with the full name General Motors']
        #     FILTER['#1', 'weighing more than 3500']
        #     UNION['#2', '#3']

        grounding = {}
        grounding[GroundingIndex(0,0,"distinctive models")] = GroundingKey.make_column_grounding("model_list", "Model")
        grounding[GroundingIndex(1,1,"which are produced by maker with the full name General Motors")] = GroundingKey.make_value_grounding("car_makers", "FullName", "General Motors")
        grounding[GroundingIndex(2,1,"weighing more than 3500")] = GroundingKey.make_comparative_grounding(">", "3500", GroundingKey.make_column_grounding("cars_data", "Weight"))
        grounding["distinct"] = ["#1"]

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev157(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 157
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: For model volvo, how many cylinders does the car with the least accelerate have?
        # sql_query:
        # SELECT T1.cylinders FROM CARS_DATA AS T1 JOIN CAR_NAMES AS T2 ON T1.Id  =  T2.MakeId WHERE T2.Model  =  'volvo' ORDER BY T1.accelerate ASC LIMIT 1;

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Cylinders
            WHERE
            {
              {
                SELECT ?cars_data_1
                WHERE
                {
                  {
                    SELECT (min(?Accelerate) AS ?min)
                    WHERE
                    {
                      {
                        SELECT ?cars_data
                        WHERE
                        {
                          ?cars_data arc:cars_data:Id:car_names:MakeId ?car_names.
                          ?car_names arc:car_names:Model ?Model.
                          FILTER(?Model = key:car_names:Model:volvo).
                        }
                        GROUP BY ?cars_data
                      }
                      ?cars_data arc:cars_data:Accelerate ?Accelerate.
                    }
                  }
                  {
                    SELECT ?cars_data_1
                    WHERE
                    {
                      ?cars_data_1 arc:cars_data:Id:car_names:MakeId ?car_names_1.
                      ?car_names_1 arc:car_names:Model ?Model_1.
                      FILTER(?Model_1 = key:car_names:Model:volvo).
                    }
                    GROUP BY ?cars_data_1
                  }
                  ?cars_data_1 arc:cars_data:Accelerate ?Accelerate_1.
                  FILTER(?Accelerate_1 = ?min).
                }
                GROUP BY ?cars_data_1
              }
              ?cars_data_1 arc:cars_data:Cylinders ?Cylinders.
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        # #1: SELECT['cars']
        # #2: PROJECT['models of #REF', '#1']
        # #3: COMPARATIVE['#1', '#2', 'is volvo']
        # #4: PROJECT['accelerate of #REF', '#3']
        # #5: SUPERLATIVE['min', '#3', '#4']
        # #6: PROJECT['cylinders of #REF', '#5']
        # #7: AGGREGATE['count', '#6']

        grounding = {}
        grounding[GroundingIndex(0,0,"cars")] = GroundingKey.make_table_grounding("cars_data")
        grounding[GroundingIndex(1,0,'models of #REF')] = GroundingKey.make_column_grounding("car_names", "Model")
        grounding[GroundingIndex(2,2,"is volvo")] = GroundingKey.make_comparative_grounding("=", "volvo", GroundingKey.make_column_grounding("car_names", "Model"))
        grounding[GroundingIndex(3,0,'accelerate of #REF')] = GroundingKey.make_column_grounding("cars_data", "Accelerate")
        grounding[GroundingIndex(4,0,"min")] = GroundingKey.make_comparative_grounding("min", "None")
        grounding[GroundingIndex(5,0,"'cylinders of #REF'")] = GroundingKey.make_column_grounding("cars_data", "Cylinders")
        grounding[GroundingIndex(6,0,"count")] = GroundingKey.make_column_grounding("cars_data", "Cylinders")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev159(unittest.TestCase):

    # @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 159
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: How many cars have a larger accelerate than the car with the largest horsepower?
        # sql_query:
        # SELECT COUNT(*) FROM CARS_DATA WHERE Accelerate  >  ( SELECT Accelerate FROM CARS_DATA ORDER BY Horsepower DESC LIMIT 1 );

        # the query produces incorrect results because the column Horsepower is of type TEXT and has null written there
        # to test the feature I'm swapping Horsepower to weight

        correct_sparql_query = textwrap.dedent("""\
            SELECT (count(?cars_data_2) AS ?count)
            WHERE
            {
                {
                    SELECT (max(?Weight) AS ?max)
                    WHERE
                    {
                        ?cars_data_1 arc:cars_data:Weight ?Weight.
                    }
                }
                ?cars_data arc:cars_data:Weight ?Weight_1.
                FILTER(?Weight_1 = ?max).
                ?cars_data arc:cars_data:Accelerate ?Accelerate.
                ?cars_data_2 arc:cars_data:Accelerate ?Accelerate_1.
                FILTER(?Accelerate_1 > ?Accelerate).
            }""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.add_aggregator(OutputColumnId.from_grounding(GroundingKey.make_table_grounding("cars_data"), schema), "count")])
        correct_sparql_query.query_has_superlative = True

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['cars']
        #     PROJECT['horsepower of #REF', '#1']
        #     SUPERLATIVE['max', '#1', '#2']
        #     PROJECT['accelerate of #REF', '#1']
        #     PROJECT['accelerate of #REF', '#3']
        #     COMPARATIVE['#1', '#4', 'is higher than #5']
        #     AGGREGATE['count', '#6']

        grounding = {}
        grounding[GroundingIndex(0,0,"cars")] = GroundingKey.make_table_grounding("cars_data")
        grounding[GroundingIndex(1,0,"horsepower of #REF")] = GroundingKey.make_column_grounding("cars_data", "Weight")
        grounding[GroundingIndex(2,0,"max")] = GroundingKey.make_comparative_grounding("max", None)
        grounding[GroundingIndex(3,0,"accelerate of #REF")] = GroundingKey.make_column_grounding("cars_data", "Accelerate")
        grounding[GroundingIndex(4,0,"accelerate of #REF")] = GroundingKey.make_column_grounding("cars_data", "Accelerate")
        grounding[GroundingIndex(5,2,"is higher than #5")] = GroundingKey.make_comparative_grounding(">", "#5")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev170(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 170
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?country_name
            WHERE
            {
                ?countries arc:countries:CountryName ?country_name.
                MINUS{
                    ?car_makers arc:car_makers:Country ?car_maker_country.
                    ?car_maker_country arc:car_makers:Country:countries:CountryId ?countries.
                }
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['cars']
        #     FILTER['#1', 'that had 8 cylinders']
        #     FILTER['#1', 'that were produced before 1980']
        #     UNION['#2', '#3']
        #     PROJECT['mpg of #REF', '#4']
        #     AGGREGATE['max', '#5']

        grounding = {}
        grounding[GroundingIndex(0,0,"cars")] = GroundingKey.make_table_grounding("cars_data")
        grounding[GroundingIndex(1,1,"that had 8 cylinders")] = GroundingKey.make_comparative_grounding("=", "8", GroundingKey.make_column_grounding("cars_data", "Cylinders"))
        grounding[GroundingIndex(2,1,"that were produced before 1980")] = GroundingKey.make_comparative_grounding("<", "1980", GroundingKey.make_column_grounding("cars_data", "Year"))
        grounding[GroundingIndex(4,0,"mpg of #REF")] = GroundingKey.make_column_grounding("cars_data", "MPG")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev173(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 173
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?country_name
            WHERE
            {
                ?countries arc:countries:CountryName ?country_name.
                MINUS{
                    ?car_makers arc:car_makers:Country ?car_maker_country.
                    ?car_maker_country arc:car_makers:Country:countries:CountryId ?countries.
                }
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['countries']
        #     FILTER['#1', 'with car maker']
        #     DISCARD['#1', '#2']
        #     PROJECT['name of #REF', '#3']

        grounding = {}
        grounding[GroundingIndex(0,0,"countries")] = GroundingKey.make_table_grounding("countries")
        grounding[GroundingIndex(1,1,"with car maker")] = GroundingKey.make_table_grounding("car_makers")
        grounding[GroundingIndex(3,0,"name of #REF")] = GroundingKey.make_column_grounding("countries", "CountryName")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev175(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 175
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # SELECT T1.Id ,  T1.Maker
        # FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker
        # GROUP BY T1.Id
        # HAVING count(*)  >=  2
        # INTERSECT
        # SELECT T1.Id ,  T1.Maker
        # FROM CAR_MAKERS AS T1 JOIN MODEL_LIST AS T2 ON T1.Id  =  T2.Maker JOIN CAR_NAMES AS T3 ON T2.model  =  T3.model
        # GROUP BY T1.Id
        # HAVING count(*)  >  3

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?car_makers ?Maker
            WHERE
            {
              ?car_makers arc:car_makers:Maker ?Maker.
              {
                SELECT ?Maker (count(?model_list) AS ?count)
                WHERE
                {
                  ?car_makers_1 arc:car_makers:Maker ?Maker.
                  ?Maker_2 arc:model_list:Maker:car_makers:Id ?car_makers_1.
                  ?model_list arc:model_list:Maker ?Maker_2.
                }
                GROUP BY ?Maker
              }
              FILTER(?count >= 2).
              {
                SELECT ?Maker (count(?car_names) AS ?count_1)
                WHERE
                {
                  ?car_makers_2 arc:car_makers:Maker ?Maker.
                  ?Maker_4 arc:model_list:Maker:car_makers:Id ?car_makers_2.
                  ?model_list_1 arc:model_list:Maker ?Maker_4.
                  ?model_list_1 arc:model_list:Model ?Model_1.
                  ?Model arc:car_names:Model:model_list:Model ?Model_1.
                  ?car_names arc:car_names:Model ?Model.
                }
                GROUP BY ?Maker
              }
              FILTER(?count_1 > 3).
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        qdmr.args[-1] = ["#9", "#8"]
        # break_program:
        #     SELECT['car makers']
        #     PROJECT['the models that #REF produce', '#1']
        #     GROUP['count', '#2', '#1']
        #     COMPARATIVE['#1', '#3', 'is at least 2']
        #     PROJECT['the car makers that #REF produce', '#1']
        #     GROUP['count', '#5', '#1']
        #     COMPARATIVE['#1', '#6', 'is more than 3']
        #     INTERSECTION['#1', '#4', '#7']
        #     PROJECT['the ids of #REF', '#8']
        #     UNION['#9', '#8']

        grounding = {}
        grounding[GroundingIndex(0,0,"car makers")] = GroundingKey.make_column_grounding("car_makers", "Maker")
        grounding[GroundingIndex(1,0,"the models that #REF produce")] = GroundingKey.make_table_grounding("model_list")
        grounding[GroundingIndex(3,2,"is at least 2")] = GroundingKey.make_comparative_grounding(">=", "2")
        grounding[GroundingIndex(4,0,"the car makers that #REF produce")] = GroundingKey.make_column_grounding("car_names", "MakeId")
        grounding[GroundingIndex(6,2,"is more than 3")] = GroundingKey.make_comparative_grounding(">", "3")
        grounding[GroundingIndex(8,0,"the ids of #REF")] = GroundingKey.make_column_grounding("car_makers", "Id")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev183(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 183
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # Question: List all airline names and their abbreviations in ""USA"
        # SELECT Airline ,  Abbreviation FROM AIRLINES WHERE Country  =  "USA"

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Airline ?Abbreviation
            WHERE
            {
                ?airlines arc:airlines:Airline ?Airline.
                ?airlines arc:airlines:Abbreviation ?Abbreviation.
                ?airlines arc:airlines:Country ?Country.
                FILTER(?Country = "USA")
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['airlines']
        #     PROJECT['names of #REF', '#1']
        #     PROJECT['abbreviations of #REF', '#2']
        #     UNION['#2', '#3']
        #     FILTER['#4', 'in USA']

        grounding = {}
        grounding[GroundingIndex(0,0,"airlines")] = GroundingKey.make_table_grounding("airlines")
        grounding[GroundingIndex(1,0,"names of #REF")] = GroundingKey.make_column_grounding("airlines", "Airline")
        grounding[GroundingIndex(2,0,"abbreviations of #REF")] = GroundingKey.make_column_grounding("airlines", "Abbreviation")
        grounding[GroundingIndex(4,1,"in USA")] = GroundingKey.make_comparative_grounding("=", "USA", GroundingKey.make_column_grounding("airlines", "Country"))

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev_switch_filter_to_comparative(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 183
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # Question: List all airline names and their abbreviations in ""USA"
        # SELECT Airline ,  Abbreviation FROM AIRLINES WHERE Country  =  "USA"

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Airline ?Abbreviation
            WHERE
            {
                ?airlines arc:airlines:Airline ?Airline.
                ?airlines arc:airlines:Abbreviation ?Abbreviation.
                ?airlines arc:airlines:Country ?Country.
                FILTER(?Country = "USA")
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        qdmr.ops[-1] = "comparative"
        qdmr.args[-1] = ["#4", "#4", "in USA"]
        # break_program:
        #     SELECT['airlines']
        #     PROJECT['names of #REF', '#1']
        #     PROJECT['abbreviations of #REF', '#2']
        #     UNION['#2', '#3']
        #     COMPARATIVE['#4', '#4','in USA']

        grounding = {}
        grounding[GroundingIndex(0,0,"airlines")] = GroundingKey.make_table_grounding("airlines")
        grounding[GroundingIndex(1,0,"names of #REF")] = GroundingKey.make_column_grounding("airlines", "Airline")
        grounding[GroundingIndex(2,0,"abbreviations of #REF")] = GroundingKey.make_column_grounding("airlines", "Abbreviation")
        grounding[GroundingIndex(4,2,"in USA")] = GroundingKey.make_comparative_grounding("=", "USA", GroundingKey.make_column_grounding("airlines", "Country"))

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev237(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 237
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Find all airlines that have flights from both airports 'APG' and 'CVO'.
        # SQL:
        # SELECT T1.Airline FROM AIRLINES AS T1 JOIN FLIGHTS AS T2 ON T1.uid  =  T2.Airline WHERE T2.SourceAirport  =  "APG"
        # INTERSECT
        # SELECT T1.Airline FROM AIRLINES AS T1 JOIN FLIGHTS AS T2 ON T1.uid  =  T2.Airline WHERE T2.SourceAirport  =  "CVO"

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        # #1: SELECT['flights']
        # #2: FILTER['#1', 'from the airport APG']
        # #3: FILTER['#1', 'from the airport CVO']
        # #4: PROJECT['airlines of #REF', '#1']
        # #5: PROJECT['airlines of #REF', '#2']
        # #6: PROJECT['airlines of #REF', '#3']
        # #7: INTERSECTION['#4', '#5', '#6']

        grounding = {}
        grounding[GroundingIndex(0,0,"flights")] = GroundingKey.make_table_grounding("flights")
        grounding[GroundingIndex(1,1,"from the airport APG")] = GroundingKey.make_comparative_grounding("=", "APG", GroundingKey.make_column_grounding("flights", "SourceAirport"))
        grounding[GroundingIndex(2,1,"from the airport CVO")] = GroundingKey.make_comparative_grounding("=", "CVO", GroundingKey.make_column_grounding("flights", "SourceAirport"))
        grounding[GroundingIndex(3,0,"airlines of #REF")] = GroundingKey.make_column_grounding("airlines", "Airline")
        grounding[GroundingIndex(4,0,"airlines of #REF")] = GroundingKey.make_table_grounding("airlines")
        grounding[GroundingIndex(5,0,"airlines of #REF")] = GroundingKey.make_table_grounding("airlines")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev261(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 261
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Sort employee names by their age in ascending order.
        # SQL: SELECT name FROM employee ORDER BY age

        # CAUTION: ages of some employees are the same, so there are multiple correct results
        # SPARQL and SQL sort differently
        # for the sake of this test, we will sort by ID instead

        correct_sparql_query = textwrap.dedent("""\
        SELECT ?Name
        {
            ?employee arc:employee:Name ?Name.
            ?employee arc:employee:Employee_ID ?Age.
        }
        ORDER BY ASC(?Age)""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("employee", "Name"))])

        qdmr = get_qdmr_from_break(split_name, i_query)
        qdmr.args[3] = ["#2", "#3"]
        # break_program:
        #   SELECT['employees']
        #   PROJECT['names of #REF', '#1']
        #   PROJECT['ages of #REF', '#1']
        #   SORT['#2', '#3']

        grounding = {}
        grounding[GroundingIndex(0, 0, "employees")] = GroundingKey.make_table_grounding("employee")
        grounding[GroundingIndex(1, 0, "names of #REF")] = GroundingKey.make_column_grounding("employee", "Name")
        grounding[GroundingIndex(2, 0, "ages of #REF")] = GroundingKey.make_column_grounding("employee", "Employee_ID")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=True,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev_non_deterministic_sort(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 261
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Sort employee names by their age in ascending order.
        # SQL: SELECT name FROM employee ORDER BY age

        # CAUTION: ages of some employees are the same, so there are multiple correct results
        # SPARQL and SQL sort differently
        # This should be dealed with in the metric

        correct_sparql_query = textwrap.dedent("""\
        SELECT ?Name
        {
            ?employee arc:employee:Name ?Name.
            ?employee arc:employee:Age ?Age.
        }
        ORDER BY ASC(?Age)""")
        # correct_sparql_query = QueryToRdf(query=correct_sparql_query,
        #     output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("employee", "Name")),
        #                  OutputColumnId.from_grounding(GroundingKey.make_column_grounding("employee", "Age"))])

        qdmr = get_qdmr_from_break(split_name, i_query)
        qdmr.args[3] = ["#2", "#3"]
        # break_program:
        #   SELECT['employees']
        #   PROJECT['names of #REF', '#1']
        #   PROJECT['ages of #REF', '#1']
        #   SORT['#2', '#3']

        grounding = {}
        grounding[GroundingIndex(0, 0, "employees")] = GroundingKey.make_table_grounding("employee")
        grounding[GroundingIndex(1, 0, "names of #REF")] = GroundingKey.make_column_grounding("employee", "Name")
        grounding[GroundingIndex(2, 0, "ages of #REF")] = GroundingKey.make_column_grounding("employee", "Age")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=True,
                                    return_message=True,
                                    schema=schema)
        self.assertTrue(equal, message)

        # in this test, we have three people with the same age:
        # ('Lee Mears', '29')
        # ('Matt Stevens', '29')
        # ('Tim Payne', '29')

        # manually switch two confusing entries to ensure thes test never passes accidentally
        a_data = None
        b_data = None
        for i_ in range(len(result_correct.data)):
            if result_correct.data[i_][0] == "Tim Payne":
                a_data = i_
            if result_correct.data[i_][0] == "Matt Stevens":
                b_data = i_
        self.assertTrue(a_data is not None and b_data is not None, "Something is wrong with the test, could not find entries to swap")
        # swap and test again
        result_correct.data[a_data], result_correct.data[b_data] = result_correct.data[b_data], result_correct.data[a_data]
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=True,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev_sort_by_tuple(self):
        """Test an entry from spider dataset - modify to test sorting item w.r.t. several keys
        """
        split_name = "dev"
        i_query = 261
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Sort employee names by their age in ascending order.
        # SQL: SELECT name FROM employee ORDER BY age

        # CAUTION: ages of some employees are the same
        # SPARQL and SQL sort differently

        correct_sparql_query = textwrap.dedent("""\
        SELECT ?Name ?Age
        {
            ?employee arc:employee:Name ?Name.
            ?employee arc:employee:Age ?Age.
        }
        ORDER BY ASC(?Age) ASC(?Name)""")
        correct_sparql_query = QueryToRdf(query=correct_sparql_query,
            output_cols=[OutputColumnId.from_grounding(GroundingKey.make_column_grounding("employee", "Name")),
                         OutputColumnId.from_grounding(GroundingKey.make_column_grounding("employee", "Age"))])

        qdmr = QdmrInstance(["select", "project", "project", "union", "union", "sort"],
            [["employees"],
                ['names of #REF', '#1'],
                ['ages of #REF', '#1'],
                ['#2', '#3'],
                ['#3', '#2'],
                ['#4', '#5']
            ])
        # break_program:
        #   SELECT['employees']
        #   PROJECT['names of #REF', '#1']
        #   PROJECT['ages of #REF', '#1']
        #   UNION['#2', '#3']
        #   UNION['#3', '#2']
        #   SORT['#4', '#5']

        grounding = {}
        grounding[GroundingIndex(0, 0, "employees")] = GroundingKey.make_table_grounding("employee")
        grounding[GroundingIndex(1, 0, "names of #REF")] = GroundingKey.make_column_grounding("employee", "Name")
        grounding[GroundingIndex(2, 0, "ages of #REF")] = GroundingKey.make_column_grounding("employee", "Age")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_to_rdf(correct_sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=True,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev266(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 266
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Find the cities that have more than one employee under age 30.
        # SQL: select city from employee where age  <  30 group by city having count(*)  >  1

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['cities']
        #     PROJECT['employees of #REF', '#1']
        #     PROJECT['ages of #REF', '#2']
        #     COMPARATIVE['#2', '#3', 'is under 30']
        #     GROUP['count', '#4', '#1']
        #     COMPARATIVE['#1', '#5', 'is more than one']

        grounding = {}
        grounding[GroundingIndex(0,0,"cities")] = GroundingKey.make_column_grounding("employee", "City")
        grounding[GroundingIndex(1,0,"employees of #REF")] = GroundingKey.make_table_grounding("employee")
        grounding[GroundingIndex(2,0,"ages of #REF")] = GroundingKey.make_column_grounding("employee", "Age")
        grounding[GroundingIndex(3,2,"is under 30")] = GroundingKey.make_comparative_grounding("<", "30")
        grounding[GroundingIndex(5,2,"is more than one")] = GroundingKey.make_comparative_grounding(">", "1")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev353(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 353
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # SQL:
        # SELECT DISTINCT T1.template_type_description
        # FROM Ref_template_types AS T1 JOIN Templates AS T2 ON T1.template_type_code  = T2.template_type_code JOIN Documents AS T3 ON T2.Template_ID  =  T3.template_ID
        # Question: What are the distinct template type descriptions for the templates ever used by any document?

        correct_sparql_query = textwrap.dedent("""\
            SELECT DISTINCT ?Template_Type_Description
            WHERE
            {
              {
                SELECT ?Templates
                WHERE
                {
                  ?Template_ID arc:Documents:Template_ID:Templates:Template_ID ?Templates.
                  ?Documents arc:Documents:Template_ID ?Template_ID.
                }
                GROUP BY ?Templates
              }
              ?Templates arc:Templates:Template_Type_Code ?Template_Type_Code.
              ?Template_Type_Code arc:Templates:Template_Type_Code:Ref_Template_Types:Template_Type_Code ?Ref_Template_Types.
              ?Ref_Template_Types arc:Ref_Template_Types:Template_Type_Description ?Template_Type_Description.
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['templates']
        #     FILTER['#1', 'used by any documents']
        #     PROJECT['type description of #REF', '#2']
        #     FILTER['#3', 'that are distinct']

        grounding = {}
        grounding[GroundingIndex(0,0,"templates")] = GroundingKey.make_table_grounding("Templates")
        grounding[GroundingIndex(1,1,"used by any documents")] = GroundingKey.make_table_grounding("Documents")
        grounding[GroundingIndex(2,0,"type description of #REF")] = GroundingKey.make_column_grounding("Ref_Template_Types", "Template_Type_Description")
        grounding["distinct"] = ["#4"]

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev367(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 367
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # question: Show all document ids and the number of paragraphs in each document. Order by document id.
        # SQL: SELECT document_id ,  count(*) FROM Paragraphs GROUP BY document_id ORDER BY document_id

        qdmr = get_qdmr_from_break(split_name, i_query)
        qdmr.args[-1] = ["#5", "#2", "ASC"]
        # break_program:
        #  #1: SELECT['documents']
        #  #2: PROJECT['document ids of #REF', '#1']
        #  #3: PROJECT['paragraphs of #REF', '#1']
        #  #4: GROUP['count', '#3', '#1']
        #  #5: UNION['#2', '#4']
        #  #6: SORT['#5', '#2', 'ASC']

        grounding = {}
        grounding[GroundingIndex(0,0,"documents")] = GroundingKey.make_table_grounding("Documents")
        grounding[GroundingIndex(1,0,"document ids of #REF")] = GroundingKey.make_column_grounding("Paragraphs", "Document_ID")
        grounding[GroundingIndex(2,0,"paragraphs of #REF")] = GroundingKey.make_table_grounding("Paragraphs")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=True,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev414(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 414
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # SQL:
        # SELECT name ,  Level_of_membership FROM visitor WHERE Level_of_membership  >  4 ORDER BY age DESC
        # Question: Find the name and membership level of the visitors whose membership level is higher than 4, and sort by their age from old to young.

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Name ?Level_of_membership
            WHERE
            {
              ?visitor arc:visitor:Level_of_membership ?Level_of_membership.
              FILTER(?Level_of_membership > 4).
              ?visitor arc:visitor:Name ?Name.
              ?visitor arc:visitor:Age ?Age.
            }
            ORDER BY DESC(?Age)""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        qdmr.args[-1] = ["#7", "#6", "from old to young"]
        # break_program:
        #     SELECT['visitors']
        #     PROJECT['membership levels of #REF', '#1']
        #     COMPARATIVE['#1', '#2', 'is higher than 4']
        #     PROJECT['names of #REF', '#3']
        #     PROJECT['membership levels of #REF', '#3']
        #     PROJECT['ages of #REF', '#3']
        #     UNION['#4', '#5']
        #     SORT['#7', '#6', 'from old to young']

        grounding = {}
        grounding[GroundingIndex(0,0,"visitors")] = GroundingKey.make_table_grounding("visitor")
        grounding[GroundingIndex(1,0,"membership levels of #REF")] = GroundingKey.make_column_grounding("visitor", "Level_of_membership")
        grounding[GroundingIndex(2,2,"is higher than 4")] = GroundingKey.make_comparative_grounding(">", "4")
        grounding[GroundingIndex(3,0,"names of #REF")] = GroundingKey.make_column_grounding("visitor", "Name")
        grounding[GroundingIndex(4,0,"membership levels of #REF")] = GroundingKey.make_column_grounding("visitor", "Level_of_membership")
        grounding[GroundingIndex(5,0,"ages of #REF")] = GroundingKey.make_column_grounding("visitor", "Age")
        grounding[GroundingIndex(7,2,"from old to young")] = GroundingKey.make_sortdir_grounding(ascending=False)

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=True,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderDev426(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 426
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # SQL:
        # SELECT t1.name FROM visitor AS t1 JOIN visit AS t2 ON t1.id  =  t2.visitor_id JOIN museum AS t3 ON t3.Museum_ID  =  t2.Museum_ID
        #   WHERE t3.open_year  <  2009
        # INTERSECT
        # SELECT t1.name FROM visitor AS t1 JOIN visit AS t2 ON t1.id  =  t2.visitor_id JOIN museum AS t3 ON t3.Museum_ID  =  t2.Museum_ID WHERE
        # t3.open_year  >  2011
        # Question: What is the name of the visitor who visited both a museum opened before 2009 and a museum opened after 2011?

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Name ?Level_of_membership
            WHERE
            {
              ?visitor arc:visitor:Level_of_membership ?Level_of_membership.
              FILTER(?Level_of_membership > 4).
              ?visitor arc:visitor:Name ?Name.
              ?visitor arc:visitor:Age ?Age.
            }
            ORDER BY DESC(?Age)""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        # #1: SELECT['museums']
        # #2: FILTER['#1', 'that opened before 2009']
        # #3: FILTER['#1', 'that opened after 2011']
        # #4: PROJECT['the visitor of #REF', '#1']
        # #5: INTERSECTION['#4', '#2', '#3']
        # #6: PROJECT['name of #REF', '#5']

        grounding = {}
        grounding[GroundingIndex(0,0,"museums")] = GroundingKey.make_table_grounding("museum")
        grounding[GroundingIndex(1,1,"that opened before 2009")] = GroundingKey.make_comparative_grounding("<", "2009", GroundingKey.make_column_grounding("museum", "Open_Year"))
        grounding[GroundingIndex(2,1,"that opened after 2011")] = GroundingKey.make_comparative_grounding(">", "2011", GroundingKey.make_column_grounding("museum", "Open_Year"))
        grounding[GroundingIndex(3,0,"the visitor of #REF")] = GroundingKey.make_table_grounding("visitor")
        grounding[GroundingIndex(5,0,"name of #REF")] = GroundingKey.make_column_grounding("visitor", "Name")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev_swap_args(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 426
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # SQL:
        # SELECT t1.name FROM visitor AS t1 JOIN visit AS t2 ON t1.id  =  t2.visitor_id JOIN museum AS t3 ON t3.Museum_ID  =  t2.Museum_ID
        #   WHERE t3.open_year  <  2009
        # INTERSECT
        # SELECT t1.name FROM visitor AS t1 JOIN visit AS t2 ON t1.id  =  t2.visitor_id JOIN museum AS t3 ON t3.Museum_ID  =  t2.Museum_ID WHERE
        # t3.open_year  >  2011
        # Question: What is the name of the visitor who visited both a museum opened before 2009 and a museum opened after 2011?

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Name ?Level_of_membership
            WHERE
            {
              ?visitor arc:visitor:Level_of_membership ?Level_of_membership.
              FILTER(?Level_of_membership > 4).
              ?visitor arc:visitor:Name ?Name.
              ?visitor arc:visitor:Age ?Age.
            }
            ORDER BY DESC(?Age)""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        # #1: SELECT['museums']
        # #2: FILTER['#1', 'that opened before 2009']
        # #3: FILTER['#1', 'that opened after 2011']
        # #4: PROJECT['the visitor of #REF', '#1']
        # #5: INTERSECTION['#4', '#2', '#3']
        # #6: PROJECT['name of #REF', '#5']

        grounding = {}
        grounding[GroundingIndex(0,0,"museums")] = GroundingKey.make_table_grounding("museum")
        grounding[GroundingIndex(1,1,"that opened before 2009")] = GroundingKey.make_comparative_grounding(">", "2011", GroundingKey.make_column_grounding("museum", "Open_Year"))
        grounding[GroundingIndex(2,1,"that opened after 2011")] = GroundingKey.make_comparative_grounding("<", "2009", GroundingKey.make_column_grounding("museum", "Open_Year"))
        grounding[GroundingIndex(3,0,"the visitor of #REF")] = GroundingKey.make_table_grounding("visitor")
        grounding[GroundingIndex(5,0,"name of #REF")] = GroundingKey.make_column_grounding("visitor", "Name")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev_intersection_via_double_filter(self):
        """Test an entry from spider dataset
        """
        split_name = "dev"
        i_query = 426
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # SQL:
        # SELECT t1.name FROM visitor AS t1 JOIN visit AS t2 ON t1.id  =  t2.visitor_id JOIN museum AS t3 ON t3.Museum_ID  =  t2.Museum_ID
        #   WHERE t3.open_year  <  2009
        # INTERSECT
        # SELECT t1.name FROM visitor AS t1 JOIN visit AS t2 ON t1.id  =  t2.visitor_id JOIN museum AS t3 ON t3.Museum_ID  =  t2.Museum_ID WHERE
        # t3.open_year  >  2011
        # Question: What is the name of the visitor who visited both a museum opened before 2009 and a museum opened after 2011?

        correct_sparql_query = textwrap.dedent("""\
            SELECT ?Name
            WHERE
            {
              {
                SELECT ?visitor
                WHERE
                {
                  {
                    SELECT ?visitor
                    WHERE
                    {
                      ?visitor_ID arc:visit:visitor_ID:visitor:ID ?visitor.
                      ?visit arc:visit:visitor_ID ?visitor_ID.
                      ?visit arc:visit:Museum_ID ?Museum_ID.
                      ?Museum_ID arc:visit:Museum_ID:museum:Museum_ID ?museum.
                      ?museum arc:museum:Open_Year ?Open_Year.
                      FILTER(?Open_Year < "2009").
                    }
                    GROUP BY ?visitor
                  }
                  ?visitor_ID_1 arc:visit:visitor_ID:visitor:ID ?visitor.
                  ?visit_1 arc:visit:visitor_ID ?visitor_ID_1.
                  ?visit_1 arc:visit:Museum_ID ?Museum_ID_1.
                  ?Museum_ID_1 arc:visit:Museum_ID:museum:Museum_ID ?museum_1.
                  ?museum_1 arc:museum:Open_Year ?Open_Year_1.
                  FILTER(?Open_Year_1 > "2011").
                }
                GROUP BY ?visitor
              }
              ?visitor arc:visitor:Name ?Name.
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        # #1: SELECT['visitor']
        # #2: FILTER['#1', 'that visited a museum opened before 2009']
        # #3: FILTER['#2', 'that visited a museum opened after 2011']
        # #4: PROJECT['name of #REF', '#3']

        qdmr = QdmrInstance(["select", "filter", "filter", "project"],
                            [["visitor"],
                             ['#1', 'that visited a museum opened before 2009'],
                             ['#2', 'that visited a museum opened after 2011'],
                             ['name of #REF', '#3']
                            ])

        grounding = {}
        grounding[GroundingIndex(0,0,"visitor")] = GroundingKey.make_table_grounding("visitor")
        grounding[GroundingIndex(1,1,"that visited a museum opened before 2009")] = GroundingKey.make_comparative_grounding("<", "2009", GroundingKey.make_column_grounding("museum", "Open_Year"))
        grounding[GroundingIndex(2,1,"that visited a museum opened after 2011")] = GroundingKey.make_comparative_grounding(">", "2011", GroundingKey.make_column_grounding("museum", "Open_Year"))
        grounding[GroundingIndex(3,0,"name of #REF")] = GroundingKey.make_column_grounding("visitor", "Name")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderTrain1353(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "train"
        i_query = 1353
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # Question: What is the sum of budgets of the Marketing and Finance departments?
        # sql_query:
        # SELECT sum(budget) FROM department WHERE dept_name  =  'Marketing' OR dept_name  =  'Finance'

        correct_sparql_query = textwrap.dedent("""\
            SELECT (?budget_1 + ?budget_2 AS ?sum)
            WHERE
            {
                ?dep_1 arc:department:budget ?budget_1.
                ?dep_1 arc:department:dept_name ?dept_name_1.
                FILTER(?dept_name_1 = key:department:dept_name:Marketing).
                ?dep_2 arc:department:budget ?budget_2.
                ?dep_2 arc:department:dept_name ?dept_name_2.
                FILTER(?dept_name_2 = key:department:dept_name:Finance).
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        #     SELECT['budgets']
        #     FILTER['#1', 'of the Marketing department']
        #     FILTER['#1', 'of the Finance department']
        #     ARITHMETIC['sum', '#2', '#3']

        grounding = {}
        grounding[GroundingIndex(0,0,"budgets")] = GroundingKey.make_column_grounding("department", "budget")
        # grounding looks like key:department:dept_name:Marketing because that value is a key in the RDF graph
        grounding[GroundingIndex(1,1,"of the Marketing department")] = GroundingKey.make_value_grounding("department", "dept_name", "Marketing")
        grounding[GroundingIndex(2,1,"of the Finance department")] = GroundingKey.make_value_grounding("department", "dept_name", "Finance")

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


class TestSpiderTrain4320(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_spider_dev(self):
        """Test an entry from spider dataset
        """
        split_name = "train"
        i_query = 4320
        db_id = get_db_id(split_name, i_query)

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = get_sql_query(split_name, i_query)
        # Question: What are the distinct grant amount for the grants where the documents were sent before '1986-08-26 20:49:27' and grant were ended after '1989-03-16 18:27:16'?
        # sql_query:
        # SELECT T1.grant_amount FROM Grants AS T1 JOIN Documents AS T2 ON T1.grant_id  =  T2.grant_id WHERE T2.sent_date  <  '1986-08-26 20:49:27'
        # INTERSECT
        # SELECT grant_amount FROM grants WHERE grant_end_date  >  '1989-03-16 18:27:16'

        # CAUTION! this query interprets dates as strings and works just by luck! need to parse dates properly
        correct_sparql_query = textwrap.dedent("""\
            SELECT DISTINCT ?grant_amount
            WHERE
            {
              {
                SELECT ?Grants
                WHERE
                {
                  {
                    SELECT ?Grants
                    WHERE
                    {
                      ?grant_id arc:Documents:grant_id:Grants:grant_id ?Grants.
                      ?Documents arc:Documents:grant_id ?grant_id.
                      ?Documents arc:Documents:sent_date ?sent_date.
                      FILTER(?sent_date < "1986-08-26 20:49:27").
                    }
                    GROUP BY ?Grants
                  }
                  ?Grants arc:Grants:grant_end_date ?grant_end_date.
                  FILTER(?grant_end_date > "1989-03-16 18:27:16").
                }
                GROUP BY ?Grants
              }
              ?Grants arc:Grants:grant_amount ?grant_amount.
            }""")

        qdmr = get_qdmr_from_break(split_name, i_query)
        # break_program:
        # #1: SELECT['grants']
        # #2: PROJECT['documents of #REF', '#1']
        # #3: PROJECT['when #REF were sent', '#2']
        # #4: PROJECT['when #REF ended', '#1']
        # #5: COMPARATIVE['#1', '#3', 'is before 1986-08-26 20:49:27']
        # #6: COMPARATIVE['#1', '#4', 'is after 1989-03-16 18:27:16']
        # #7: INTERSECTION['#1', '#5', '#6']
        # #8: PROJECT['distinct grant amounts of #REF', '#7']

        grounding = {}
        grounding[GroundingIndex(0,0,"grants")] = GroundingKey.make_table_grounding("Grants")
        grounding[GroundingIndex(1,0,"documents of #REF")] = GroundingKey.make_table_grounding("Documents")
        grounding[GroundingIndex(2,0,"when #REF were sent")] = GroundingKey.make_column_grounding("Documents", "sent_date")
        grounding[GroundingIndex(3,0,"when #REF ended")] = GroundingKey.make_column_grounding("Grants", "grant_end_date")

        grounding[GroundingIndex(4,2,"is before 1986-08-26 20:49:27")] = GroundingKey.make_comparative_grounding("<", "1986-08-26 20:49:27", GroundingKey.make_column_grounding("Documents", "sent_date"))
        grounding[GroundingIndex(5,2,"is after 1989-03-16 18:27:16")] = GroundingKey.make_comparative_grounding(">", "1989-03-16 18:27:16", GroundingKey.make_column_grounding("Grants", "grant_end_date"))
        grounding[GroundingIndex(7,0,"distinct grant amounts of #REF")] = GroundingKey.make_column_grounding("Grants", "grant_amount")
        grounding["distinct"] = ["#8"]

        sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

        result_correct = QueryResult.execute_query_sql(sql_query, schema)
        result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
        equal, message = result.is_equal_to(result_correct,
                                    require_column_order=True,
                                    require_row_order=False,
                                    return_message=True)
        self.assertTrue(equal, message)


# class TestSpiderTrain1384(unittest.TestCase):

#     @timeout(ONE_TEST_TIMEOUT)
#     def test_spider_dev(self):
#         """Test an entry from spider dataset
#         """
#         split_name = "train"
#         i_query = 1384
#         db_id = get_db_id(split_name, i_query)

#         rdf_graph, schema = get_graph_and_schema(split_name, db_id)

#         sql_query = get_sql_query(split_name, i_query)
#         # Question: Find the name of the students and their department names sorted by their total credits in ascending order.
#         # sql_query:
#         # SELECT name ,  dept_name FROM student ORDER BY tot_cred

#         # CAUTION: this test works fine but is super slow (large database) so I'm commenting it out by default

#         correct_sparql_query = textwrap.dedent("""\
#             SELECT ?name ?dept_name ?tot_cred
#             WHERE
#             {
#                 ?student arc:student:name ?name.
#                 ?student arc:student:dept_name ?dept_name.
#                 ?student arc:student:tot_cred ?tot_cred.
#             }
#             ORDER BY ASC(?tot_cred)""")

#         qdmr = get_qdmr_from_break(split_name, i_query)
#         qdmr.args[3] = ["#1", "#3", "in ascending order"]
#         # break_program:
#         #     SELECT['students']
#         #     PROJECT['credits of #REF', '#1']
#         #     GROUP['sum', '#2', '#1']
#         #     SORT['#1', '#3', 'in ascending order']
#         #     PROJECT['names of #REF', '#4']
#         #     PROJECT['departments of #REF', '#4']
#         #     PROJECT['names of #REF', '#6']
#         #     UNION['#5', '#7']

#         grounding = {}
#         grounding[GroundingIndex(0,0,"students")] = GroundingKey.make_table_grounding("student")
#         grounding[GroundingIndex(1,0,"credits of #REF")] = GroundingKey.make_column_grounding("student", "tot_cred")
#         grounding[GroundingIndex(3,2,"in ascending order")] = GroundingKey.make_sortdir_grounding(ascending=True)
#         grounding[GroundingIndex(4,0,"names of #REF")] = GroundingKey.make_column_grounding("student", "name")
#         grounding[GroundingIndex(5,0,"departments of #REF")] = GroundingKey.make_column_grounding("student", "dept_name")
#         grounding[GroundingIndex(6,0,"names of #REF")] = GroundingKey.make_column_grounding("student", "dept_name")

#         sparql_query = create_sparql_query_from_qdmr(qdmr, schema, rdf_graph, grounding)

#         result_correct = QueryResult.execute_query_sql(sql_query, schema)
#         result = QueryResult.execute_query_to_rdf(sparql_query, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)
#         equal, message = result.is_equal_to(result_correct,
#                                     require_column_order=True,
#                                     require_row_order=True,
#                                     return_message=True)
#         self.assertTrue(equal, message)


class TestSql2SqlLimits(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_dog_bug(self):
        db_id = "dog_kennels"
        split_name = "dev"
        
        rdf_graph, schema = get_graph_and_schema(split_name, db_id)
        
        sql_query_1 = "SELECT max(age) FROM Dogs"
        sql_query_2 = "SELECT Dogs.age FROM Dogs ORDER BY Dogs.age DESC LIMIT 1"

        result_1 = QueryResult.execute_query_sql(sql_query_1, schema)
        result_2 = QueryResult.execute_query_sql(sql_query_2, schema)

        equal, message = result_1.is_equal_to(result_2,
                                              require_column_order=False,
                                              require_row_order=False,
                                              weak_mode_argmax=True,
                                              return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_limits(self):
        db_id = "dog_kennels"
        split_name = "dev"

        rdf_graph, schema = get_graph_and_schema(split_name, db_id)
        
        sql_query_1 = "SELECT Dogs.age FROM Dogs ORDER BY Dogs.age DESC LIMIT 1"
        sql_query_2 = "SELECT Dogs.age FROM Dogs ORDER BY Dogs.age DESC LIMIT 2"

        result_1 = QueryResult.execute_query_sql(sql_query_1, schema)
        result_2 = QueryResult.execute_query_sql(sql_query_2, schema)

        equal, message = result_1.is_equal_to(result_2,
                                              require_column_order=False,
                                              require_row_order=False,
                                              weak_mode_argmax=True,
                                              return_message=True)
        self.assertTrue(not equal, message)

        equal, message = result_2.is_equal_to(result_1,
                                              require_column_order=False,
                                              require_row_order=False,
                                              weak_mode_argmax=True,
                                              return_message=True)
        self.assertTrue(not equal, message)


class TestWeakArgMaxMode(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_sql2sql(self):
        db_id = "concert_singer"
        split_name = "dev"
        rdf_graph, schema = get_graph_and_schema(split_name, db_id)
        
        sql_query_1 = "SELECT concert_Name, Year FROM concert ORDER BY Year ASC LIMIT 1"
        sql_query_2 = "SELECT concert_Name, Year FROM concert"
        sql_query_3 = "SELECT concert_Name, Year FROM concert where Year = (SELECT Year FROM concert ORDER BY Year ASC LIMIT 1)"
        sql_query_4 = "SELECT concert_Name, Year FROM concert where Year = (SELECT min(Year) FROM concert)"

        result_1 = QueryResult.execute_query_sql(sql_query_1, schema)
        result_2 = QueryResult.execute_query_sql(sql_query_2, schema)
        result_3 = QueryResult.execute_query_sql(sql_query_3, schema)
        result_4 = QueryResult.execute_query_sql(sql_query_4, schema)

        equal, message = result_1.is_equal_to(result_2,
                                              require_column_order=False,
                                              require_row_order=False,
                                              weak_mode_argmax=True,
                                              return_message=True)
        self.assertTrue(not equal, message)

        equal, message = result_1.is_equal_to(result_3,
                                              require_column_order=False,
                                              require_row_order=False,
                                              weak_mode_argmax=True,
                                              return_message=True)
        self.assertTrue(equal, message)

        equal, message = result_1.is_equal_to(result_4,
                                              require_column_order=False,
                                              require_row_order=False,
                                              weak_mode_argmax=True,
                                              return_message=True)
        self.assertTrue(equal, message)

    @timeout(ONE_TEST_TIMEOUT)
    def test_qdmr2sql(self):
        db_id = "concert_singer"
        split_name = "dev"
        rdf_graph, schema = get_graph_and_schema(split_name, db_id)

        sql_query = "SELECT concert_Name, Year FROM concert ORDER BY Year ASC LIMIT 1"
        result_sql = QueryResult.execute_query_sql(sql_query, schema)
        
        qdmr_0 = QdmrInstance(["select", "project", "union"],
                              [["concert_Name"],
                              ["Year", "#1"],
                              ["#1", "#2"],
                             ])
        grounding_0 = { GroundingIndex(0,0,"concert_Name") : GroundingKey.make_column_grounding("concert", "concert_Name"),
                        GroundingIndex(1,0,"Year") : GroundingKey.make_column_grounding("concert", "Year"),
                      }

        sparql_query_0 = create_sparql_query_from_qdmr(qdmr_0, schema, rdf_graph, grounding_0)

        result_qdmr_0 = QueryResult.execute_query_to_rdf(sparql_query_0, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)

        equal, message = result_sql.is_equal_to(result_qdmr_0,
                                                require_column_order=False,
                                                require_row_order=False,
                                                weak_mode_argmax=True,
                                                return_message=True)
        self.assertTrue(not equal, message)


        qdmr_1 = QdmrInstance(["select", "project", "superlative", "union"],
                              [["concert_Name"],
                              ["Year", "#1"],
                              ["min", "#1", "#2"],
                              ["#3", "#2"],
                             ])
        grounding_1 = { GroundingIndex(0,0,"concert_Name") : GroundingKey.make_column_grounding("concert", "concert_Name"),
                        GroundingIndex(1,0,"Year") : GroundingKey.make_column_grounding("concert", "Year"),
                        GroundingIndex(2,0,"min") : GroundingKey.make_comparative_grounding("min", None),
                      }

        sparql_query_1 = create_sparql_query_from_qdmr(qdmr_1, schema, rdf_graph, grounding_1)

        result_qdmr_1 = QueryResult.execute_query_to_rdf(sparql_query_1, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)

        equal, message = result_sql.is_equal_to(result_qdmr_1,
                                                require_column_order=False,
                                                require_row_order=False,
                                                weak_mode_argmax=True,
                                                return_message=True)
        self.assertTrue(equal, message)


        qdmr_2 = QdmrInstance(["select", "project", "comparative", "union"],
                              [["concert_Name"],
                              ["Year", "#1"],
                              ["#1", "#2", "min"],
                              ["#3", "#2"],
                             ])
        grounding_2 = { GroundingIndex(0,0,"concert_Name") : GroundingKey.make_column_grounding("concert", "concert_Name"),
                        GroundingIndex(1,0,"Year") : GroundingKey.make_column_grounding("concert", "Year"),
                        GroundingIndex(2,2,"min") : GroundingKey.make_comparative_grounding("min", None),
                      }

        sparql_query_2 = create_sparql_query_from_qdmr(qdmr_2, schema, rdf_graph, grounding_2)

        result_qdmr_2 = QueryResult.execute_query_to_rdf(sparql_query_2, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)

        equal, message = result_sql.is_equal_to(result_qdmr_2,
                                                require_column_order=False,
                                                require_row_order=False,
                                                weak_mode_argmax=True,
                                                return_message=True)
        self.assertTrue(equal, message)


        qdmr_3 = QdmrInstance(["select", "project", "aggregate", "comparative", "union"],
                              [["concert_Name"],
                              ["Year", "#1"],
                              ["min", "#2"],
                              ["#1", "#2", "=min"],
                              ["#4", "#2"],
                             ])
        grounding_3 = { GroundingIndex(0,0,"concert_Name") : GroundingKey.make_column_grounding("concert", "concert_Name"),
                        GroundingIndex(1,0,"Year") : GroundingKey.make_column_grounding("concert", "Year"),
                        GroundingIndex(3,2,"=min") : GroundingKey.make_comparative_grounding("=", "#3"),
                      }

        sparql_query_3 = create_sparql_query_from_qdmr(qdmr_3, schema, rdf_graph, grounding_3)

        result_qdmr_3 = QueryResult.execute_query_to_rdf(sparql_query_3, rdf_graph, schema, virtuoso_server=VIRTUOSO_SPARQL_SERVICE)

        equal, message = result_sql.is_equal_to(result_qdmr_3,
                                                require_column_order=False,
                                                require_row_order=False,
                                                weak_mode_argmax=True,
                                                return_message=True)
        self.assertTrue(equal, message)


class TestTimeToTimeComparison(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_sql2sql(self):
        db_id = "voter_1"
        split_name = "dev"
        rdf_graph, schema = get_graph_and_schema(split_name, db_id)
        
        sql_query_1 = "SELECT max(created) FROM votes WHERE state  =  'CA'"
        sql_query_2 = 'SELECT VOTES.created FROM VOTES WHERE VOTES.state = "CA" ORDER BY VOTES.created DESC LIMIT 1'

        result_1 = QueryResult.execute_query_sql(sql_query_1, schema)
        result_2 = QueryResult.execute_query_sql(sql_query_2, schema)

        equal, message = result_1.is_equal_to(result_2,
                                              require_column_order=False,
                                              require_row_order=False,
                                              weak_mode_argmax=True,
                                              return_message=True)
        self.assertTrue(equal, message)


class TestWta1(unittest.TestCase):

    @timeout(ONE_TEST_TIMEOUT)
    def test_sql2sql(self):
        db_id = "wta_1"
        split_name = "dev"
        rdf_graph, schema = get_graph_and_schema(split_name, db_id)
        
        sql_query_1 = "SELECT first_name ,  last_name FROM players ORDER BY birth_date"
        sql_query_2 = "SELECT players.first_name , players.last_name FROM players ORDER BY players.birth_date ASC"

        result_1 = QueryResult.execute_query_sql(sql_query_1, schema)
        result_2 = QueryResult.execute_query_sql(sql_query_2, schema)

        equal, message = result_1.is_equal_to(result_2,
                                              require_column_order=False,
                                              require_row_order=False,
                                              weak_mode_argmax=True,
                                              return_message=True)
        self.assertTrue(equal, message)


if __name__ == '__main__':
    datasets_break = {}
    datasets_spider = {}

    script_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(script_path, ".."))
    spider_path = os.path.join(root_path, "data", "spider")
    db_path = os.path.join(spider_path, "database")
    qdmr_path = os.path.join(root_path, "data", "break", "logical-forms")

    for split_name in ['dev', 'train']:
        datasets_break[split_name] = DatasetBreak(qdmr_path, split_name)
        datasets_spider[split_name] = DatasetSpider(spider_path, split_name)


    def get_db_id(subset, i_query):
        query_name, sql_data = datasets_spider[subset][i_query]
        db_id = sql_data["db_id"]
        return db_id


    @lru_cache()
    def get_graph_and_schema(subset, db_id):
        dataset_spider = datasets_spider[subset]
        table_data = dataset_spider.table_data
        schema = dataset_spider.schemas[db_id]

        assert db_id in table_data, f"Could not find database {db_id} in any subset"
        table_data = table_data[db_id]

        schema.load_table_data(db_path)
        rdf_graph = RdfGraph(schema)

        return rdf_graph, schema


    def get_qdmr_from_break(subset, i_query):
        qdmr = datasets_break[subset].get_qdmr_by_subset_indx(i_query, "SPIDER")
        # qdmr_name = dataset_break.get_name_by_subset_indx(args.spider_idx)
        return qdmr

    def get_sql_query(subset, i_query):
        query_name, sql_data = datasets_spider[subset][i_query]
        sql_query = sql_data["query"]
        return sql_query


    unittest.main()
