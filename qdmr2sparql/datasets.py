import os
import json
import csv

from qdmr2sparql.structures import DatabaseSchema, QdmrInstance


class DatasetBreak():
    correct_header = ("question_id", "question_text", "decomposition", "program", "operators", "split")

    def __init__(self, dataset_path, subset=None, target_file=None, filter_subset="spider"):
        self.subset = subset
        if subset == "train":
            if filter_subset == "spider":
                data_file = os.path.join(dataset_path, "train_spider.csv")
            else:
                data_file = os.path.join(dataset_path, "train.csv")
        elif subset == "dev":
            if filter_subset == "spider":
                data_file = os.path.join(dataset_path, "dev_spider.csv")
            else:
                data_file = os.path.join(dataset_path, "dev.csv")
        elif subset is None:
            assert target_file is not None, "Either subset or target_file should be specified"
            data_file = target_file
        else:
            raise ValueError(f"Unknown subset of BREAK: {subset}")

        # read the data file
        self.qdmrs, self.questions, self.qdmr_full_table = self.read_parse_break_csv_file(data_file, filter_subset)
        self.names = list(self.qdmrs.keys())

    def __len__(self):
        return len(self.names)

    def __getitem__(self, key):
        if not isinstance(key, str):
            name = self.names[key]
        else:
            name = key
        return name, self.qdmrs[name]

    def make_iterator(self, start_break_idx=None, end_break_idx=None):

        if start_break_idx is None:
            start_break_idx = 0
        if end_break_idx is None:
            end_break_idx = max(self.get_index_from_name(name) for name in self.names) + 1

        list_of_names = []
        for idx in range(start_break_idx, end_break_idx):
            try:
                name = self.names[idx]
                list_of_names.append(name)
            except IndexError as e:
                # this example is not present - skip it
                print(f"WARNING: example with index {idx} is not present in subset {self.subset}")

        def generate_examples():
            for name in list_of_names:
                yield self[name]

        return generate_examples()

    @staticmethod
    def get_item_name(dataset_keyword, subset, indx):
        return f"{dataset_keyword}_{subset}_{indx}"

    def get_qdmr_by_subset_indx(self, indx, dataset_keyword, subset=None):
        name = self.get_name_by_subset_indx(indx, dataset_keyword, subset)
        return self.qdmrs[name]

    def get_name_by_subset_indx(self, indx, dataset_keyword, subset=None):
        subset = self.subset if subset is None else subset
        name = self.get_item_name(dataset_keyword, subset, indx)
        assert name in self.qdmrs, f"Cound not find item {name}"
        return name

    @staticmethod
    def get_index_from_name(name):
        index_start = name.find("_")
        index_start = name.find("_", index_start + 1)
        return name[index_start+1:]

    @staticmethod
    def get_dataset_keyword_from_name(name):
        index_start = name.find("_")
        return name[:index_start]

    def get_question_by_subset_indx(self, indx, dataset_keyword, subset=None):
        name = self.get_name_by_subset_indx(indx, dataset_keyword, subset)
        return self.questions[name]

    @staticmethod
    def read_parse_break_csv_file(file, filter_subset):
        read_csv = []
        for line in csv.reader(open(file, 'r', encoding="utf-8")):
            read_csv.append(line)

        # check the header
        header = tuple(read_csv[0])
        assert header == DatasetBreak.correct_header, f"The header of {file} is incorrect, expected {DatasetBreak.correct_header}, found {header}"
        qdmr_full_table = read_csv[1:]

        id_position = header.index("question_id")
        question_position = header.index("question_text")
        qdmr_position = header.index("program")
        ops_position = header.index("operators")

        qdmr_full_table = [entry for entry in qdmr_full_table if entry[id_position].lower().find(filter_subset) >= 0]

        questions = {}
        qdmrs = {}
        for entry in qdmr_full_table:
            name = entry[id_position]
            questions[name] = entry[question_position]
            if entry[qdmr_position] and entry[ops_position]:
                ops, args = QdmrInstance.parse_break_program(entry[qdmr_position], entry[ops_position])
                qdmrs[name] = QdmrInstance(ops, args)
            else:
                qdmrs[name] = None

        return qdmrs, questions, qdmr_full_table

    def save_break_to_csv_file(self, output_file):
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.correct_header)

            id_position = self.correct_header.index("question_id")
            question_position = self.correct_header.index("question_text")
            qdmr_position = self.correct_header.index("program")
            ops_position = self.correct_header.index("operators")

            for entry in self.qdmr_full_table:
                entry_name = entry[id_position]
                qdmr = self.qdmrs[entry_name]
                question = self.questions[entry_name]
                qdmr_str, op_str = qdmr.get_strs_for_saving()
                entry[qdmr_position] = qdmr_str
                entry[ops_position] = op_str
                entry[question_position] = question
                writer.writerow(entry)


class DatasetSpider():
    def __init__(self, dataset_path, subset=None, target_file=None):
        table_path = os.path.join(dataset_path, "tables.json")

        self.dataset_path = dataset_path
        self.subset = subset
        if subset == "train":
            sql_path = os.path.join(dataset_path, "train_spider.json")
            sql_link_annotation_path = os.path.join(dataset_path, "slsql_train.json")
        elif subset == "dev":
            sql_path = os.path.join(dataset_path, "dev.json")
            sql_link_annotation_path = os.path.join(dataset_path, "slsql_dev.json")
        elif subset is None:
            assert target_file is not None, "Either subset or target_file should be specified"
            sql_path = target_file
            sql_link_annotation_path = None
        else:
            raise ValueError(f"Unknown subset of SPIDER: {subset}")

        # read the data file
        self.sql_data, self.schemas, self.table_data, self.sql_linking_dict\
            = self.read_parse_spider_json_file(table_path, sql_path, self.subset, sql_link_annotation_path=sql_link_annotation_path)
        self.names = list(self.sql_data.keys())

    def __len__(self):
        return len(self.names)

    def __getitem__(self, key):
        if not isinstance(key, str):
            name = self.names[key]
        else:
            name = key
        return name, self.sql_data[name]

    @staticmethod
    def get_item_name(dataset_keyword, subset, indx):
        return f"{dataset_keyword}_{subset}_{indx}"

    def get_sql_data_by_subset_indx(self, indx, subset=None, dataset_keyword="SPIDER"):
        name = self.get_name_by_subset_indx(indx, subset, dataset_keyword)
        return self.sql_data[name]

    def get_name_by_subset_indx(self, indx, subset=None, dataset_keyword="SPIDER"):
        subset = self.subset if subset is None else subset
        name = self.get_item_name(dataset_keyword, subset, indx)
        assert name in self.sql_data, f"Cound not find item {name}"
        return name

    @staticmethod
    def read_parse_spider_json_file(table_path, sql_path, subset, sql_link_annotation_path=None):

        with open(sql_path) as inf:
            sql_data_list = json.load(inf)

        sql_data_dict = {}
        used_tables = set()
        for spider_idx, sql in enumerate(sql_data_list):
            name = DatasetBreak.get_item_name("SPIDER", subset, spider_idx)
            sql_data_dict[name] = sql
            used_tables.add(sql["db_id"])

        with open(table_path) as inf:
            table_data = json.load(inf)
        table_data = {table['db_id']: table for table in table_data if table['db_id'] in used_tables}
        schemas = {db_id : DatabaseSchema(table) for db_id, table in table_data.items()}
        assert not (used_tables - set(schemas.keys())), f"Data for some tables was not found: {used_tables - set(table_data.keys())}"

        sql_linking_dict = {}
        if sql_link_annotation_path:
            with open(sql_link_annotation_path) as inf:
                sql_linking_list = json.load(inf)

            for linking in sql_linking_list:
                name = DatasetBreak.get_item_name("SPIDER", subset, linking["id"])
                sql_linking_dict[name] = linking

        return sql_data_dict, schemas, table_data, sql_linking_dict
