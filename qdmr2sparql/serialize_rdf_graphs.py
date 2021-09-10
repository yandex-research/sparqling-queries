import os
import argparse


from datasets import DatasetBreak, DatasetSpider
from structures import RdfGraph


def main(args):
    datasets_spider = {}

    # setup paths
    script_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(script_path, ".."))
    spider_path = os.path.join(root_path, "data","spider")
    db_path = os.path.join(spider_path, "database")
    target_path = os.path.join(root_path, args.target_path)

    # load the Spider dataset
    spider_splits = ['dev', 'train']
    for split_name in spider_splits:
        datasets_spider[split_name] = DatasetSpider(spider_path, split_name)

    target_databases_with_split = []
    if not args.databases:
        for split_name in spider_splits:
            for db_id in datasets_spider[split_name].schemas.keys():
                target_databases_with_split.append((db_id, split_name))
    else:
        for db_id in args.databases:
            for split_name in spider_splits:
                if db_id in datasets_spider[split_name].schemas:
                    target_databases_with_split.append((db_id, split_name))
                    continue

    # serialize data
    for db_id, split_name in target_databases_with_split:
        dataset_spider = datasets_spider[split_name]
        table_data = dataset_spider.table_data
        schema = dataset_spider.schemas[db_id]

        assert db_id in table_data, f"Could not find database {db_id} in any subset"
        table_data = table_data[db_id]

        schema.load_table_data(db_path)
        rdf_graph = RdfGraph(schema)

        target_file = os.path.join(target_path, f"{db_id}.ttl")
        rdf_graph.g.serialize(target_file, format="ttl")

        # create an additional file to define the graph name in Virtuoso
        with open(target_file + ".graph", "w") as graph_file:
            graph_file.write(db_id)


def parse_args():
    parser = argparse.ArgumentParser(description="Build grounding between QDMR and SQL.")
    parser.add_argument('--target_path', type=str, default="data/spider/database_rdf",
        help="Path to save the serialized RDF graphs, relative with respect to the root of this package")
    parser.add_argument('--databases', type=str, default=None, nargs='+',
        help="Names of the database to serialize, default - serialize all SPIDER databases")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
