import os
import shutil
import sqlite3

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Build grounding between QDMR and SQL.')
    parser.add_argument('--spider_path', type=str, help='path to spider dataset')
    parser.add_argument('--database_path', type=str, default="database", help='path to spider databases (starting from --spider_path)')
    parser.add_argument('--database', type=str, default=None, help='Database to fix, by default will try to fix everything that is implemented')
    args = parser.parse_args()
    return args


def fix_database_car_1(sqlite_file):
    print("Editing database", sqlite_file)
    conn = sqlite3.connect(sqlite_file)
    conn.text_factory = lambda b: b.decode(errors = 'ignore')
    c = conn.cursor()

    # get all the tables in the database
    query_get_all_tables = "SELECT name FROM sqlite_master WHERE type='table'"
    c.execute(query_get_all_tables)
    table_names = c.fetchall()

    # print(table_names)

    # we want to fix table "cars_data"
    tbl_name = "cars_data"
    c.execute(f"PRAGMA table_info({tbl_name});")
    col_data = c.fetchall()
    # print(col_data)

    c.execute(f"PRAGMA foreign_key_list({tbl_name});")
    foreign_keys_out = c.fetchall()

    # for (tname,) in table_names:
    #     c.execute(f"PRAGMA foreign_key_list({tname});")
    #     foreign_keys = c.fetchall()
    #     print(f"Foreign keys of {tname}:", foreign_keys)

    # get content
    c.execute(f"SELECT * FROM {tbl_name}")
    tbl_data = c.fetchall()

    # fix column data
    for i in range(len(col_data)):
        d = col_data[i]
        if d[1] in ["MPG", "Horsepower"]:
            d_new = tuple(d[:2] + ("REAL",) + d[3:])
            col_data[i] = d_new

    # turn off checking the correctness of foreign keys
    c.execute("PRAGMA foreign_keys = OFF;")
    c.execute("PRAGMA legacy_alter_table=ON;")

    # drop the old table
    c.execute(f"DROP TABLE {tbl_name};")

    # create new table
    tbl_name_new = tbl_name
    # column extry: NAME TYPE [PRIMARY KEY] [NOT NULL] [DEFAULT 0]
    column_list = [f"{col[1]} {col[2]}" for col in col_data]

    # add primary keys
    column_list.append(f"PRIMARY KEY ({', '.join(col[1] for col in col_data if col[5])})")

    for f in foreign_keys_out:
        line = f"FOREIGN KEY({f[3]}) REFERENCES {f[2]}({f[4]})"
        column_list.append(line)

    tbl_command = f"CREATE TABLE {tbl_name_new} ( {', '.join(column_list)} );"
    # print(tbl_command)

    c.execute(tbl_command)

    # insert data to the new table
    column_names = [t[1] for t in col_data]
    for d in tbl_data:
        data_command = f"INSERT INTO {tbl_name_new} ( {', '.join(column_names)} ) VALUES ( {', '.join(str(dd) for dd in d)} );"
        c.execute(data_command)

    # turn on checking the correctness of foreign keys
    c.execute("PRAGMA foreign_keys = ON;")
    c.execute("PRAGMA legacy_alter_table=OFF;")

    # for (tname,) in table_names:
    #     c.execute(f"PRAGMA foreign_key_list({tname});")
    #     foreign_keys = c.fetchall()
    #     print(f"Foreign keys of {tname}:", foreign_keys)

    conn.commit()
    conn.close()


def fix_database_add_foreign_keys(sqlite_file, foreign_key_list, keys_to_delete=None):
    print("Editing database", sqlite_file)
    conn = sqlite3.connect(sqlite_file)
    conn.text_factory = lambda b: b.decode(errors = 'ignore')
    c = conn.cursor()

    # get all the tables in the database
    query_get_all_tables = "SELECT name FROM sqlite_master WHERE type='table'"
    c.execute(query_get_all_tables)
    table_names = c.fetchall()

    # get column names
    col_names = {}
    for (tbl_name,) in table_names:
        c.execute(f"PRAGMA table_info({tbl_name});")
        col_data = c.fetchall()
        col_names[tbl_name] = [t[1] for t in col_data]

    foreign_keys_per_table = {}
    for (tbl_name,) in table_names:
        c.execute(f"PRAGMA foreign_key_list({tbl_name});")
        foreign_keys_per_table[tbl_name] = c.fetchall()
        # print(f"Foreign keys of {tbl_name}:", foreign_keys_per_table[tbl_name])

    # some keys to delete:
    def delete_keys(key_list):
        for foreign_key in key_list:
            tbl_src, col_src, tbl_tgt, col_tgt = foreign_key
            if tbl_src in foreign_keys_per_table:
                to_delete = []
                for i in range(len(foreign_keys_per_table[tbl_src])):
                    key = foreign_keys_per_table[tbl_src][i]
                    if key[2] == tbl_tgt and key[3] == col_src and key[4] == col_tgt:
                        to_delete.append(i)
                        # print(f"deleting {tbl_src}: {key}")
                foreign_keys_per_table[tbl_src] = [k for i, k in enumerate(foreign_keys_per_table[tbl_src]) if i not in to_delete]

    if keys_to_delete:
        delete_keys(keys_to_delete)

    # add missing foreign keys
    # not all the fields here might be correct, but we are using only names in the end of the day
    # print(f"Adding: {foreign_key_list}")
    for foreign_key in foreign_key_list:
        tbl_src, col_src, tbl_tgt, col_tgt = foreign_key
        assert (tbl_src,) in table_names, f"Could not find table {tbl_src} in database {sqlite_file}"
        assert (tbl_tgt,) in table_names, f"Could not find table {tbl_tgt} in database {sqlite_file}"
        assert col_src in col_names[tbl_src], f"Could not find column {col_src} in table {tbl_src} in database {sqlite_file}"
        assert col_tgt in col_names[tbl_tgt], f"Could not find column {col_tgt} in table {tbl_tgt} in database {sqlite_file}"

        key_found = False
        for key in foreign_keys_per_table[tbl_src]:
            if key[2] == tbl_tgt and key[3] == col_src and key[4] == col_tgt:
                key_found = True
                break
        if not key_found:
            foreign_keys_per_table[tbl_src].append( (len(foreign_keys_per_table[tbl_src]), 0,\
                                                    tbl_tgt, col_src, col_tgt, 'NO ACTION', 'NO ACTION', 'NONE') )
        # foreign_keys_per_table["flights"].append( (len(foreign_keys_per_table["flights"]), 0,\
        #                                           'airlines', 'Airline', 'uid', 'NO ACTION', 'NO ACTION', 'NONE') )

    # turn off checking the correctness of foreign keys
    c.execute("PRAGMA foreign_keys = OFF;")
    c.execute("PRAGMA legacy_alter_table=ON;")

    # first we want to remove trailing white spaces from all the tables
    for (tbl_name,) in table_names:
        c.execute(f"PRAGMA table_info({tbl_name});")
        col_data = c.fetchall()
        # print(col_data)

        foreign_keys_out = foreign_keys_per_table[tbl_name]

        # get content
        c.execute(f"SELECT * FROM {tbl_name}")
        tbl_data = c.fetchall()

        # fix column data
        for i_row in range(len(tbl_data)):
            row = tbl_data[i_row]
            for i in range(len(row)):
                d = row[i]
                if col_data[i][2] == "TEXT":
                    d = d.strip()
                    row = row[:i] + (d,) + row[i+1:]
            tbl_data[i_row] = row

        c.execute(f"DROP TABLE {tbl_name};")

        # create new table
        tbl_name_new = tbl_name
        # column extry: NAME TYPE [PRIMARY KEY] [NOT NULL] [DEFAULT 0]
        column_list = [f"{col[1]} {col[2]}" for col in col_data]

        # add primary keys
        pr_keys = [col[1] for col in col_data if col[5]]
        if pr_keys:
            column_list.append(f"PRIMARY KEY ({', '.join(pr_keys)})")

        for f in foreign_keys_out:
            line = f"FOREIGN KEY({f[3]}) REFERENCES {f[2]}({f[4]})"
            column_list.append(line)

        tbl_command = f"CREATE TABLE {tbl_name_new} ( {', '.join(column_list)} );"
        # print(tbl_command)
        c.execute(tbl_command)

        # insert data to the new table
        column_names = [t[1] for t in col_data]
        for d in tbl_data:
            def clear_str(dd):
                return str(dd).replace("'", "''")

            # d = [f"'{clear_str(dd)}'" if any(t in col_data[i][2].lower() for t in ["text", "char", "date", "time"]) else str(dd) for i, dd in enumerate(d)]
            d = [f"'{clear_str(dd)}'"  for i, dd in enumerate(d)]
            value_str = ', '.join(d)
            data_command = f"INSERT INTO {tbl_name_new} ( {', '.join(column_names)} ) VALUES ( {value_str} );"
            c.execute(data_command)

    # turn on checking the correctness of foreign keys
    c.execute("PRAGMA foreign_keys = ON;")
    c.execute("PRAGMA legacy_alter_table=OFF;")

    # for (tname,) in table_names:
    #     c.execute(f"PRAGMA foreign_key_list({tname});")
    #     foreign_keys = c.fetchall()
    #     print(f"Foreign keys of {tname}:", foreign_keys)

    conn.commit()
    conn.close()


def fix_database_cre_Doc_Template_Mgt(sqlite_file):
    print("Editing database", sqlite_file)
    conn = sqlite3.connect(sqlite_file)
    conn.text_factory = lambda b: b.decode(errors = 'ignore')
    c = conn.cursor()

    # get all the tables in the database
    query_get_all_tables = "SELECT name FROM sqlite_master WHERE type='table'"
    c.execute(query_get_all_tables)
    table_names = c.fetchall()

    # print(table_names)

    foreign_keys_per_table = {}
    for (tbl_name,) in table_names:
        c.execute(f"PRAGMA foreign_key_list({tbl_name});")
        foreign_keys_per_table[tbl_name] = c.fetchall()
        # print(f"Foreign keys of {tbl_name}:", foreign_keys_per_table[tbl_name])

    # turn off checking the correctness of foreign keys
    c.execute("PRAGMA foreign_keys = OFF;")
    c.execute("PRAGMA legacy_alter_table=ON;")

    # first we want to add some data to column "Other_Details" of table "Paragraphs:"
    tbl_name = "Paragraphs"
    c.execute(f"PRAGMA table_info({tbl_name});")
    col_data = c.fetchall()
    # print(col_data)

    foreign_keys_out = foreign_keys_per_table[tbl_name]
    # get content
    c.execute(f"SELECT * FROM {tbl_name}")
    tbl_data = c.fetchall()

    # fix column data
    i_paragraph_text = [i for i in range(len(col_data)) if col_data[i][1] == "Paragraph_Text"]
    i_paragraph_text = i_paragraph_text[0]

    for i_row in range(len(tbl_data)):
        row = tbl_data[i_row]
        for i in range(len(row)):
            d = row[i]
            if col_data[i][1] == "Other_Details":
                paragraph_text = row[i_paragraph_text]
                d = "Other details for " + paragraph_text
                row = row[:i] + (d,) + row[i+1:]
        tbl_data[i_row] = row

    c.execute(f"DROP TABLE {tbl_name};")

    # create new table
    tbl_name_new = tbl_name
    # column extry: NAME TYPE [PRIMARY KEY] [NOT NULL] [DEFAULT 0]
    column_list = [f"{col[1]} {col[2]}" for col in col_data]

    # add primary keys
    column_list.append(f"PRIMARY KEY ({', '.join(col[1] for col in col_data if col[5])})")

    for f in foreign_keys_out:
        line = f"FOREIGN KEY({f[3]}) REFERENCES {f[2]}({f[4]})"
        column_list.append(line)

    tbl_command = f"CREATE TABLE {tbl_name_new} ( {', '.join(column_list)} );"
    # print(tbl_command)
    c.execute(tbl_command)

    # insert data to the new table
    column_names = [t[1] for t in col_data]
    for d in tbl_data:
        d = [f"'{dd}'" if col_data[i][2] == "TEXT" or "VARCHAR" in col_data[i][2] else str(dd) for i, dd in enumerate(d)]
        value_str = ', '.join(d)
        data_command = f"INSERT INTO {tbl_name_new} ( {', '.join(column_names)} ) VALUES ( {value_str} );"
        c.execute(data_command)

    # turn on checking the correctness of foreign keys
    c.execute("PRAGMA foreign_keys = ON;")
    c.execute("PRAGMA legacy_alter_table=OFF;")

    # for (tname,) in table_names:
    #     c.execute(f"PRAGMA foreign_key_list({tname});")
    #     foreign_keys = c.fetchall()
    #     print(f"Foreign keys of {tname}:", foreign_keys)

    conn.commit()
    conn.close()


def fix_database_wta_1(sqlite_file):
    print("Editing database", sqlite_file)
    conn = sqlite3.connect(sqlite_file)
    conn.text_factory = lambda b: b.decode(errors = 'ignore')
    c = conn.cursor()

    # get all the tables in the database
    query_get_all_tables = "SELECT name FROM sqlite_master WHERE type='table'"
    c.execute(query_get_all_tables)
    table_names = c.fetchall()

    # print(table_names)

    foreign_keys_per_table = {}
    for (tbl_name,) in table_names:
        c.execute(f"PRAGMA foreign_key_list({tbl_name});")
        foreign_keys_per_table[tbl_name] = c.fetchall()
        # print(f"Foreign keys of {tbl_name}:", foreign_keys_per_table[tbl_name])

    # turn off checking the correctness of foreign keys
    c.execute("PRAGMA foreign_keys = OFF;")
    c.execute("PRAGMA legacy_alter_table=ON;")

    # we will downsample the table by removing all players who did not play any matches_to_tables
    c.execute(f"SELECT loser_id, winner_id FROM matches")
    ids_player_with_matches = c.fetchall()
    ids_player_with_matches = [r[0] for r in ids_player_with_matches] + [r[1] for r in ids_player_with_matches]
    ids_player_with_matches = set(ids_player_with_matches)

    columns_with_ids = {"matches": ["loser_id", "winner_id"],
                        "players": ["player_id"],
                        "rankings": ["player_id"]}

    # first we want to remove trailing white spaces from all the tables
    for (tbl_name,) in table_names:
        c.execute(f"PRAGMA table_info({tbl_name});")
        col_data = c.fetchall()
        # print(col_data)

        foreign_keys_out = foreign_keys_per_table[tbl_name]

        # get content
        c.execute(f"SELECT * FROM {tbl_name}")
        tbl_data = c.fetchall()

        c.execute(f"DROP TABLE {tbl_name};")

        # create new table
        tbl_name_new = tbl_name
        # column extry: NAME TYPE [PRIMARY KEY] [NOT NULL] [DEFAULT 0]
        column_list = [f"{col[1]} {col[2]}" for col in col_data]

        # add primary keys
        cols_with_pr_keys = [col[1] for col in col_data if col[5]]
        if cols_with_pr_keys:
            column_list.append(f"PRIMARY KEY ({', '.join(cols_with_pr_keys)})")

        for f in foreign_keys_out:
            line = f"FOREIGN KEY({f[3]}) REFERENCES {f[2]}({f[4]})"
            column_list.append(line)

        tbl_command = f"CREATE TABLE {tbl_name_new} ( {', '.join(column_list)} );"
        # print(tbl_command)
        c.execute(tbl_command)

        # insert data to the new table
        column_names = [t[1] for t in col_data]
        for d in tbl_data:
            flags_ids = [d[i_col] in ids_player_with_matches for i_col, col in enumerate(col_data) if col[1] in columns_with_ids[tbl_name]]
            if not any(flags_ids):
                # skip this row
                continue

            bad_ch = "'"
            d = [f"'{dd.replace(bad_ch, '')}'" if col_data[i][2] == "TEXT" else str(dd) for i, dd in enumerate(d)]
            value_str = ', '.join(d)
            data_command = f"INSERT INTO {tbl_name_new} ( {', '.join(column_names)} ) VALUES ( {value_str} );"
            # print(data_command)
            c.execute(data_command)

    # turn on checking the correctness of foreign keys
    c.execute("PRAGMA foreign_keys = ON;")
    c.execute("PRAGMA legacy_alter_table=OFF;")

    # for (tname,) in table_names:
    #     c.execute(f"PRAGMA foreign_key_list({tname});")
    #     foreign_keys = c.fetchall()
    #     print(f"Foreign keys of {tname}:", foreign_keys)

    conn.commit()
    conn.close()


def fix_database_soccer_1(sqlite_file):
    print("Editing database", sqlite_file)
    conn = sqlite3.connect(sqlite_file)
    conn.text_factory = lambda b: b.decode(errors = 'ignore')
    c = conn.cursor()

    # get all the tables in the database
    query_get_all_tables = "SELECT name FROM sqlite_master WHERE type='table'"
    c.execute(query_get_all_tables)
    table_names = c.fetchall()

    # print(table_names)

    foreign_keys_per_table = {}
    for (tbl_name,) in table_names:
        c.execute(f"PRAGMA foreign_key_list({tbl_name});")
        foreign_keys_per_table[tbl_name] = c.fetchall()
        # print(f"Foreign keys of {tbl_name}:", foreign_keys_per_table[tbl_name])

    # turn off checking the correctness of foreign keys
    c.execute("PRAGMA foreign_keys = OFF;")
    c.execute("PRAGMA legacy_alter_table=ON;")

    # we will downsample the number of players - we fo not need that many, although we might need to get some players back if they are needed for queries
    c.execute(f"SELECT player_api_id FROM Player")
    ids_to_keep = c.fetchall()
    ids_to_keep = [r[0] for r in ids_to_keep]
    # the table has 11k Players, subsampling by a factor of 100
    ids_to_keep = ids_to_keep[::100]
    ids_to_keep = set(ids_to_keep)
    # adding players back
    # for SPIDER_dev_1300
    ids_to_keep.add(30690)
    ids_to_keep.add(30834)

    table_to_subsample = {"Player": ["player_api_id"],
                          "Player_Attributes": ["player_api_id"]}

    # first we want to remove trailing white spaces from all the tables
    for (tbl_name,) in table_names:
        c.execute(f"PRAGMA table_info({tbl_name});")
        col_data = c.fetchall()
        # print(col_data)

        foreign_keys_out = foreign_keys_per_table[tbl_name]

        # get content
        c.execute(f"SELECT * FROM {tbl_name}")
        tbl_data = c.fetchall()

        if tbl_name == "sqlite_sequence":
            continue
        c.execute(f"DROP TABLE {tbl_name};")

        # create new table
        tbl_name_new = tbl_name
        # column extry: NAME TYPE [PRIMARY KEY] [NOT NULL] [DEFAULT 0]
        column_list = [f"{col[1]} {col[2]}" for col in col_data]

        # add primary keys
        cols_with_pr_keys = [col[1] for col in col_data if col[5]]
        if cols_with_pr_keys:
            column_list.append(f"PRIMARY KEY ({', '.join(cols_with_pr_keys)})")

        for f in foreign_keys_out:
            line = f"FOREIGN KEY({f[3]}) REFERENCES {f[2]}({f[4]})"
            column_list.append(line)

        tbl_command = f"CREATE TABLE {tbl_name_new} ( {', '.join(column_list)} );"
        # print(tbl_command)
        c.execute(tbl_command)

        # insert data to the new table
        column_names = [t[1] for t in col_data]
        for d in tbl_data:
            if tbl_name in table_to_subsample:
                flags_ids = [d[i_col] in ids_to_keep for i_col, col in enumerate(col_data) if col[1] in table_to_subsample[tbl_name]]
                if not any(flags_ids):
                    # skip this row
                    continue

            bad_ch = "'"
            d_new = []
            for i, dd in enumerate(d):
                if dd is None:
                    dd_new = "null"
                elif col_data[i][2] == "TEXT":
                    dd_new = f"'{dd.replace(bad_ch, '')}'"
                else:
                    dd_new = str(dd)
                d_new.append(dd_new)

            value_str = ', '.join(d_new)
            data_command = f"INSERT INTO {tbl_name_new} ( {', '.join(column_names)} ) VALUES ( {value_str} );"
            # print(data_command)
            c.execute(data_command)

    # turn on checking the correctness of foreign keys
    c.execute("PRAGMA foreign_keys = ON;")
    c.execute("PRAGMA legacy_alter_table=OFF;")

    # for (tname,) in table_names:
    #     c.execute(f"PRAGMA foreign_key_list({tname});")
    #     foreign_keys = c.fetchall()
    #     print(f"Foreign keys of {tname}:", foreign_keys)

    conn.commit()
    conn.close()


def fix_database_local_govt_mdm(sqlite_file):
    print("Editing database", sqlite_file)
    conn = sqlite3.connect(sqlite_file)
    conn.text_factory = lambda b: b.decode(errors = 'ignore')
    c = conn.cursor()

    data_command = f"INSERT INTO Council_Tax ( council_tax_id, cmi_cross_ref_id ) VALUES ( 6,  102 );"
    c.execute(data_command)
    data_command = f"INSERT INTO Council_Tax ( council_tax_id, cmi_cross_ref_id ) VALUES ( 10,  105 );"
    c.execute(data_command)

    conn.commit()
    conn.close()


implemented_database_fixes = {}
implemented_database_fixes["car_1"] = ('dev', fix_database_car_1)
implemented_database_fixes["wta_1"] = ('dev', fix_database_wta_1)
implemented_database_fixes["cre_Doc_Template_Mgt"] = ('dev', fix_database_cre_Doc_Template_Mgt)
implemented_database_fixes["soccer_1"] = ('train', fix_database_soccer_1)
implemented_database_fixes["local_govt_mdm"] = ('train', fix_database_local_govt_mdm)

databases_add_foreign_keys = {}
databases_delete_foreign_keys = {}
# each key is a list tbl_src, col_src, tbl_tgt, col_tgt
databases_add_foreign_keys["flight_2"] = ('dev', [["flights", "Airline", "airlines", "uid"]])

databases_add_foreign_keys["academic"] = ('train', [["author", "oid", "organization", "oid"]])

databases_add_foreign_keys["insurance_fnol"] = ('train', [["First_Notification_of_Loss", "Customer_ID", "Customers", "Customer_ID"],
                                                        ["First_Notification_of_Loss", "Policy_ID", "Available_Policies", "Policy_ID"]])

databases_add_foreign_keys["store_product"] = ('train', [["store_product", "Product_ID", "product", "product_id"]])

databases_add_foreign_keys["student_assessment"] = ('train', [["Student_Course_Attendance", "student_id", "Students", "student_id"],
                                                              ["Student_Course_Attendance", "course_id", "Courses", "course_id"]])

databases_add_foreign_keys["solvency_ii"] = ('train', [["Assets_in_Events", "Asset_ID", "Assets", "Asset_ID"],
                                                       ["Events", "Channel_ID", "Channels", "Channel_ID"]])

databases_add_foreign_keys["local_govt_and_lot"] = ('train', [["Residents_Services", "resident_id", "Residents", "resident_id"],
                                                              ["Residents_Services", "property_id", "Properties", "property_id"],
                                                              ["Customer_Events", "resident_id", "Residents", "resident_id"],
                                                              ["Customer_Events", "property_id", "Properties", "property_id"],
                                                              ["Customer_Event_Notes", "resident_id", "Residents", "resident_id"],
                                                              ["Customer_Event_Notes", "property_id", "Properties", "property_id"],
                                                              ["Organizations", "parent_organization_id", "Organizations", "organization_id"]])

databases_add_foreign_keys["cre_Doc_Control_Systems"] = ('train', [["Draft_Copies", "document_id", "Documents", "document_id"],
                                                                   ["Draft_Copies", "draft_number", "Document_Drafts", "draft_number"],
                                                                   ["Circulation_History", "document_id", "Documents", "document_id"],
                                                                   ["Circulation_History", "draft_number", "Document_Drafts", "draft_number"],
                                                                   ["Circulation_History", "copy_number", "Draft_Copies", "copy_number"]])

databases_add_foreign_keys["coffee_shop"] = ('train', [["happy_hour", "HH_ID", "happy_hour_member", "HH_ID"],
                                                        ["happy_hour_member", "HH_ID", "happy_hour", "HH_ID"]])

databases_add_foreign_keys["customers_card_transactions"] = ('train', [["Accounts", "customer_id", "Customers", "customer_id"],
                                                                       ["Customers_Cards", "customer_id", "Customers", "customer_id"]])

databases_add_foreign_keys["musical"] = ('train', [["actor", "Musical_ID", "musical", "Musical_ID"]])
databases_delete_foreign_keys ["musical"] = [["actor", "Musical_ID", "actor", "Actor_ID"]]

databases_add_foreign_keys["insurance_and_eClaims"] = ('train', [["Claims_Processing", "Claim_Stage_ID", "Claims_Processing_Stages", "Claim_Stage_ID"]])

databases_add_foreign_keys["local_govt_mdm"] = ('train', [["Benefits_Overpayments", "council_tax_id", "Council_Tax", "council_tax_id"],
                                                          ["Parking_Fines", "council_tax_id", "Council_Tax", "council_tax_id"],
                                                          ["Rent_Arrears", "council_tax_id", "Council_Tax", "council_tax_id"]])

databases_add_foreign_keys["hr_1"] = ('train', [["departments", "LOCATION_ID", "locations", "LOCATION_ID"]])

databases_add_foreign_keys["sakila_1"] = ('train', [["film_text", "film_id", "film", "film_id"],
                                                    ["staff", "store_id", "store", "store_id"]])

databases_add_foreign_keys["scholar"] = ('train', [["paperDataset", "datasetId", "dataset", "datasetId"],
                                                   ["paperDataset", "paperId", "paper", "paperId"]])

databases_add_foreign_keys["formula_1"] = ('train', [["results", "statusId", "status", "statusId"],
                                                     ["races", "year", "seasons", "year"]])

databases_add_foreign_keys["loan_1"] = ('train', [["loan", "cust_ID", "customer", "cust_ID"]])

databases_add_foreign_keys["cre_Drama_Workshop_Groups"] = ('train', [["Bookings", "Store_ID", "Stores", "Store_ID"],
                                                                     ["Invoices", "Product_ID", "Products", "Product_ID"],
                                                                     ["Invoices", "Order_Item_ID", "Order_Items", "Order_Item_ID"],
                                                                     ["Invoice_Items", "Order_ID", "Bookings", "Booking_ID"],
                                                                     ["Invoice_Items", "Product_ID", "Services", "Service_ID"]])

databases_add_foreign_keys["geo"] = ('train', [["lake", "state_name", "state", "state_name"]])

databases_add_foreign_keys["customers_and_products_contacts"] = ('train', [["Contacts", "customer_id", "Customers", "customer_id"]])

databases_add_foreign_keys["company_1"] = ('train', [["dept_locations", "Dnumber", "department", "Dnumber"],
                                                     ["works_on", "Pno", "project", "Pnumber"],
                                                     ["works_on", "Essn", "employee", "Ssn"],
                                                     ["department", "Mgr_ssn", "employee", "Ssn"],
                                                     ["dependent", "Essn", "employee", "Ssn"],
                                                     ["employee", "Super_ssn", "employee", "Ssn"],
                                                     ])

databases_add_foreign_keys["product_catalog"] = ('train', [["Catalog_Contents_Additional_Attributes", "attribute_id", "Attribute_Definitions", "attribute_id"],
                                                           ["Catalog_Contents", "parent_entry_id", "Catalog_Contents", "catalog_entry_id"],
                                                           ["Catalog_Contents", "previous_entry_id", "Catalog_Contents", "catalog_entry_id"],
                                                           ["Catalog_Contents", "next_entry_id", "Catalog_Contents", "catalog_entry_id"]])


def get_path_and_make_backup(db, spider_path, database_path="database"):
    db_path = os.path.join(spider_path, database_path)
    db_path = os.path.join(db_path, db, f"{db}.sqlite")

    backup_file = f"{db_path}.backup"

    # backup the file or restore from the backup
    if os.path.isfile(backup_file):
        shutil.copyfile(backup_file, db_path)
    else:
        shutil.copyfile(db_path, backup_file)

    return db_path


def main(args):
    print(args)

    if args.database:
        databases_to_fix = [args.database]
    else:
        databases_to_fix = set(list(implemented_database_fixes.keys()) + list(databases_add_foreign_keys.keys()))

    for db in databases_to_fix:
        if db not in implemented_database_fixes and db not in databases_add_foreign_keys:
            raise NotImplementedError(f"Do not know how to fix database {db}")

        db_path = get_path_and_make_backup(db, args.spider_path, database_path = args.database_path)
        if db in implemented_database_fixes:
            subset, fixer = implemented_database_fixes[db]
            fixer(db_path)

        if db in databases_add_foreign_keys:
            subset, foreign_key_list = databases_add_foreign_keys[db]
            fix_database_add_foreign_keys(db_path, foreign_key_list,
                                          keys_to_delete=databases_delete_foreign_keys[db] if db in databases_delete_foreign_keys else None)
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
