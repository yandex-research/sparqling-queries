import os

markdown_bold = lambda x: '**' + x + '**' if x else x
markdown_italic = lambda x: '*' + x + '*' if x else x
markdown_header = """ | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |\n """ + \
                """ | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | """


class Visualizer():
    def __init__(self, writer, logdir):
        self.writer = writer
        self.logdir = logdir
        self.visualized_db_info = {}

    def visualize_schema(self, db):
        schema_info = []
        tab_name = ''
        for i, el_or in enumerate(db['column_names_original']):
            schema_info.append([str(i)])
            idx =  el_or[0]
            if idx == -1 or tab_name != db['table_names_original'][idx]:
                tab_name = db['table_names_original'][idx] if idx >= 0 else ''
                schema_info[i].append(markdown_bold(tab_name))
            else:
                schema_info[i].append(' ')
            schema_info[i].append(el_or[-1])
            schema_info[i] += [' ', ' '] # PK, FK

        for k in db['primary_keys']:
            schema_info[k][3] = '+'

        for k, kk in db['foreign_keys']:
            schema_info[k][4] = '--> ' + str(kk)

        text = """ | Idx | Table      | Column | Primary Key | Foreign Key | \n""" + \
                """ | ----------- | ----------- | ----------- | ----------- | ----------- | \n """ 

        for el in schema_info:
            text += ' | ' + ' | '.join(el) + ' | \n'
        return text

    def visualize_results(self, break_item, qdmr_info, gold_qdmr_info):
        if break_item.orig_spider_entry:
            sql = break_item.orig_spider_entry['query']
        else:
            sql = ''
        qdmr, distinct_idx = qdmr_info
        if gold_qdmr_info:
            gold_qdmr, gold_distinct_idx = gold_qdmr_info
        else:
            gold_qdmr, gold_distinct_idx = None, None

        def format_qdmr(qdmr, distinct_idx):
            if qdmr is None:
                return ''
            assert len(qdmr.ops) == len(qdmr.args)
            text = ''
            for i, op in enumerate(qdmr.ops):
                text_arg = ', '.join(qdmr.args[i])
                text_arg = text_arg.replace('|', ':')
                text += str(i + 1) + '.'
                if distinct_idx and '#' + str(i + 1) in distinct_idx:
                    text += markdown_italic('(distinct)')
                text += ' ' + op.upper() + '[' + text_arg + '] <br>'
            return text

        text = ' | ' + break_item.full_name 
        text += ' | '  + break_item.text
        text += ' | '  + sql
        text += ' | '  + format_qdmr(gold_qdmr, gold_distinct_idx)
        text += ' | '  + format_qdmr(qdmr, distinct_idx) + ' | '
        return text

    def visualization(self, break_item, qdmr_info, gold_qdmr_info, res, sql_hardness=None):
        if break_item.schema is not None:
            db_id = break_item.schema.db_id
        else:
            db_id = break_item.subset_name
        file_name = os.path.join(self.logdir, db_id + '_results.md')
        if db_id not in self.visualized_db_info.keys():
            if break_item.schema:
                md_schema = self.visualize_schema(break_item.schema.orig) + " \n "
                if self.writer:
                    self.writer.add_text(db_id, md_schema, 100000)
            else:
                md_schema = ''
            
            with open(file_name, 'w+') as f:
                md_schema += markdown_header
                f.write(md_schema + '\n')
            self.visualized_db_info[db_id] = {'res': 0., 'count': 0.}

        md_res = self.visualize_results(break_item, qdmr_info, gold_qdmr_info)
        self.visualized_db_info[db_id]['count'] += 1
        if res:
            md_res += '+ | {} | \n '.format(sql_hardness) 
            self.visualized_db_info[db_id]['res'] += 1
        else:
            md_res += '- | {} | \n '.format(sql_hardness) 
        if self.writer:
            self.writer.add_text(db_id, markdown_header + ' \n ' + md_res, break_item.subset_idx)
        with open(file_name, 'a') as f:
            f.write(md_res.replace(':', ':\u200b'))

    def finalize(self):
        for db_id, info in self.visualized_db_info.items():
            file_name = os.path.join(self.logdir, db_id + '_results.md')
            res = "{:.4f}".format(info['res'] / info['count'])
            text = '***\n Exec acc: ' + markdown_bold(res) + '\n'
            if self.writer:
                self.writer.add_text(db_id, text, 100000)
            with open(file_name, 'a') as f:
                f.write(text)