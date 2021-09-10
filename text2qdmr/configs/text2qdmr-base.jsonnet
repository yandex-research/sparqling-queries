function(args) {
    local save_name = args.save_name,

    local lr_s = '%0.1e' % args.lr,
    local bert_lr_s = '%0.1e' % args.bert_lr,
    local end_lr_s = if args.end_lr == 0 then '0e0' else '%0.1e' % args.end_lr,

    local base_bert_enc_size = 1024,
    local enc_size =  base_bert_enc_size,

    model_name: 'bs=%(bs)d,lr=%(lr)s,bert_lr=%(bert_lr)s,end_lr=%(end_lr)s,att=%(att)d' % (args + {
        lr: lr_s,
        bert_lr: bert_lr_s,
        end_lr: end_lr_s,
    }),

    use_online_data_processing: if std.objectHas(args, 'use_online_data_processing') then args.use_online_data_processing else false,
    num_dataloading_workers: if std.objectHas(args, 'num_dataloading_workers') then args.num_dataloading_workers else 0,

    data: {
        train: {
            name: 'qdmr',
            paths: {
                spider_path: args.spider_data_path + 'train_spider.json',
                tables_path: [
                    args.spider_data_path + 'tables.json',
                    ],
                db_path: args.spider_data_path + 'database',
                break_logic_form_path: args.break_data_path  + 'logical-forms-fixed/train_spider.csv',
                grounding_path: args.grounding_path + 'grnd_list_positive_train.json',
                extracted_value_path: if std.objectHas(args, 'extracted_value_path') then args.extracted_value_path + '/extracted_values_train.json' else '',
            },
            partition: 'train',
            extract_value: {
                max_values_from_database: if std.objectHas(args, 'train_max_values_from_database') then args.train_max_values_from_database else 50,
                value_order:  if std.objectHas(args, 'value_order') then args.value_order else 'sort',
            },
            augment_at_iter_shuffle_tables: if std.objectHas(args, 'augment_at_iter_shuffle_tables') then args.augment_at_iter_shuffle_tables else false,
            augment_at_iter_shuffle_columns: if std.objectHas(args, 'augment_at_iter_shuffle_columns') then args.augment_at_iter_shuffle_columns else false,
            augment_at_iter_shuffle_values: if std.objectHas(args, 'augment_at_iter_shuffle_values') then args.augment_at_iter_shuffle_values else false,
            augment_at_iter_shuffle_qdmr_ordering: if std.objectHas(args, 'augment_at_iter_shuffle_qdmr_ordering') then args.augment_at_iter_shuffle_qdmr_ordering else false,
            augment_at_iter_shuffle_sort_dir: if std.objectHas(args, 'augment_at_iter_shuffle_sort_dir') then args.augment_at_iter_shuffle_sort_dir else false,
            augment_at_iter_shuffle_compsup_op: if std.objectHas(args, 'augment_at_iter_shuffle_compsup_op') then args.augment_at_iter_shuffle_compsup_op else false,
        },
        val: {
            name: 'qdmr',
            paths: {
                spider_path: args.spider_data_path + 'dev.json',
                tables_path: [args.spider_data_path + 'tables.json'],
                db_path: args.spider_data_path + 'database',
                break_logic_form_path: args.break_data_path  + 'logical-forms-fixed/dev_spider.csv',
                grounding_path: args.grounding_path + 'grnd_list_positive_dev.json',
                extracted_value_path: if std.objectHas(args, 'extracted_value_path') then args.extracted_value_path + '/extracted_values_val.json' else '',
            },
            partition: 'dev',
            extract_value: {
                max_values_from_database: if std.objectHas(args, 'eval_max_values_from_database') then args.eval_max_values_from_database else 50,
                matching: false,
            },
        },
        test: {
            name: 'qdmr',
            paths: {
                spider_path: args.spider_data_path + 'dev.json',
                tables_path: [args.spider_data_path + 'tables.json'],
                db_path: args.spider_data_path + 'database',
                break_logic_form_path: args.break_data_path  + 'logical-forms/test_spider.csv',
                grounding_path: '',
                extracted_value_path: if std.objectHas(args, 'extracted_value_path') then args.extracted_value_path + '/extracted_values_test.json' else '',
            },
            partition: 'test',
            extract_value: {
                max_values_from_database: if std.objectHas(args, 'eval_max_values_from_database') then args.eval_max_values_from_database else 50,
                matching: false,
            },
        },
    },
    [if std.objectHas(args, 'add_full_break') then 'full_data' else null]: {
            train: {
                name: 'qdmr-full',
                paths: {
                    spider_path: args.spider_data_path + 'train_spider.json',
                    tables_path: [
                        args.spider_data_path + 'tables.json',
                        ],
                    db_path: args.spider_data_path + 'database',
                    break_logic_form_path: args.break_data_path  + 'logical-forms-fixed/train.csv',
                    grounding_path: args.grounding_path + 'grnd_full_break_train.json',
                },
                partition: 'train',
                extract_value: {
                    max_values_from_database: if std.objectHas(args, 'train_max_values_from_database') then args.train_max_values_from_database else 50,
                    matching: if std.objectHas(args, 'matching') then args.matching else false,
                },
                exclude_names: ['SPIDER'],
            },
    },

    model: {
        name: 'EncDec',
        encoder: {
            name: 'text2qdmr',
            update_config:  {
                name: if std.objectHas(args, 'update_name') then args.update_name else 'relational_transformer',
                num_layers: 8,
                num_heads: 8,
                sc_link: args.sc_link,
                cv_link: args.cv_link,
                ct_foreign_key: args.use_graph_relations,
                cc_foreign_key: args.use_graph_relations,
                cc_table_match: args.use_graph_relations,
                ct_table_match: args.use_graph_relations,
                tc_foreign_key: args.use_graph_relations,
                tc_table_match: args.use_graph_relations,
                tt_foreign_key: args.use_graph_relations,
                qv_token_match: args.sc_link && !args.merge_sc_link,
                qv_default: args.use_type_relations,
                full_grnd_link: if std.objectHas(args, 'full_grnd_link') then args.full_grnd_link else false,
                merge_sc_link: if std.objectHas(args, 'merge_sc_link') then args.merge_sc_link else false,
                default_graph_link: if std.objectHas(args, 'default_graph_link') then args.default_graph_link else false,
                default_link_and_graph: if std.objectHas(args, 'default_link_and_graph') then args.default_link_and_graph else false,
            },
            summarize_header: 'avg',
            bert_token_type: args.pretrained_version == 'bert',
            include_in_memory: ['question', 'grounding'],
            use_relations: if std.objectHas(args, 'use_relations') then args.use_relations else false,
        },   
        decoder: {
            name: 'text2qdmr',
            dropout: 0.20687225956012834,
            desc_attn: 'mha',
            enc_recurrent_size: enc_size,
            recurrent_size : 512,
            loss_type: 'label_smooth',
            use_align_mat: true,
            use_align_loss: false,
            share_pointers: true,
            share_pointer_type: 'dotprod',
            exclude_rules_loss: ['SupUnknownOp', 'UnknownOp', 'UnknownColumnGrounding'],
        },
        encoder_preproc: {
            db_path: args.spider_data_path + "database",
            compute_sc_link: args.sc_link,
            compute_cv_link: args.cv_link,
            fix_issue_16_primary_keys: true,
            pretrained_version: args.pretrained_version,
            save_path: '%s%s,emb=bert,cvlink' % [args.save_data_path, save_name],
            use_bert_unlimited_length: false,
            use_column_type: false,
            use_general_grounding: if std.objectHas(args, 'use_general_grounding') then args.use_general_grounding else true,
            use_graph_relations: args.use_graph_relations,
            use_type_relations: if std.objectHas(args, 'use_type_relations') then args.use_type_relations else true,
            add_cellmatch: if std.objectHas(args, 'add_cellmatch') then args.add_cellmatch else false,
            merge_sc_link: if std.objectHas(args, 'merge_sc_link') then args.merge_sc_link else false,
            construct_general_grounding: if std.objectHas(args, 'construct_general_grounding') then args.construct_general_grounding else false,
            use_bert_masks: if std.objectHas(args, 'use_bert_masks') then args.use_bert_masks else false,
            include_table_name_in_column: if std.objectHas(args, 'include_table_name_in_column') then args.include_table_name_in_column else false,
        },
        decoder_preproc: self.encoder_preproc {
            grammar: {
                name: 'qdmr',
                version: if std.objectHas(args, 'grammar_version') then args.grammar_version else 1,
            },
            use_seq_elem_rules: true,

            include_table_name_in_column:: null,

            save_path: '%s%s,emb=bert,cvlink' % [args.save_data_path, save_name],
            compute_sc_link:: null,
            compute_cv_link:: null,
            db_path:: null,
            fix_issue_16_primary_keys:: null,
            pretrained_version:: null,
            use_bert_unlimited_length:: null,
            use_column_type:: null,
            use_general_grounding:: null,
            use_graph_relations:: null,
            use_type_relations:: null,
            add_cellmatch:: null,
            merge_sc_link:: null,
            construct_general_grounding:: null,
            use_bert_masks:: null,
        },
    },

    train: {
        eval_batch_size: 50,

        keep_every_n: 1000,
        eval_every_n: 100,
        save_every_n: 100,
        report_every_n: 10,
        max_keep: 12,

        num_eval_items: 50,

        batch_size: args.bs,
        num_batch_accumulated: args.num_batch_accumulated,
        clip_grad: 1,

        model_seed: args.att,
        data_seed:  args.att,
        init_seed:  args.att,

        max_steps: args.max_steps,
    },
    optimizer: {
        name: 'bertAdamw',
        lr: 0.0,
        bert_lr: 0.0,
    },
    
    lr_scheduler: {
        name: 'bert_warmup_polynomial_group',
        decay_steps: $.train.max_steps - self.num_warmup_steps,
        power: 0.5,
        start_lrs: [args.lr, args.bert_lr],
        start_lr: 1e-3,
        end_lr: args.end_lr,
        num_warmup_steps: $.train.max_steps / 8,
    },

    log: {
        reopen_to_flush: true,
    }
}
