import collections
import itertools
import json
import os
import copy

import attr
import numpy as np
import torch.nn.functional as F
import torch

from text2qdmr.model.modules import abstract_preproc, attention, variational_lstm
from text2qdmr.utils import registry
from text2qdmr.utils import serialization
from text2qdmr.utils.serialization import ComplexEncoder, ComplexDecoder
from text2qdmr.utils import vocab
from text2qdmr.model.modules.decoder_utils import get_field_presence_info, TreeState, lstm_init
from text2qdmr.model.decoder_tree.infer_tree_traversal import InferenceTreeTraversal
from text2qdmr.model.decoder_tree.train_tree_traversal import TrainTreeTraversal
from text2qdmr.model.decoder_tree.tree_traversal import TreeTraversal
from text2qdmr.model.decoder_tree.infer_rules import *

@attr.s
class BreakDecoderPreprocItem:
    tree = attr.ib()
    orig_code = attr.ib()
    spider_idx = attr.ib()

class BreakDecoderPreproc(abstract_preproc.AbstractPreproc):
    def __init__(
            self,
            grammar,
            save_path,
            min_freq=3,
            max_count=5000,
            use_seq_elem_rules=False):
        self.grammar = registry.construct('grammar', grammar)
        self.ast_wrapper = self.grammar.ast_wrapper

        self.observed_productions_path = os.path.join(save_path, 'observed_productions.json')
        self.grammar_rules_path = os.path.join(save_path, 'grammar_rules.json')
        self.data_dir = os.path.join(save_path, 'dec')

        self.use_seq_elem_rules = use_seq_elem_rules

        self.items = collections.defaultdict(list)
        self.sum_type_constructors = collections.defaultdict(set)
        self.field_presence_infos = collections.defaultdict(set)
        self.seq_lengths = collections.defaultdict(set)
        self.primitive_types = set()

        self.all_rules = None
        self.rules_mask = None

    def validate_item(self, item, section, value_unit_dict=None):
        if item.qdmr_code:
            num_choices = len(item.qdmr_code)
            assert len(item.values) == len(item.qdmr_code)
        else:
            assert section != 'train'
            num_choices = 0

        all_results, all_parsed = [], []
        for i in range(num_choices):
            parsed = self.grammar.parse(item.qdmr_code[i], item.schema, value_unit_dict, item.column_data[i], section)

            if parsed:
                self.ast_wrapper.verify_ast(parsed)
                all_results.append(True)
                all_parsed.append(parsed)
            
        if num_choices == 0 and section == 'test':
            all_results.append(True) 
            all_parsed.append(None)
        return all_results, all_parsed

    def add_item(self, item, section, idx_to_add, validation_info):
        for root in np.array(validation_info)[idx_to_add]:
            if root:
                self._record_productions(root)
        
        dec_item = self.preprocess_item(item, idx_to_add, validation_info)
        self.items[section].append(dec_item)
    
    @classmethod
    def preprocess_item(cls, item, idx_to_add, validation_info):
        if item.qdmr_code:
            all_qdmr_code = np.array(item.qdmr_code, dtype=object)[idx_to_add]
            dec_item = BreakDecoderPreprocItem(
                        tree=validation_info,
                        orig_code=all_qdmr_code.tolist(),
                        spider_idx=None)
        else:
            dec_item = BreakDecoderPreprocItem(
                        tree=validation_info,
                        orig_code=None,
                        spider_idx=None)
        return dec_item

    def clear_items(self):
        self.items = collections.defaultdict(list)
        
    def save(self, partition=None):
        os.makedirs(self.data_dir, exist_ok=True)

        for section, items in self.items.items():
            with open(os.path.join(self.data_dir, section + '.jsonl'), 'w') as f:
                for item in items:
                    f.write(json.dumps(item, cls=ComplexEncoder) + '\n')
        if partition is None:
            # observed_productions
            self.sum_type_constructors = serialization.to_dict_with_sorted_values(
                self.sum_type_constructors)
            self.field_presence_infos = serialization.to_dict_with_sorted_values(
                self.field_presence_infos, key=str)
            self.seq_lengths = serialization.to_dict_with_sorted_values(
                self.seq_lengths)
            self.primitive_types = sorted(self.primitive_types)
            with open(self.observed_productions_path, 'w') as f:
                json.dump({
                    'sum_type_constructors': self.sum_type_constructors,
                    'field_presence_infos': self.field_presence_infos,
                    'seq_lengths': self.seq_lengths,
                    'primitive_types': self.primitive_types,
                }, f, indent=2, sort_keys=True)

            # grammar
            self.all_rules, self.rules_mask = self._calculate_rules()
            with open(self.grammar_rules_path, 'w') as f:
                json.dump({
                    'all_rules': self.all_rules,
                    'rules_mask': self.rules_mask,
                }, f, indent=2, sort_keys=True)

    def load(self):
        observed_productions = json.load(open(self.observed_productions_path))
        self.sum_type_constructors = observed_productions['sum_type_constructors']
        self.field_presence_infos = observed_productions['field_presence_infos']
        self.seq_lengths = observed_productions['seq_lengths']
        self.primitive_types = observed_productions['primitive_types']

        grammar = json.load(open(self.grammar_rules_path))
        self.all_rules = serialization.tuplify(grammar['all_rules'])
        self.rules_mask = grammar['rules_mask']

    def dataset(self, section):
        return [
            BreakDecoderPreprocItem(**json.loads(line, cls=ComplexDecoder))
            for line in open(os.path.join(self.data_dir, section + '.jsonl'))]

    def _record_productions(self, tree):
        queue = [(tree, False)]
        while queue:
            node, is_seq_elem = queue.pop()
            node_type = node['_type']

            # Rules of the form:
            # expr -> Attribute | Await | BinOp | BoolOp | ...
            # expr_seq_elem -> Attribute | Await | ... | Template1 | Template2 | ...
            for type_name in [node_type] + node.get('_extra_types', []):
                if type_name in self.ast_wrapper.constructors:
                    sum_type_name = self.ast_wrapper.constructor_to_sum_type[type_name]
                    if is_seq_elem and self.use_seq_elem_rules:
                        self.sum_type_constructors[sum_type_name + '_seq_elem'].add(type_name)
                    else:
                        self.sum_type_constructors[sum_type_name].add(type_name)

            # Rules of the form:
            # FunctionDef
            # -> identifier name, arguments args
            # |  identifier name, arguments args, stmt* body
            # |  identifier name, arguments args, expr* decorator_list
            # |  identifier name, arguments args, expr? returns
            # ...
            # |  identifier name, arguments args, stmt* body, expr* decorator_list, expr returns
            assert node_type in self.ast_wrapper.singular_types
            field_presence_info = get_field_presence_info(
                self.ast_wrapper,
                node,
                self.ast_wrapper.singular_types[node_type].fields)
            self.field_presence_infos[node_type].add(field_presence_info)

            for field_info in self.ast_wrapper.singular_types[node_type].fields:
                field_value = node.get(field_info.name, [] if field_info.seq else None)
                to_enqueue = []
                if field_info.seq:
                    # Rules of the form:
                    # stmt* -> stmt
                    #        | stmt stmt
                    #        | stmt stmt stmt
                    self.seq_lengths[field_info.type + '*'].add(len(field_value))
                    to_enqueue = field_value
                else:
                    to_enqueue = [field_value]
                for child in to_enqueue:
                    if isinstance(child, collections.abc.Mapping) and '_type' in child:
                        queue.append((child, field_info.seq))
                    else:
                        self.primitive_types.add(type(child).__name__)

    def _calculate_rules(self):
        offset = 0

        all_rules = []
        rules_mask = {}

        # Rules of the form:
        # expr -> Attribute | Await | BinOp | BoolOp | ...
        # expr_seq_elem -> Attribute | Await | ... | Template1 | Template2 | ...
        for parent, children in sorted(self.sum_type_constructors.items()):
            assert not isinstance(children, set)
            rules_mask[parent] = (offset, offset + len(children))
            offset += len(children)
            all_rules += [(parent, child) for child in children]

        # Rules of the form:
        # FunctionDef
        # -> identifier name, arguments args
        # |  identifier name, arguments args, stmt* body
        # |  identifier name, arguments args, expr* decorator_list
        # |  identifier name, arguments args, expr? returns
        # ...
        # |  identifier name, arguments args, stmt* body, expr* decorator_list, expr returns
        for name, field_presence_infos in sorted(self.field_presence_infos.items()):
            assert not isinstance(field_presence_infos, set)
            rules_mask[name] = (offset, offset + len(field_presence_infos))
            offset += len(field_presence_infos)
            all_rules += [(name, presence) for presence in field_presence_infos]

        # Rules of the form:
        # stmt* -> stmt
        #        | stmt stmt
        #        | stmt stmt stmt
        for seq_type_name, lengths in sorted(self.seq_lengths.items()):
            assert not isinstance(lengths, set)
            rules_mask[seq_type_name] = (offset, offset + len(lengths))
            offset += len(lengths)
            all_rules += [(seq_type_name, i) for i in lengths]

        return tuple(all_rules), rules_mask

    def _all_tokens(self, root):
        queue = [root]
        while queue:
            node = queue.pop()
            type_info = self.ast_wrapper.singular_types[node['_type']]

            for field_info in reversed(type_info.fields):
                field_value = node.get(field_info.name)
                if field_info.type in self.grammar.general_pointers:
                    pass
                elif field_info.type in self.ast_wrapper.primitive_types:
                    for token in self.grammar.tokenize_field_value(field_value):
                        yield token
                elif isinstance(field_value, (list, tuple)):
                    queue.extend(field_value)
                elif field_value is not None:
                    queue.append(field_value)


@registry.register('decoder', 'text2qdmr')
class BreakDecoder(torch.nn.Module):
    Preproc = BreakDecoderPreproc

    def __init__(
            self,
            device,
            preproc,
            #
            rule_emb_size=128,
            node_embed_size=64,
            # TODO: This should be automatically inferred from encoder
            enc_recurrent_size=256,
            recurrent_size=256,
            dropout=0.,
            desc_attn='bahdanau',
            copy_pointer=None,
            multi_loss_type='logsumexp',
            sup_att=None,
            use_align_mat=False,
            use_align_loss=False,
            enumerate_order=False,
            loss_type="softmax",
            share_pointers=False,
            share_pointer_type=None,
            exclude_rules_loss=()):
        super().__init__()
        self._device = device
        self.preproc = preproc
        self.ast_wrapper = preproc.ast_wrapper

        self.rule_emb_size = rule_emb_size
        self.node_emb_size = node_embed_size
        self.enc_recurrent_size = enc_recurrent_size
        self.recurrent_size = recurrent_size

        self.rules_index = {v: idx for idx, v in enumerate(self.preproc.all_rules)}
        self.use_align_mat = use_align_mat
        self.use_align_loss = use_align_loss
        self.enumerate_order = enumerate_order
        self.share_pointers = share_pointers
        self.share_pointer_type = share_pointer_type
        self.mask_values = True # TODO delete this

        if use_align_mat:
            from text2qdmr.model.modules import decoder_utils
            self.compute_align_loss = lambda *args: \
                decoder_utils.compute_align_loss(self, *args)
            self.compute_pointer_with_align = lambda *args: \
                decoder_utils.compute_pointer_with_align(self, *args)

        if self.preproc.use_seq_elem_rules:
            self.node_type_vocab = vocab.Vocab(
                sorted(self.preproc.primitive_types) +
                sorted(self.ast_wrapper.custom_primitive_types) +
                sorted(self.preproc.sum_type_constructors.keys()) +
                sorted(self.preproc.field_presence_infos.keys()) +
                sorted(self.preproc.seq_lengths.keys()),
                special_elems=())
        else:
            self.node_type_vocab = vocab.Vocab(
                sorted(self.preproc.primitive_types) +
                sorted(self.ast_wrapper.custom_primitive_types) +
                sorted(self.ast_wrapper.sum_types.keys()) +
                sorted(self.ast_wrapper.singular_types.keys()) +
                sorted(self.preproc.seq_lengths.keys()),
                special_elems=())

        self.state_update = variational_lstm.RecurrentDropoutLSTMCell(
            input_size=self.rule_emb_size * 2 + self.enc_recurrent_size + self.recurrent_size + self.node_emb_size,
            hidden_size=self.recurrent_size,
            dropout=dropout)

        self.attn_type = desc_attn
        if desc_attn == 'bahdanau':
            self.desc_attn = attention.BahdanauAttention(
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size,
                proj_size=50)
        elif desc_attn == 'mha':
            self.desc_attn = attention.MultiHeadedAttention(
                h=8,
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size)
        elif desc_attn == 'mha-1h':
            self.desc_attn = attention.MultiHeadedAttention(
                h=1,
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size)
        elif desc_attn == 'sep':
            self.question_attn = attention.MultiHeadedAttention(
                h=1,
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size)
            self.schema_attn = attention.MultiHeadedAttention(
                h=1,
                query_size=self.recurrent_size,
                value_size=self.enc_recurrent_size)
        else:
            # TODO: Figure out how to get right sizes (query, value) to module
            self.desc_attn = desc_attn
        self.sup_att = sup_att

        self.rule_logits = torch.nn.Sequential(
            torch.nn.Linear(self.recurrent_size, self.rule_emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.rule_emb_size, len(self.rules_index)))
        self.rule_embedding = torch.nn.Embedding(
            num_embeddings=len(self.rules_index),
            embedding_dim=self.rule_emb_size)

        assert share_pointers
        self.pointer_action_emb_proj = torch.nn.Linear(self.enc_recurrent_size, self.rule_emb_size)
        if share_pointer_type == 'dotprod':
            self.pointer = attention.ScaledDotProductPointer(
                query_size=self.recurrent_size,
                key_size=self.enc_recurrent_size)
        elif share_pointer_type == 'bahdanau':
            self.pointer = attention.BahdanauPointer(
                query_size=self.recurrent_size,
                key_size=self.enc_recurrent_size,
                proj_size=50)
        else:
            raise Exception(f'Unknown share_pointer_type {share_pointer_type}')

        max_num_refs = 20

        self.ref_logits = torch.nn.Sequential(
            torch.nn.Linear(self.recurrent_size, self.rule_emb_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.rule_emb_size, max_num_refs))
        self.ref_embedding = torch.nn.Embedding(
            num_embeddings=max_num_refs,
            embedding_dim=self.rule_emb_size)

        if multi_loss_type == 'logsumexp':
            self.multi_loss_reduction = lambda logprobs: -torch.logsumexp(logprobs, dim=1)
        elif multi_loss_type == 'mean':
            self.multi_loss_reduction = lambda logprobs: -torch.mean(logprobs, dim=1)

        self.node_type_embedding = torch.nn.Embedding(
            num_embeddings=len(self.node_type_vocab),
            embedding_dim=self.node_emb_size)

        # TODO batching
        self.zero_rule_emb = torch.zeros(1, self.rule_emb_size, device=self._device)
        self.zero_recurrent_emb = torch.zeros(1, self.recurrent_size, device=self._device)
        if loss_type == "softmax":
            self.xent_loss = torch.nn.CrossEntropyLoss(reduction='none')
        elif loss_type == "label_smooth":
            self.xent_loss = self.label_smooth_loss

        self.ref_loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.exclude_rules_loss = exclude_rules_loss

    def label_smooth_loss(self, X, target, smooth_value=0.1, reduction=-1):
        if self.training and X.shape[1] != 1:
            logits = torch.log_softmax(X, dim=1)
            batch_size, size = X.size()
            assert target.numel() == batch_size, f"Have {target.numel()} targets for the batch of {batch_size} elements, should be equal"
            target = target.view(-1).unsqueeze(1)
            one_hot = torch.full(X.size(), smooth_value / (size - 1)).to(X.device)
            one_hot.scatter_(1, target, 1 - smooth_value)
            reduction = "batchmean" if reduction is -1 else reduction
            loss = F.kl_div(logits, one_hot, reduction=reduction)
            return loss.unsqueeze(0)
        else:
            reduction = "none" if reduction is -1 else reduction
            return torch.nn.functional.cross_entropy(X, target, reduction="none")

    @classmethod
    def _calculate_rules(cls, preproc):
        offset = 0

        all_rules = []
        rules_mask = {}

        # Rules of the form:
        # expr -> Attribute | Await | BinOp | BoolOp | ...
        # expr_seq_elem -> Attribute | Await | ... | Template1 | Template2 | ...
        for parent, children in sorted(preproc.sum_type_constructors.items()):
            assert parent not in rules_mask
            rules_mask[parent] = (offset, offset + len(children))
            offset += len(children)
            all_rules += [(parent, child) for child in children]

        # Rules of the form:
        # FunctionDef
        # -> identifier name, arguments args
        # |  identifier name, arguments args, stmt* body
        # |  identifier name, arguments args, expr* decorator_list
        # |  identifier name, arguments args, expr? returns
        # ...
        # |  identifier name, arguments args, stmt* body, expr* decorator_list, expr returns
        for name, field_presence_infos in sorted(preproc.field_presence_infos.items()):
            assert name not in rules_mask
            rules_mask[name] = (offset, offset + len(field_presence_infos))
            offset += len(field_presence_infos)
            all_rules += [(name, presence) for presence in field_presence_infos]

        # Rules of the form:
        # stmt* -> stmt
        #        | stmt stmt
        #        | stmt stmt stmt
        for seq_type_name, lengths in sorted(preproc.seq_lengths.items()):
            assert seq_type_name not in rules_mask
            rules_mask[seq_type_name] = (offset, offset + len(lengths))
            offset += len(lengths)
            all_rules += [(seq_type_name, i) for i in lengths]

        return all_rules, rules_mask

    def compute_loss_batched(self, enc_inputs, examples, desc_encs, execution_maps, debug):
        assert not (self.enumerate_order and self.training)

        mle_losses = self.compute_loss_given_execution_plan_batched(execution_maps, desc_encs)

        assert not self.use_align_loss
        return mle_losses

    @staticmethod
    def compute_decoder_input(enc_input, example, desc_enc, 
                                    exclude_rules_loss,
                                    preproc,
                                    sup_att,
                                    attn_type,
                                    debug=False):
        grnd_idx = desc_enc.grnd_idx
        debug_for_excluding = len(exclude_rules_loss) != 0
        rules_index = {v: idx for idx, v in enumerate(preproc.all_rules)}
        traversal = TrainTreeTraversal(preproc, desc_enc, rules_index, exclude_rules_loss=exclude_rules_loss, debug=debug_for_excluding)

        traversal.step(None)
        queue = [
            TreeState(
                node=example.tree[grnd_idx],
                parent_field_type=preproc.grammar.root_type,
            )
        ]
        while queue:
            item = queue.pop()
            node = item.node
            parent_field_type = item.parent_field_type

            if isinstance(node, (list, tuple)):
                node_type = parent_field_type + '*'
                rule = (node_type, len(node))
                rule_idx = rules_index[rule]
                assert traversal.cur_item.state == TreeTraversal.State.LIST_LENGTH_APPLY
                traversal.step(rule_idx)

                if preproc.use_seq_elem_rules and parent_field_type in preproc.ast_wrapper.sum_types:
                    parent_field_type += '_seq_elem'

                for i, elem in reversed(list(enumerate(node))):
                    queue.append(
                        TreeState(
                            node=elem,
                            parent_field_type=parent_field_type,
                        ))
                continue

            if parent_field_type in preproc.grammar.ref:
                assert isinstance(node, int)
                assert traversal.cur_item.state == TreeTraversal.State.REF_APPLY
                traversal.step(node)
                continue

            if parent_field_type in preproc.grammar.general_pointers:
                assert isinstance(node, int)
                assert traversal.cur_item.state == TreeTraversal.State.GENERAL_POINTER_APPLY
                pointer_map = desc_enc.pointer_maps.get(parent_field_type)
                if pointer_map:
                    values = pointer_map[node]
                    assert sup_att != '1h'
                    traversal.step(values[0], values[1:])
                else:
                    traversal.step(node)
                continue

            if parent_field_type in preproc.ast_wrapper.primitive_types:
                # value
                field_type = type(node).__name__
                assert traversal.cur_item.state == TreeTraversal.State.GEN_TOKEN_APPLY
                traversal.step(node)
                continue

            type_info = preproc.ast_wrapper.singular_types[node['_type']]

            if parent_field_type in preproc.sum_type_constructors:
                # ApplyRule, like expr -> Call
                rule = (parent_field_type, type_info.name)
                rule_idx = rules_index[rule]
                assert traversal.cur_item.state == TreeTraversal.State.SUM_TYPE_APPLY
                extra_rules = [
                    rules_index[parent_field_type, extra_type]
                    for extra_type in node.get('_extra_types', [])]
                traversal.step(rule_idx, extra_rules)

            if type_info.fields:
                # ApplyRule, like Call -> expr[func] expr*[args] keyword*[keywords]
                # Figure out which rule needs to be applied
                present = get_field_presence_info(preproc.ast_wrapper, node, type_info.fields)
                rule = (node['_type'], tuple(present))
                rule_idx = rules_index[rule]
                assert traversal.cur_item.state == TreeTraversal.State.CHILDREN_APPLY
                traversal.step(rule_idx)

            # reversed so that we perform a DFS in left-to-right order
            for field_info in reversed(type_info.fields):
                if field_info.name not in node:
                    continue

                queue.append(
                    TreeState(
                        node=node[field_info.name],
                        parent_field_type=field_info.type,
                    ))

        traversal_step_info = {}
        traversal.traversal_step_log.append(traversal_step_info)
        
        traversal_step_info["prev_action_emb_type"] = traversal.prev_action_emb_type
        traversal_step_info["prev_action_emb_idx"] = traversal.prev_action_emb_idx

        return copy.deepcopy(traversal.traversal_step_log)

    def compute_loss_given_execution_plan_batched(self, decoder_inputs_batch, desc_enc_batch):

        model = self

        # sort by decreasing lengths
        batch_size = len(decoder_inputs_batch)
        decoder_length = [len(plan) for plan in decoder_inputs_batch]
        new_batch_order = [i_ for l_, i_ in sorted(zip(decoder_length, list(range(batch_size))), reverse=True)]
        decoder_inputs_batch = [decoder_inputs_batch[i_] for i_ in new_batch_order]
        desc_enc_batch = [desc_enc_batch[i_] for i_ in new_batch_order]
        decoder_length = [len(plan) for plan in decoder_inputs_batch] # recompute for the reordered batch
        max_num_steps = decoder_length[0] - 1 # the last item is a barrier

        # merge encoder memory
        encoder_lens = [desc_enc.memory.shape[1] for desc_enc in desc_enc_batch]
        encoder_memory = torch.nn.utils.rnn.pad_sequence([desc_enc.memory.squeeze(0) for desc_enc in desc_enc_batch], batch_first=True)
        encoder_memory_mask = torch.arange(max(encoder_lens))[None, :] < torch.tensor(encoder_lens)[:, None]
        encoder_memory_mask = encoder_memory_mask.to(device=encoder_memory.device)

        # merge pointer memory
        if not self.use_align_mat:
            pointer_memories_lens = {}
            pointer_memories = {}
            pointer_memories_mask = {}
            for node_type in desc_enc_batch[0].pointer_memories.keys():
                pointer_memories_lens[node_type] = [desc_enc.pointer_memories[node_type].shape[1] for desc_enc in desc_enc_batch]
                pointer_memories[node_type] = torch.nn.utils.rnn.pad_sequence([desc_enc.pointer_memories[node_type].squeeze(0) for desc_enc in desc_enc_batch], batch_first=True)
                pointer_memories_mask[node_type] = torch.arange(max(pointer_memories_lens[node_type]))[None, :] < torch.tensor(pointer_memories_lens[node_type])[:, None]
                pointer_memories_mask[node_type] = pointer_memories_mask[node_type].to(device=pointer_memories[node_type].device)

        if self.use_align_mat:
            # merge desc_enc_batch[:].m2c_align_mat for the pointer loss
            m2c_align_mat_max_size = (
                max(desc_enc.m2c_align_mat.size(i_dim) for desc_enc in desc_enc_batch) for i_dim in range(2)
            )
            m2c_align_mat_batch = torch.zeros([batch_size] + list(m2c_align_mat_max_size),
                                            device=desc_enc_batch[0].m2c_align_mat.device,
                                            dtype=desc_enc_batch[0].m2c_align_mat.dtype)
            for i_b in range(batch_size):
                m2c_align_mat_batch[i_b,
                                    :desc_enc_batch[i_b].m2c_align_mat.size(0),
                                    :desc_enc_batch[i_b].m2c_align_mat.size(1)] =\
                                        desc_enc_batch[i_b].m2c_align_mat

        # prepare node embeddings
        # node_type_emb shape: batch (=1) x emb_size
        node_type_idx_for_step_and_batch = []
        for i_b in range(batch_size):
            node_type_idx_batch = [] 
            for i_step in range(len(decoder_inputs_batch[i_b]) - 1): # the last item is a barrier
                node_type = decoder_inputs_batch[i_b][i_step]["node_type"] # type: str
                node_type_idx = model._index(model.node_type_vocab, node_type) # type: int tensor
                node_type_idx_batch.append(node_type_idx)
            node_type_idx_batch = torch.cat(node_type_idx_batch, dim=0)
            node_type_idx_for_step_and_batch.append(node_type_idx_batch)
        # cat and pad with some embeddings index - it should never be actually used
        padding_value = model._index(model.node_type_vocab, "root").item() # type: int
        node_type_idx_for_step_and_batch = torch.nn.utils.rnn.pad_sequence(node_type_idx_for_step_and_batch,
                                                                           batch_first=False,
                                                                           padding_value=padding_value)

        # prepare action embeddings and a loss map
        action_embeddings_all_dict = {"zero_rule_emb" : model.zero_rule_emb}
        action_embeddings_index_for_key = {"zero_rule_emb" : 0}
        action_embeddings_key_for_index = ["zero_rule_emb"]
        action_emb_keys_for_batch_and_step = []
        loss_type_for_batch_and_step = []
        loss_target_for_batch_and_step = []
        ref_count_steps_for_batch_and_step = []
        ref_count_step_barrier = -1
        for i_b in range(batch_size):
            action_emb_key_for_step = ["zero_rule_emb"] # add a barrier action at the beginning
            loss_type_for_step = [0]
            loss_target_for_step = [0]
            ref_count_steps_for_step = [ref_count_step_barrier]
            desc_enc = desc_enc_batch[i_b]
            for i_step in range(len(decoder_inputs_batch[i_b]) - 1): # the last item is a barrier
                step_input = decoder_inputs_batch[i_b][i_step]
                node_type = step_input["node_type"]
                step_input_next = decoder_inputs_batch[i_b][i_step + 1] # different step index here is an artifact of information collection - this is ugly indeed
                action_emb_type = step_input_next["prev_action_emb_type"]
                action_emb_idx = step_input_next["prev_action_emb_idx"]
                ref_count_steps = ref_count_step_barrier
                if action_emb_type == "zero_rule_emb":
                    action_emb = model.zero_rule_emb
                    action_emb_key = action_emb_type
                    loss_type = 0
                    loss_target = 0
                elif action_emb_type == "_update_prev_action_emb_apply_rule":
                    rule_idx = model._tensor([action_emb_idx])
                    action_emb = model.rule_embedding(rule_idx)
                    action_emb_key = (action_emb_type, action_emb_idx)
                    loss_type = 1
                    loss_target = action_emb_idx

                    # add excluded items
                    if model.preproc.all_rules[action_emb_idx][-1] in model.exclude_rules_loss:
                        # simply ignore this position for the loss
                        loss_type = 0
                elif action_emb_type == "_update_prev_action_emb_pointer":
                    pointer_action_emb_proj = model.pointer_action_emb_proj[node_type]\
                                                if not model.share_pointers else model.pointer_action_emb_proj
                    action_emb = pointer_action_emb_proj(desc_enc.pointer_memories[node_type][:, action_emb_idx])
                    action_emb_key = (action_emb_type, action_emb_idx, i_b, i_step)
                    loss_type = 2
                    loss_target = action_emb_idx
                elif action_emb_type == "_update_prev_action_emb_gen_ref":
                    ref_idx = model._tensor([action_emb_idx])
                    action_emb = model.ref_embedding(ref_idx)
                    action_emb_key = (action_emb_type, action_emb_idx)
                    loss_type = 3
                    loss_target = action_emb_idx
                    assert "ref_count_steps" in step_input
                    ref_count_steps = step_input["ref_count_steps"]
                else:
                    raise RuntimeError(f"Unimplemeted embedding type {action_emb_type} with index {action_emb_idx}")

                if action_emb_key not in action_embeddings_all_dict:
                    assert len(action_embeddings_key_for_index) == len(action_embeddings_all_dict)
                    action_embeddings_key_for_index.append(action_emb_key)
                    action_embeddings_index_for_key[action_emb_key] = len(action_embeddings_all_dict)
                    action_embeddings_all_dict[action_emb_key] = action_emb
                else:
                    assert (action_embeddings_all_dict[action_emb_key] - action_emb).norm().item() < 1e-8
                
                action_emb_key_for_step.append(action_emb_key)
                loss_type_for_step.append(loss_type)
                loss_target_for_step.append(loss_target)
                ref_count_steps_for_step.append(ref_count_steps)

            action_emb_keys_for_batch_and_step.append(action_emb_key_for_step)
            loss_type_for_batch_and_step.append(model._tensor(loss_type_for_step))
            loss_target_for_batch_and_step.append(model._tensor(loss_target_for_step))
            ref_count_steps_for_batch_and_step.append(model._tensor(ref_count_steps_for_step))
        
        # prepare action embedding layer
        action_emb_weights = []
        for i_ in range(len(action_embeddings_index_for_key)):
            emb_key = action_embeddings_key_for_index[i_]
            action_emb_weights.append(action_embeddings_all_dict[emb_key].view(-1))
        action_emb_weights = torch.stack(action_emb_weights, dim=0)

        action_embedding_layer = torch.nn.Embedding(num_embeddings=len(action_embeddings_index_for_key),
                                                    embedding_dim=model.rule_emb_size,
                                                    _weight=action_emb_weights)

        # prepare action embedding indices
        action_emb_idx_for_step_and_batch = []
        for i_b in range(batch_size):
            action_emb_idx_for_step = [action_embeddings_index_for_key[key] for key in action_emb_keys_for_batch_and_step[i_b]]
            action_emb_idx_for_step = model._tensor(action_emb_idx_for_step)
            action_emb_idx_for_step_and_batch.append(action_emb_idx_for_step)
        # cat and pad with some embeddings index - it should never be actually used
        padding_value = action_embeddings_index_for_key["zero_rule_emb"] # type: int
        action_emb_idx_for_step_and_batch = torch.nn.utils.rnn.pad_sequence(action_emb_idx_for_step_and_batch,
                                                                            batch_first=False,
                                                                            padding_value=padding_value)
        
        # prepare indices for parent actions
        parent_node_idx_for_step_and_batch = []
        for i_b in range(batch_size):
            parent_node_idx_for_step = []
            for i_step in range(len(decoder_inputs_batch[i_b]) - 1): # the last item is a barrier
                parent_node_idx = decoder_inputs_batch[i_b][i_step]["parent_h_idx"]
                parent_node_idx = parent_node_idx + 1 # adding a barrier node at the beginning
                parent_node_idx_for_step.append(parent_node_idx)
            parent_node_idx_for_step = model._tensor(parent_node_idx_for_step)
            parent_node_idx_for_step_and_batch.append(parent_node_idx_for_step)
        # cat and pad with some embeddings index - it should never be actually used
        padding_value = 0 # type: int
        parent_node_idx_for_step_and_batch = torch.nn.utils.rnn.pad_sequence(parent_node_idx_for_step_and_batch,
                                                                             batch_first=False,
                                                                             padding_value=padding_value)
        padding_value = 0 # type: int
        loss_type_for_batch_and_step = torch.nn.utils.rnn.pad_sequence(loss_type_for_batch_and_step,
                                                                       batch_first=False,
                                                                       padding_value=padding_value)
        loss_target_for_batch_and_step = torch.nn.utils.rnn.pad_sequence(loss_target_for_batch_and_step,
                                                                         batch_first=False,
                                                                         padding_value=padding_value)
        ref_count_steps_for_batch_and_step = torch.nn.utils.rnn.pad_sequence(ref_count_steps_for_batch_and_step,
                                                                         batch_first=False,
                                                                         padding_value=ref_count_step_barrier)
                                                                         

        # init sequence decoding 
        recurrent_state = lstm_init(
            model._device, None, model.recurrent_size, batch_size
        )
        outputs = recurrent_state[0].unsqueeze(0) # add dummy dimension for the sequence length
        action_embs = model.zero_rule_emb.expand(batch_size, -1).unsqueeze(0) # clone for the batch and add dummy dimension for the sequence length
        action_emb = action_embs[-1]

        batch_size_cur = batch_size # batch size will get smaller when shorter sequences end
        decoder_inputs_batch_cur = decoder_inputs_batch
        encoder_memory_cur = encoder_memory
        encoder_memory_mask_cur = encoder_memory_mask
        recurrent_state_cur = recurrent_state
        node_type_idx_for_step_and_batch_cur = node_type_idx_for_step_and_batch
        action_emb_idx_for_step_and_batch_cur = action_emb_idx_for_step_and_batch
        parent_node_idx_for_step_and_batch_cur = parent_node_idx_for_step_and_batch
        for i_step in range(max_num_steps):
            while i_step >= decoder_length[batch_size_cur - 1] - 1:
                batch_size_cur = batch_size_cur - 1
                assert batch_size_cur >= 1, f"Trying to do too many steps ({i_step}) for decoder lengths {decoder_length}"
                decoder_inputs_batch_cur = decoder_inputs_batch_cur[:-1]
                encoder_memory_cur = encoder_memory_cur[:-1] 
                encoder_memory_mask_cur = encoder_memory_mask_cur[:-1]
                recurrent_state_cur = [r[:-1] for r in recurrent_state_cur]
                node_type_idx_for_step_and_batch_cur = node_type_idx_for_step_and_batch_cur[:, :-1]
                action_emb_idx_for_step_and_batch_cur = action_emb_idx_for_step_and_batch_cur[:, :-1]
                parent_node_idx_for_step_and_batch_cur = parent_node_idx_for_step_and_batch_cur[:, :-1]

            step_input = [dec[i_step] for dec in decoder_inputs_batch_cur]
            # prepare current embeddings
            node_types = [s["node_type"] for s in step_input]
            # desc_context shape: batch (=1) x emb_size
            desc_context, attention_logits = model._desc_attention_batched(recurrent_state_cur,
                                                                           encoder_memory_cur,
                                                                           encoder_memory_mask_cur.unsqueeze(1)) # prepare the dimensions for multi-headed attention
            
            # get node embeddings
            node_type_idx_step = node_type_idx_for_step_and_batch_cur[i_step]
            node_type_emb = model.node_type_embedding(node_type_idx_step) # for torch.cat(dim=1): batch_size x feature_dim

            # get action embeddings
            action_emb_idx = action_emb_idx_for_step_and_batch_cur[i_step] # no + 1 increment to i_step
            action_emb = action_embedding_layer(action_emb_idx) # for torch.cat(dim=1): batch_size x feature_dim

            # prepare parent embeddings
            parent_node_idx = parent_node_idx_for_step_and_batch_cur[i_step]

            parent_action_idx = torch.gather(action_emb_idx_for_step_and_batch_cur, 0, parent_node_idx.unsqueeze(0)).squeeze(0) # batch_size
            parent_action_emb = action_embedding_layer(parent_action_idx) # for torch.cat(dim=1): batch_size x feature_dim

            outputs_cur = outputs[:,:batch_size_cur]
            idx_for_gather = parent_node_idx.unsqueeze(0).unsqueeze(2).expand(1, outputs_cur.size(1), outputs.size(2)) # 1 x batch_size x feature_dim
            parent_h = torch.gather(outputs_cur, 0, idx_for_gather).squeeze(0) # for torch.cat(dim=1): batch_size x feature_dim

            # prepare input to the decoder cell
            state_input = torch.cat(
                (
                    action_emb,  # a_{t-1}: rule_emb_size
                    desc_context,  # c_t: enc_recurrent_size
                    parent_h,  # s_{p_t}: recurrent_size
                    parent_action_emb,  # a_{p_t}: rule_emb_size
                    node_type_emb,  # n_{f-t}: node_emb_size
                ),
                dim=-1)

            # update decoder state
            recurrent_state_cur = model.state_update(
                # state_input shape: batch (=1) x (emb_size * 5)
                state_input, recurrent_state_cur)
            
            output = recurrent_state_cur[0]
            if output.size(0) < batch_size:
                # pad output vector with zeros
                output = torch.cat([output,
                                    torch.zeros((batch_size - output.size(0), output.size(1)), device=output.device, dtype=output.dtype)],
                                    dim=0)
            outputs = torch.cat([outputs, output.unsqueeze(0)], dim=0) # add current output to the tensor of all outputs

        # accumulate loss matrix from several losses
        loss_matrix = torch.zeros(loss_type_for_batch_and_step.shape,
                                  device=outputs.device,
                                  dtype=output.dtype)
        batch_index = torch.arange(batch_size, device=outputs.device).unsqueeze(0).expand(loss_matrix.size())

        # loss_type == 1
        loss_mask = loss_type_for_batch_and_step == 1
        if loss_mask.sum().item() > 0: # have any elements with this loss
            outputs_for_loss = torch.masked_select(outputs, loss_mask.unsqueeze(2))
            outputs_for_loss = outputs_for_loss.view(-1, outputs.size(-1))
            loss_targets = loss_target_for_batch_and_step[loss_mask]
            
            rule_logits = model.rule_logits(outputs_for_loss)
            losses_this_type = model.xent_loss(rule_logits, loss_targets, reduction="none")
            if losses_this_type.shape == (1, rule_logits.shape[0], rule_logits.shape[1]):
                losses_this_type = losses_this_type.sum(2).view(-1) # carefull with the dimensions here!
            elif losses_this_type.shape == (rule_logits.shape[0],):
                pass
            else:
                raise RuntimeError(f"Cannot process the output of the loss of the shape {losses_this_type.shape}")
                
            loss_matrix[loss_mask] = losses_this_type

        # loss_type == 2
        loss_mask = loss_type_for_batch_and_step == 2
        if loss_mask.sum().item() > 0: # have any elements with this loss
            outputs_for_loss = torch.masked_select(outputs, loss_mask.unsqueeze(2))
            outputs_for_loss = outputs_for_loss.view(-1, outputs.size(-1))
            loss_targets = loss_target_for_batch_and_step[loss_mask]
            
            # assert model.use_align_mat
            # compute_pointer = self.model.compute_pointer_with_align if self.model.use_align_mat else self.model.compute_pointer

            node_types = node_type_idx_for_step_and_batch[loss_mask[1:]] # exclude the barrier element
            assert model.share_pointers, "Implemented only this way"
            pointer = model.pointers[node_type] if not model.share_pointers else model.pointer 

            # implementing for only one node type
            good_node_types = [model.node_type_vocab.elem_to_id[key] for key in ["grounding", "column"] if key in model.node_type_vocab.elem_to_id]
            assert len(good_node_types) == 1, f"Expecting only nodes of node_type 'grounding' for the pointer loss"
            assert (node_types == good_node_types[0]).all().item(), f"Expecting only nodes of node_type 'grounding' for the pointer loss"
            
            batch_index_for_loss = batch_index[loss_mask]
            if self.use_align_mat:
                # encoder_memory
                encoder_memory_for_loss = torch.index_select(encoder_memory, 0, batch_index_for_loss)
                attn_mask_for_loss = ~torch.index_select(encoder_memory_mask, 0, batch_index_for_loss) # need inverted mask here

                memory_pointer_logits = pointer(
                        outputs_for_loss, encoder_memory_for_loss, attn_mask_for_loss)
                memory_pointer_probs = torch.nn.functional.softmax(
                    memory_pointer_logits, dim=1)

                m2c_align_mat_batch_for_loss = torch.index_select(m2c_align_mat_batch, 0, batch_index_for_loss)
                pointer_probs = torch.bmm(memory_pointer_probs.unsqueeze(1), m2c_align_mat_batch_for_loss).squeeze(1)
                pointer_probs = pointer_probs.clamp(min=1e-9)
                pointer_logits = torch.log(pointer_probs)
            else:
                pointer_memories_for_loss = torch.index_select(pointer_memories['grounding'], 0, batch_index_for_loss)
                attn_mask_for_loss = ~torch.index_select(pointer_memories_mask['grounding'], 0, batch_index_for_loss) # need inverted mask here
                
                pointer_logits = pointer(
                        outputs_for_loss, pointer_memories_for_loss, attn_mask_for_loss)
                pointer_logits = pointer_logits.clamp(min=torch.log(torch.full([1], 1e-9, dtype=pointer_logits.dtype)).item())

            losses_this_type = model.xent_loss(pointer_logits, loss_targets, reduction="none")
            if losses_this_type.shape == (1, pointer_logits.shape[0], pointer_logits.shape[1]):
                # carefull with the dimensions here!
                losses_this_type = losses_this_type[0]
                losses_this_type = losses_this_type.sum(1).view(-1)
            elif losses_this_type.shape == (pointer_logits.shape[0],):
                pass
            else:
                raise RuntimeError(f"Cannot process the output of the loss of the shape {losses_this_type.shape}")

            loss_matrix[loss_mask] = losses_this_type

        # loss_type == 3
        loss_mask = loss_type_for_batch_and_step == 3
        if loss_mask.sum().item() > 0: # have any elements with this loss
            outputs_for_loss = torch.masked_select(outputs, loss_mask.unsqueeze(2))
            outputs_for_loss = outputs_for_loss.view(-1, outputs.size(-1))
            loss_targets = loss_target_for_batch_and_step[loss_mask]
            
            ref_logits = model.ref_logits(outputs_for_loss)

            ref_count_steps_for_loss = ref_count_steps_for_batch_and_step[loss_mask]
            assert (ref_count_steps_for_loss != ref_count_step_barrier).all(), f"Value {ref_count_step_barrier} should not appear in the ref loss"

            ref_logits_mask_inf = torch.arange(ref_logits.size(1), device=ref_logits.device)[None, :] >= (ref_count_steps_for_loss[:, None] - 1)
            ref_logits.masked_fill_(ref_logits_mask_inf, float('-inf'))

            losses_this_type = model.ref_loss(ref_logits, loss_targets)
            loss_matrix[loss_mask] = losses_this_type

        losses_new_order = loss_matrix.sum(0, keepdim=False)
        losses = [0.0] * batch_size
        # forward transform: decoder_inputs_batch = [decoder_inputs_batch[i_] for i_ in new_batch_order]
        # reverse transform:
        for j_, i_ in enumerate(new_batch_order):
            losses[i_] = losses_new_order[j_]
        return losses

    def begin_inference(self, desc_enc):
        rules_index = {v: idx for idx, v in enumerate(self.preproc.all_rules)}
        traversal = InferenceTreeTraversal(self, desc_enc, rules_index=rules_index)
        choices = traversal.step(None)
        return traversal, choices

    def _desc_attention(self, prev_state, desc_enc):
        # prev_state shape:
        # - h_n: batch (=1) x emb_size
        # - c_n: batch (=1) x emb_size
        query = prev_state[0]
        if self.attn_type != 'sep':
            return self.desc_attn(query, desc_enc.memory, attn_mask=None)
        else:
            question_context, question_attention_logits = self.question_attn(query, desc_enc.question_memory)
            schema_context, schema_attention_logits = self.schema_attn(query, desc_enc.schema_memory)
            return question_context + schema_context, schema_attention_logits

    def _desc_attention_batched(self, prev_state, memory_batched, attn_mask_batched):
        # prev_state shape:
        # - h_n: batch (=1) x emb_size
        # - c_n: batch (=1) x emb_size
        query = prev_state[0]
        if self.attn_type != 'sep':
            return self.desc_attn(query, memory_batched, attn_mask=attn_mask_batched)
        else:
            raise NotImplementedError("Did not implement this type of batched attention")

    def _tensor(self, data, dtype=None):
        return torch.tensor(data, dtype=dtype, device=self._device)

    def _index(self, vocab, word):
        return self._tensor([vocab.index(word)])

    def _update_state(
            self,
            node_type,
            prev_state,
            prev_action_emb, prev_action_emb_type, prev_action_emb_idx,
            parent_h, parent_h_idx,
            parent_action_emb,
            desc_enc):
        # desc_context shape: batch (=1) x emb_size
        desc_context, attention_logits = self._desc_attention(prev_state, desc_enc)
        # node_type_emb shape: batch (=1) x emb_size
        node_type_emb = self.node_type_embedding(
            self._index(self.node_type_vocab, node_type))

        if hasattr(self, "lstm_inputs"):
            lstm_inputs = {}
            self.lstm_inputs.append(lstm_inputs)
            
            lstm_inputs["prev_action_emb_type"] = prev_action_emb_type
            lstm_inputs["prev_action_emb_idx"] = prev_action_emb_idx
            lstm_inputs["node_type"] = node_type
            lstm_inputs["parent_h_idx"] = parent_h_idx

        state_input = torch.cat(
            (
                prev_action_emb,  # a_{t-1}: rule_emb_size
                desc_context,  # c_t: enc_recurrent_size
                parent_h,  # s_{p_t}: recurrent_size
                parent_action_emb,  # a_{p_t}: rule_emb_size
                node_type_emb,  # n_{f-t}: node_emb_size
            ),
            dim=-1)

        new_state = self.state_update(
            # state_input shape: batch (=1) x (emb_size * 5)
            state_input, prev_state)

        return new_state, attention_logits

    def apply_rule(
            self,
            node_type,
            prev_state,
            prev_action_emb, prev_action_emb_type, prev_action_emb_idx,
            parent_h, parent_h_idx,
            parent_action_emb,
            desc_enc):
        new_state, _ = self._update_state(
            node_type, prev_state,
            prev_action_emb, prev_action_emb_type, prev_action_emb_idx,
            parent_h, parent_h_idx,
            parent_action_emb,
            desc_enc)
        # output shape: batch (=1) x emb_size
        output = new_state[0]
        # rule_logits shape: batch (=1) x num choices
        rule_logits = self.rule_logits(output)

        return output, new_state, rule_logits

    def rule_infer(self, node_type, rule_logits, is_train=False, step_history=None, grounding=None):
        rule_logprobs = torch.nn.functional.log_softmax(rule_logits, dim=-1)
        rules_start, rules_end = self.preproc.rules_mask[node_type]

        if not is_train and node_type == 'step' and step_history[0][0] == 'root':
            assert grounding is None, grounding
            return [(self.select_index,
            rule_logprobs[0, self.select_index])]

        if not is_train and step_history is not None:
            if node_type in ['comp_op_type', 'column_type', 'superlative_op_type']:
                is_comparative = node_type in ['comp_op_type', 'column_type'] 


                is_filter = False
                col_type_match = True

                if is_comparative:
                    # check ref1 == ref2 - then filter, else comparative
                    refs = [idx for rule, idx in step_history if rule == 'ref']
                    assert len(refs) == 2, refs
                    is_filter = refs[0] == refs[1]
                    has_comp_op = any([idx for rule, idx in step_history if rule == 'CompOp'])


                    lhs_val_type = set([grnd.data_type for grnd in grounding[refs[1]]])
                    if self.val_types_wo_cols:
                        assert lhs_val_type, lhs_val_type
                        col_type_match = len(self.val_types_wo_cols.intersection(lhs_val_type)) != 0
                        # if filter and no_column - should be NoOp (and NoColGrnd because of no_column) -> tbls, cols
                        # if filter and not no_column - should be ColGrnd if CompOp -> tbls, cols if NoOp or val+col if CompOp  or NoOp and ColGrnd
                        # if comp and no_column - can be NoOp/CompOp (and NoColGrnd because of no_column) -> ref
                        # if comp and not no_column - if CompOp, ColGrnd -> val + col
                        #                           - if NoOp, ColGrnd -> val + col
                        #                           - if NoOp/CompOp, NoColGrnd -> ref

                # remove unknown rules
                check_unknown = lambda end_node: end_node.find('Unknown') >= 0 
                # if no values, do not predict column, do not predict comp op in filter
                check_no_values = lambda end_node: self.no_vals \
                                                    and (end_node == 'ColumnGrounding' or is_filter and end_node == 'CompOp')
                # if all values with column, column prediction is necessary in filter with comp_op
                check_required_column = lambda end_node: is_filter and has_comp_op and self.required_column \
                                                        and end_node == 'NoColumnGrounding' 
                # if all values without column, no_column prediction is necessary 
                check_no_column = lambda end_node: self.no_column and end_node == 'ColumnGrounding' 
                # if types of ref columns and values wo columns are different and is filter:
                # if no_column - should be NoOp, otherwise - should be ColGrnd if CompOp
                check_col_type_match = lambda end_node: is_filter and not col_type_match \
                                                    and (self.no_column and end_node == 'CompOp' \
                                                    or not self.no_column and has_comp_op and end_node == 'NoColumnGrounding')

                indices = []
                logprobs = []
                for (start_node, end_node), idx in self.rules_index.items():
                    if idx < rules_start or idx >= rules_end:
                        continue
                    assert start_node == node_type, (start_node, node_type)
                    if check_unknown(end_node) or check_no_values(end_node) or \
                        check_required_column(end_node) or check_no_column(end_node) or check_col_type_match(end_node):
                        continue
        
                    indices.append(idx)
                    logprobs.append(rule_logprobs[0, idx])

                assert not is_comparative or not self.no_vals \
                    or self.no_vals and (len(indices) == 1 or node_type != 'column_type'), (self.no_vals, node_type)

                logprobs = torch.stack(logprobs)
                return list(zip(indices, logprobs))
        
            if node_type == 'comp_val':
                # comp_val = CompGrounding(grounding grounding) | CompRef(ref ref) 

                refs = [idx for rule, idx in step_history if rule == 'ref']
                assert len(refs) == 2, refs
                is_filter = refs[0] == refs[1]
                lhs_val_type = set([grnd.data_type for grnd in grounding[refs[1]]])

                # do not allow ref grounding after column prediction or if it is filter
                col_grnd = any([r for r, i in step_history if r == 'ColumnGrounding']) 
                if col_grnd or is_filter:
                    rules_end -= 1

                 # if not filter and no values or NoColumnGrounding prediction with all values+columns, 
                 # or NoColumnGrounding prediction while types of ref columns and values wo columns are different 
                 # do not allow comp grounding
                if not is_filter and (self.no_vals or not col_grnd and (self.required_column \
                    or self.val_types_wo_cols and not self.val_types_wo_cols.intersection(lhs_val_type))):
                    rules_start += 1
                assert rules_start < rules_end, (rules_start, rules_end)

        # TODO: Mask other probabilities first?
        return list(zip(
            range(rules_start, rules_end),
            rule_logprobs[0, rules_start:rules_end]))

    def gen_ref(
            self,
            node_type,
            prev_state,
            prev_action_emb, prev_action_emb_type, prev_action_emb_idx,
            parent_h, parent_h_idx,
            parent_action_emb,
            desc_enc,
            count_steps):
        new_state, attention_logits = self._update_state(
            node_type, prev_state,
            prev_action_emb, prev_action_emb_type, prev_action_emb_idx,
            parent_h, parent_h_idx,
            parent_action_emb,
            desc_enc)
        # output shape: batch (=1) x emb_size
        output = new_state[0]

        if hasattr(self, "lstm_inputs"):
            self.lstm_inputs[-1]["ref_count_steps"] = count_steps

         # ref_logits shape: batch (=1) x num refs
        ref_logits = self.ref_logits(output)
        ref_logits[:, (count_steps - 1):] += float('-inf')

        return output, new_state, ref_logits

    def ref_infer(self, node_type, ref_logits, count_steps, step_history=None, grounding=None):
        ref_logprobs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
        
        if step_history and step_history[0][0] == 'NextStepComp':
            ref = [idx for rule, idx in step_history if rule == 'ref']

            if len(ref) == 1 and self.no_vals:
                return [(ref[0],
                ref_logprobs[0, ref[0]])]

            elif step_history[-1][0] ==  'CompRef':
                possible_steps = get_comp_refs(ref_logprobs, count_steps, step_history, grounding)
                return list(zip(
                    possible_steps,
                    ref_logprobs[0, possible_steps]))
        
        return list(zip(
            range(count_steps - 1),
            ref_logprobs[0, :(count_steps - 1)]))

    def compute_pointer(
            self,
            node_type,
            prev_state,
            prev_action_emb, prev_action_emb_type, prev_action_emb_idx,
            parent_h, parent_h_idx,
            parent_action_emb,
            desc_enc):
        new_state, attention_logits = self._update_state(
            node_type, prev_state,
            prev_action_emb, prev_action_emb_type, prev_action_emb_idx,
            parent_h, parent_h_idx,
            parent_action_emb,
            desc_enc)
        # output shape: batch (=1) x emb_size
        output = new_state[0]
        # pointer_logits shape: batch (=1) x num choices
        pointer = self.pointers[node_type] if not self.share_pointers else self.pointer
        pointer_logits = pointer(
            output, desc_enc.pointer_memories[node_type])
        return output, new_state, pointer_logits, attention_logits

    def pointer_infer(self, node_type, logits):
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        return list(zip(
            # TODO batching
            range(logits.shape[1]),
            logprobs[0]))
