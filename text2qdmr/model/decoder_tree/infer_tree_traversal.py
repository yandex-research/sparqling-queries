import attr
import pyrsistent
import torch

from text2qdmr.model.decoder_tree.tree_traversal import TreeTraversal
from text2qdmr.model.decoder_tree.infer_rules import *
from text2qdmr.model.modules import decoder_utils
from text2qdmr.model.grammar.qdmr import AGG_TYPES_F

AGG_OPS = list(AGG_TYPES_F.keys())

def grnd_from_choices(model, idx):
    grnd_choice = model.ids_to_grounding_choices[idx]
    if grnd_choice.choice_type == 'table':
        tbl_name = model.schema.tables[idx].orig_name
        if model.schema.tables[idx].primary_keys:
            col_name = model.schema.tables[idx].primary_keys[0].orig_name
        else:
            col_name = '{}-primary_key'.format(tbl_name)
        return (tbl_name, col_name)
    elif grnd_choice.choice_type == 'value':
        for val_unit in grnd_choice.choice:
            if val_unit.table:
                return val_unit.table, val_unit.column
        return None, None
    elif grnd_choice.choice_type == 'column':
        return grnd_choice.choice

class InferenceTreeTraversal(TreeTraversal):
    class TreeAction:
        pass

    @attr.s(frozen=True)
    class SetParentField(TreeAction):
        parent_field_name = attr.ib()
        node_type = attr.ib()
        node_value = attr.ib(default=None)

    @attr.s(frozen=True)
    class NodeFinished(TreeAction):
        pass

    @attr.s(frozen=True)
    class OutputGrounding:
        data_type = attr.ib()
        column = attr.ib(default=None)
        table = attr.ib(default=None)

    class InferHandler(TreeTraversal.Handler):
        handlers = {}

    def __init__(self, model, desc_enc, example=None, rules_index=None):
        if model is None:
            return
        super().__init__(model.preproc, desc_enc, rules_index)
        self.handlers = self.InferHandler.handlers
        for key in super().Handler.handlers:
            if key not in self.handlers:
                self.handlers[key] = super().Handler.handlers[key]

        self.model = model
        model.state_update.set_dropout_masks(batch_size=1)
        self.recurrent_state = decoder_utils.lstm_init(
            model._device, None, self.model.recurrent_size, 1
        )
        self.prev_action_emb = model.zero_rule_emb
        self.update_prev_action_emb = InferenceTreeTraversal._update_prev_action_emb_apply_rule

        self.cur_item = attr.evolve(self.cur_item,
            parent_action_emb=self.model.zero_rule_emb,
            parent_h=self.model.zero_recurrent_emb)
 
        self.example = example
        self.actions = pyrsistent.pvector()
        self.count_steps = 0

    def clone(self):
        super_clone = super().clone()
        super_clone.example = self.example
        super_clone.model = self.model
        super_clone.actions = self.actions
        super_clone.handlers = self.handlers
        return super_clone

    def rule_choice(self, node_type, rule_logits=None, step_history=None, grounding=None):
        pass

    def ref_choice(self, node_type, ref_logits=None, step_history=None, grounding=None):
        pass

    def pointer_choice(self, node_type, logits=None, attention_logits=None, step_history=None, grounding=None):
        pass

    def compute_grounding(self, step_history, grounding):

        def get_grounding(rule_type):
            for rule, idx in step_history:
                if rule in rule_type:
                    if rule == 'grounding':
                        tbl_name, col_name = grnd_from_choices(self.model, idx)
                        data_type = self.model.column_data[tbl_name].get(col_name, 'number')
                        return self.OutputGrounding(column=col_name, table=tbl_name, data_type=data_type)
                    else:
                        return grounding[idx]

        if step_history[0][0] == 'NextStepSelect':
            # grounding of arg
            if grounding is None:
                grounding = []
            output = get_grounding('grounding')
            
        elif step_history[0][0] == 'NextStepAgg' or step_history[0][0] == 'NextStepGroup':
            # None if agg_type is op otherwise column
            # TODO case with last op = group
            output = None
            output = get_grounding('grounding')
            if output is None:
                assert step_history[4][0] == 'agg_ops', step_history
                agg_op = AGG_OPS[step_history[4][1]]
                if agg_op in ('min', 'max'):
                    assert step_history[5][0] == 'ref', step_history
                    idx = step_history[5][1]
                    output = grounding[idx]
                else:
                    output = self.OutputGrounding(data_type='number')

        elif step_history[0][0] == 'NextStepProject': 
            # grounding of 1st arg or ref
            output = get_grounding(('grounding', 'ref'))

        elif step_history[0][0] == 'NextStepUnion':
            # if groundings are the same, keep one, otherwise concat all
            args = []
            for rule, idx in step_history:
                if rule == 'ref':
                    grnd = grounding[idx]
                    if grnd not in args or (len(grnd) == 1 and grnd[0].column is None):
                        args.append(grnd)
            if len(args) <= 1:
                output = args[0] 
            else:
                output = [arg for sublist in args for arg in sublist]

        elif step_history[0][0] in ('NextStepIntersect', 'NextStepDiscard', 
                                    'NextStepSort', 'NextStepComp', 'NextStepSup'):
            # 1st arg
            output = get_grounding('ref')

        elif step_history[0][0] == 'NextStepArithmetic':
            output = self.OutputGrounding(data_type='number')

        if isinstance(output, list):
            grounding.append(output)
        else:
            grounding.append([output])
        return grounding

    def step(self, last_choice, extra_choice_info=None, strict_decoding=False):
        # update step history and compute current groundings
        if strict_decoding:
            if self.cur_item.step_history is None:
                step_history = [(self.cur_item.node_type, last_choice)]
                self.cur_item = attr.evolve(self.cur_item, step_history=step_history)
            elif self.cur_item.node_type == 'step':
                if self.cur_item.step_history[0][0] != 'root':
                    step_history = self.cur_item.step_history + [(self.cur_item.node_type, last_choice)]
                    grounding = self.compute_grounding(step_history, self.cur_item.grounding)
                else:
                    grounding = None
                self.cur_item = attr.evolve(self.cur_item, step_history=None, grounding=grounding)
            else:
                step_history = self.cur_item.step_history + [(self.cur_item.node_type, last_choice)]
                self.cur_item = attr.evolve(self.cur_item, step_history=step_history)
        return super().step(last_choice, extra_choice_info)

    def update_using_last_choice(self, last_choice, extra_choice_info):
        super().update_using_last_choice(last_choice, extra_choice_info)
        # Record actions
        # CHILDREN_INQUIRE
        if self.cur_item.state == TreeTraversal.State.CHILDREN_INQUIRE:
            self.actions = self.actions.append(
                self.SetParentField(
                    self.cur_item.parent_field_name,  self.cur_item.node_type))
            type_info = self.model.ast_wrapper.singular_types[self.cur_item.node_type]
            if not type_info.fields:
                self.actions = self.actions.append(self.NodeFinished())

        elif self.cur_item.state == TreeTraversal.State.REF_APPLY:
            self.actions = self.actions.append(self.SetParentField(
                    self.cur_item.parent_field_name,
                    node_type=None,
                    node_value=last_choice))

        elif self.cur_item.state == TreeTraversal.State.GENERAL_POINTER_APPLY:
            self.actions = self.actions.append(self.SetParentField(
                    self.cur_item.parent_field_name,
                    node_type=None,
                    node_value=last_choice))

        # NODE_FINISHED
        elif self.cur_item.state == TreeTraversal.State.NODE_FINISHED:
            self.actions = self.actions.append(self.NodeFinished())

    @classmethod
    def _update_prev_action_emb_apply_rule(cls, self, last_choice, extra_choice_info):
        rule_idx = self.model._tensor([last_choice])
        self.prev_action_emb = self.model.rule_embedding(rule_idx)
        TreeTraversal._update_prev_action_emb_apply_rule(self, last_choice, extra_choice_info)

    @classmethod
    def _update_prev_action_emb_pointer(cls, self, last_choice, extra_choice_info):
        # TODO batching
        pointer_action_emb_proj = self.model.pointer_action_emb_proj[
            self.cur_item.node_type
        ] if not self.model.share_pointers else self.model.pointer_action_emb_proj
        self.prev_action_emb = pointer_action_emb_proj(self.desc_enc.pointer_memories[self.cur_item.node_type][:, last_choice])
        TreeTraversal._update_prev_action_emb_pointer(self, last_choice, extra_choice_info)

    @classmethod
    def _update_prev_action_emb_gen_ref(cls, self, last_choice, extra_choice_info):
        ref_idx = self.model._tensor([last_choice])
        self.prev_action_emb = self.model.ref_embedding(ref_idx)
        TreeTraversal._update_prev_action_emb_gen_ref(self, last_choice, extra_choice_info)

    @InferHandler.register_handler(TreeTraversal.State.SUM_TYPE_INQUIRE)
    def process_sum_inquire(self, last_choice):
        # 1. ApplyRule, like expr -> Call
        # a. Ask which one to choose
        output, self.recurrent_state, rule_logits = self.model.apply_rule(
            self.cur_item.node_type,
            self.recurrent_state,
            self.prev_action_emb, self.prev_action_emb_type, self.prev_action_emb_idx,
            self.cur_item.parent_h, self.cur_item.parent_h_idx,
            self.cur_item.parent_action_emb,
            self.desc_enc,
        )

        super().process_sum_inquire(last_choice)
        self.cur_item = attr.evolve(self.cur_item, 
            parent_h=output, 
            parent_h_idx=len(self.model.lstm_inputs) - 1 if hasattr(self.model, "lstm_inputs") else self.cur_item.parent_h_idx,)

        self.update_prev_action_emb = (
            InferenceTreeTraversal._update_prev_action_emb_apply_rule
        )
        
        choice = self.model.rule_infer(self.cur_item.node_type, rule_logits,
                        step_history=self.cur_item.step_history,
                        grounding=self.cur_item.grounding)

        return choice, False

    @InferHandler.register_handler(TreeTraversal.State.SUM_TYPE_APPLY)
    def process_sum_apply(self, last_choice):
        # b. Add action, prepare for #2
        super().process_sum_apply(last_choice)
        self.cur_item = attr.evolve(self.cur_item,
            parent_action_emb=self.prev_action_emb)
        return None, True

    @InferHandler.register_handler(TreeTraversal.State.CHILDREN_INQUIRE)
    def process_children_inquire(self, last_choice):
        # 2. ApplyRule, like Call -> expr[func] expr*[args] keyword*[keywords]
        # Check if we have no children
        type_info = self.preproc.ast_wrapper.singular_types[
            self.cur_item.node_type
        ]
        if not type_info.fields:
            if self.pop():
                last_choice = None
                return last_choice, True
            else:
                return None, False

        super().process_children_inquire(last_choice)

        # a. Ask about presence
        output, self.recurrent_state, rule_logits = self.model.apply_rule(
            self.cur_item.node_type,
            self.recurrent_state,
            self.prev_action_emb, self.prev_action_emb_type, self.prev_action_emb_idx,
            self.cur_item.parent_h, self.cur_item.parent_h_idx,
            self.cur_item.parent_action_emb,
            self.desc_enc,
        )
        self.cur_item = attr.evolve(self.cur_item,
            parent_h=output,
            parent_h_idx=len(self.model.lstm_inputs) - 1 if hasattr(self.model, "lstm_inputs") else self.cur_item.parent_h_idx)

        self.update_prev_action_emb = (
            InferenceTreeTraversal._update_prev_action_emb_apply_rule
        )

        choice = self.model.rule_infer(self.cur_item.node_type, rule_logits,
                        step_history=self.cur_item.step_history,
                        grounding=self.cur_item.grounding)
        return choice, False

    @InferHandler.register_handler(TreeTraversal.State.GENERAL_POINTER_INQUIRE)
    def process_general_pointer_inquire(self, last_choice):
        # a. Ask which one to choose
        compute_pointer = self.model.compute_pointer_with_align if self.model.use_align_mat else self.model.compute_pointer
        output, self.recurrent_state, logits, _ = compute_pointer(
            self.cur_item.node_type,
            self.recurrent_state,
            self.prev_action_emb, self.prev_action_emb_type, self.prev_action_emb_idx,
            self.cur_item.parent_h, self.cur_item.parent_h_idx,
            self.cur_item.parent_action_emb,
            self.desc_enc,
        )
        super().process_general_pointer_inquire(last_choice)
        self.cur_item = attr.evolve(self.cur_item,
            parent_h=output,
            parent_h_idx=len(self.model.lstm_inputs) - 1 if hasattr(self.model, "lstm_inputs") else self.cur_item.parent_h_idx)

        pointer_logprobs = self.model.pointer_infer(self.cur_item.node_type, logits)
        # Group them based on pointer map
        pointer_map = self.desc_enc.pointer_maps.get(self.cur_item.node_type)
        if not pointer_map:
            pointer_logprobs = pointer_logprobs

        pointer_logprobs = dict(pointer_logprobs)
        pointer_logprobs = [
                (orig_index, torch.logsumexp(
                    torch.stack(
                        tuple(pointer_logprobs[i] for i in mapped_indices),
                        dim=0),
                    dim=0))
                for orig_index, mapped_indices in pointer_map.items()
            ]

        pointer_choice = get_general_pointer_probs(self.model, pointer_logprobs, 
                                                self.cur_item.step_history, self.cur_item.grounding)
        
        self.update_prev_action_emb = (
            InferenceTreeTraversal._update_prev_action_emb_pointer
        )
        return pointer_choice, False

    @InferHandler.register_handler(TreeTraversal.State.REF_INQUIRE)
    def process_ref_inquire(self, last_choice):
        output, self.recurrent_state, ref_logits = self.model.gen_ref(
            self.cur_item.node_type,
            self.recurrent_state,
            self.prev_action_emb, self.prev_action_emb_type, self.prev_action_emb_idx,
            self.cur_item.parent_h, self.cur_item.parent_h_idx,
            self.cur_item.parent_action_emb,
            self.desc_enc,
            self.count_steps
        )
        super().process_ref_inquire(last_choice)
        self.cur_item = attr.evolve(self.cur_item,
            parent_h=output,
            parent_h_idx=len(self.model.lstm_inputs) - 1 if hasattr(self.model, "lstm_inputs") else self.cur_item.parent_h_idx)

        choice = self.model.ref_infer(self.cur_item.node_type, ref_logits, self.count_steps, 
                                    step_history=self.cur_item.step_history, grounding=self.cur_item.grounding)
        
        self.update_prev_action_emb = (
            InferenceTreeTraversal._update_prev_action_emb_gen_ref
        )
        return choice, False

    def finalize(self):
        root = current = None
        stack = []
        for i, action in enumerate(self.actions):
            if isinstance(action, self.SetParentField):
                if action.node_value is None:
                    new_node = {'_type': action.node_type}
                else:
                    new_node = action.node_value

                if action.parent_field_name is None:
                    # Initial node in tree.
                    assert root is None
                    root = current = new_node
                    stack.append(root)
                    continue

                existing_list = current.get(action.parent_field_name)
                if existing_list is None:
                    current[action.parent_field_name] = new_node
                else:
                    assert isinstance(existing_list, list)
                    current[action.parent_field_name].append(new_node)

                if action.node_value is None:
                    stack.append(current)
                    current = new_node

            elif isinstance(action, self.NodeFinished):
                current = stack.pop()

            else:
                raise ValueError(action)

        assert not stack
        return root, self.model.preproc.grammar.unparse(root, self.model.schema, self.model.value_unit_dict, self.model.column_data)

    def save_step_to_log(self, node_type, parent_h_idx, prev_action_emb_type, prev_action_emb_idx, count_steps=None):
       pass