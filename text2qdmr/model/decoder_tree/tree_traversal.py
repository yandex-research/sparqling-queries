import enum

import attr
import pyrsistent
from copy import deepcopy

@attr.s
class TreeState:
    node = attr.ib()
    parent_field_type = attr.ib()

class TreeTraversal:
    class Handler:
        handlers = {}

        @classmethod
        def register_handler(cls, func_type):
            if func_type in cls.handlers:
                raise RuntimeError(f"{func_type} handler is already registered")

            def inner_func(func):
                cls.handlers[func_type] = func.__name__
                return func

            return inner_func

    @attr.s(frozen=True)
    class QueueItem:
        item_id = attr.ib()
        state = attr.ib()
        node_type = attr.ib()
        parent_h_idx = attr.ib()
        parent_field_name = attr.ib()
        parent_action_emb = attr.ib(default=None)
        parent_h = attr.ib(default=None)
        step_history = attr.ib(default=None)
        grounding = attr.ib(default=None)

        def to_str(self):
            return f"<state: {self.state}, node_type: {self.node_type}, parent_field_name: {self.parent_field_name}>"

    class State(enum.Enum):
        SUM_TYPE_INQUIRE = 0
        SUM_TYPE_APPLY = 1
        CHILDREN_INQUIRE = 2
        CHILDREN_APPLY = 3
        REF_INQUIRE = 4
        REF_APPLY = 5
        GENERAL_POINTER_INQUIRE = 6
        GENERAL_POINTER_APPLY = 7
        NODE_FINISHED = 8

    def __init__(self, preproc, desc_enc, rules_index):
        if preproc is None:
            return

        self.preproc = preproc
        self.desc_enc = desc_enc
        self.rules_index = rules_index

        self.prev_action_emb_type = "zero_rule_emb"
        self.prev_action_emb_idx = 0

        root_type = self.preproc.grammar.root_type
        if root_type in self.preproc.ast_wrapper.sum_types:
            initial_state = TreeTraversal.State.SUM_TYPE_INQUIRE
        else:
            initial_state = TreeTraversal.State.CHILDREN_INQUIRE

        self.queue = pyrsistent.pvector()
        self.cur_item = TreeTraversal.QueueItem(
            item_id=0,
            state=initial_state,
            node_type=root_type,
            parent_h_idx=-1,
            parent_field_name=None,
        )
        self.next_item_id = 1
        self.count_steps = 0
        self.last_handler_name = ''
        self.update_prev_action_emb = TreeTraversal._update_prev_action_emb_apply_rule
        
        self.traversal_step_log = []

    def clone(self):
        other = self.__class__(None, None, None)
        other.preproc = self.preproc
        other.desc_enc = self.desc_enc
        other.rules_index = self.rules_index
        other.recurrent_state = self.recurrent_state
        other.prev_action_emb = self.prev_action_emb
        other.queue = self.queue
        other.cur_item = deepcopy(self.cur_item)
        other.next_item_id = self.next_item_id
        other.actions = self.actions
        other.count_steps = self.count_steps
        other.last_handler_name = self.last_handler_name
        other.update_prev_action_emb = self.update_prev_action_emb
        other.traversal_step_log = self.traversal_step_log
        return other

    def step(self, last_choice, extra_choice_info=None):
        all_next_steps = [idx for rule, idx in self.rules_index.items() if rule[0] == 'step']
        if self.last_handler_name == 'process_sum_inquire':
            self.count_steps += (last_choice in all_next_steps) # TODO 

        while True:
            self.update_using_last_choice(last_choice, extra_choice_info)
            handler_name = self.handlers[self.cur_item.state]
            handler = getattr(self, handler_name)
            self.last_handler_name = handler_name
            choices, continued = handler(last_choice)

            if continued:
                last_choice = choices
                continue
            else:
                return choices

    def pop(self):
        if self.queue:
            last_item = self.cur_item
            self.cur_item = self.queue[-1]
            if last_item.step_history:
                self.cur_item = attr.evolve(self.cur_item, step_history=last_item.step_history, 
                                            grounding=last_item.grounding)
            self.queue = self.queue.delete(-1)
            return True
        return False

    def update_using_last_choice(self, last_choice, extra_choice_info):
        if last_choice is not None:
            self.update_prev_action_emb(self, last_choice, extra_choice_info)

    @classmethod
    def _update_prev_action_emb_apply_rule(cls, self, last_choice, extra_choice_info):
        self.prev_action_emb_type = "_update_prev_action_emb_apply_rule"
        self.prev_action_emb_idx = last_choice

    @classmethod
    def _update_prev_action_emb_pointer(cls, self, last_choice, extra_choice_info):
        self.prev_action_emb_type = "_update_prev_action_emb_pointer"
        self.prev_action_emb_idx = last_choice

    @classmethod
    def _update_prev_action_emb_gen_ref(cls, self, last_choice, extra_choice_info):
        # ref_idx = self.model._tensor([last_choice])
        # self.prev_action_emb = self.model.ref_embedding(ref_idx)

        self.prev_action_emb_type = "_update_prev_action_emb_gen_ref"
        self.prev_action_emb_idx = last_choice

    @Handler.register_handler(State.SUM_TYPE_INQUIRE)
    def process_sum_inquire(self, last_choice):
        # 1. ApplyRule, like expr -> Call
        # a. Ask which one to choose
        self.save_step_to_log(self.cur_item.node_type, self.cur_item.parent_h_idx, self.prev_action_emb_type, self.prev_action_emb_idx)

        self.cur_item = attr.evolve(self.cur_item,
            state=TreeTraversal.State.SUM_TYPE_APPLY,
            parent_h_idx=len(self.traversal_step_log)-1)

        self.update_prev_action_emb = (
            TreeTraversal._update_prev_action_emb_apply_rule
        )
        choices = self.rule_choice(self.cur_item.node_type,
                        step_history=self.cur_item.step_history,
                        grounding=self.cur_item.grounding)
        return choices, False

    @Handler.register_handler(State.SUM_TYPE_APPLY)
    def process_sum_apply(self, last_choice):
        # b. Add action, prepare for #2
        sum_type, singular_type = self.preproc.all_rules[last_choice]
        assert sum_type == self.cur_item.node_type

        self.cur_item = attr.evolve(
            self.cur_item,
            node_type=singular_type,
            state=TreeTraversal.State.CHILDREN_INQUIRE,
        )
        return None, True

    @Handler.register_handler(State.CHILDREN_INQUIRE)
    def process_children_inquire(self, last_choice):
        # 2. ApplyRule, like Call -> expr[func] expr*[args] keyword*[keywords]
        # Check if we have no children
        # a. Ask about presence
        type_info = self.preproc.ast_wrapper.singular_types[
            self.cur_item.node_type
        ]
        if not type_info.fields:
            if self.pop():
                last_choice = None
                return last_choice, True
            else:
                return None, False

        self.save_step_to_log(self.cur_item.node_type, self.cur_item.parent_h_idx, self.prev_action_emb_type, self.prev_action_emb_idx)

        self.cur_item = attr.evolve(self.cur_item,
            state=TreeTraversal.State.CHILDREN_APPLY,
            parent_h_idx=len(self.traversal_step_log) - 1)
        
        self.update_prev_action_emb = (
            TreeTraversal._update_prev_action_emb_apply_rule
        )
        choices = self.rule_choice(self.cur_item.node_type, 
                                step_history=self.cur_item.step_history, 
                                grounding=self.cur_item.grounding)
        return choices, False

    @Handler.register_handler(State.CHILDREN_APPLY)
    def process_children_apply(self, last_choice):
        # b. Create the children
        node_type, children_presence = self.preproc.all_rules[last_choice]
        assert node_type == self.cur_item.node_type

        self.queue = self.queue.append(
            TreeTraversal.QueueItem(
                item_id=self.cur_item.item_id,
                state=TreeTraversal.State.NODE_FINISHED,
                node_type=None,
                parent_action_emb=None,
                parent_h=None,
                parent_h_idx=None,
                parent_field_name=None,
                step_history=self.cur_item.step_history,
                grounding=self.cur_item.grounding,
            )
        )
        for field_info, present in reversed(
                list(
                    zip(
                        self.preproc.ast_wrapper.singular_types[node_type].fields,
                        children_presence,
                    )
                )
        ):
            if not present:
                continue

            child_type = field_type = field_info.type
            if field_info.seq:
                raise NotImplementedError
            if field_type in self.preproc.grammar.ref:
                child_state = TreeTraversal.State.REF_INQUIRE
            elif field_type in self.preproc.ast_wrapper.sum_types:
                child_state = TreeTraversal.State.SUM_TYPE_INQUIRE
            elif field_type in self.preproc.ast_wrapper.product_types:
                assert self.preproc.ast_wrapper.product_types[field_type].fields
                child_state = TreeTraversal.State.CHILDREN_INQUIRE
            elif field_type in self.preproc.grammar.general_pointers:
                child_state = TreeTraversal.State.GENERAL_POINTER_INQUIRE
            elif field_type in self.preproc.ast_wrapper.primitive_types:
                raise NotImplementedError
            else:
                raise ValueError(f"Unable to handle field type {field_type}")

            self.queue = self.queue.append(
                TreeTraversal.QueueItem(
                    item_id=self.next_item_id,
                    state=child_state,
                    node_type=child_type,
                    parent_action_emb=self.prev_action_emb,
                    parent_h=self.cur_item.parent_h,
                    parent_h_idx=self.cur_item.parent_h_idx,
                    parent_field_name=field_info.name,
                    step_history=self.cur_item.step_history,
                    grounding=self.cur_item.grounding
                )
            )
            self.next_item_id += 1

        advanced = self.pop()
        assert advanced
        last_choice = None
        return last_choice, True

    @Handler.register_handler(State.GENERAL_POINTER_INQUIRE)
    def process_general_pointer_inquire(self, last_choice):
        # a. Ask which one to choose
        self.save_step_to_log(self.cur_item.node_type, self.cur_item.parent_h_idx, self.prev_action_emb_type, self.prev_action_emb_idx)

        self.cur_item = attr.evolve(self.cur_item,
            state=TreeTraversal.State.GENERAL_POINTER_APPLY,
            parent_h_idx=len(self.traversal_step_log)-1)

        self.update_prev_action_emb = (
            TreeTraversal._update_prev_action_emb_pointer
        )
        choices = self.pointer_choice(
            self.cur_item.node_type, step_history=self.cur_item.step_history, grounding=self.cur_item.grounding
        )
        return choices, False

    @Handler.register_handler(State.GENERAL_POINTER_APPLY)
    def process_general_pointer_apply(self, last_choice):
        if self.pop():
            last_choice = None
            return last_choice, True
        else:
            return None, False

    @Handler.register_handler(State.REF_INQUIRE)
    def process_ref_inquire(self, last_choice):
        self.save_step_to_log(self.cur_item.node_type, self.cur_item.parent_h_idx, self.prev_action_emb_type, self.prev_action_emb_idx, count_steps=self.count_steps)
        self.cur_item = attr.evolve(self.cur_item,
            state=TreeTraversal.State.REF_APPLY,
            parent_h_idx=len(self.traversal_step_log)-1)
        self.update_prev_action_emb = (
            TreeTraversal._update_prev_action_emb_gen_ref
        )
        choices = self.ref_choice(self.cur_item.node_type, step_history=self.cur_item.step_history, grounding=self.cur_item.grounding)
        return choices, False

    @Handler.register_handler(State.REF_APPLY)
    def process_ref_apply(self, last_choice):
        if self.pop():
            last_choice = None
            return last_choice, True
        else:
            return None, False

    @Handler.register_handler(State.NODE_FINISHED)
    def process_node_finished(self, last_choice):
        if self.pop():
            last_choice = None
            return last_choice, True
        else:
            return None, False

    def rule_choice(self, node_type, rule_logits=None, step_history=None, grounding=None):
        raise NotImplementedError

    def pointer_choice(self, node_type, logits=None, attention_logits=None, step_history=None, grounding=None):
        raise NotImplementedError

    def ref_choice(self, node_type, ref_logits=None, step_history=None, grounding=None):
        raise NotImplementedError
