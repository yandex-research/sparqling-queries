import operator

import attr
import pyrsistent
import torch

from text2qdmr.model.decoder_tree.tree_traversal import TreeTraversal


@attr.s
class ChoiceHistoryEntry:
    rule_left = attr.ib()
    choices = attr.ib()
    probs = attr.ib()
    valid_choices = attr.ib()
class TrainTreeTraversal(TreeTraversal):
    @attr.s(frozen=True)
    class XentChoicePoint:
        logits = attr.ib()
        pointer = attr.ib(default=False)
        def compute_loss(self, outer, idx, extra_indices):
            if extra_indices:
                raise RuntimeError(f"Not implemented, can ever get here?")
            else:
                # idx shape: batch (=1)
                idx = outer.model._tensor([idx])
                # loss_piece shape: batch (=1)
                loss = outer.model.xent_loss(self.logits, idx)
            return loss
    @attr.s(frozen=True)
    class RefChoicePoint:
        logits = attr.ib()
        def compute_loss(self, outer, idx, extra_indices):
            assert not extra_indices
            # idx shape: batch (=1)
            idx = outer.model._tensor([idx])
            # loss_piece shape: batch (=1)
            loss = outer.model.ref_loss(self.logits, idx)
            return loss

    def __init__(self, preproc, desc_enc, rules_index, exclude_rules_loss=None, debug=False):
        super().__init__(preproc, desc_enc, rules_index)
        self.handlers = super().Handler.handlers

        self.choice_point = None
        self.prev_action_emb = None
        self.loss = pyrsistent.pvector()

        self.debug = debug
        self.history = pyrsistent.pvector()

    def clone(self):
        super_clone = super().clone()
        super_clone.choice_point = self.choice_point
        super_clone.loss = self.loss
        super_clone.debug = self.debug
        super_clone.history = self.history
        return super_clone

    def save_step_to_log(self, node_type, parent_h_idx, prev_action_emb_type, prev_action_emb_idx, count_steps=None):
        traversal_step_info = {}
        self.traversal_step_log.append(traversal_step_info)
        
        traversal_step_info["prev_action_emb_type"] = prev_action_emb_type
        traversal_step_info["prev_action_emb_idx"] = prev_action_emb_idx
        traversal_step_info["node_type"] = node_type
        traversal_step_info["parent_h_idx"] = parent_h_idx
        if count_steps is not None:
            traversal_step_info["ref_count_steps"] = count_steps

    def rule_choice(self, node_type, step_history=None, grounding=None):
        pass

    def pointer_choice(self, node_type, logits=None, attention_logits=None, step_history=None, grounding=None):
        self.choice_point = self.XentChoicePoint(logits, pointer=True)
        self.attention_choice = self.XentChoicePoint(attention_logits, pointer=True)

    def ref_choice(self, node_type, ref_logits=None, step_history=None, grounding=None):
        self.choice_point = self.RefChoicePoint(ref_logits)

    def update_using_last_choice(self, last_choice, extra_choice_info):
        super().update_using_last_choice(last_choice, extra_choice_info)
        if last_choice is None:
            return