import attr
import operator

import torch

@attr.s
class TreeState:
    node = attr.ib()
    parent_field_type = attr.ib()

def lstm_init(device, num_layers, hidden_size, *batch_sizes):
    init_size = batch_sizes + (hidden_size,)
    if num_layers is not None:
        init_size = (num_layers,) + init_size
    init = torch.zeros(*init_size, device=device)
    return (init, init)


def maybe_stack(items, dim=None):
    to_stack = [item for item in items if item is not None]
    if not to_stack:
        return None
    elif len(to_stack) == 1:
        return to_stack[0].unsqueeze(dim)
    else:
        return torch.stack(to_stack, dim)


def accumulate_logprobs(d, keys_and_logprobs):
    for key, logprob in keys_and_logprobs:
        existing = d.get(key)
        if existing is None:
            d[key] = logprob
        else:
            d[key] = torch.logsumexp(
                torch.stack((logprob, existing), dim=0),
                dim=0)


def get_field_presence_info(ast_wrapper, node, field_infos):
    present = []
    for field_info in field_infos:
        field_value = node.get(field_info.name)
        is_present = field_value is not None and field_value != []

        maybe_missing = field_info.opt or field_info.seq
        is_builtin_type = field_info.type in ast_wrapper.primitive_types

        if maybe_missing and is_builtin_type:
            # TODO: make it possible to deal with "singleton?"
            present.append(is_present and type(field_value).__name__)
        elif maybe_missing and not is_builtin_type:
            present.append(is_present)
        elif not maybe_missing and is_builtin_type:
            present.append(type(field_value).__name__)
        elif not maybe_missing and not is_builtin_type:
            assert is_present
            present.append(True)
    return tuple(present)

def compute_align_loss(model, desc_enc, example):
    '''model: a nl2code decoder'''
    # find relevant columns
    root_node = example.tree
    rel_cols = list(reversed([val for val in model.ast_wrapper.find_all_descendants_of_type(root_node, "column")]))
    rel_tabs = list(reversed([val for val in model.ast_wrapper.find_all_descendants_of_type(root_node, "table")]))

    rel_cols_t = torch.LongTensor(sorted(list(set(rel_cols)))).to(model._device)
    rel_tabs_t = torch.LongTensor(sorted(list(set(rel_tabs)))).to(model._device)
    
    if rel_cols:
        mc_att_on_rel_col = desc_enc.m2c_align_mat.index_select(1, rel_cols_t)
        mc_max_rel_att, _ = mc_att_on_rel_col.max(dim=0)
        mc_max_rel_att.clamp_(min=1e-9)

    if rel_tabs:
        mt_att_on_rel_tab = desc_enc.m2t_align_mat.index_select(1, rel_tabs_t)
        mt_max_rel_att, _ = mt_att_on_rel_tab.max(dim=0)
        mt_max_rel_att.clamp_(min=1e-9)

    align_loss = 0
    if rel_cols:
        align_loss = - torch.log(mc_max_rel_att).mean() 
    if rel_tabs:
        align_loss -= torch.log(mt_max_rel_att).mean()
    return align_loss


def compute_pointer_with_align(
        model,
        node_type,
        prev_state,
        prev_action_emb, HACKING_prev_action_emb_type, HACKING_prev_action_emb_idx,
        parent_h, HACKING_parent_h_idx,
        parent_action_emb,
        desc_enc):
    new_state, attention_weights = model._update_state(
        node_type, prev_state,
        prev_action_emb,  HACKING_prev_action_emb_type, HACKING_prev_action_emb_idx,
        parent_h, HACKING_parent_h_idx,
        parent_action_emb,
        desc_enc)
    # output shape: batch (=1) x emb_size
    output = new_state[0]
    pointer = model.pointers[node_type] if not model.share_pointers else model.pointer 
    memory_pointer_logits = pointer(
            output, desc_enc.memory)
    memory_pointer_probs = torch.nn.functional.softmax(
        memory_pointer_logits, dim=1)
    # pointer_logits shape: batch (=1) x num choices
    if node_type == "column" or node_type == "grounding":
        pointer_probs = torch.mm(memory_pointer_probs, desc_enc.m2c_align_mat)
    elif node_type == "table":
        pointer_probs = torch.mm(memory_pointer_probs, desc_enc.m2t_align_mat)
    elif node_type == "value":
        pointer_probs = torch.mm(memory_pointer_probs, desc_enc.m2v_align_mat)
    else:
        raise RuntimeError
    pointer_probs = pointer_probs.clamp(min=1e-9)
    pointer_logits = torch.log(pointer_probs)
    return output, new_state, pointer_logits, attention_weights

@attr.s
class Hypothesis:
    inference_state = attr.ib()
    next_choices = attr.ib()
    score = attr.ib(default=0)

    choice_history = attr.ib(factory=list)
    score_history = attr.ib(factory=list)

def beam_search(model, preproc_item, beam_size, max_steps, strict_decoding=False):
    inference_state, next_choices = model.begin_inference(preproc_item)
    beam = [Hypothesis(inference_state, next_choices)]
    finished = []
    
    for step in range(max_steps):
        # Check if all beams are finished
        if len(finished) == beam_size:
            break

        candidates = []

        # For each hypothesis, get possible expansions
        # Score each expansion
        for hyp in beam:
            candidates += [(hyp, choice, choice_score.item(),
                            hyp.score + choice_score.item())
                           for choice, choice_score in hyp.next_choices]

        # Keep the top K expansions
        candidates.sort(key=operator.itemgetter(3), reverse=True)
        candidates = candidates[:beam_size - len(finished)]

        # Create the new hypotheses from the expansions
        beam = []
        for hyp, choice, choice_score, cum_score in candidates:
            inference_state = hyp.inference_state.clone()
            next_choices = inference_state.step(choice, strict_decoding=strict_decoding)
            if next_choices is None:
                finished.append(Hypothesis(
                    inference_state,
                    None,
                    cum_score,
                    hyp.choice_history + [choice],
                    hyp.score_history + [choice_score]))
            else:
                beam.append(
                    Hypothesis(inference_state, next_choices, cum_score,
                               hyp.choice_history + [choice],
                               hyp.score_history + [choice_score]))

    finished.sort(key=operator.attrgetter('score'), reverse=True)
    return finished
