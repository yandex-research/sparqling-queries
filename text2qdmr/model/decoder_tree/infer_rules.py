def get_select_project_grnd_probs(model, pointer_logprobs):
    # Filter all tables, columns and values from schema for grounding in select and GroundingProjectArg in project
    # For project after select check project and select groundings - they should be different
    keep_pointer_logprobs = []
    
    for idx, grnd_choice in model.ids_to_grounding_choices.items():
        assert pointer_logprobs[idx][0] == idx

        if grnd_choice.choice_type != 'value':
            keep_pointer_logprobs.append(pointer_logprobs[idx])
        else:
            for val_unit in grnd_choice.choice:
                if val_unit.column and val_unit.table:
                    keep_pointer_logprobs.append(pointer_logprobs[idx])
                    break
    assert keep_pointer_logprobs, model.ids_to_grounding_choices
    return keep_pointer_logprobs


def get_column_agg_probs(model, pointer_logprobs):
    # Filter all columns for UseColumn rule in agg/group
    keep_pointer_logprobs = []
    for idx, grnd_choice in model.ids_to_grounding_choices.items():
        assert pointer_logprobs[idx][0] == idx

        if grnd_choice.choice_type == 'column':
            keep_pointer_logprobs.append(pointer_logprobs[idx])
    
    assert keep_pointer_logprobs, model.ids_to_grounding_choices
    return keep_pointer_logprobs

def get_column_grnd_probs(model, pointer_logprobs):
    # Filter all columns with proper types for ColumnGrounding in comparative
    assert len(model.ids_to_grounding_choices) == len(pointer_logprobs), (len(model.ids_to_grounding_choices), len(pointer_logprobs))
    keep_pointer_logprobs = []

    for idx, grnd_choice in model.ids_to_grounding_choices.items():
        if grnd_choice.choice_type == 'column':
            assert pointer_logprobs[idx][0] == idx
            tbl_name, col_name = grnd_choice.choice
            val_type = model.column_data[tbl_name][col_name]
            if (tbl_name, col_name) in model.value_columns:
                keep_pointer_logprobs.append(pointer_logprobs[idx])
    assert keep_pointer_logprobs, (model.value_columns, model.ids_to_grounding_choices)
    return keep_pointer_logprobs

def get_comp_grnd_probs(model, pointer_logprobs, step_history, grounding):
    # Filter CompGrounding in comparative (3 arg)
    assert len(model.ids_to_grounding_choices) == len(pointer_logprobs)
    keep_pointer_logprobs = []

    # check ref1 == ref2 - then filter, else comparative
    refs = [idx for rule, idx in step_history if rule == 'ref']
    assert len(refs) == 2, refs
    is_filter = refs[0] == refs[1]
    lhs_val_type = set([grnd.data_type for grnd in grounding[refs[1]]])

    # find column grounding and type of this column
    col_name, val_type = None, None
    for rule, idx in step_history:
        if rule == 'grounding':
            column_grnd = model.ids_to_grounding_choices[idx]
            assert column_grnd.choice_type == 'column'
            tbl_name, col_name = column_grnd.choice
            val_type = model.column_data[tbl_name][col_name]
            break

    has_column = val_type is not None
    has_comp_op = any([idx for rule, idx in step_history if rule == 'CompOp'])

    for idx, grnd_choice in model.ids_to_grounding_choices.items():
        if grnd_choice.choice_type == 'value':
            assert pointer_logprobs[idx][0] == idx
            if has_column:
                # choose values with correct type
                for val_unit in grnd_choice.choice:
                    if val_unit.value_type == val_type and val_unit.column is None \
                        or val_unit.column == col_name and val_unit.table == tbl_name:
                        keep_pointer_logprobs.append(pointer_logprobs[idx])
                        break
            else:
                # choose values without column
                for val_unit in grnd_choice.choice:
                    if val_unit.column is None and val_unit.value_type in lhs_val_type:
                        keep_pointer_logprobs.append(pointer_logprobs[idx]) 
        elif not has_column and not has_comp_op and is_filter:
            keep_pointer_logprobs.append(pointer_logprobs[idx]) 

    assert keep_pointer_logprobs, (model.ids_to_grounding_choices, val_type, col_name)
    return keep_pointer_logprobs

def get_comp_refs(ref_logprobs, count_steps, step_history, grounding):
     # Filter CompRef in comparative (3 arg)
    assert len(grounding) == (count_steps - 1), (grounding, count_steps)

    refs = [idx for rule, idx in step_history if rule == 'ref']
    assert len(refs) == 2, refs
    lhs = refs[1]
    lhs_val_type = set([grnd.data_type for grnd in grounding[lhs]])

    possible_steps = []
    for num_step in range(count_steps - 1):
        if num_step != lhs:
            rhs_val_type = set([grnd.data_type for grnd in grounding[num_step]])
            if rhs_val_type == lhs_val_type:
                possible_steps.append(num_step)

    if not possible_steps:
        print('CANNOT FIND GOOD COMP REF')
        return range(count_steps - 1)
    return possible_steps


def get_general_pointer_probs(model, pointer_logprobs, step_history=None, grounding=None):
    assert all([idx == orig_index for idx, (orig_index, _) in zip(range(len(pointer_logprobs)), pointer_logprobs)])
    
    if step_history is not None:
        last_rule_type = step_history[-1][0]

        if last_rule_type == 'ColumnGrounding':
            pointer_logprobs = get_column_grnd_probs(model, pointer_logprobs)
        elif last_rule_type == 'CompGrounding':
            pointer_logprobs = get_comp_grnd_probs(model, pointer_logprobs, step_history, grounding)
        elif last_rule_type == 'GroundingProjectArg' or step_history[0][0] == 'NextStepSelect':
            pointer_logprobs = get_select_project_grnd_probs(model, pointer_logprobs)
        elif last_rule_type == 'UseColumn':
            pointer_logprobs = get_column_agg_probs(model, pointer_logprobs)

    return pointer_logprobs