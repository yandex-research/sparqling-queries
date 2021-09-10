import attr 
import json
import numpy as np
from text2qdmr.datasets.utils.extract_values import GroundingKey, ValueUnit
from text2qdmr.datasets.qdmr import QDMRStepArg

def to_dict_with_sorted_values(d, key=None):
    return {k: sorted(v, key=key) for k, v in d.items()}


def to_dict_with_set_values(d):
    result = {}
    for k, v in d.items():
        hashable_v = []
        for v_elem in v:
            if isinstance(v_elem, list):
                hashable_v.append(tuple(v_elem))
            else:
                hashable_v.append(v_elem)
        result[k] = set(hashable_v)
    return result


def tuplify(x):
    if not isinstance(x, (tuple, list)):
        return x
    return tuple(tuplify(elem) for elem in x)


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, GroundingKey) or isinstance(obj, ValueUnit):
            return obj.__dict__
        elif isinstance(obj, QDMRStepArg) and obj.arg_type == 'ref':
             return attr.asdict(obj)
        elif isinstance(obj, QDMRStepArg) and obj.arg_type == 'grounding':
            dict_obj = attr.asdict(obj)
            if isinstance(obj, GroundingKey):
                dict_obj['arg'] = obj.arg.__dict__
            return dict_obj
        elif attr.has(obj):
            return attr.asdict(obj)
        elif isinstance(obj, list):
            obj = tuple(obj)
        return super().default(obj)

class ComplexDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if isinstance(obj, dict) and 'type' in obj and 'keys' in obj:
            return GroundingKey(grounding_type=obj['type'], keys=obj['keys'])
        elif isinstance(obj, dict) and 'value' in obj and 'from_qdmr' in obj:
            return ValueUnit(**obj)
        elif isinstance(obj, dict) and 'arg_type' in obj and 'arg' in obj:
            obj['arg'] = self.object_hook(obj['arg'])
            return QDMRStepArg(**obj)

        return obj
