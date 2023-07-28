# from
# https://github.com/DFKI-NLP/sam/blob/main/sam/utils.py
from typing import Dict

from allennlp.nn.util import masked_max, masked_mean, get_final_encoder_states
from torch import Tensor, sum, BoolTensor
import torch


def pool(vector: Tensor,
         mask: BoolTensor,
         dim: int,
         pooling: str,
         is_bidirectional: bool) -> Tensor:
    if pooling == "max":
        return masked_max(vector, mask, dim)
    elif pooling == "mean":
        return masked_mean(vector, mask, dim)
    elif pooling == "sum":
        return torch.sum(vector, dim)
    elif pooling == "final":
        return get_final_encoder_states(vector, mask, is_bidirectional)
    else:
        raise ValueError(f"'{pooling}' is not a valid pooling operation.")


def flatten_dict(d, _keys=()):
    if not isinstance(d, dict):
        return d
    res = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_children = flatten_dict(v, _keys=_keys+(k,))
            res.update(flat_children)
        else:
            res[_keys+(k,)] = v
    return res


def unflatten_dict(d, _res=None):
    if not isinstance(d, dict):
        return d
    if _res is None:
        _res = {}

    for k, v in d.items():
        if len(k) == 1:
            _res[k[0]] = v
        else:
            unflat = unflatten_dict({k[1:]: v}, _res.get(k[0], None))
            if k[0] not in _res:
                _res[k[0]] = unflat
    return _res


class TensorDict:

    ROOT_KEY = 'root'

    def __init__(self, d, is_flat=False):
        if not isinstance(d, dict):
            d = {self.ROOT_KEY: d}
            assert not is_flat, 'TensorDict can not be initialised with a non-dict and is_flat'
        else:
            assert self.ROOT_KEY not in d.keys(), f'ROOT_KEY={self.ROOT_KEY} not allowed as key'
        #assert isinstance(d, dict), f'd has to be a dict, but it is: {type(d)}'
        self._d = None
        self._d_flat = None
        if not is_flat:
            self._d = d
        else:
            self._d_flat = d

    @property
    def flat(self):
        if self._d_flat is None:
            self._d_flat = flatten_dict(self._d)
        return self._d_flat

    @property
    def d(self):
        if self._d is None:
            self._d = unflatten_dict(self._d_flat)
        if self.ROOT_KEY in self._d:
            return self._d[self.ROOT_KEY]
        else:
            return self._d

    def split(self, dim=0):
        n_dim = None
        res = None
        not_split = {}
        for k, v in self.flat.items():
            # split only, if split dim in range
            if isinstance(v, Tensor) and len(v.size()) > dim:
                if res is None:
                    n_dim = v.size()[dim]
                    res = [{} for _ in range(n_dim)]
                else:
                    assert n_dim == v.size()[dim], 'dimension does not match'
                for i, d in enumerate(res):
                    if dim != 0:
                        raise NotImplementedError('split is only implemented for dim=0')
                    d[k] = v[i:i+1]
            else:
                not_split[k] = v
        for d in res:
            d.update(not_split)
        return [TensorDict(d, is_flat=True) for d in res]

    @staticmethod
    def merge(tensordicts, dim=0, non_tensor_handling_func=None):
        tensordicts_flat = [d.flat if isinstance(d, TensorDict) else TensorDict(d).flat for d in tensordicts]
        dict_of_lists = {k: [dic[k] for dic in tensordicts_flat] for k in tensordicts_flat[0]}
        res = {}
        for k, v in dict_of_lists.items():
            if isinstance(v[0], Tensor):
                res[k] = torch.cat(v, dim=dim)
            else:
                if non_tensor_handling_func is None:
                    raise Exception(f"key={k}: merge for type {type(v)} not implemented. "
                                    f"Please provide a non_tensor_handling_func that merge")
                else:
                    res[k] = non_tensor_handling_func(v)

        return TensorDict(res, is_flat=True)

    def __eq__(self, other):
        s = self.flat
        o = other.flat
        if s.keys() != o.keys():
            return False
        for k, v in s.items():
            if isinstance(v, Tensor) and isinstance(o[k], Tensor):
                if not v.equal(o[k]):
                    return False
            else:
                if v != o[k]:
                    return False
        return True


def normalize_span_f1_result(metric_dict: Dict[str, float]):
    res = {}
    # tp, fp, fn added to mapping so that we can get their count in output metric
    mapping = {'recall-': 'recall', 'precision-': 'precision', 'f1-measure-': 'f1','tp-':'tp','fp-':'fp','fn-':'fn'}
    for k, v in metric_dict.items():
        new_k = None
        for x, y in mapping.items():
            if k.startswith(x):
                new_k = f'{k[len(x):]}/{y}'
                break
        assert new_k is not None, f'unknown metric key: {k} (value: {v})'
        res[new_k] = v

    return res