from dataclasses import dataclass

import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Tree:
    _values: tuple
    _keys: tuple

    @classmethod
    def from_dict(cls, d: dict):
        keys = tuple(d.keys())
        values = tuple(
            cls.from_dict(v) if isinstance(v, dict) else v for v in d.values()
        )
        return cls(values, keys)

    def tree_flatten(self):
        return self._values, self._keys

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(values, keys)

    def as_dict(self):
        return {
            k: v.as_dict() if isinstance(v, Tree) else v
            for k, v in zip(self._keys, self._values)
        }
