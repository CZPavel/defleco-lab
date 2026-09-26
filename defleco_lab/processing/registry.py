from .base import ProcessingMethod


class MethodRegistry:
    def __init__(self):
        self._types: dict[str, type[ProcessingMethod]] = {}

    def register(self, method: type[ProcessingMethod]):
        if method.info.id in self._types:
            raise ValueError(f"duplicate method id: {method.info.id}")
        self._types[method.info.id] = method
        return method

    def create(self, method_id: str, **params):
        return self._types[method_id](**params)

    def infos(self):
        return tuple(t.info for t in self._types.values())


registry = MethodRegistry()
