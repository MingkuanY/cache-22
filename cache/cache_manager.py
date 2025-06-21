class CacheManager:
    def __init__(self):
        self._store = set()

    def check_cache(self, vec):
        return vec in self._store

    def add_to_cache(self, vec):
        self._store.add(vec)
