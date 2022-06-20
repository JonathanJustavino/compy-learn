
class IndexCache(object):

    def __init__(self, fn_load_item, max_cache_size=1000):
        self.fn_load_item = fn_load_item
        self.max_cache_size = max_cache_size
        self.index_list = []
        self.item_by_index = {}

    def get(self, index):
        item = self.item_by_index.get(index)
        if not item:
            item = self.fn_load_item(index)
            self.set(index, item)
        return item

    def set(self, index, item):
        self.item_by_index[index] = item
        self.index_list.append(index)
        self.ensure_cache_size()

    def ensure_cache_size(self):
        while len(self.index_list) > self.max_cache_size:
            oldest_index = self.index_list.pop(0)
            print("Deleted", oldest_index, end="\r")
            del self.item_by_index[oldest_index]
