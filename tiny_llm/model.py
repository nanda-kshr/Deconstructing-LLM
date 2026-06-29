class Model:
    def __init__(self):
        self.d = {}
    def train(self, examples):
        for context, target in examples:
            context = tuple(context)
            self.d.setdefault(context, {})
            if target in self.d[context]:
                self.d[context][target] = self.d[context][target] + 1
            else:
                self.d[context][target] = 1
        return self.d
    
    def predict(self, context):
        if tuple(context) not in self.d:
            return {}
        counts = self.d[tuple(context)]
        total = sum(counts.values())

        return {
            token: count / total for token, count in counts.items()
        }

e = [
    (["I"], "love"),
    (["I", "love"], "pizza"),
    (["I"], "eat"),
    (["I", "love"], "pasta"),
]

m = Model()
m.train(e)
print(m.predict(['We']))