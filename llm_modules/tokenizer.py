
class Tokenizer:

    def tokenize(self, text):
        return [i.split() for i in text.split("\n")]