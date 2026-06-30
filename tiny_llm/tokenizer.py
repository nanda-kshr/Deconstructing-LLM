import re


class Tokenizer:

    def __init__(self, file="data/char_train.txt"):
        self.text = self.load_text(file)
        self.fit(self.text)

    def fit(self, text):
        self.vocab = self.build_vocab()
        self.vocab_size = len(self.vocab)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

    def load_text(self, file):
        with open(file, "r") as f:
            text = f.read()
            return text
        
    def build_vocab(self):
        tokens = sorted(set(self.text)) # re.findall(r'\w+|[^\w\s]', self.text)
        return sorted(tokens)
    
    def encode(self, text):
        return [self.token_to_id[ch] for ch in text]
    
    def decode(self, ids):
        return "".join([self.id_to_token[idx] for idx in ids if idx in self.id_to_token])
        



# t = Tokenizer()
# print(t.text)
# print(t.vocab)
# print(t.token_to_id)
# print(t.id_to_token)
# print(t.encode("hello"))
# print(t.decode([7,18,22]))