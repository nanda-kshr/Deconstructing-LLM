from tokenizer import Tokenizer
from dataset import Dataset

t = Tokenizer()
encoded_ids = t.encode("hello")
print(encoded_ids)
print(t.decode(encoded_ids))

d = Dataset(encoded_ids)
print(d.training_set)