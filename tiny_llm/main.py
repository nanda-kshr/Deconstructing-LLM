from tokenizer import Tokenizer
from dataset import Dataset
from embedding import Embedding

t = Tokenizer()
encoded_ids = t.encode("hello")
print(encoded_ids)
print(t.decode(encoded_ids))

d = Dataset(encoded_ids)
training_examples = d.examples
print(training_examples)

e = Embedding(t.vocab_size, 10)
embeddings = e.forward(training_examples[2][0])
print(embeddings)


