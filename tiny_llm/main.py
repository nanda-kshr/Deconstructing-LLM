from tokenizer import Tokenizer
from dataset import Dataset
from embedding import Embedding
from attention import Attention

print("Tokenizer")
t = Tokenizer()
encoded_ids = t.encode("hello")
print(encoded_ids)
print(t.decode(encoded_ids))

print("Dataset")
d = Dataset(encoded_ids)
training_examples = d.examples
print(training_examples)

print("Embedding")
embedding_dim = 10
e = Embedding(t.vocab_size, embedding_dim)
embeddings = e.forward(training_examples[2][0])
print(embeddings)

print("Attention")
a = Attention(embedding_dim)
print(a.forward(embeddings))