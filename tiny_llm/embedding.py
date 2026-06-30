import random

class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_table = [[random.random() * 2 - 1 for _ in range(self.embedding_dim)] for _ in range(self.vocab_size)]
    
    def forward(self, input_ids):
        return [self.embedding_table[i] for i in input_ids]