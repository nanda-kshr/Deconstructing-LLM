
import random
import utils

class FeedForward:
    def __init__(self, embedding_dim, hidden_dim):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.W1 = [[random.random() * 2 - 1 for _ in range(hidden_dim)] for _ in range(embedding_dim)]
        self.b1 = [random.random() * 2 - 1 for _ in range(hidden_dim)]
        self.W2 = [[random.random() * 2 - 1 for _ in range(embedding_dim)] for _ in range(hidden_dim)]
        self.b2 = [random.random() * 2 - 1 for _ in range(embedding_dim)]
    
    def relu(self, vector):
        return [max(0, x) for x in vector]
    
    def forward(self, embeddings):
        outputs = []
        for embedding in embeddings:

            hidden = utils.matmul(embedding, self.W1)
            hidden = utils.add_vectors(hidden, self.b1)

            hidden = self.relu(hidden)
    
            output = utils.matmul(hidden, self.W2)
            output = utils.add_vectors(output, self.b2)
            outputs.append(output)
        return outputs