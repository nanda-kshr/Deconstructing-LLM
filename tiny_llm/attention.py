import math
import random
from utils import matmul, multiply_vector, add_vectors, dot

class Attention:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.Wq = [[random.random() * 2 - 1 for i in range(embedding_dim)] for i in range(embedding_dim)]
        self.Wk = [[random.random() * 2 - 1 for i in range(embedding_dim)] for i in range(embedding_dim)]
        self.Wv = [[random.random() * 2 - 1 for i in range(embedding_dim)] for i in range(embedding_dim)]
    
    def forward(self, embeddings):
        queries = []
        keys = []
        values = []
        for embedding in embeddings:
            queries.append(matmul(embedding, self.Wq))
            keys.append(matmul(embedding, self.Wk))
            values.append(matmul(embedding, self.Wv))
        scores = self.attention_scores(queries, keys)
        normalized_scores = self.normalize_scores(scores)
        outputs = self.apply_attention(normalized_scores, values)
        return outputs

    
    def attention_scores(self, queries, keys):
        scores = []
        for q in queries:
            score_row = []
            for k in keys:
                score_row.append(dot(q, k) / math.sqrt(self.embedding_dim))
            scores.append(score_row)
        return scores
    
    def softmax(self, scores):
        max_score = max(scores)
        exp_values = [
            math.exp(score - max_score)
            for score in scores
        ]
        sum_exp = sum(exp_values)
        return [exp / sum_exp for exp in exp_values]
    
    def normalize_scores(self, scores):
        normalized_scores = []
        for score_row in scores:
            normalized_scores.append(self.softmax(score_row))
        return normalized_scores
    
    
    def apply_attention(self, probabilities, values):
        outputs = []
        for probability in probabilities:
            output_vector = [0] * self.embedding_dim
            for prob, value in zip(probability, values):
                output_vector = add_vectors(output_vector, multiply_vector(value, prob))
            outputs.append(output_vector)
        return outputs

