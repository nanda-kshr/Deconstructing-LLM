import utils



class TransformerBlock:

    def __init__(self, attention, feedforward, layernorm, residual):
        self.attention = attention
        self.feedforward = feedforward
        self.layernorm = layernorm
        self.residual = residual

    def forward(self, embeddings):

        x = self.attention.forward(embeddings)
        x = self.residual(x, embeddings)

        x = self.layernorm(x)

        x = self.feedforward.forward(x)

        x = self.residual(x, previous)

        x = self.layernorm(x)

        return x
    
    def residual(self, x, original):
        return utils.add_vectors(x, original)
    
    def layer_norm(self, embeddings):
        return embeddings

