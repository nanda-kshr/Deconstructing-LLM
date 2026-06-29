words = [
    "I",
    "love",
    "pizza",
    "eat",
    "pasta"
]

class Embedding:
    def __init__(self):
        self.word_to_id = {
            "I": 0,
            "love": 1,
            "pizza": 2,
            "eat": 3,
            "pasta": 4
        }

        self.embedding_table = [
            [0.2, 0.1],   # I
            [0.5, 0.7],   # love
            [0.9, 0.4],   # pizza
            [0.3, 0.8],   # eat
            [0.8, 0.3]    # pasta
        ]

    def get_embedding(self, word):
        word_row = self.word_to_id[word]
        return self.embedding_table[word_row]
    
    def similarity(self, word1, word2):
        v1 = self.get_embedding(word1)
        v2 = self.get_embedding(word2)

        return (
            v1[0] * v2[0] +
            v1[1] * v2[1]
        )


e = Embedding()

print(e.get_embedding("pizza"))
