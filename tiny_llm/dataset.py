class Dataset:

    def __init__(self, encoded_ids):
        self.encoded_ids = encoded_ids
        self.examples = self.generate_training_set()

    def generate_training_set(self):
        examples = []

        for i in range(1, len(self.encoded_ids)):
            context = self.encoded_ids[:i]
            target = self.encoded_ids[i]
            examples.append((context, target))

        return examples