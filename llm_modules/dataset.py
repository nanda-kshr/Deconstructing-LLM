
def create_training_examples(tokens):
    l = []
    for i in range(1, len(tokens)):
        l.append((tokens[0:i],tokens[i]))
    return l