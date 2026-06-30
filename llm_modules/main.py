from tokenizer import Tokenizer

with open("data.txt", "r") as file:
    data = file.read()

t = Tokenizer()

data = t.tokenize(data)
print(data)