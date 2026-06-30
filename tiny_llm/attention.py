sentence = ["I", "love", "pizza"]

scores = {
    "I": 0.1,
    "love": 0.2,
    "pizza": 0.7
}


I = [0.2, 0.1]
love = [0.5, 0.7]
pizza = [0.9, 0.4]

new_x = (
    0.1 * I[0] +
    0.2 * love[0] +
    0.7 * pizza[0]
)

new_y = (
    0.1 * I[1] +
    0.2 * love[1] +
    0.7 * pizza[1]
)

