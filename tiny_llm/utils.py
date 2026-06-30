
def matmul(vector, matrix):
        result = [0] * len(matrix[0])
        for i in range(len(matrix[0])):
            for j in range(len(vector)):
                result[i] += vector[j] * matrix[j][i]
        return result

def multiply_vector( vector, scalar):
    return [x * scalar for x in vector]

def add_vectors(a, b):
    return [x + y for x, y in zip(a, b)]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))
    
