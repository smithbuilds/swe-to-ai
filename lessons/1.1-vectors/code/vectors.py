import math

def scale(vector, scalar):
    result = []
    for num in vector:
        v = num * scalar
        result.append(v)
    return(result)

def add(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Can't add vectors")
    return [a + b for a, b in zip(v1, v2)]

def dot(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Can't dot vectors")
    return sum(a * b for a, b in zip(v1, v2))

def magnitude(v):
    result = math.sqrt(dot(v, v))
    return result

def normalize(v):
    result = scale(v, 1/magnitude(v))
    return result

print(scale([2, 3], 3))
print(add([1,1], [2,2]))
print(dot([1,2], [3, 4]))
print(magnitude([3,4]))
print(normalize([3, 4]))