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
    mag = magnitude(v)
    if mag == 0:
        raise ValueError("Cannot normalize the zero vector")
    result = scale(v, 1/magnitude(v))
    return result

def subtract(v1, v2):
    v2 = scale(v2, -1)
    result = add(v1, v2)
    return result

def verifymag(v):
    if(magnitude(normalize(v)) == 1):
        return True
    else:
        return False
    
def are_orthogonal(v1, v2):
    result = dot(v1, v2)
    if result == 0:
        return True
    else: 
        return False

def linear_combination(vectors, scalars):
    length = len(vectors)
    x = 0
    weighted_sum = [0] * len(vectors[0])
    while x < length:
       weighted_sum = add(scale(vectors[x], scalars[x]), weighted_sum)
       x += 1
    return weighted_sum

print(scale([2, 3], 3))
print(add([1,1], [2,2]))
print(dot([1,2], [3, 4]))
print(magnitude([3,4]))
print(normalize([3, 4]))
print(subtract([4,6],[1,2]))
print(verifymag([4,50,6]))
print(verifymag([6, 8, 20]))
print(verifymag([1,2,3]))
print(linear_combination([[1,2],[2,3],[4,4]], [1, 2, 5]))
      