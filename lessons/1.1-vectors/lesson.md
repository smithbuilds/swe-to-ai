# 1.1 — Vectors Aren't Just Arrays

## Why This Matters

Every piece of data in a machine learning system is a vector. When a language model processes a word, it becomes a vector. When an image goes into a neural network, it becomes a vector. When you compare two documents for similarity, you're measuring the angle between vectors.

If you treat vectors as just arrays of numbers, you'll be able to run the code — but you won't understand what it's doing. The geometric intuition is what separates someone who uses ML from someone who can debug it, improve it, and reason about why it fails.

---

## The Intuition

In computer science, a vector is a resizable array. In math, a vector is an **arrow** — it has a direction and a magnitude (length), with its tail fixed at the origin.

`[2, 3]` doesn't mean "a list containing 2 and 3." It means "go 2 units right and 3 units up." The numbers are coordinates of where the arrow's tip lands.

This distinction matters because you can do things to arrows that you can't do to arbitrary arrays:
- You can scale them (stretch or shrink)
- You can add them (tip-to-tail)
- You can measure the angle between them
- You can ask whether they span a space

### Basis Vectors

Every vector in 2D space is a combination of two special vectors:
- **î** (i-hat) = `[1, 0]` — one unit right
- **ĵ** (j-hat) = `[0, 1]` — one unit up

`[2, 3]` is just `2 * î + 3 * ĵ`. The basis vectors are a coordinate system — a shared language for describing where things are in space.

Different choices of basis vectors give you different coordinate systems. This matters in ML: embedding spaces have their own basis, and the choice of basis affects what operations mean.

### Span

The **span** of a set of vectors is every point you can reach using linear combinations of them — any combination of scaling and adding.

- Two vectors pointing in different directions → span is the entire 2D plane
- Two vectors pointing in the same direction → span is just a line (one is redundant)

### Linear Independence

Vectors are **linearly independent** if none of them can be expressed as a combination of the others. Each vector adds new "reach" to the span.

Vectors are **linearly dependent** if one is redundant — it lives in the span of the others. `[1, 2]` and `[2, 4]` are dependent: the second is just `2 * [1, 2]`.

---

## The Math

Two operations define everything in linear algebra:

**Scaling:** multiply every component by a scalar

    s · [v₁, v₂, ..., vₙ] = [s·v₁, s·v₂, ..., s·vₙ]

**Addition:** add corresponding components

    [v₁, v₂, ..., vₙ] + [w₁, w₂, ..., wₙ] = [v₁+w₁, v₂+w₂, ..., vₙ+wₙ]

From these two, everything else follows.

**Dot product:** multiply corresponding components, then sum

    v · w = (v₁·w₁) + (v₂·w₂) + ... + (vₙ·wₙ)

The dot product measures how much two vectors point in the same direction:
- Positive → same-ish direction
- Zero → perpendicular
- Negative → opposite directions

**Magnitude:** the length of the arrow

    |v| = sqrt(v · v) = sqrt(v₁² + v₂² + ... + vₙ²)

**Normalization:** scale a vector to magnitude 1 (unit vector)

    v̂ = v / |v|

---

## Let's Build It

No numpy. Just Python lists so you feel every operation.

```python
import math

def scale(vector, scalar):
    result = []
    for num in vector:
        v = num * scalar
        result.append(v)
    return result

def add(v1, v2):
    # Vectors must have the same dimension to add
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension")
    # zip pairs up elements: zip([1,2], [3,4]) -> [(1,3), (2,4)]
    return [a + b for a, b in zip(v1, v2)]

def dot(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension")
    # Multiply corresponding elements, sum the results
    return sum(a * b for a, b in zip(v1, v2))

def magnitude(v):
    # Length of the arrow = sqrt of dot product with itself
    return math.sqrt(dot(v, v))

def normalize(v):
    # Scale by 1/magnitude to make the arrow length 1
    return scale(v, 1 / magnitude(v))
```

Test cases:
```python
print(scale([2, 3], 3))       # [6, 9]
print(add([1, 1], [2, 2]))    # [3, 3]
print(dot([1, 2], [3, 4]))    # 11  (1*3 + 2*4)
print(magnitude([3, 4]))      # 5.0 (3-4-5 triangle)
print(normalize([3, 4]))      # [0.6, 0.8] — same direction as [3,4], but magnitude(result) == 1.0
```

---

## The Mistakes I Made

**Returning strings for errors instead of raising exceptions.**
My first version of `add` returned `"Can't add vectors"` on dimension mismatch. That's a silent failure — the string propagates through the program and causes a confusing crash somewhere else. `raise ValueError(...)` fails loudly at the source.

**Confusing "length" with "dimension."**
In Python, `len(v)` gives you the number of elements. In math, that's the vector's *dimension*, not its length. Length (magnitude) is `sqrt(dot(v, v))`. Two different things with similar names — using the wrong one in conversation will cause confusion.

**Floating point surprises.**
`normalize([3, 4])` returns `[0.6000000000000001, 0.8]`, not `[0.6, 0.8]`. Computers represent numbers in binary, and `0.6` has no exact binary representation. This isn't a bug — it's a property of floating point arithmetic you'll encounter constantly in ML.

---

## Try It Yourself

1. Write a `subtract(v1, v2)` function. Don't use a loop — use `scale` and `add`.
2. Verify that `magnitude(normalize(v))` returns `1.0` for any non-zero vector.
3. What happens when you call `normalize([0, 0])`? Why? How would you handle it?
4. Write `are_orthogonal(v1, v2)` — returns `True` if two vectors are perpendicular. Use `dot`.
5. Write `linear_combination(vectors, scalars)` — takes a list of vectors and a list of scalars, returns the weighted sum. This is the core operation behind every neural network layer.

---

## The Takeaway

A vector is an arrow in space, not a list of numbers. The numbers are just how we write it down. Everything in machine learning — embeddings, weights, activations, gradients — is a vector, and the geometry of those vectors (direction, magnitude, angle between them) is what encodes meaning. Scaling and addition are the two primitives; everything else is built from them.

---

## Resources

- [3Blue1Brown — Essence of Linear Algebra, Ch. 1-2](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Mathematics for Machine Learning — Ch. 2](https://mml-book.github.io)
- [Stanford CS229 Linear Algebra Review](https://cs229.stanford.edu/section/cs229-linalg.pdf)
