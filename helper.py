import sympy
from sympy import symbols

def internal_nodes(dim):
    indices = []

    original_dim = dim + 2
    for row in range(dim):
        for col in range(dim):
            indices.append(
                original_dim
                + original_dim * row
                + 1
                + col
            )

    return indices

def build_linear_basis_function(p1, ps):
    p2, p3 = ps

    A = sympy.Matrix([
        [ p1[0], p1[1], 1 ],
        [ p2[0], p2[1], 1 ],
        [ p3[0], p3[1], 1 ],
    ])

    b = sympy.Matrix(
        3, 1,
        [ 1, 0, 0 ],
    )

    a, b, c = tuple(A.LUsolve(b))
    x, y = symbols("x y")

    return a*x + b*y + c
