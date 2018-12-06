from mpl_toolkits.mplot3d import Axes3D
from sympy import Symbol, symbols
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import sympy

def build_triangulation(size):
    h = 1 / (size + 1)

    x = []
    y = []

    triangles = []
    for row in range(size + 2):
        for col in range(size + 2):
            x.append(col * h)
            y.append(row * h)

    for cell_row in range(size + 1):
        for cell_col in range(size + 1):
            LL = (size + 2) * cell_row + cell_col
            LR = LL + 1
            UL = LL + (size + 2)
            UR = UL + 1
            triangles.extend([
                [LL, LR, UR],
                [LL, UR, UL],
            ])

    triangulation = tri.Triangulation(x, y, triangles)
    return triangulation

def build_z(size, values):
    z = []
    z.extend(0 for _ in range(size+2))
    for row in range(size):
        z.append(0)
        z.extend(values[row*size:(row+1)*size])
        z.append(0)
    z.extend(0 for _ in range(size+2))

    return z


def render():
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    size = 3
    z = [ 1 for _ in range(size**2) ]
    ax.plot_trisurf(build_triangulation(size), build_z(size, z), linewidth=0.2, antialiased=True)

    plt.show()

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

class FEM(object):
    def __init__(self, dim):
        self.dim = dim
        self.h = 1 / (dim + 1)

    def N(self, t):
        row = t // (self.dim + 2)
        col = t % (self.dim + 2)

        h = Symbol("h")
        return (col * h, row * h)

    def T(self, alpha, n):
        row = (n // 2) // (self.dim + 1)
        col = (n // 2) % (self.dim + 1)

        LL = (self.dim + 2) * row + col
        LR = LL + 1
        UL = LL + (self.dim + 2)
        UR = UL + 1

        if alpha == 1:
            return LL

        if n % 2 == 0:
            if alpha == 2:
                return UR
            elif alpha == 3:
                return UL
        else:
            if alpha == 2:
                return LR
            elif alpha == 3:
                return UR

        raise Exception

    def psi(self, alpha, n):
        t1 = self.T(1, n)
        t2 = self.T(2, n)
        t3 = self.T(3, n)

        n1 = self.N(t1)
        n2 = self.N(t2)
        n3 = self.N(t3)

        if alpha == 1:
            p, ps = n1, [n2, n3]
        elif alpha == 2:
            p, ps = n2, [n1, n3]
        elif alpha == 3:
            p, ps = n3, [n1, n2]
        else:
            raise Exception

        psi = build_linear_basis_function(p, ps)
        return psi

    def gradient_psi(self, alpha, n):
        psi = self.psi(alpha, n)
        x, y = symbols("x y")
        return (sympy.diff(psi, x), sympy.diff(psi, y))

    def domain(self, n):
        t1 = self.T(1, n)
        t2 = self.T(2, n)
        t3 = self.T(3, n)

        n1 = self.N(t1)
        n2 = self.N(t2)
        n3 = self.N(t3)

        x = symbols("x")
        if n % 2 == 0:
            return ((n1[0], n2[0]), (n1[1] + (x - n1[0]), n2[1]))
        else:
            return ((n1[0], n2[0]), (n1[1], n3[1] - (n2[0] - x)))

def integrate(f, domain):
    x, y = symbols("x y")
    x_domain, y_domain = domain
    result = sympy.integrate(
        f,
        (y, y_domain[0], y_domain[1]),
        (x, x_domain[0], x_domain[1]),
    )
    return result

def build_elemental_b(fem, n, f):
    b = [ 0, 0, 0 ]
    for alpha in range(3):
        b[alpha] = integrate(
            fem.psi(alpha + 1, n) * f,
            fem.domain(n)
        )
    return b

def build_element_stiffness(fem, n):
    A = [
        [ 0, 0, 0 ],
        [ 0, 0, 0 ],
        [ 0, 0, 0 ],
    ]

    for alpha in range(3):
        for beta in range(3):
            alpha_gradient = fem.gradient_psi(alpha + 1, n)
            beta_gradient = fem.gradient_psi(beta + 1, n)
            A[alpha][beta] = integrate(
                alpha_gradient[0] * beta_gradient[0] + \
                alpha_gradient[1] * beta_gradient[1],
                fem.domain(n)
            )

    return A

import pytest

def test_T():
    fem = FEM(2)

    assert fem.T(1, 0) == 0
    assert fem.T(2, 0) == 5
    assert fem.T(3, 0) == 4

    assert fem.T(1, 1) == 0
    assert fem.T(2, 1) == 1
    assert fem.T(3, 1) == 5

    assert fem.T(3, 15) == 14

def test_N():
    fem = FEM(2)

    h = Symbol("h")

    assert fem.N(fem.T(1, 0)) == (0, 0)
    assert fem.N(fem.T(2, 0)) == (h, h)
    assert fem.N(fem.T(3, 11)) == (3*h, 2*h)

def test_build_linear_basis_function():
    p1 = [0, 0]

    h, x = symbols("h x")
    p2 = [h, 0]
    p3 = [h, h]

    z = build_linear_basis_function(p1, [p2, p3])
    assert z == 1 - x/h

def test_psi():
    fem = FEM(2)

    h, x, y = symbols("h x y")

    psi = fem.psi(1, 8)
    assert psi == 2 - y/h

    psi = fem.psi(2, 8)
    assert psi == -1 + x/h

def test_domain():
    fem = FEM(2)
    h, x = symbols("h x")

    xs, ys = fem.domain(10)
    assert xs == (2*h, 3*h)
    assert ys == (h + (x - 2*h), 2*h)

    xs, ys = fem.domain(13)
    assert xs == (0, h)
    assert ys == (2*h, 3*h - (h - x))

def test_elemental_b():
    fem = FEM(2)
    h, x = symbols("h x")

    f = x

    b = build_elemental_b(fem, 0, f)
    # assert b == [ h**2 / 2, h**2 / 2, -h**2 / 2 ]

def build_internal_A(A, dim):
    A_internal = np.zeros((dim**2, dim**2))

    nodes = internal_nodes(dim)
    for row in range(dim**2):
        for col in range(dim**2):
            A_internal[row][col] = A[nodes[row]][nodes[col]]
    return A_internal

def build_internal_b(b, dim):
    b_internal = np.zeros((dim**2, 1))

    nodes = internal_nodes(dim)
    for row in range(dim**2):
        b_internal[row][0] = b[nodes[row]][0]
    return b_internal

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

def test_internal_nodes():
    assert internal_nodes(1) == [4]
    assert internal_nodes(2) == [5,6,9,10]
    assert internal_nodes(3) == [6,7,8, 11, 12, 13, 16, 17, 18]

if __name__ == "__main__":
    dim = 3
    node_count = (dim + 2) ** 2
    triangle_count = 2 * (dim + 1) ** 2

    fem = FEM(dim)
    x, h = symbols("x h")
    f = 10

    A_n = []
    b_n = []
    for n in range(triangle_count):
        A_n.append(build_element_stiffness(fem, n))
        b_n.append(build_elemental_b(fem, n, f))

    A = np.zeros((node_count, node_count))
    b = np.zeros((node_count, 1))

    for n in range(triangle_count):
        for alpha in range(3):
            for beta in range(3):
                A[fem.T(alpha+1, n)][fem.T(beta+1, n)] += A_n[n][alpha][beta]
            b[fem.T(alpha+1, n)][0] += b_n[n][alpha].subs(h, fem.h)

    # print (A)
    # print (b)
    etas = np.linalg.solve(A, b)
    # print(etas)

    A_internal = build_internal_A(A, dim)
    b_internal = build_internal_b(b, dim)
    print(A_internal)
    print(b_internal)

    etas_internal = np.linalg.solve(A_internal, b_internal)
    print(etas_internal)
    # render()
