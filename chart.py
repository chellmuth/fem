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

    def domain(self, n):
        t1 = self.T(1, n)
        t2 = self.T(2, n)
        t3 = self.T(3, n)

        n1 = self.N(t1)
        n2 = self.N(t2)
        n3 = self.N(t3)

        if n % 2 == 0:
            return ((n1[0], n2[0]), (n1[1], n2[1]))
        else:
            return ((n1[0], n2[0]), (n1[1], n3[1]))

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
    h = symbols("h")

    xs, ys = fem.domain(10)
    assert xs == (2*h, 3*h)
    assert ys == (h, 2*h)

    xs, ys = fem.domain(13)
    assert xs == (0, h)
    assert ys == (2*h, 3*h)

if __name__ == "__main__":
    render()
