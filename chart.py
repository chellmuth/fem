from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, pi, sin
import numpy as np
import sympy

import render
from helper import internal_nodes, build_linear_basis_function

class FEM(object):
    def __init__(self, dim):
        self.dim = dim
        self.h = 1 / (dim + 1)

    def N(self, t):
        row = t // (self.dim + 2)
        col = t % (self.dim + 2)

        h = symbols("h")
        return ((col * h).subs({"h": self.h}), (row * h).subs({"h": self.h}))

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

    def w(self, n, etas):
        t1 = self.T(1, n)
        t2 = self.T(2, n)
        t3 = self.T(3, n)

        eta1 = etas[t1][0]
        eta2 = etas[t2][0]
        eta3 = etas[t3][0]

        n1 = self.N(t1)
        n2 = self.N(t2)
        n3 = self.N(t3)

        x, y = symbols("x y")

        if n % 2 == 0:
            domain = ((x>= n1[0]) & (x<=n2[0]) & ((y-n1[1])>=(x-n1[0])) & (y<=n2[1]))
        else:
            domain = ((x>= n1[0]) & (x<=n2[0]) & ((y-n1[1])<=(x-n1[0])) & (y>=n2[1]))

        return sympy.Piecewise(
            (
                (self.psi(1, n) * eta1) + (self.psi(2, n) * eta2) + (self.psi(3, n) * eta3),
                domain
            ),
            (0, True)
        )

    def w_internal(self, n, etas):
        internals = internal_nodes(self.dim)

        t1 = self.T(1, n)
        t2 = self.T(2, n)
        t3 = self.T(3, n)

        eta1 = 0
        eta2 = 0
        eta3 = 0
        if t1 in internals:
            eta1 = etas[internals.index(t1)][0]
        if t2 in internals:
            eta2 = etas[internals.index(t2)][0]
        if t3 in internals:
            eta3 = etas[internals.index(t3)][0]

        n1 = self.N(t1)
        n2 = self.N(t2)
        n3 = self.N(t3)

        x, y = symbols("x y")

        if n % 2 == 0:
            domain = ((x>= n1[0]) & (x<=n2[0]) & ((y-n1[1])>=(x-n1[0])) & (y<=n2[1]))
        else:
            domain = ((x>= n1[0]) & (x<=n2[0]) & ((y-n1[1])<=(x-n1[0])) & (y>=n2[1]))

        return sympy.Piecewise(
            (
                (self.psi(1, n) * eta1) + (self.psi(2, n) * eta2) + (self.psi(3, n) * eta3),
                domain
            ),
            (0, True)
        )

from scipy.integrate import dblquad

def integrate(f, domain):
    x, y = symbols("x y")
    x_domain, y_domain = domain
    x_low, x_high = x_domain
    y_low, y_high = y_domain

    y_low_converted = sympy.lambdify(x, y_low)
    y_high_converted = sympy.lambdify(x, y_high)

    # result1 = sympy.integrate(
    #     f,
    #     (y, y_domain[0], y_domain[1]),
    #     (x, x_domain[0], x_domain[1]),
    # )

    ff = sympy.lambdify((x, y), f)
    result2 = dblquad(ff, x_low, x_high, y_low_converted, y_high_converted)

    return result2[0]

def build_elemental_b(fem, n, f):
    b = [ 0, 0, 0 ]
    for alpha in range(3):
        b[alpha] = integrate(
            (fem.psi(alpha + 1, n).subs({"h": fem.h}) * f),
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
            integrand = alpha_gradient[0] * beta_gradient[0] + \
                        alpha_gradient[1] * beta_gradient[1]

            A[alpha][beta] = integrate(
                integrand.subs({"h": fem.h}),
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

def test_internal_nodes():
    assert internal_nodes(1) == [4]
    assert internal_nodes(2) == [5,6,9,10]
    assert internal_nodes(3) == [6,7,8, 11, 12, 13, 16, 17, 18]




if __name__ == "__main__":
    dim = 3
    node_count = (dim + 2) ** 2
    triangle_count = 2 * (dim + 1) ** 2

    fem = FEM(dim)

    render.render_u(FEM(20))

    x, y = symbols("x y")

    f = -8*pi*sin(2*pi*x)*sin(2*pi*y)

    A_n = []
    b_n = []
    for n in range(triangle_count):
        print(n)
        A_n.append(build_element_stiffness(fem, n))
        b_n.append(build_elemental_b(fem, n, f))

    A = np.zeros((node_count, node_count))
    b = np.zeros((node_count, 1))

    for n in range(triangle_count):
        print(n)
        for alpha in range(3):
            for beta in range(3):
                A[fem.T(alpha+1, n)][fem.T(beta+1, n)] += A_n[n][alpha][beta]
            b[fem.T(alpha+1, n)][0] += b_n[n][alpha]

    A_internal = build_internal_A(A, dim)
    b_internal = build_internal_b(b, dim)

    etas_internal = np.linalg.solve(A_internal, b_internal)

    ws = []
    for n in range(triangle_count):
        w = fem.w_internal(n, etas_internal)
        if w != 0:
            ws.append(w)

    render.render_internal(fem, etas_internal)

    # Global render
    if False:
        etas = np.linalg.solve(A, b)

        ws = []
        for n in range(triangle_count):
            w = fem.w(n, etas)
            if w != 0:
                ws.append(w)

        render.render(fem, etas)

