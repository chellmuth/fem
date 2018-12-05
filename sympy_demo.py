from sympy import cos, symbols, pi, Piecewise
from sympy.plotting import plot3d
x, y = symbols('x y')

vh = Piecewise((1 - (x + y), (0 <= x) & (x + y <=1) & (0 <= y)), (0, True))
u = 1/4 * (-cos(2 * pi * x) + 1) * (-cos(2 * pi * y) + 1)

plot3d(vh, (x, 0, 1), (y, 0, 1))
