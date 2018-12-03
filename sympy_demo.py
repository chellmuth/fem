from sympy import cos, symbols, pi
from sympy.plotting import plot3d
x, y = symbols('x y')

plot3d(1/4 * (-cos(2 * pi * x) + 1) * (-cos(2 * pi * y) + 1), (x, 0, 1), (y, 0, 1))
