from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

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

if __name__ == "__main__":
    render()
