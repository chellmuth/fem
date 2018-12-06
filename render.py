import matplotlib.pyplot as plt
import matplotlib.tri as tri

from helper import internal_nodes

def build_triangulation(fem):
    x = []
    y = []

    triangles = []
    for row in range(fem.dim + 2):
        for col in range(fem.dim + 2):
            x.append(col * fem.h)
            y.append(row * fem.h)

    for n in range(2 * (fem.dim + 1) ** 2):
        triangles.append(
            [
                fem.T(1, n),
                fem.T(2, n),
                fem.T(3, n),
            ]
        )

    triangulation = tri.Triangulation(x, y, triangles)
    return triangulation

def build_z(fem, etas):
    z = []

    # f = -8*pi*sin(2*pi*x)*sin(2*pi*y)

    # for row in range(fem.dim + 2):
    #     for col in range(fem.dim + 2):
    #         value = f.subs(
    #             {
    #                 "x": fem.h*col,
    #                 "y": fem.h*row
    #             }
    #         )
    #         z.append(float(value))

    internals = internal_nodes(fem.dim)
    for node in range((fem.dim + 2) ** 2):
        eta = 0
        if node in internals:
            eta = etas[internals.index(node)][0]
        z.append(eta)

    return z

def render(fem, etas):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    triangulation = build_triangulation(fem)
    z = build_z(fem, etas)

    ax.plot_trisurf(
        triangulation,
        z,
        linewidth=0.2, antialiased=True
    )

    plt.show()
