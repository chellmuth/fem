from collections import namedtuple

class Point(namedtuple("Point", ["x", "y"])):
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __truediv__(self, a):
        if a == 0:
            raise Exception("divide by zero")

        return Point(self.x / a, self.y / a)

class Line(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def midpoint(self):
        return (self.p1 + self.p2) / 2

class Quad(object):
    def __init__(self, points):
        self.points = points

    def triangulate(self):
        p1, p2, p3, p4 = self.points
        return [
            Triangle(p1, p2, p3),
            Triangle(p3, p4, p1),
        ]


class Triangle(object):
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def subdivide(self):
        side1 = Line(self.p1, self.p2)
        side2 = Line(self.p2, self.p3)
        side3 = Line(self.p3, self.p1)

        mid1 = side1.midpoint()
        mid2 = side2.midpoint()
        mid3 = side3.midpoint()

        return [
            Triangle(self.p1, mid1, mid3),
            Triangle(self.p2, mid2, mid1),
            Triangle(self.p3, mid3, mid2),
            Triangle(mid1, mid2, mid3),
        ]

    def __repr__(self):
        return repr([
            self.p1,
            self.p2,
            self.p3,
        ])

if __name__ == "__main__":
    quad = Quad([
        Point(0, 0),
        Point(1, 0),
        Point(1, 1),
        Point(0, 1),
    ])

    triangles = quad.triangulate()
    print(triangles)

    triangles = [ sub_t for t in triangles for sub_t in t.subdivide() ]
    print(len(triangles))
    print(triangles)
