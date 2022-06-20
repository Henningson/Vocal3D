import numpy as np

class Ray:
    def __init__(self, origin, direction):
        self._origin = origin
        self._direction = direction

    def origin(self):
        return self._origin

    def direction(self):
        return self._direction

    def __add__(self, other):
        return Ray(self._origin, self._direction + other._direction)

    def __sub__(self, other):
        return Ray(self._origin, self._direction - other._direction)

    def __mul__(self, other):
        return Ray(self._origin, self._direction * other)

    def shape(self):
        return self._direction.shape

    def __len__(self):
        return self._direction.size


class Plane:
    def __init__(self, normal, origin):
        self._normal = normal
        self._origin = origin

    def normal(self):
        return self._normal

    def origin(self):
        return self._origin

    def rayIntersection(self, ray):
        denom = np.sum(self._normal * ray._direction, axis=1)

        denom = np.where(np.abs(denom) < 0.000001, np.nan, denom)
        t = np.sum((self._origin - ray._origin) * self._normal, axis=1) / denom

        return np.expand_dims(t, -1)


class Line:
    def __init__(self, p1, p2):
        self._p1 = p1
        self._p2 = p2

    def p1(self):
        return self._p1

    def p2(self):
        return self._p2