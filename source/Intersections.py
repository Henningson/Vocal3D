import numpy as np

def rayPlane(ray, plane):
    denom = np.sum(plane.normal() * ray.direction(), axis=1)

    denom = np.where(np.abs(denom) < 0.000001, np.nan, denom)
    t = np.sum((plane.origin() - ray.origin()) * plane.normal(), axis=1) / denom

    return np.expand_dims(t, -1)

def lineLine(line1, line2, epsilon=1e-4):
    p1 = line1._p1
    p2 = line1._p2
    p3 = line2._p1
    p4 = line2._p2

    if len(p1.shape) != 2:
        p1 = np.expand_dims(p1, 0)
        p2 = np.expand_dims(p2, 0)
        p3 = np.expand_dims(p3, 0)
        p4 = np.expand_dims(p4, 0)

    p13 = p1 - p3
    p43 = p4 - p3
    p21 = p2 - p1

    d1343 = np.sum(p13 * p43, axis=1)
    d4321 = np.sum(p43 * p21, axis=1)
    d1321 = np.sum(p13 * p21, axis=1)
    d4343 = np.sum(p43 * p43, axis=1)
    d2121 = np.sum(p21 * p21, axis=1)

    denom = d2121 * d4343 - d4321 * d4321    
    numer = d1343 * d4321 - d1321 * d4343
    mua = numer / denom
    mub = (d1343 + d4321 * mua) / d4343

    pa = p1 + np.expand_dims(mua, -1) * p21
    pb = p3 + np.expand_dims(mub, -1) * p43
    return pa, pb, np.linalg.norm(pb - pa, axis=1)


def pointLineSegmentDistance(linePointA, linePointB, point):
    l2 = np.sum((linePointB - linePointA) ** 2)

    if l2 == 0.0:
        return np.linalg.norm(point - linePointA)

    t = max(0, min(1, np.dot(point - linePointA, linePointB - linePointA) / l2))
    projection = linePointA + t * (linePointB - linePointA)
    return np.linalg.norm(point - projection)