
import math
import numpy as np

def deg2rad(num):
    return num / 180 * math.pi


def rotateLine(vec,  deg):
    rad = deg2rad(deg)
    return np.array([np.cos(rad) * vec[0] - np.sin(rad) * vec[1], np.sin(rad) * vec[0] + np.cos(rad) * vec[1]])


def rad2deg(rad):
	return rad * 180.0 / math.pi


class M52D:
    def __init__(self, rZero, T, psi, xLength, isLeft=True):
        self.rZero = rZero
        self.T = T
        self.psi = deg2rad(psi)
        self.xLength = xLength
        self.subdivisions = 3

        self.rPsi = self.rZero / (1.0 - np.sin(self.psi / 2.0))
        self.rL = self.T / 2.0
        self.r40 = self.T / 2.0
        self.b = np.sqrt(2.0) * self.rPsi / np.sqrt(1.0 + np.sin(self.psi / 2.0))
        self.Q1 = (self.T - self.rPsi) * (1.0 / np.cos(self.psi / 2.0)) + (self.rPsi - self.rL) * np.tan(self.psi / 2.0)
        self.Q2 = self.rL * np.sin(self.psi / 2.0)
        self.Q3 = self.Q1 * np.cos(self.psi / 2.0)
        self.Q4 = self.rZero
        self.Q5 = self.rL * np.sin(50)

        self.isLeft = isLeft

        self.generate()

    def subdivideSemicircle(self, center, direction, angle, subdivisions):
        pos = list()
        
        for i in range(subdivisions):
            pos.append(center + rotateLine(direction, (angle / subdivisions) * i))
        
        return pos

    def subdivideLine(self, a, b,  subdivisions):
        pos = list()
        
        direction = b - a
        length = np.sqrt(direction[0] * direction[0] + direction[1] * direction[1])
        direction = direction / length

        for i in range(subdivisions):
            pos.append(a + direction * ((length / subdivisions) * i))

        return pos

    def generate(self):
        start = np.array([0, 0])
        startEnd = np.array([self.xLength - self.rPsi, 0])

        pos1 = self.subdivideLine(start, startEnd, self.subdivisions)

        startFirstSemicircle = startEnd
        firstSemicircleOrigin = startEnd + np.array([0.0, self.rPsi])
        endFirstSemiCircle = firstSemicircleOrigin + rotateLine(np.array([0.0, -self.rPsi]), 90 - rad2deg(self.psi) / 2)
        pos2 = self.subdivideSemicircle(firstSemicircleOrigin, np.array([0.0, -self.rPsi]), 90 - rad2deg(self.psi) / 2, self.subdivisions - 1)

        startQ1 = endFirstSemiCircle
        endQ1 = startQ1 + rotateLine(np.array([self.Q1, 0.0]), 90 - rad2deg(self.psi) / 2)
        pos3 = self.subdivideLine(startQ1, endQ1, self.subdivisions - 1)

        directionToSecondCircleOrigin = rotateLine(np.array([self.rL, 0.0]), 180 - rad2deg(self.psi) / 2)
        secondSemiCircleOrigin = endQ1 + directionToSecondCircleOrigin
        secondSemiCircleEnd = secondSemiCircleOrigin + rotateLine(-directionToSecondCircleOrigin, 50 + (rad2deg(self.psi) / 2))
        pos4 = self.subdivideSemicircle(secondSemiCircleOrigin, -directionToSecondCircleOrigin, 50 + (rad2deg(self.psi) / 2), self.subdivisions - 1)

        lengthLastLine = (secondSemiCircleEnd[0] - start[0]) / np.cos(deg2rad(40.0))
        end = secondSemiCircleEnd + rotateLine(np.array([lengthLastLine, 0]), 180 - 40)
        pos5 = self.subdivideLine(secondSemiCircleEnd, end, self.subdivisions)
        pos6 = self.subdivideLine(end, start, self.subdivisions)

        self.vertices = pos1 + pos2 + pos3 + pos4 + pos5 + pos6

        maxX = 0
        if self.isLeft:
            for vertex in self.vertices:
                maxX = maxX if vertex[0] < maxX else vertex[0]
            for i in range(len(self.vertices)):
                self.vertices[i][0] -= maxX
        else:
            for vertex in self.vertices:
                maxX = maxX if vertex[0] < maxX else vertex[0]
            for i in range(len(self.vertices)):
                self.vertices[i][0] = -self.vertices[i][0] + maxX

        for i in range(len(self.vertices)):
            self.vertices[i][1] = -self.vertices[i][1]

        return self.vertices

    def translate(self, vec):        
        for i in range(len(self.vertices)):
            self.vertices[i] += vec

    def getVertices(self):
        return self.vertices