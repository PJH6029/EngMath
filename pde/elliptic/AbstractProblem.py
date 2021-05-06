from pde.geometric.Line import Line


# 직접 호출하지 않음
# super class
class AbstractProblem:
    @staticmethod
    def solve(f, x0, y0, h, *boundary):
        # f: Laplace of U = f(x, y), x0, y0 = start coordinate, h: gap
        # boundary: [P, U or U', P, U or U', ..., U or U']

        lines = AbstractProblem.makeLines(boundary)

    @staticmethod
    def makeLines(boundary):
        lines = []

        if len(boundary) % 2 != 0:  # 길이가 짝수여야 함
            raise Exception("경계 조건 인자가 홀수개입니다.")

        for i in range(0, len(boundary) - 2, 2):
            lines.append(Line(boundary[i], boundary[i + 2], boundary[i + 1]))
        lines.append(Line(boundary[-2], boundary[0], boundary[-1]))

        return lines
