from utils.utils import Utils


class Euler:
    @staticmethod
    def eulerMethod(f, xn, yn, h):
        return yn + h * f(xn, yn)

    @staticmethod
    def improvedEulerMethod(f, xn, yn, h):
        k1 = h * f(xn, yn)
        k2 = h * f(xn + h, yn + k1)
        y_next = yn + (1 / 2) * (k1 + k2)
        return y_next

    @staticmethod
    def eulerMethodForSystem(scalarF, xn, yn_vec, h):  # scalar F
        F_vec = Utils.get_F_from_ODE(scalarF)
        y_next_vec = yn_vec + h * F_vec(xn, *yn_vec)
        return y_next_vec

    @staticmethod
    def improvedEulerMethodForSystem(scalarF, xn, yn_vec, h):
        F_vec = Utils.get_F_from_ODE(scalarF)
        k1_vec = h * F_vec(xn, *yn_vec)
        k2_vec = h * F_vec(xn + h, *(yn_vec + k1_vec))
        y_next_vec = yn_vec + (1 / 2) * (k1_vec + k2_vec)
        return y_next_vec
