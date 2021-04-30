import numpy as np
np.set_printoptions(precision=12)

class Iteration:
    @staticmethod
    def GaussSeidelMethod(coeff_mat, b_vec,
                          x_vec=None, epoch=100):
        # mat * x_vec = b_vec, mat: nxn
        # 식: 위키피디아
        n = len(b_vec)
        x_vec_res = np.array([0 for _ in range(n)])
        if x_vec:
            x_vec_res = x_vec

        L_star = np.tril(coeff_mat)
        L_star_inv = np.linalg.inv(L_star)
        U = np.triu(coeff_mat, 1)

        for i in range(epoch):
            x_vec_res = np.matmul(L_star_inv, b_vec - np.matmul(U, x_vec_res))

        return x_vec_res

    @staticmethod
    def JacobiMethod(coeff_mat, b_vec, x_vec=None, epoch=100):
        # mat * x_vec = b_vec, mat: nxn
        # 식: 위키피디아
        n = len(b_vec)
        x_vec_res = np.array([0 for _ in range(n)])
        if x_vec:
            x_vec_res = np.array([0 for _ in range(n)])

        D = np.diag(np.diag(coeff_mat))
        D_inv = np.linalg.inv(D)
        L_U = coeff_mat - D

        for i in range(epoch):
            x_vec_res = np.matmul(D_inv, b_vec - np.matmul(L_U, x_vec_res))

        return x_vec_res