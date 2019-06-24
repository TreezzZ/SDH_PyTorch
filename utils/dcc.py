import torch


def solve(B, W, Y, F_X, nu):
    """Solve DCC(Discrete Cyclic Coordinate Descent) problem
    """
    for i in range(B.shape[0]):
        Q = (Y @ W.t() + nu * F_X).t()

        q = Q[i, :]
        v = W[i, :]
        W_prime = torch.cat((W[:i, :], W[i+1:, :]))
        B_prime = torch.cat((B[:i, :], B[i+1:, :]))

        B[i, :] = (q - B_prime.t() @ W_prime @ v).sign()

    return B