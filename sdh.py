import torch

from sklearn.metrics.pairwise import rbf_kernel
from loguru import logger
from utils.evaluate import mean_average_precision, pr_curve


def train(
        train_data,
        train_targets,
        query_data,
        query_targets,
        retrieval_data,
        retrieval_targets,
        code_length,
        num_anchor,
        max_iter,
        lamda,
        nu,
        sigma,
        device,
        topk,
        ):
    """
    Training model.

    Args
        train_data(torch.Tensor): Training data.
        train_targets(torch.Tensor): Training targets.
        query_data(torch.Tensor): Query data.
        query_targets(torch.Tensor): Query targets.
        retrieval_data(torch.Tensor): Retrieval data.
        retrieval_targets(torch.Tensor): Retrieval targets.
        code_length(int): Hash code length.
        num_anchor(int): Number of anchors.
        max_iter(int): Number of iterations.
        lamda, nu, sigma(float): Hyper-parameters.
        device(torch.device): GPU or CPU.
        topk(int): Compute mAP using top k retrieval result.

    Returns
        checkpoint(dict): Checkpoint.
    """
    # Initialization
    n = train_data.shape[0]
    L = code_length
    m = num_anchor
    t = max_iter
    X = train_data.t()
    Y = train_targets.t()
    B = torch.randn(L, n).sign()

    # Permute data
    perm_index = torch.randperm(n)
    X = X[:, perm_index]
    Y = Y[:, perm_index]

    # Randomly select num_anchor samples from the training data
    anchor = X[:, :m]

    # Map training data via RBF kernel
    phi_x = torch.from_numpy(rbf_kernel(X.numpy().T, anchor.numpy().T, sigma)).t()

    # Training
    B = B.to(device)
    Y = Y.to(device)
    phi_x = phi_x.to(device)
    for it in range(t):
        # G-Step
        W = torch.pinverse(B @ B.t() + lamda * torch.eye(code_length, device=device)) @ B @ Y.t()

        # F-Step
        P = torch.pinverse(phi_x @ phi_x.t()) @ phi_x @ B.t()
        F_X = P.t() @ phi_x

        # B-Step
        B = solve_dcc(B, W, Y, F_X, nu)

    # Evaluate
    query_code = generate_code(query_data.t(), anchor, P, sigma)
    retrieval_code = generate_code(retrieval_data.t(), anchor, P, sigma)

    # Compute map
    mAP = mean_average_precision(
        query_code.t().to(device),
        retrieval_code.t().to(device),
        query_targets.to(device),
        retrieval_targets.to(device),
        device,
        topk,
    )

    # PR curve
    Precision, R = pr_curve(
        query_code.t().to(device),
        retrieval_code.t().to(device),
        query_targets.to(device),
        retrieval_targets.to(device),
        device,
    )

    # Save checkpoint
    checkpoint = {
        'tB': B,
        'tL': train_targets,
        'qB': query_code,
        'qL': query_targets,
        'rB': retrieval_code,
        'rL': retrieval_targets,
        'anchor': anchor,
        'projection': P,
        'P': Precision,
        'R': R,
        'map': mAP,
    }

    return checkpoint


def solve_dcc(B, W, Y, F_X, nu):
    """Solve DCC(Discrete Cyclic Coordinate Descent) problem
    """
    for i in range(B.shape[0]):
        Q = W @ Y + nu * F_X

        q = Q[i, :]
        v = W[i, :]
        W_prime = torch.cat((W[:i, :], W[i+1:, :]))
        B_prime = torch.cat((B[:i, :], B[i+1:, :]))

        B[i, :] = (q - B_prime.t() @ W_prime @ v).sign()

    return B
   

def generate_code(data, anchor, P, sigma):
    """
    Generate hash code from data using projection matrix.

    Args
        data(torch.Tensor): Data.
        anchor(torch.Tensor): Anchor points.
        P(torch.Tensor): Projection matrix.
        sigma(float): RBF kernel width.

    Returns
        code(torch.Tensor): Hash code.
    """
    phi_x = torch.from_numpy(rbf_kernel(data.cpu().numpy().T, anchor.cpu().numpy().T, sigma)).t().to(P.device)
    return (P.t() @ phi_x).sign()

