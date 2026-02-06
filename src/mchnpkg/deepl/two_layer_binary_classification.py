import torch


def binary_classification(
    d: int,
    n: int,
    epochs: int = 10000,
    eta: float = 0.001,
    seed: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    X = torch.randn(n, d, dtype=torch.float32, device=device)
    Y = (X.sum(dim=1, keepdim=True) > 2).float()

    def init_W(n_in, n_out):
        std = 1.0 / (n_in ** 0.5)
        W = torch.randn(n_in, n_out, device=device, dtype=torch.float32) * std
        return torch.nn.Parameter(W)

    W1 = init_W(d, 48)
    W2 = init_W(48, 16)
    W3 = init_W(16, 32)
    W4 = init_W(32, 1)

    params = [W1, W2, W3, W4]

    def bce(yhat, y, eps=1e-7):
        yhat = torch.clamp(yhat, eps, 1 - eps)
        return -(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat)).mean()

    W1_hist = torch.zeros((epochs, d, 48), device="cpu", dtype=torch.float32)
    W2_hist = torch.zeros((epochs, 48, 16), device="cpu", dtype=torch.float32)
    W3_hist = torch.zeros((epochs, 16, 32), device="cpu", dtype=torch.float32)
    W4_hist = torch.zeros((epochs, 32, 1), device="cpu", dtype=torch.float32)

    losses = []

    for i in range(epochs):
        a1 = torch.sigmoid(X @ W1)
        a2 = torch.sigmoid(a1 @ W2)
        a3 = torch.sigmoid(a2 @ W3)
        yhat = torch.sigmoid(a3 @ W4)

        loss = bce(yhat, Y)
        losses.append(loss.item())

        loss.backward()

        with torch.no_grad():
            for p in params:
                p -= eta * p.grad
                p.grad.zero_()

        W1_hist[i] = W1.detach().cpu().clone()
        W2_hist[i] = W2.detach().cpu().clone()
        W3_hist[i] = W3.detach().cpu().clone()
        W4_hist[i] = W4.detach().cpu().clone()

    return (
        W1.detach().cpu(),
        W2.detach().cpu(),
        W3.detach().cpu(),
        W4.detach().cpu(),
        losses,
        W1_hist,
        W2_hist,
        W3_hist,
        W4_hist,
    )
