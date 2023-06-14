from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import seaborn as sns

sns.set_theme()


def toDataFrame(t: torch.Tensor, origin: str):
    t = t.cpu().detach().numpy()
    df = pd.DataFrame(data=t, columns=(f"x{ix}" for ix in range(t.shape[1])))
    df['ix'] = df.index * 1.
    df["origin"] = origin
    return df



def scatterplots(samples: List[Tuple[str, torch.Tensor]], col_wrap=4):
    """Draw the 

    Args:
        samples (List[Tuple[str, torch.Tensor]]): The list of samples with their names
        col_wrap (int, optional): Number of columns in the graph. Defaults to 4.

    Raises:
        NotImplementedError: If the dimension of the data is not supported
    """
    # Convert data into pandas dataframes
    _, dim = samples[0][1].shape
    samples = [toDataFrame(sample, name) for name, sample in samples]
    data = pd.concat(samples, ignore_index=True)

    g = sns.FacetGrid(data, height=2, col_wrap=col_wrap, col="origin", sharex=False, sharey=False)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    if dim == 1:
        g.map(sns.kdeplot, "distribution")
        plt.show()
    elif dim == 2:
        g.map(sns.scatterplot, "x0", "x1", alpha=0.6)
        plt.show()
    else:
        raise NotImplementedError()


def iter_data(dataset: Dataset, bs):
    """Infinite iterator on dataset"""
    while True:
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
        yield from iter(loader)


class MLP(nn.Module):
    """Simple 4 layer MLP"""
    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )
    def forward(self, x):
        return self.net(x)


# --- Modules de base

class FlowModule(nn.Module):
    def __init__(self):
        super().__init__()

    def decoder(self, y) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns [f^-1(x)] and log |det J_f^-1(x)|"""
        ...

    def encoder(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns [f(x)] and log |det J_f(x)|"""
        ...

    def check(self, x: torch.Tensor):
        with torch.no_grad():
            (y, ), logdetj_1 = self.encoder(x)
            (hat_x, ), logdetj = self.decoder(y)

            # Check inverse
            delta = (x - hat_x).abs().mean()
            assert  delta < 1e-6, f"f^{{-1}}(f(x)) not equal to x (mean abs. difference = {delta})"

            # Check logdetj
            delta_logdetj = (logdetj_1 + logdetj).abs().mean()
            assert  delta_logdetj < 1e-6, f"log | J | not equal to -log |J^-1| (mean abs. difference = {delta_logdetj})"


class FlowSequential(FlowModule):
    """A container for a succession of flow modules"""
    def __init__(self, *flows: FlowModule):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def apply(self, modules_iter, caller, x):
        m, _ = x.shape
        logdet = torch.zeros(m, device=x.device)
        zs = [x]
        for module in modules_iter:
            gx, _logdet = caller(module, x)
            zs.extend(gx)
            logdet += _logdet

            x = gx[-1]
        return zs, logdet            

    def modulenames(self, decoder=False):
        return [f"L{ix} {module.__class__.__name__}" for ix, module in enumerate(reversed(self.flows) if decoder else self.flows)]

    def encoder(self, x):
        """Returns the sequence (z_K, ..., z_0) and the log det"""
        zs, logdet = self.apply(self.flows, (lambda m, x: m.encoder(x)), x)
        return zs, logdet

    def decoder(self, y):
        """Returns the sequence (z_0, ..., z_K) and the log det"""
        zs, logdet = self.apply(reversed(self.flows), (lambda m, y: m.decoder(y)), y)
        return zs, logdet


class FlowModel(FlowSequential):
    """Flow model = prior + flow modules"""
    def __init__(self, prior: torch.distributions.Distribution, *flows: FlowModule):
        super().__init__(*flows)
        self.prior = prior

    def encoder(self, x):
        # Computes [z_K, ..., z_0] and the sum of log det | f |
        zs, logdet = super().encoder(x)

        # Just computes the prior of $z_0$
        logprob = self.prior.log_prob(zs[-1])

        return logprob, zs, logdet

    def plot(self, data: torch.Tensor, n: int):
        """Plot samples together with ground truth (data)"""
        with torch.no_grad():
            d = data[list(np.random.choice(range(len(data)), n)), :]
            z0 = self.prior.sample((n, ))
            zs, _ = self.decoder(z0)

            data = [("data", d), ("dist", z0)] + list(zip(self.modulenames(decoder=True), zs[1:]))    
            scatterplots(data, col_wrap=4)




class Invertible1x1Conv(FlowModule):
    """ 
    As introduced in Glow paper.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))

        # Decompose Q in P (L + Id) (S + U)

        # https://pytorch.org/docs/stable/generated/torch.lu_unpack.html
        P, L, U = torch.lu_unpack(*Q.lu())

        # Not optimizated
        self.P = nn.Parameter(P, requires_grad=False)

        # Lower triangular
        self.L = nn.Parameter(L)

        # Diagonal
        self.S = nn.Parameter(U.diag())

        self.U = nn.Parameter(torch.triu(U, diagonal=1))

    def _assemble_W(self):
        """Computes W from P, L, S and U"""

        # https://pytorch.org/docs/stable/generated/torch.tril.html
        # Excludes the diagonal
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim, device=self.L.device))

        # https://pytorch.org/docs/stable/generated/torch.triu.html
        # Excludes the diagonal
        U = torch.triu(self.U, diagonal=1)

        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def decoder(self, y) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns [f^-1(x)] and log |det J_f^-1(x)|"""
        ...

    def encoder(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns [f(x)] and log |det J_f(x)|"""
        ...
