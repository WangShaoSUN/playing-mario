import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.distributions.normal import Normal
import numpy as np
from math import pi
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def eig_stable(A, tol=1e-10, out=None):
    # L, V = torch.eig(A, eigenvectors=True)
    # L=L[:,0].view(-1)

    L, V = torch.symeig(A, eigenvectors=True)
    keep = L > tol
    # print(out)
    # print("A",A)
    # print(keep)
    return (L[keep], V[:, keep])


def sample_MVG2(M, U, V):
    Lu, Pu = eig_stable(U, out="U")
    Lv, Pv = eig_stable(V, out="V")

    E = torch.randn((Lu.shape[0], Lv.shape[0])).to(device)

    # print(torch.diag(torch.sqrt(Lv)))
    # print(Pv.transpose(0,1))

    return M + Pu @ torch.diag(torch.sqrt(Lu)) @ E @ torch.diag(torch.sqrt(Lv)) @ Pv.transpose(0, 1)


def sample_MVG(M, U, v):
    # U is a matrix, v is the diagonal of V
    # M: rxc
    # U: rxr
    # V: cxc
    try:
      A = torch.cholesky(U + 1e-3 * torch.diag(torch.ones(U.shape[0]).to(device)))
    except:
      A =torch.eye(U.shape[0],U.shape[1]).to(device)
    # A = torch.cholesky(U)
    B = torch.diag(v.sqrt())
    E = torch.randn(M.shape).to(device)

    return M + torch.mm(torch.mm(A, E), B)

class NoisyLayer_with_MVG(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(NoisyLayer_with_MVG,self).__init__()

        self.dim_in = dim_in + 1
        self.dim_out = dim_out

        self.weight = nn.Parameter(torch.Tensor(self.dim_out, self.dim_in))
        self.W_in_logvar = nn.Parameter(torch.Tensor(self.dim_in))  # U
        self.W_out_logvar = nn.Parameter(torch.Tensor(self.dim_out))  # V

        self.init_parameters()

    def init_parameters(self):
        # init means
        self.weight.data.normal_(0, .05)

        # init variances
        self.W_in_logvar.data.normal_(-5, .05)
        self.W_out_logvar.data.normal_(-5, .05)

    def forward(self, x, sample=True):
        # print(x.device)
        # local reparameterization trick
        # x=x.cuda()
        x = torch.cat((torch.ones(x.shape[0], 1).to(device), x), dim=1) # Add column of ones since no bias
        # x =x.to(device)
        mu_activations = F.linear(x, self.weight)
        if not sample:
            active = mu_activations
        else:
            u = self.W_in_logvar.exp()  # Diagonal of U
            v = self.W_out_logvar.exp()  # Diagonal of V

            var_in_activations = torch.mm(torch.mm(x, torch.diag(u)), x.transpose(0, 1))
            var_out_activations = v
            active = sample_MVG(mu_activations, var_in_activations, var_out_activations)
            # active = sample_MVG(mu_activations, var_in_activations, torch.diag(var_out_activations))
        return active
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    # bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)