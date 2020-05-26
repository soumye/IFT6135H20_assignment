"""
Template for Question 2 of hwk3.
@author: Samuel Lavoie
"""
import torch
import math
import q2_sampler
import q2_model
import numpy as np
from matplotlib import pyplot as plt

def lp_reg(x, y, critic, device='cpu'):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm. This is important to consider,
    because we make the assumption that your implementation follows this notation when testing your function. ***

    :param x: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty
    """
    sampler = iter(q2_sampler.distribution1(0, x.size(0)))
    data = next(sampler)
    t = torch.tensor(data[:, 1], dtype=torch.float32).unsqueeze(1)

    x_hat = t*x + (1-t)*y
    x_hat.requires_grad = True
    fx_hat = critic(x_hat)
    fx_hat.backward(torch.ones_like(fx_hat))

    grads = torch.autograd.grad(fx_hat, x_hat, grad_outputs=torch.ones_like(x_hat), create_graph=True)

    norm_grad = torch.norm(grads, dim=1, p=2)
    zero = torch.Tensor([0.])

    lp = torch.max(zero, norm_grad - 1).pow(2)

    # sampler = iter(q2_sampler.distribution1(0, x.size(0)))
    # data = next(sampler)
    # t = torch.tensor(data[:, 1], requires_grad=True, dtype=torch.float32, device=device).view(-1, 1, 1, 1)

    # x_hat = t*x + (1-t)*y
    # fx_hat = critic(x_hat)

    # grads = torch.autograd.grad(fx_hat, x_hat, grad_outputs=torch.ones_like(fx_hat), create_graph=True)[0]

    # norm_grad = torch.norm(grads, dim=1, p=2)
    # zero = torch.Tensor([0.]).to(device)

    # lp = torch.max(zero, norm_grad - 1).pow(2)

    return lp.mean()


def vf_wasserstein_distance(p, q, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    fp = critic(p)
    fq = critic(q)
    obj = fp.mean() - fq.mean()

    return torch.unsqueeze(obj, dim=0)



def vf_squared_hellinger(x, y, critic):
    """
    Complete me. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Nowazin et al: https://arxiv.org/pdf/1606.00709.pdf
    In other word, x are samples from the distribution P and y are samples from the distribution Q. The critic is the
    equivalent of T in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x 2) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Peason Chi square.
    :return: (FloatTensor) - shape: (1,) - Estimate of the Squared Hellinger
    """ 
    gf_p = 1. - torch.exp(- critic(x))
    gf_q = 1. - torch.exp(- critic(y))

    conj_q = gf_q / (1 - gf_q)
    obj = gf_p.mean() - conj_q.mean()

    return torch.unsqueeze(obj, dim=0)


def Hellinger(theta):
    # Example of usage of the code provided for answering Q2.5 as well as recommended hyper parameters.
    model = q2_model.Critic(2)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    sampler1 = iter(q2_sampler.distribution1(0, 512))
    theta = 0
    sampler2 = iter(q2_sampler.distribution1(theta, 512))
    lambda_reg_lp = 50 # Recommended hyper parameters for the lipschitz regularizer.
    steps = 150
    for step in range(steps):
        data1 = torch.from_numpy(next(sampler1)).float()
        data2 = torch.from_numpy(next(sampler2)).float()
        loss = -vf_squared_hellinger(data1, data2, model)
        print('Step {} : loss {}'.format(step, loss))
        optim.zero_grad()
        loss.backward()
        optim.step()
    data1 = torch.from_numpy(next(sampler1)).float()
    data2 = torch.from_numpy(next(sampler2)).float()
    return vf_squared_hellinger(data1, data2, model)

def Wasserstein(theta):
    # Example of usage of the code provided for answering Q2.5 as well as recommended hyper parameters.
    model = q2_model.Critic(2)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    sampler1 = iter(q2_sampler.distribution1(0, 512))
    theta = 0
    sampler2 = iter(q2_sampler.distribution1(theta, 512))
    lambda_reg_lp = 50 # Recommended hyper parameters for the lipschitz regularizer.
    steps = 150
    for step in range(steps):
        data1 = torch.from_numpy(next(sampler1)).float()
        data2 = torch.from_numpy(next(sampler2)).float()
        loss = -vf_wasserstein_distance(data1, data2, model) + lambda_reg_lp*lp_reg(data1, data2, model)
        print('Step {} : loss {}'.format(step, loss))
        optim.zero_grad()
        loss.backward()
        optim.step()
    data1 = torch.from_numpy(next(sampler1)).float()
    data2 = torch.from_numpy(next(sampler2)).float()
    return vf_wasserstein_distance(data1, data2, model)    

if __name__ == '__main__':
    # X = np.arange(0,2,0.1)
    # Y = [Hellinger(x) for x in X]
    # plt.plot(X,Y)
    # plt.title('Square Hellinger Distance')
    # plt.xlabel('theta')
    # plt.ylabel('Distance')
    # plt.show()
    
    X = np.arange(0,2,0.1)
    Y = [Wasserstein(x) for x in X]
    plt.plot(X,Y)
    plt.title('Wasserstein Distance')
    plt.xlabel('theta')
    plt.ylabel('Distance')
    plt.show()
