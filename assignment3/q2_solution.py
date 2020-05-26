"""
Template for Question 2 of hwk3.
@author: Samuel Lavoie
"""
import torch
import q2_sampler
import q2_model
from torch.distributions.uniform import Uniform
import numpy as np
from matplotlib import pyplot as plt


def lp_reg(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm. This is important to consider,
    because we make the assumption that your implementation follows this notation when testing your function. ***

    :param x: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty
    """
    t = Uniform(0.0,1.0).rsample((x.shape[0],))
    x_cap = torch.einsum('bf,b->bf', x, t) + torch.einsum('bf,b->bf', y, 1-t)
    x_cap = torch.autograd.Variable(x_cap,requires_grad=True)
    grad = torch.autograd.grad(outputs=critic(x_cap), inputs=x_cap, grad_outputs=torch.ones(x.shape[0]), create_graph=True, retain_graph=True)
    return (torch.relu(torch.norm(grad[0],dim=1) - 1.0)**2).mean()


def vf_wasserstein_distance(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    return (critic(x) - critic(y)).mean()

def g_helinger(v):
    return 1.0 - torch.exp(-v)

def fenchel_helinger(t):
    return t/(1.0-t)
    
def vf_squared_hellinger(x, y, critic):
    """
    Complete me. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Nowazin et al: https://arxiv.org/pdf/1606.00709.pdf
    In other word, x are samples from the distribution P and y are samples from the distribution Q. Please note that the Critic is unbounded. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Squared Hellinger.
    :return: (FloatTensor) - shape: (1,) - Estimate of the Squared Hellinger
    """
    return (g_helinger(critic(x)) - fenchel_helinger(g_helinger(critic(y)))).mean()

def Hellinger(theta):
    # Example of usage of the code provided for answering Q2.5 as well as recommended hyper parameters.
    model = q2_model.Critic(2)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    sampler1 = iter(q2_sampler.distribution1(0, 512))
    sampler2 = iter(q2_sampler.distribution1(theta, 512))
    lambda_reg_lp = 50 # Recommended hyper parameters for the lipschitz regularizer.
    steps = 500
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
    sampler2 = iter(q2_sampler.distribution1(theta, 512))
    lambda_reg_lp = 50 # Recommended hyper parameters for the lipschitz regularizer.
    steps = 50
    for step in range(steps):
        data1 = torch.from_numpy(next(sampler1)).float()
        data2 = torch.from_numpy(next(sampler2)).float()
        loss = vf_wasserstein_distance(data1, data2, model) + lambda_reg_lp*lp_reg(data1, data2, model)
        print('Step {} : loss {}'.format(step, loss))
        optim.zero_grad()
        loss.backward()
        optim.step()
    data1 = torch.from_numpy(next(sampler1)).float()
    data2 = torch.from_numpy(next(sampler2)).float()
    return vf_wasserstein_distance(data1, data2, model)    

if __name__ == '__main__':
    torch.manual_seed(911)
    np.random.seed(10)
    


    X = np.arange(0,2,0.1)
    Y = [Wasserstein(x) for x in X]
    plt.plot(X,Y)
    plt.title('Wasserstein Distance')
    plt.xlabel('theta')
    plt.ylabel('Distance')
    plt.show()
