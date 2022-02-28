from torch import nn
import torch
from torch.functional import F
import numpy as np

###################### OPTIMIZATION FUNCTION #########################
class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, numPixels, numSpots):
        super().__init__()
        # initialize weights with random numbers
        vis = torch.rand(numPixels, numSpots)
        vis = torch.where(vis > 0.5, 1.0, 0.0)
        
        # make weights torch parameters
        self.vis = nn.Parameter(vis, requires_grad = True)      
        
    def forward(self, hists):
        """Implement function to be optimised. In this case, an exponential decay
        function (a + exp(-k * X) + b),
        """
#         vis = self.vis
#         self.a, self.b, self.c = torch.tensor(0., requires_grad=True), 
#         torch.tensor(1., requires_grad=True), torch.tensor(0.5, requires_grad=True)
#         if torch.min(self.vis) != torch.max(self.vis):
        if False:
            vis = (self.vis - torch.min(self.vis)) / (torch.max(self.vis) - torch.min(self.vis))
        if True: 
            vis = torch.sigmoid(self.vis)
        # TRY 
            # sigmoid 
            # gumbel-softmax
            # e.g.  spectral clustering (integer optimization w/ real number relaxation)
            # if p^Tp is well conditioned, problem will become least squares
            # linear regression w/ l1 constraint (lasso)
            # elastic net (l1 / l2 regularizer w/ tradeoff)
            # integer programming / optimization (but it's hard)
            # it can be a graph cut problem
        if False:
            relu = MyReLU.apply
            vis = relu(vis)
#         self.debug_v = torch.where(self.vis > 0.5, 1.0, 0.0)
#         print(self.debug_v.requires_grad)
#         self.debug_v = torch.where(self.vis > self.c, self.a, self.b)
#         self.debug_v.requires_grad = True
#         self.debug_v = self.vis
#         self.debug_v.retain_grad()
        obs = torch.sum(hists * vis.unsqueeze(-1), axis=1)
        return obs

def training_loop(model, histograms, observations, optimizer, thresh, lam, n=1000):
    "Training loop for torch model."
    losses = []
    prev_loss = 0
    for i in range(n):
        preds = model(histograms)
        loss1 = F.mse_loss(preds, observations) 
        loss2 = torch.sum(torch.square(torch.diff(model.vis, n=1, axis=0))) 
        loss = loss1 + lam * loss2
        cur_loss = loss.detach().numpy()
        losses.append(cur_loss)  
        if np.abs(prev_loss-cur_loss) < thresh:
            break
        loss.backward()
#         print(model.debug_v.grad.data)
        optimizer.step()
        optimizer.zero_grad()
    return losses

# def bruteForce(v_initial, hists, observations):
#     numPixels, numSources, _ = hists.shape
#     v_optim = v_initial
#     for i in range(1):
#         # compute optimal parameters
#         obs_optim = np.sum(np.expand_dims(v_optim, 2) * hists, 1)
#         loss_optim = np.sum(np.abs(obs_optim - observations), 1)

#         # flip bits for pixel
#         v_test = v_optim
#         v_test[:, i] = 1-v_optim[:, i]
        
#         # see if flipping bits improves loss for all pixels
#         obs_test = np.sum(np.expand_dims(v_optim, 2) * hists, 1)
#         loss_test = np.sum(np.abs(obs_test - observations), 1)
#         flip_criteria = loss_test < loss_optim
#         print(flip_criteria)
        
#         # compute new optimal parameters
#         print(v_optim[:, i])
#         v_optim[:, i] = np.abs(v_optim[:, i] - flip_criteria)
#         print(v_optim[:, i])
        
#         flippedBits = np.sum(flip_criteria)
#         if flippedBits != 0:
#             print(str(i) + ': ' + str(flippedBits) + ' bits flipped')    
#     return v_optim

def bruteForce(v_initial, hists, observations):
    numPixels, numSources, _ = hists.shape
    v_optim = np.copy(v_initial)
    for j in range(3):
        for i in range(numSources):
            # compute optimal parameters
            obs_optim = np.sum(np.expand_dims(v_optim, 2) * hists, 1)
            loss_optim = np.sum(np.abs(obs_optim - observations), 1)

            # flip bits for pixel
            v_test = np.copy(v_optim)
            v_test[:, i] = 1-v_test[:, i]

            # see if flipping bits improves loss for all pixels
            obs_test = np.sum(np.expand_dims(v_test, 2) * hists, 1)
            loss_test = np.sum(np.abs(obs_test - observations), 1)
            flip_criteria = (loss_test < loss_optim).astype(int)

            # compute new optimal parameters
            v_optim[:, i] = np.abs(v_optim[:, i] - flip_criteria)
            loss_optim = np.copy(loss_test)

            flippedBits = np.sum(flip_criteria)
#             if flippedBits != 0:
#                 print(str(i) + ': ' + str(flippedBits) + ' bits flipped')    
    return v_optim

class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
#         return input.clamp(min=0.0)
        return torch.where(input > 0.5, 1.0, 0.0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        o = 0.49
        a = 0.5 - o; b = 0.5 + o
        m = 1/(b-a)
        input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input < 0] = 0.0
#         print(type(torch.where((input < a or input > b), 0.0, m)))
        return grad_output * torch.where((input < a) & (input > b), 0.0, m)