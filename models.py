import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.args.args import argchoice, argignore
import numpy as np
import pdb
import utils
from torchvision.utils import save_image, make_grid
from torchvision import transforms

from modules import ReversibleFlow, Network, AffineFlowStep, Unitary

# from hessian import jacobian

class GlowPrediction(nn.Module):
  def __init__(self, dataset, flow:argchoice=[ReversibleFlow]):
    self.__dict__.update(locals())
    super(GlowPrediction, self).__init__()
    example = torch.stack([dataset[i][0][0] for i in range(1)])
    self.flow = flow(examples=example)


  def forward(self, step, x, y):
    xflat = x.reshape(-1, x.size(2), x.size(3), x.size(4))
    return self.flow(step, xflat)

  def logger(self, step, data, out):
    if step % 100 == 0:
      x, _ = data
      zcat, zlist, logdet = out
      temperature = 0.7
      xhat, _ = self.flow.decode([temperature * torch.randn_like(z) for z in zlist])
      return {':xsample': self.dataset.denormalize(xhat), # generated samples
              ':xtruth': self.dataset.denormalize(x[:, 0]), # true samples, 15 long (10 past + 5 future)
              ':temperature': temperature} # smaller means more realistic samples


class StableSVD(nn.Module):
  def __init__(self, dimension, fast=True):
    super(StableSVD, self).__init__()
    self.U = Unitary(dim=dimension, fast=fast)
    self.V = Unitary(dim=dimension, fast=fast)
    self.alpha = nn.Parameter(torch.Tensor(dimension))
    self.reset_parameters()

  def reset_parameters(self):
    torch.nn.init.constant_(self.alpha, 0.001)

  def get(self):
    U = self.U.get()
    V = self.V.get()
    S = 1 - torch.clamp(torch.abs(self.alpha), min=0, max=1)
    A = U.matmul(S[:, None] * V)
    return A

class StableRealJordanForm(nn.Module):
  def __init__(self, dimension):
    super(StableRealJordanForm, self).__init__()
    assert dimension % 2 == 0, 'dimension {} must be even'.format(dimension)
    self.eig_alpha = nn.Parameter(torch.Tensor(dimension // 2).double())
    self.eig_beta = nn.Parameter(torch.Tensor(dimension // 2).double())
    self.reset_parameters()

  def reset_parameters(self):
    torch.nn.init.constant_(self.eig_alpha, 0.5)
    torch.nn.init.constant_(self.eig_beta, 0.5)

  def get(self):
    # To avoid sqrt(0) when alpha = 1, we use doubles for eig_alpha and eig_beta, and a very small
    # epsilon, it is cast back to a 32-bit float when the assignment happens to build the A matrix
    alpha = torch.clamp((1 - 1e-14) - torch.abs(self.eig_alpha), min=0)
    beta = torch.clamp(1 - torch.abs(self.eig_beta), min=0) * torch.sqrt(1 - alpha**2)

    with torch.cuda.device(alpha.device):
      A = torch.cuda.FloatTensor(self.eig_alpha.size(0) * 2, self.eig_alpha.size(0) * 2)

    A.fill_(0)
    A.view(-1)[::A.size(1) * 2 + 2] = alpha
    A.view(-1)[A.size(1) + 1::A.size(1) * 2 + 2] = alpha
    A.view(-1)[1::A.size(1) * 2 + 2] = beta
    A.view(-1)[A.size(1)::A.size(1) * 2 + 2] = -beta

    return A

class Unconstrained(nn.Module):
  def __init__(self, dimension):
    super(Unconstrained, self).__init__()
    self.A = nn.Parameter(torch.Tensor(dimension, dimension))
    self.reset_parameters()

  def reset_parameters(self):
    torch.nn.init.xavier_uniform_(self.A)
    u, s, v = torch.svd(self.A)
    self.A.data = u.matmul(v.t())


  def get(self):
    return self.A

class FramePredictionBase(nn.Module):
  def __init__(self, dataset, extra_hidden_dim=8, l2_regularization=.001,
               future_sequence_length=5,
               max_hidden_dim=128,
               log_all=True,
               prediction_alpha=1,
               #true_hidden_dim=32,
               network:argchoice=[AffineFlowStep],
               A:argchoice=[StableRealJordanForm, StableSVD, Unconstrained],
               flow:argchoice=[ReversibleFlow], inner_minimization_as_constant=True):
    self.__dict__.update(locals())
    super(FramePredictionBase, self).__init__()

    example = torch.stack([dataset[i][0][0] for i in range(1)])

    self.flow = flow(examples=example)
    self.observation_dim = torch.numel(example[0])
    self.example = example[0].unsqueeze(0)

    # round up to the nearest power of two, makes multi_matmul significantly faster
    self.hidden_dim = min(self.observation_dim + extra_hidden_dim, max_hidden_dim)

    self.C = nn.Parameter(torch.Tensor(self.observation_dim, self.hidden_dim))
    self.A = A(self.hidden_dim)

    self.reset_parameters()

  def __repr__(self):
    summary =  '                Name : {}\n'.format(self.__class__.__name__)
    summary += '    Hidden Dimension : {}\n'.format(self.hidden_dim)
    summary += 'Obervation Dimension : {}\n'.format(self.observation_dim)
    summary += '              A type : {}\n'.format(self.A.__class__.__name__)
    summary += '         A Dimension : {0}x{0} = {1}\n'.format(self.hidden_dim, self.hidden_dim**2)
    # summary += '         C Dimension : {0}x{1} = {2}\n'.format(self.observation_dim,
    #                                                            self.hidden_dim,
    #                                                            self.C.numel())
    return summary

  def reset_parameters(self):
    # Initialize a full-rank C
    torch.nn.init.xavier_uniform_(self.C)
    u, s, v = torch.svd(self.C)
    self.C.data = u.matmul(v.t())

    self.A.reset_parameters()

  @utils.profile
  def forward(self, step, x, y):
    # fold sequence into batch
    xflat = x.reshape(-1, x.size(2), x.size(3), x.size(4))
    M = float(np.prod(xflat.shape[1:]))

    # encode each frame of each sequence in parallel
    _, (zcat, zlist, logdet) = self.flow(step, xflat, loss=False)

    # unfold batch back into sequences
    # y is (batch, sequence, observation_dim)
    y = zcat.reshape(x.size(0), x.size(1), -1)#.detach()

    # stack sequence into tall vector
    # Y is tall (batch, sequence * observation_dim) with stacked y's
    Y = y.reshape(y.size(0), -1)

    # the scaling factor should not be centered (i.e not std or var, which are computed using centered values)
    scaling_lambda = y.abs().mean()

    # Find x0*
    A, C = self.A.get(), self.C

    term = C.t().matmul(C)
    MtM = term
    #b = y.matmul(C)

    # O = [C, C@A, C@C@A, C^3 @ A, ...] (the observation matrices over time)
    O = [C]
    for i in range(x.size(1) - 1):
      O.append(O[-1].matmul(A))
      MtM += O[-1].t().matmul(O[-1])
      #b[:, i+1:] = b[:, i+1:].matmul(A)

    O = torch.cat(O, dim=0) # we can avoid constructing O if we need to
    U = torch.cholesky(MtM, upper=False)
    rhs = Y.matmul(O) #b.sum(1)
    z = torch.trtrs(rhs.t(), U, transpose=False, upper=False)[0]
    x0 = torch.trtrs(z, U, transpose=True, upper=False)[0]

    # Dynamic linear system evolution
    # x is the state, y is the observation
    xtemp = x0
    yhat = [C.matmul(xtemp)]
    for i in range(x.size(1) - 1):
      xtemp = A.matmul(xtemp)
      yhat.append(C.matmul(xtemp))

    # Yhat is the frame prediction, what we actually care about
    Yhat = torch.cat(yhat, dim=0)

    # Why are we comparing Y.T and Yhat? Aren't they the same? Where's Y_truth?
    # this accounts for the scaling factor, but learns an unscaled dynamic system
    w = (Y.t() - Yhat) / scaling_lambda

    prediction_error = (w * w).mean()

    log_likelihood = (logdet / M).mean() - torch.log(scaling_lambda)
    loss = -log_likelihood + self.prediction_alpha * prediction_error #+ reconstruction_loss

    if np.isnan(loss.detach().cpu().numpy()):
      print('loss is nan')
      import pdb
      pdb.set_trace()

    return loss, (loss, w, Y, Yhat, y, A, x0, logdet / M, prediction_error, zlist, zcat, O, C, scaling_lambda)

  def decode(self, Y, sequence_length, zlist):
    # heaven help me if I need to change how this indexing works
    # this is awkward, but we are converting shapes so that we go from
    # Yhat to Y to y to zcat to zlist then decoding and reshaping to compare with x
    s = sequence_length
    n = Y.size(0)

    zcat_hat = Y.reshape(-1, self.observation_dim)
    # print([m[:n * s].view(n * s, -1).size(1) for m in zlist])
    indexes = np.cumsum([0] + [np.prod(m.shape[1:]) for m in zlist])
    zlist_hat = [zcat_hat[:, a:b].reshape(c[:n * s].size()) for a, b, c in zip(indexes[:-1], indexes[1:], zlist)]
    xhat, _ = self.flow.decode(zlist_hat)

    return xhat

  @utils.profile
  def logger(self, step, data, out, force_log=False):
    self.eval()

    x, x_future = data
    b, s, c, h, w = x.shape
    loss, w, Y, Yhat, y, A, x0, logdet, prediction_error, zlist, zcat, O, C, scaling_lambda = out
    stats = {':logdet': logdet.mean(),
             ':w_mean': w.mean(),
             ':w_std': w.std(),
             ':scaling_lambda': scaling_lambda,
             ':normed_prederr': ((w * w).reshape(y.size()) / y.norm(dim=2, keepdim=True)).mean()
             #':Atheta': torch.arccos(.5 * (torch.trace(A) - 1)) # still untested: https://math.stackexchange.com/questions/261617/find-the-rotation-axis-and-angle-of-a-matrix
             }

    stats.update({':tmem_alloc': torch.cuda.memory_allocated() * 1e-9,
                  ':tmaxmem_alloc': torch.cuda.max_memory_allocated() * 1e-9,
                  ':tmem_cache': torch.cuda.memory_cached() * 1e-9,
                  ':tmaxmem_cache': torch.cuda.max_memory_cached() * 1e-9})

    errnorm = w.t().reshape(y.size()).norm(dim=2)

    stats.update({'pe:prediction_error': prediction_error,
                  ':y_mean': y.mean(),
                  ':y_max': y.max(),
                  ':y_min': y.min(),
                  ':y_var': y.var(),
                  ':y_mean_norm': y.norm(dim=2).mean(),
                  # ':log10_first_errnorm': torch.log(errnorm[:, 0].mean()),
                  # ':log10_last_errnorm': torch.log(errnorm[:, -1].mean())
                  })

    if step % 100 == 0 or force_log:
      xhat = self.decode(Yhat.t()[:1], s, zlist)

      #render out into the future
      state = x0[:, 0:1]
      for i in range(s):
        state = A.matmul(state)
      yfuture = [C.matmul(state)]
      future_sequence_length = self.future_sequence_length
      for i in range(future_sequence_length - 1):
        state = A.matmul(state)
        yfuture.append(C.matmul(state))
      yfuture = torch.cat(yfuture, dim=0).t()
      xfuture = self.decode(yfuture, future_sequence_length, zlist)
      yfuture = yfuture.reshape(1, future_sequence_length, *x[0:1].shape[2:])

      # prepare the ys for logging
      recon_error = self.dataset.denormalize(x[0] - xhat)
      ytruth =      Y[0:1].reshape(y[0:1].size()).reshape(x[0].size())
      yhat = Yhat.t()[0:1].reshape(y[0:1].size()).reshape(x[0].size())

      xerror = recon_error.abs().max(1)[0].detach()
      xerror = xerror.reshape(xerror.size(0), -1)
      xerror = xerror.clamp(min=0, max=1)
      xerror = xerror.reshape(recon_error.size(0), 1, recon_error.size(2), recon_error.size(3))

      recon_error = self.dataset.denormalize(x_future[0] - xfuture) # (actual - pred)
      xerror_future = recon_error.abs().max(1)[0].detach() 
      xerror_future = xerror_future.reshape(xerror_future.size(0), -1).clamp(min=0, max=1)
      xerror_future = xerror_future.reshape(recon_error.size(0), 1, recon_error.size(2), recon_error.size(3))

      def grid(imgs):
        return make_grid(imgs, nrow=10, padding=1, pad_value=imgs.max())


      stats.update({':present_yerr': grid(yhat - ytruth),
                    ':present_xerror': grid(xerror),
                    ':future_xerror': grid(xerror_future),
                    ':present_xhat': grid(self.dataset.denormalize(xhat)), # linear system model of past frames
                    ':future_xhat': grid(self.dataset.denormalize(xfuture)), # linear system model of future frames
                    ':present_xtruth': grid(self.dataset.denormalize(x[0])), # actual past frames
                    ':future_xtruth': grid(self.dataset.denormalize(x_future[0])) # actual future frames
                    })

      # import pdb; pdb.set_trace()

      # Calculate PSNR and SSIM
      present_gt = self.dataset.denormalize(x[0]).detach().cpu()
      present_hat = self.dataset.denormalize(xhat).detach().cpu()
      present_psnrs = np.array([utils.sklearn_psnr(gt, hat) for gt, hat in zip(present_gt, present_hat)])
      present_ssims = np.array([utils.sklearn_ssim(gt, hat) for gt, hat in zip(present_gt, present_hat)])

      future_gt = self.dataset.denormalize(x_future[0]).detach().cpu()
      future_hat = self.dataset.denormalize(xfuture).detach().cpu()
      future_psnrs = np.array([utils.sklearn_psnr(gt, hat) for gt, hat in zip(future_gt, future_hat)])
      future_ssims = np.array([utils.sklearn_ssim(gt, hat) for gt, hat in zip(future_gt, future_hat)])

      stats.update({
        ':present_psnr': present_psnrs.mean(),
        ':present_ssim': present_ssims.mean(),
        ':future_first_psnr': future_psnrs[0],
        ':future_fifth_psnr': future_psnrs[4],
        ':future_first_ssim': future_ssims[0],
        ':future_fifth_ssim': future_ssims[4]
      })


      # temperature = 0.7
      # xhat_samples, _ = self.flow.decode([temperature * torch.randn_like(z) for z in zlist])
      # stats.update({':xsample': self.dataset.denormalize(xhat_samples), # generated samples
      #         # ':xtruth': self.dataset.denormalize(x[:, 0]), # true samples, 15 long (10 past + 5 future)
      #         ':temperature': temperature}) # smaller means more realistic samples

      save_image(stats[':present_xtruth'], './images/mnist_x_present_truth.png', nrow=10, padding=1, pad_value=100)
      save_image(stats[':future_xtruth'], './images/mnist_x_future_truth.png', nrow=10, padding=1, pad_value=100)
      save_image(stats[':present_xtruth'], './images/mnist_x_present_hat.png', nrow=10, padding=1, pad_value=100)
      save_image(stats[':future_xhat'], './images/mnist_x_future_hat.png', nrow=10, padding=1, pad_value=100)
      save_image(stats[':present_xerror'], './images/mnist_x_present_error.png', nrow=10, padding=1, pad_value=100)
      save_image(stats[':future_xerror'], './images/mnist_x_future_error.png', nrow=10, padding=1, pad_value=100)

      # import pdb; pdb.set_trace()

      # # Save the training sequence as raw images so we can transfer it to Beyond-MSE.
      # for i in range(x.shape[1]):
      #   save_image(x[0,i], './images/raw_sequence/seq{}.png'.format(i), padding=0)
      # for i in range(x_future.shape[1]):
      #   save_image(x_future[0,i], './images/raw_sequence/seq{}.png'.format(i + x.shape[1]), padding=0)

    self.train()

    return stats

# stage 3
  # Scale up image resolution
  # profile: factor out O? use .half() for C'C computation? cusolver?
  # fully connected f functions in glow
  # pca / wavelet / DCT on datasets
  # is actnorm still failing?
  # get docker container up to speed
  # look at how small changes in the parameters are causing (possibly) large swings in the yhat values

# stage 4
  # complicated dataset testing
  # comparison with autoencoder
  # comparison with s.o.t.a



