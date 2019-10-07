import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from intl.models.iw.model import *

from intl.models.iw.model import MLP, to_var


class MlpIWA(object):

  def __init__(self,
               train_dataloader,
               test_dataloader,
               validation_dataloader,
               input_dim,
               hidden_dim=100,
               output_dim=2,
               hyper_parameters=None,
               verbose=False,
               **kwargs):
    self.tr_dl = train_dataloader
    self.te_dl = test_dataloader
    self.va_dl = validation_dataloader
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.verbose = verbose

    self.hyper_parameters = {
      'lr': 1e-2,
      'momentum': 0.9,
      'batch_size': 100,
      'num_iterations': 8000,
      'num_epochs': 1000
    }

    self.criterion = nn.CrossEntropyLoss()

    if hyper_parameters and type(hyper_parameters) == 'dict':
      self.hyper_parameters.update(hyper_parameters)

  def build_model(self):
    net = MLP(n_in=self.input_dim, n_hidden=self.hidden_dim, n_out=self.output_dim)
    # if torch.cuda.is_available():
    #   net.cuda()
    #   torch.backends.cudnn.benchmark = True

    opt = torch.optim.SGD(net.params(), lr=1e-3)

    return net, opt

  def train(self):
    net, opt = self.build_model()
    num_iterations = 1
    lst_acc = []
    lst_loss = []
    lr = 0.01
    for epoch in tqdm(range(num_iterations)):
      net.train()

      tr_X, tr_y = next(iter(self.tr_dl))
      va_X, va_y = next(iter(self.va_dl))

      meta_net = MLP(n_in=self.input_dim, n_hidden=self.hidden_dim, n_out=self.output_dim)
      meta_net.load_state_dict(net.state_dict())

      # if torch.cuda.is_available():
      #   meta_model.cuda()


      # tr_y = tr_y.view(-1, 1)
      # va_y = va_y.view(-1, 1)

      tr_X = to_var(tr_X, requires_grad=False)
      tr_y = to_var(tr_y, requires_grad=False)
      va_X = to_var(va_X, requires_grad=False)
      va_y = to_var(va_y, requires_grad=False)

      y_f_hat = meta_net(tr_X)
      cost = F.cross_entropy(y_f_hat, tr_y, reduction='none')
      eps = to_var(torch.zeros(cost.size()), requires_grad=True)
      l_f_meta = torch.sum(cost * eps)

      meta_net.zero_grad()

      grads = torch.autograd.grad(l_f_meta, (meta_net.parameters()), create_graph=True)
      meta_net.update_params(lr, source_params=grads)

      y_g_hat = meta_net(va_X)
      l_g_meta = F.cross_entropy(y_g_hat, va_y)

      grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, allow_unused=True)[0]

      print(grad_eps)

      print('----------------')


if __name__ == '__main__':
  from intl.tdata.amazon_dataset_combined import AmazonDatasetCombined

  source = 'dvd'
  target = 'electronics'
  target_train_ratio = 0.05
  random_state = 0

  params = {
    'source': source,
    'target': target,
    'target_train_ratio': target_train_ratio,
    'max_features': 5000,
    'random_state': random_state,
    'y_dtype': 'long',
    'device': 'cpu'
  }

  params['partition'] = 'tr'
  tr_data = AmazonDatasetCombined(**params)

  params['partition'] = 'te'
  te_data = AmazonDatasetCombined(**params)

  params['partition'] = 'va'
  va_data = AmazonDatasetCombined(**params)

  batch_size = 50

  tr_loader = DataLoader(tr_data, batch_size=50, shuffle=True)
  te_loader = DataLoader(te_data, batch_size=50, shuffle=True)
  va_loader = DataLoader(va_data, batch_size=50, shuffle=True)

  iw = MlpIWA(tr_loader,
              te_loader,
              va_loader,
              input_dim=5000,
              output_dim=2)

  iw.train()
