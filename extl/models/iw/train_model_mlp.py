from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from intl.models.iw.model import *
import numpy as np
from intl.tdata.amazon_dataset_combined import AmazonDatasetCombined
from intl.tdata.amazon_dataset import AmazonDataset

hyperparameters = {
  'lr': 1e-2,
  'momentum': 0.9,
  'batch_size': 100,
  'num_iterations': 100000,
}



def build_model():
  net = MLP(n_in=4000, n_hidden=100, n_out=1)
  #
  # if torch.cuda.is_available():
  #   net.cuda()
  #   torch.backends.cudnn.benchmark = True

  opt = torch.optim.SGD(net.params(), lr=1e-3)

  return net, opt


def train_lre():

  source = 'books'
  target = 'dvd'
  target_train_ratio = 0.1
  random_state = 0

  params = {
    'source': source,
    'target': target,
    'target_train_ratio': target_train_ratio,
    'max_features': 5000,
    'random_state': random_state,
    'y_dtype': 'float',
    'device': 'cpu'
  }

  params['partition'] = 'tr'
  tr_data = AmazonDatasetCombined(**params)

  params['partition'] = 'te'
  te_data = AmazonDatasetCombined(**params)

  params['partition'] = 'va'
  va_data = AmazonDatasetCombined(**params)

  batch_size = 100
  input_size = 4000

  tr_loader = DataLoader(tr_data, batch_size=batch_size, shuffle=True)
  te_loader = DataLoader(te_data, batch_size=batch_size)
  va_loader = DataLoader(va_data, batch_size=batch_size, shuffle=True)

  source_data = AmazonDataset(source=source, target=target, partition_name='source',
                              max_features=input_size, target_train_ratio=0.1)
  target_data = AmazonDataset(source=source, target=target, partition_name='target',
                              max_features=input_size, target_train_ratio=0.1)
  validation_data = AmazonDataset(source=source, target=target, partition_name='validation',
                              max_features=input_size, target_train_ratio=0.1)

  tr_loader = DataLoader(source_data, batch_size=batch_size, shuffle=True)
  te_loader = DataLoader(target_data, batch_size=batch_size)
  va_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

  net, opt = build_model()

  meta_losses_clean = []
  net_losses = []
  plot_step = 1000

  smoothing_alpha = 0.9

  meta_l = 0
  net_l = 0
  accuracy_log = []
  for i in range(hyperparameters['num_iterations']):
    net.train()
    # Line 2 get batch of data
    X1, y1 = next(iter(tr_loader))
    X2, y2 = next(iter(va_loader))
    # since validation data is small I just fixed them instead of building an iterator
    # initialize a dummy network for the meta learning of the weights
    meta_net = MLP(n_in=4000, n_hidden=100, n_out=1)
    meta_net.load_state_dict(net.state_dict())

    # if torch.cuda.is_available():
    #   meta_net.cuda()

    y1 = y1.view(-1,1)
    y2 = y2.view(-1, 1)

    X1 = to_var(X1, requires_grad=False)
    y1 = to_var(y1, requires_grad=False)

    X2 = to_var(X2, requires_grad=False)
    y2 = to_var(y2, requires_grad=False)


    # Lines 4 - 5 initial forward pass to compute the initial weighted loss
    y_f_hat = meta_net(X1)
    cost = F.binary_cross_entropy_with_logits(y_f_hat, y1, reduce=False)
    eps = to_var(torch.zeros(cost.size()))
    l_f_meta = torch.sum(cost * eps)

    meta_net.zero_grad()

    # Line 6 perform a parameter update
    grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
    meta_net.update_params(hyperparameters['lr'], source_params=grads)

    # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
    y_g_hat = meta_net(X2)

    l_g_meta = F.binary_cross_entropy_with_logits(y_g_hat, y2)

    grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

    # Line 11 computing and normalizing the weights
    w_tilde = torch.clamp(-grad_eps, min=0)
    norm_c = torch.sum(w_tilde)

    if norm_c != 0:
      w = w_tilde / norm_c
    else:
      w = w_tilde

    # Lines 12 - 14 computing for the loss with the computed weights
    # and then perform a parameter update
    y_f_hat = net(X1)
    cost = F.binary_cross_entropy_with_logits(y_f_hat, y1, reduce=False)
    l_f = torch.sum(cost * w)

    opt.zero_grad()
    l_f.backward()
    opt.step()

    meta_l = smoothing_alpha * meta_l + (1 - smoothing_alpha) * l_g_meta.item()
    meta_losses_clean.append(meta_l / (1 - smoothing_alpha ** (i + 1)))

    net_l = smoothing_alpha * net_l + (1 - smoothing_alpha) * l_f.item()
    net_losses.append(net_l / (1 - smoothing_alpha ** (i + 1)))

    if i % plot_step == 0:
      net.eval()

      acc = []
      for itr, (test_img, test_label) in enumerate(te_loader):
        test_label = test_label.view(-1,1)

        test_img = to_var(test_img, requires_grad=False)
        test_label = to_var(test_label, requires_grad=False)

        output = net(test_img)
        predicted = (F.sigmoid(output) > 0.5).int()

        acc.append((predicted.int() == test_label.int()).float())

      accuracy = torch.cat(acc, dim=0).mean()
      accuracy_log.append(np.array([i, accuracy])[None])

      acc_log = np.concatenate(accuracy_log, axis=0)
      print(i,accuracy)

  return accuracy
  # return np.mean(acc_log[-6:-1, 1])


def main():
  accuracy = train_lre()


if __name__ == '__main__':
  main()
