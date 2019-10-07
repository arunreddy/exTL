from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from data_loader import MNISTImbalanced
from model import *
import numpy as np


hyperparameters = {
  'lr': 1e-3,
  'momentum': 0.9,
  'batch_size': 100,
  'num_iterations': 4,
}


def get_mnist_loader(batch_size, classes=[9, 4], n_items=5000, proportion=0.9, n_val=5, mode='train'):
  """Build and return data loader."""

  dataset = MNISTImbalanced(classes=classes, n_items=n_items, proportion=proportion, n_val=n_val, mode=mode)

  shuffle = False
  if mode == 'train':
    shuffle = True
  shuffle = True
  data_loader = DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle)
  return data_loader


def build_model():
  net = MLP(n_out=2)

  if torch.cuda.is_available():
    net.cuda()
    torch.backends.cudnn.benchmark = True

  opt = torch.optim.SGD(net.params(), lr=hyperparameters["lr"])

  return net, opt


def train_lre():
  data_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=0.995, mode="train")
  test_loader = get_mnist_loader(hyperparameters['batch_size'], classes=[9, 4], proportion=0.5, mode="test")

  val_data = to_var(data_loader.dataset.data_val, requires_grad=False)
  val_labels = to_var(data_loader.dataset.labels_val, requires_grad=False)

  x1 = np.random.randn(10, 10) + 1.0
  x2 = np.random.randn(10, 10) - 1.0

  X1 = np.vstack([x1, x2])
  y1 = np.asarray([1] * 10 + [0] * 10)

  x1 = np.random.randn(10, 10) + 1.0
  x2 = np.random.randn(10, 10) - 1.0

  X2 = np.vstack([x1, x2])
  y2 = np.asarray([1] * 10 + [0] * 10)

  X1 = torch.tensor(X1, dtype=torch.float, requires_grad=False)
  y1 = torch.tensor(y1, dtype=torch.long, requires_grad=False)

  X2 = torch.tensor(X2, dtype=torch.float, requires_grad=False)
  y2 = torch.tensor(y2, dtype=torch.long, requires_grad=False)

  net, opt = build_model()

  meta_losses_clean = []
  net_losses = []
  plot_step = 100

  smoothing_alpha = 0.9

  meta_l = 0
  net_l = 0
  accuracy_log = []
  for i in tqdm(range(hyperparameters['num_iterations'])):
    net.train()
    # Line 2 get batch of data
    # image, labels = next(iter(data_loader))
    # since validation data is small I just fixed them instead of building an iterator
    # initialize a dummy network for the meta learning of the weights
    meta_net = MLP(n_out=2)
    meta_net.load_state_dict(net.state_dict())

    # if torch.cuda.is_available():
    #   meta_net.cuda()

    # image = to_var(X1, requires_grad=False)
    # labels = to_var(y1, requires_grad=False)



    # Lines 4 - 5 initial forward pass to compute the initial weighted loss
    y_f_hat = meta_net(X1)

    cost = F.cross_entropy(y_f_hat, y1, reduce=False)
    eps = to_var(torch.zeros(cost.size()))
    l_f_meta = torch.sum(cost * eps)

    meta_net.zero_grad()

    # Line 6 perform a parameter update
    grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
    meta_net.update_params(hyperparameters['lr'], source_params=grads)

    # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
    y_g_hat = meta_net(X2)

    l_g_meta = F.cross_entropy(y_g_hat, y2)

    grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

    # Line 11 computing and normalizing the weights
    w_tilde = torch.clamp(-grad_eps, min=0)
    norm_c = torch.sum(w_tilde)

    if norm_c != 0:
      w = w_tilde / norm_c
    else:
      w = w_tilde


    print(w)

    # Lines 12 - 14 computing for the loss with the computed weights
    # and then perform a parameter update
    y_f_hat = net(X1)
    cost = F.cross_entropy(y_f_hat, y1, reduce=False)
    l_f = torch.sum(cost * w)

    opt.zero_grad()
    l_f.backward()
    opt.step()

    meta_l = smoothing_alpha * meta_l + (1 - smoothing_alpha) * l_g_meta.item()
    meta_losses_clean.append(meta_l / (1 - smoothing_alpha ** (i + 1)))

    net_l = smoothing_alpha * net_l + (1 - smoothing_alpha) * l_f.item()
    net_losses.append(net_l / (1 - smoothing_alpha ** (i + 1)))

    # if i % plot_step == 0:
    #   net.eval()
    #
    #   acc = []
    #   for itr, (test_img, test_label) in enumerate(test_loader):
    #     test_img = to_var(test_img, requires_grad=False)
    #     test_label = to_var(test_label, requires_grad=False)
    #
    #     output = net(test_img)
    #     predicted = (F.sigmoid(output) > 0.5).int()
    #
    #     acc.append((predicted.int() == test_label.int()).float())
    #
    #   accuracy = torch.cat(acc, dim=0).mean()
    #   accuracy_log.append(np.array([i, accuracy])[None])
    #
    #   acc_log = np.concatenate(accuracy_log, axis=0)
    #   print(accuracy)

    # return accuracy
  # return np.mean(acc_log[-6:-1, 1])


def main():
  accuracy = train_lre()


if __name__ == '__main__':
  main()
