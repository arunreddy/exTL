import torch
import torch.nn as nn

from intl.models.iw.model import MetaModule, MetaLinear


class MLP(MetaModule):
  def __init__(self, input_dim, hidden_dim=100, output_dim=2):
    super(MLP, self).__init__()
    self.layers = nn.Sequential(
      MetaLinear(input_dim, hidden_dim),
      nn.ReLU(),
      MetaLinear(hidden_dim, output_dim)
    )

  def forward(self, x):
    x = self.layers(x)
    return x


class MinibatchMLP(object):
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
    model = MLP(input_dim=self.input_dim, hidden_dim=self.hidden_dim, output_dim=self.output_dim)
    opt = torch.optim.SGD(model.params(), lr=self.hyper_parameters["lr"])
    return model, opt

  def train(self):
    model, optimizer = self.build_model()

    data = []
    n_iter = 0
    for epoch in range(self.hyper_parameters['num_epochs']):
      for i, (tr_X, tr_y) in enumerate(self.tr_dl):

        # Forward Propagation
        y_pred = model(tr_X)
        # Compute and print loss
        loss = self.criterion(y_pred, tr_y)

        # Zero the gradients
        optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        optimizer.step()
        n_iter += 1

        if n_iter % 1000 == 0:

          # Accuracy on training data.
          correct = 0
          total = 0
          for j, (te_X, te_y) in enumerate(self.tr_dl):
            outputs = model(te_X)
            _, predicted = torch.max(outputs.data, 1)
            total += te_y.size(0)
            correct += (predicted.cpu() == te_y.cpu()).sum()

          tr_accuracy = correct.numpy() / total

          # Accuracy on test data.
          correct = 0
          total = 0
          for j, (te_X, te_y) in enumerate(self.te_dl):
            outputs = model(te_X)
            _, predicted = torch.max(outputs.data, 1)
            total += te_y.size(0)
            correct += (predicted.cpu() == te_y.cpu()).sum()

          te_accuracy = correct.numpy() / total

          data.append([epoch, n_iter, tr_accuracy, te_accuracy, loss.item()])

          if self.verbose:
            print(
              'Accuracy and loss after [{}]{} iterations is Train:{:0.3f} and Test:{:0.3f} and {:0.3f} '.format(epoch,
                                                                                                                n_iter,
                                                                                                                tr_accuracy,
                                                                                                                te_accuracy,
                                                                                                                loss.item()))
    return data

  def predict(self):
    pass

  def results(self):
    pass
