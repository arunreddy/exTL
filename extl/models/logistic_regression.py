import os
from datetime import datetime

import itertools
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils import data

from intl.data.amazon_dataset import AmazonDataset

domains = ['books', 'dvd', 'electronics', 'kitchen']


class LogisticRegression(nn.Module):
  def __init__(self, input_size, num_classes):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(input_size, num_classes)

  def forward(self, x):
    out = self.linear(x)
    return out


for source_domain, target_domain in itertools.permutations(domains, 2):

  print(source_domain, target_domain)

  log_dir = os.path.join(os.getenv('HOME'), 'logdir', 'logit', '{}_{}'.format(source_domain, target_domain),
                         datetime.now().strftime('%m%d%H%M%S'))
  writer = SummaryWriter(log_dir=log_dir)

  input_size = 4000
  num_classes = 2
  learning_rate = 0.001
  num_epochs = 1000
  batch_size = 50

  writer.add_scalar('input_size', input_size, 0)
  writer.add_scalar('num_epochs', num_epochs, 0)
  writer.add_scalar('batch_size', batch_size, 0)

  source_data = AmazonDataset(source=source_domain, target=target_domain, partition_name='source',
                              max_features=input_size, target_train_ratio=0.1, y_dtype='long')
  target_data = AmazonDataset(source=source_domain, target=target_domain, partition_name='target',
                              max_features=input_size, target_train_ratio=0.1, y_dtype='long')

  src_loader = data.DataLoader(source_data, batch_size=batch_size, shuffle=True)
  tgt_loader = data.DataLoader(target_data, batch_size=batch_size)

  model = LogisticRegression(input_size, num_classes)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  n_iter = 0
  # # Training the Model
  for epoch in range(num_epochs):
    for i, (X, y) in enumerate(src_loader):

      images = Variable(X)
      labels = Variable(y)

      # Forward + Backward + Optimize
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      n_iter += 1

      if n_iter % 1000 == 0:

        correct = 0
        total = 0

        for j, (Xt, yt) in enumerate(tgt_loader):
          _xt = Variable(Xt)
          _yt = Variable(yt)

          outputs = model(_xt)
          _, predicted = torch.max(outputs.data, 1)

          total += labels.size(0)

          #  USE GPU FOR MODEL  #

          # Total correct predictions

          correct += (predicted.cpu() == _yt.cpu()).sum()

        accuracy = correct.numpy() / total

        writer.add_scalar('metrics/loss', loss.item(), n_iter)
        writer.add_scalar('metrics/accuracy', accuracy, n_iter)

        print(loss.item(), accuracy)

  print('----- Completed ----- ')
  writer.close()
