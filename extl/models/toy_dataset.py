import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from intl.data.toy_dataset import ToyDataset
import matplotlib.pyplot as plt

class LogisticRegression(nn.Module):
  def __init__(self, input_size, num_classes):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(input_size, num_classes)

  def forward(self, x):
    out = self.linear(x)
    return out


dataset_type = 'medium'

tr_data = ToyDataset(dataset_type=dataset_type, partition_name='train')
te_data = ToyDataset(dataset_type=dataset_type, partition_name='test')

tr_loader = data.DataLoader(tr_data, batch_size=10, shuffle=True)
te_loader = data.DataLoader(te_data, batch_size=10)

input_size = 2
num_classes = 2
learning_rate = 0.001
num_epochs = 50

model = LogisticRegression(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_iter = 0
# # Training the Model
acc = []
for epoch in range(num_epochs):
  for i, (tr_X, tr_y) in enumerate(tr_loader):

    # Forward + Backward + Optimize
    optimizer.zero_grad()
    outputs = model(tr_X)
    loss = criterion(outputs, tr_y)
    loss.backward()
    optimizer.step()

    n_iter += 1

    if n_iter % 100 == 0:
      correct = 0
      total = 0
      for j, (te_X, te_y) in enumerate(te_loader):
        outputs = model(te_X)
        print(outputs.shape)
        _, predicted = torch.max(outputs.data, 1)

        total += te_y.size(0)

        correct += (predicted.cpu() == te_y.cpu()).sum()

      accuracy = correct.numpy() / total
      acc.append(accuracy)

      print("{:04d} - {:0.3f} : {:0.3f}".format(n_iter, loss.item(), accuracy))



print(model.linear.in_features)
print(model.linear.out_features)
print(model.linear.weight)
print(model.linear.bias)

# plt.figure(figsize=(10,6))
plt.plot(np.arange(len(acc)), acc, label=dataset_type)
plt.legend()
plt.show()