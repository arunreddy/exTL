import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable


#LOADING DATASET

train_dataset = dsets.MNIST(root='/tmp/data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='/tmp/data',
                           train=False,
                           transform=transforms.ToTensor())

batch_size = 100
n_iters = 10000
num_epochs = n_iters // (len(train_dataset) // batch_size)
all_loss = []
all_accuracy = []
## From The Docs...


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class LogisticRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 28*28
output_dim = 10

model = LogisticRegressionModel(input_dim, output_dim)

#  USE GPU FOR MODEL

if torch.cuda.is_available():
    model.cuda()
    print("Running On GPU")

#LOSS CLASS
criterion = nn.CrossEntropyLoss()
#OPTIMIZER CLASS
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# TRAIN THE MODEL
iter = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        #  USE GPU FOR MODEL

        if torch.cuda.is_available():
            images = Variable(images.view(-1, 28 * 28).cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images.view(-1, 28 * 28))
            labels = Variable(labels)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:

            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test tdata

            for images, labels in test_loader:
                #  USE GPU FOR MODEL  #
                images = Variable(images.view(-1, 28 * 28).cuda())

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)

                # Total number of labels
                total += labels.size(0)

                #  USE GPU FOR MODEL  #

                # Total correct predictions

                correct += (predicted.cpu() == labels.cpu()).sum()

            accuracy = 100 * correct / total

            all_loss.append(loss.item())
            all_accuracy.append(accuracy)

            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))


print(all_loss , all_accuracy)

