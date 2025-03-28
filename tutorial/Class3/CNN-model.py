import os
import torch
import torchvision
import torch.utils.data.dataloader as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import datasets, transforms
'''
This is a very very simple CNN model based on MNIST dataset, which is specialized
for recognizing written numbers. You can have a glimpse of what an artificial 
agent is, and how researchers train model.

Usage:
    open your bash or terminal in the MNISTdemo folder, then type in:
        pip install torch torchvision torchaudio (if you don't have CUDA support)
        (Well, if you have CUDA support, please visit https://developer.nvidia.com/cuda-toolkit )
        pip install matplotlib numpy
        python model.py

### Step1: import packages, training data and testing data'
'''
train_data = torchvision.datasets.MNIST(
    'dataset/mnist-pytorch', train=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    download=True
)
test_data = torchvision.datasets.MNIST(
    'dataset/mnist-pytorch', train=False,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
)

'''
### Step2: define global variables and other command that's not included in 
           any class or function
'''
batch_size = 128
epochs = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
### Step3: define your class for CNN, model/agent, testing. This is similar to 
           C/C++. To make your code clear and readable, you may define more than 
           one class corresponding to what they implement.
'''
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  
        x = self.fc_layers(x)
        return x

class Tester:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    def evaluate(self, epoch):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'epoch={epoch + 1} accuracy={accuracy:.02f}%')
        return accuracy

    def show_predictions(self, num_images=64, images_per_row=8):
            self.model.eval()
            images, labels = next(iter(self.test_loader))
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            
            rows = num_images//images_per_row
            fig, axes = plt.subplots(rows, images_per_row, figsize=(15, 2 * rows))
            for i in range(num_images):
                row, col = divmod(i, images_per_row)
                axes[row, col].imshow(images[i].cpu().squeeze(), cmap='gray')
                axes[row, col].set_title(f'True: {labels[i].item()}\nPred: {predicted[i].item()}')
                axes[row, col].axis('off')
            plt.tight_layout(pad=3.0)  # Add padding between image
            plt.show()


class Model:
    '''
    When training your model, you should first complete forward propagation, 
    then calculate loss of your outputs and the real lables of the dataset.
    Almost every model has the goal of minimize the loss!
    To optimize your model, use back propagation.
    
    '''
    def __init__(self):
        self.model = Net().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size)
        self.tester = Tester(self.model, self.test_loader, device)
        self.loss_values = []
        print(self.model)

    def train(self):
        for epoch in range(epochs):
            sum_loss = 0.0
            self.model.train()
            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.optimizer.step()

                sum_loss += loss.item()
                if (i + 1) % 100 == 0:
                    print(f'epoch={epoch + 1}, batch={i + 1} loss: {sum_loss:.4f}')
                if (i + 1) % 20 == 0:
                    self.loss_values.append(sum_loss)
                    sum_loss = 0.0

            self.tester.evaluate(epoch)

    def plot_loss(self):
        # iterations = [i * 100 for i in range(len(self.loss_values))] 
        plt.plot(range(len(self.loss_values)), self.loss_values, label='Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Iterations')
        plt.legend()
        plt.savefig('loss.png')  
        plt.close()

'''
Declare main function. It's a bit different from what we do when coding C/C++.
Search on the browser to find out why we don't use "def main()".

'''
if __name__ == "__main__":
    model = Model()
    model.train()
    model.plot_loss()
    model.tester.show_predictions()
