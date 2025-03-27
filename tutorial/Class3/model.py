import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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

### Step1: import packages, training data and testing data
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
epochs = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
### Step3: define your class for CNN, model/agent, testing. This is similar to 
           C/C++. To make your code clea and readable, you may define more than 
           one class corresponding to what they implement.
'''

class Net(torch.nn.Module):
    '''
    Question: how many hidden layers are there in the Net? What do they do?
    
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(784, 512),  # 将输入的28*28维数据映射到512维的隐藏空间
            torch.nn.BatchNorm1d(512),  # resist overfitting 批归一化
            torch.nn.ReLU(),            # what if we use other activation func such as Sigmoid() ? 引入非线性
            torch.nn.Linear(512, 10),   #将隐藏层的512维映射到输出层的10维
        )

    def forward(self, x): # forward propagation前向传播
        x = x.view(-1, 784) #将输入数据(如图像)展平为1D784维向量
        x = self.dense(x)   #通过隐藏层和输出层
        return x

class Tester:
    '''
    Just a tester to see how accurate your model is. You can simply ignore it.
    
    '''
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

class Model:
    '''
    When training your model, you should first complete forward propagation, 
    then calculate loss of your outputs and the real lables of the dataset.
    Almost every model has the goal of minimize the loss!
    To optimize your model, use back propagation.
    
    '''
    def __init__(self):
        self.model = Net().to(device)
        '''
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10)
        ).to(device)
        '''
        self.optimizer = torch.optim.Adam(self.model.parameters())  # Adam is a built-in optimizer of pytorch
        self.loss_func = torch.nn.CrossEntropyLoss()                # why is an "entropy" here?
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

                self.optimizer.zero_grad() # 1
                outputs = self.model(inputs) #2
                loss = self.loss_func(outputs, labels) #3
                loss.backward()            # 4
                self.optimizer.step()      # 5 : five common statement for optimizing and back propagation

                sum_loss += loss.item()
                if (i + 1) % 100 == 0:
                    print(f'epoch={epoch + 1}, batch={i + 1} loss: {sum_loss:.4f}') #training loss
                    self.loss_values.append(sum_loss)
                    sum_loss = 0.0

            self.tester.evaluate(epoch)
    '''
    Draw a loss-iterations chart using matplotlib package.
    You can check your chart at MINST-intro directory.
    
    '''
    def plot_loss(self):
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