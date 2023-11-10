import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms,models
import matplotlib.pyplot as plt
import os

#对图片进行变换
#RandomHorizontalFlip：随机水平翻转给定的图片,概率为0.5。即：一半的概率翻转，一半的概率不翻转
#ToTensor：将图片转为Tensor格式
#Normalize：：用给定的均值和标准差分别对每个通道的数据进行正则化。
#   给定均值(M1,…,Mn)，给定标准差(S1,…,Sn)，其中n是通道数（一般是3），对每个通道进行
#       output[channel] = (input[channel] - mean[channel]) / std[channel]
data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

trainset = datasets.ImageFolder('faces', data_transform) #数据集
#加载数据集的Dataloader，每次随机打乱，加载5张图片
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True)

#展示并保存图片
def imshow(inputs, picname):
    plt.ion()
    inputs = inputs / 2 + 0.5 #加载数据集的时候使用了正则化，需要恢复一下
    inputs = inputs.numpy().transpose((1, 2, 0)) #pytorch加载图片通道数在前，我们展示图片图片的时候通道数在后
    plt.imshow(inputs)
    # plt.show()
    plt.pause(5)
    plt.savefig(picname+'.jpg') #保存图片
    plt.close()

#鉴别器
class D(nn.Module):
    def __init__(self, nc, ndf):
        super(D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1, 256*6*6)
        out = self.fc(out).reshape(-1)
        return out

#生成器
class G(nn.Module):
    def __init__(self, nc, ngf, nz, feature_size):
        super(G, self).__init__()
        self.prj = nn.Linear(feature_size, nz*6*6)
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.prj(x)
        out = out.view(-1, 1024, 6, 6)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

#inputs, _ = next(iter(trainloader))
#imshow(torchvision.utils.make_grid(inputs), "RealDataSample")

d = D(3, 32) #定义鉴别器，输入通道为3，输出通道为32
g = G(3, 128, 1024, 100) #定义生成器，最后一层输出通道为3，输入通道为128，线性层输出通道为128，输入通道为100

criterion = nn.BCELoss() #使用交叉熵作为损失函数
#生成器和鉴别器均采用Adam优化器
d_optimizer = torch.optim.Adam(d.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(g.parameters(), lr=0.0003)

def train(d, g, criterion, d_optimizer, g_optimizer, epochs=1, show_every=1000, print_every=10):
    iter_count = 0
    for epoch in range(epochs):
        for inputs,_ in trainloader:
            real_inputs = inputs #5张真实图片
            fake_inputs = g(torch.randn(5, 100)) #5张假图片

            #print("real inputs:",real_inputs.shape)
            #print("fake inputs:",fake_inputs.shape)

            #shape: [5, 3, 96, 96]

            #对应标签
            real_labels = torch.ones(real_inputs.size(0))
            fake_labels = torch.zeros(5)
            print("real labels:",real_labels.shape)

            #鉴别真实图片
            real_outputs = d(real_inputs)
            #print("read output:",real_outputs)
            d_loss_real = criterion(real_outputs, real_labels)
            real_scores = real_outputs

            #鉴别假图片
            fake_outputs = d(fake_inputs)
            d_loss_fake = criterion(fake_outputs, fake_labels)
            fake_scores = fake_outputs

            d_loss = d_loss_fake + d_loss_real #计算总损失
            #反向传播
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step() #更新鉴别器参数

            #生成假图片，并用更新后的鉴别器进行鉴别
            fake_inputs = g(torch.randn(5, 100))
            outputs = d(fake_inputs) #鉴别器的标签
            real_labels = torch.ones(outputs.size(0)) #真实的标签
            g_loss = criterion(outputs, real_labels)
            #反向传播
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step() #更新鉴别器

            #保存图片
            if iter_count % show_every == 0:
                print('Epoch:{}, Iter:{}, D:{:.4}, G:{:.4}'.format(epoch, iter_count, d_loss.item(), g_loss.item()))

                picname = "Epoch_"+str(epoch)+"Iter_"+str(iter_count)
                imshow(torchvision.utils.make_grid(fake_inputs.data), picname)
            #打印损失函数
            if iter_count % print_every == 0:
                print('Epoch:{}, Iter:{}, D:{:.4}, G:{:.4}'.format(epoch, iter_count, d_loss.item(), g_loss.item()))

            iter_count += 1
    print("Finsh")

train(d, g, criterion, d_optimizer, g_optimizer, epochs=100)
