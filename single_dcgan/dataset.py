import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets,transforms,models
from options import *


def imshow(inputs, picname):
    plt.ion()
    inputs = inputs / 2 + 0.5 
    inputs = inputs.numpy().transpose((1, 2, 0)) 
    plt.imshow(inputs)
    # plt.show()
    plt.pause(5)
    plt.savefig(picname+'.jpg') #保存图片
    plt.close()

data_transform = transforms.Compose([
    transforms.Resize(img_height),  
    #transforms.Resize(int(img_height * 1.12)),  
    #transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),                            
    transforms.ToTensor(),                                         
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),         
])

class MyDataSet():
  def __init__(self,batch_size):
    self.trainsetA = datasets.ImageFolder('%s/facades/train/trainC' % datapath, data_transform) 
    self.trainsetB = datasets.ImageFolder('%s/facades/train/trainD' % datapath, data_transform) 
    self.testA = datasets.ImageFolder('%s/facades/test/testC' % datapath, data_transform) 
    self.testB = datasets.ImageFolder('%s/facades/test/testD' % datapath, data_transform) 

    self.trainloaderA = DataLoader(self.trainsetA, batch_size=batch_size, shuffle=True)
    self.trainloaderB = DataLoader(self.trainsetB, batch_size=batch_size, shuffle=True)
    self.testloaderA = DataLoader(self.testA, batch_size=5, shuffle=True)
    self.testloaderB = DataLoader(self.testB, batch_size=5, shuffle=True)

def test():
  ds = MyDataSet(5)

  inputs, labels = next(iter(ds.trainloaderA))
  print("inputs:",inputs.shape)
  imshow(torchvision.utils.make_grid(inputs), "RealDataSample")

  inputs, labels = next(iter(ds.trainloaderB))
  print("inputs:",inputs.shape)
  imshow(torchvision.utils.make_grid(inputs), "RealDataSample")

  inputs, labels = next(iter(ds.testloaderA))
  print("inputs:",inputs.shape)
  imshow(torchvision.utils.make_grid(inputs), "RealDataSample")

  inputs, labels = next(iter(ds.testloaderB))
  print("inputs:",inputs.shape)
  imshow(torchvision.utils.make_grid(inputs), "RealDataSample")

#test()
