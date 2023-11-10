import torch
import torch.nn as nn
import sys
import itertools
import os

from dataset import *
from net import *
from cycle_replay_buffer import *
from torchvision.utils import save_image, make_grid

input_shape = (channels, img_height, img_width)

g = G(input_shape)
d = D(input_shape)

#d = D(3, 32) #定义鉴别器，输入通道为3，输出通道为32
#g = G(3, 128, 1024, 96) #定义生成器，最后一层输出通道为3，输入通道为128，线性层输出通道为128，输入通道为100

criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()
criterion_cross = nn.BCELoss() 

g.apply(weights_init_normal)
d.apply(weights_init_normal)
print("g:",g)
print("D:",d)

optimizer_G = torch.optim.Adam(
    itertools.chain(g.parameters()), lr=lr, betas=(b1, b2)
)
optimizer_D = torch.optim.Adam(d.parameters(), lr=lr, betas=(b1, b2))

ds = MyDataSet(batch_size)

def sample_images(return_sample=False):      
    batch =  next(iter(zip(ds.testloaderA)))
    g.eval()

    inputsA, labelsA = batch[0]
    print("input a shape:",inputsA.shape)
    fake = g(inputsA)
    print("fake:",fake.shape)
    rows = inputsA.shape[0]

    real_A = make_grid(inputsA, nrow=rows, normalize=True)
    fake_A = make_grid(fake, nrow=rows, normalize=True)

    image_grid = torch.cat((real_A, fake_A), 1)

    if return_sample == False: 
      imshow(image_grid, "RealDataSample")
    else:
      return image_grid

def save_images(epoch):      
    image_grid = sample_images(True)
    if not os.path.exists('images/%s/%s' % (datasetname, epoch)):
      os.makedirs('images/%s/%s' % (datasetname,epoch))
    save_image(image_grid, "images/%s/%s.png" % (datasetname, epoch), normalize=False)

def save_model(epoch):
  path = 'checkpoints/%s/%d' % (datasetname, epoch)

  if not os.path.exists(path):
    os.makedirs(path)

  torch.save(g.state_dict(), "%s/G.pth" % path)
  torch.save(d.state_dict(), "%s/D.pth" % path)

  save_images(epoch)

def my_train():
  for epoch in range(0, n_epochs):       
      save_model(epoch)
      for i, batch in enumerate(zip(ds.trainloaderA, ds.trainloaderB)): 
        inputsA, labelsA = batch[0]
        inputsB, labelsB = batch[1]

        real_A = inputsA
        real_B = inputsB
        fake_A = g(real_A)

        real_labels = torch.ones(inputsA.size(0), *d.output_shape)
        fake_labels = torch.zeros(inputsA.size(0), *d.output_shape)

        loss_fake = criterion_mse(d(fake_A), fake_labels)
        loss_real = criterion_mse(d(real_B), real_labels) 
        d_loss = (loss_fake + loss_real)/2

        optimizer_D.zero_grad()
        d_loss.backward()     
        optimizer_D.step()   

        fake_A = g(real_A)
        real_labels = torch.ones(inputsA.size(0), *d.output_shape)
        g_loss = criterion_mse(d(fake_A), real_labels)

        optimizer_G.zero_grad()                                       
        g_loss.backward()                                            
        optimizer_G.step()                                          

        sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(ds.trainloaderA),
                    d_loss.item(),
                    g_loss.item()
                )
        )
        sample_images()


my_train()
