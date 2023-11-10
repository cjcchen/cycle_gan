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

G = GeneratorResNet(input_shape, n_residual_blocks)
D = Discriminator(input_shape)

criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()

G.apply(weights_init_normal)
D.apply(weights_init_normal)
print("g:",G)
print("D:",D)

optimizer_G = torch.optim.Adam(
    itertools.chain(G.parameters()), lr=lr, betas=(b1, b2)
)
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(b1, b2))

fake_A_buffer = ReplayBuffer()
#fake_B_buffer = ReplayBuffer()
ds = MyDataSet(batch_size)

def sample_images(return_sample=False):      
    batch =  next(iter(zip(ds.testloaderA)))
    G.eval()

    inputsA, labelsA = batch[0]
    print("input a shape:",inputsA.shape)
    fake = G(inputsA)
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

  torch.save(G.state_dict(), "%s/G.pth" % path)
  torch.save(D.state_dict(), "%s/D.pth" % path)

  save_images(epoch)

def my_train():
  for epoch in range(0, n_epochs):       
      save_model(epoch)
      for i, batch in enumerate(zip(ds.trainloaderA, ds.trainloaderB)): 
        inputsA, labelsA = batch[0]
        inputsB, labelsB = batch[1]

        real_A = inputsA
        real_B = inputsB
        real_label = torch.ones(inputsA.size(0), *D.output_shape)
        fake_label = torch.zeros(inputsA.size(0), *D.output_shape)
        
        #print("batch A:",inputsA.shape, real_A_labels.shape)
        #print("batch B:",inputsB.shape, real_B_labels.shape)

        ## Self loss
        fake_A = G(real_A) # A creates A, 
        loss_identity = criterion_l1(fake_A, real_A)          

        ## GAN loss
        loss_GAN = criterion_mse(D(fake_A), real_label) # fake_b shold be the same as b
    
        #print("loss iden:",loss_identity)
        #print("loss gan:",loss_GAN)

        loss_G = loss_GAN + lambda_id * loss_identity
        optimizer_G.zero_grad()                                       
        loss_G.backward()                                            
        optimizer_G.step()                                          

        loss_real = criterion_mse(D(real_B), real_label)
        buffer_fake_A = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_mse(D(buffer_fake_A), fake_label)
        loss_D = (loss_real + loss_fake) / 2
        optimizer_D.zero_grad()
        loss_D.backward()     
        optimizer_D.step()   

        sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, identity: %f]"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(ds.trainloaderA),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_identity.item(),
                )
        )
        sample_images()


my_train()
