import torch
import torch.nn as nn
import sys
import itertools
import os

from cycle_dataset import *
from cycle_net import *
from cycle_replay_buffer import *
from torchvision.utils import save_image, make_grid

input_shape = (channels, img_height, img_width)

G_AB = GeneratorResNet(input_shape, n_residual_blocks)
G_BA = GeneratorResNet(input_shape, n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

criterion_mse = nn.MSELoss()
criterion_l1 = nn.L1Loss()

G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()
ds = MyDataSet(batch_size)

def sample_images(return_sample=False):      
    batch =  next(iter(zip(ds.testloaderA, ds.testloaderB)))
    G_AB.eval()
    G_BA.eval()

    inputsA, labelsA = batch[0]
    inputsB, labelsB = batch[1]

    fake_B = G_AB(inputsA)           
    fake_A = G_BA(inputsB)           
    rows = inputsA.shape[0]

    real_A = make_grid(inputsA, nrow=rows, normalize=True)
    real_B = make_grid(inputsB, nrow=rows, normalize=True)
    fake_A = make_grid(fake_A, nrow=rows, normalize=True)
    fake_B = make_grid(fake_B, nrow=rows, normalize=True)

    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)

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

  torch.save(G_AB.state_dict(), "%s/G_AB.pth" % path)
  torch.save(G_BA.state_dict(), "%s/G_BA.pth" % path)
  torch.save(D_A.state_dict(), "%s/D_A.pth" % path)
  torch.save(D_B.state_dict(), "%s/D_B.pth" % path)

  save_images(epoch)

def my_train():
  for epoch in range(0, n_epochs):       
      save_model(epoch)
      for i, batch in enumerate(zip(ds.trainloaderA, ds.trainloaderB)): 
        inputsA, labelsA = batch[0]
        inputsB, labelsB = batch[1]

        real_A = inputsA
        real_B = inputsB
        real_label = torch.ones(inputsA.size(0), *D_A.output_shape)
        fake_label = torch.zeros(inputsB.size(0), *D_A.output_shape)
        
        #print("batch A:",inputsA.shape, real_A_labels.shape)
        #print("batch B:",inputsB.shape, real_B_labels.shape)

        ## Self loss
        real_ouptut_A = G_BA(real_A) # A creates A, 
        real_output_B = G_AB(real_B) # B creates B,

        loss_A = criterion_l1(real_ouptut_A, real_A)          
        loss_B = criterion_l1(real_output_B, real_B)

        loss_identity = (loss_A + loss_B) / 2 # self creation should be as small as possible

        ## GAN loss
        fake_B = G_AB(real_A) # A creates B 
        fake_A = G_BA(real_B) # B creates A

        loss_GAN_AB = criterion_mse(D_B(fake_B), real_label) # fake_b shold be the same as b
        loss_GAN_BA = criterion_mse(D_A(fake_A), real_label) # fake_b shold be the same as b
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2 
    
        #print("loss iden:",loss_identity)
        #print("loss gan:",loss_GAN)

        # Cycle loss 
        recov_A = G_BA(fake_B)                                        
        recov_B = G_AB(fake_A)

        loss_cycle_A = criterion_l1(recov_A, real_A)             
        loss_cycle_B = criterion_l1(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity
        optimizer_G.zero_grad()                                       
        loss_G.backward()                                            
        optimizer_G.step()                                          


        loss_real = criterion_mse(D_A(real_A), real_label)
        buffer_fake_A = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_mse(D_A(buffer_fake_A), fake_label)
        loss_D_A = (loss_real + loss_fake) / 2
        optimizer_D_A.zero_grad()
        loss_D_A.backward()     
        optimizer_D_A.step()   

        loss_real = criterion_mse(D_B(real_B), real_label)
        buffer_fake_B = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_mse(D_B(buffer_fake_B), fake_label)
        loss_D_B = (loss_real + loss_fake) / 2

        optimizer_D_B.zero_grad()                                     
        loss_D_B.backward()                                      
        optimizer_D_B.step()                                    
        loss_D = (loss_D_A + loss_D_B) / 2

        sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f]"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(ds.trainloaderA),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                )
        )
        #sample_images()


my_train()
