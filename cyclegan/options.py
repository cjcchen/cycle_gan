
img_height=256
img_width=256
channels = 3

n_residual_blocks = 9
n_epochs = 100
b1=0.5
b2=0.999
lr=0.00005
lambda_id=5
lambda_cyc=10
datasetname="face"
batch_size=20
decay_start_epoch=50
pretrain_path="./checkpoints/face"
datapath="../../datasets/facades/"
datapath="../../datasets/facades/"
path_trainA=datapath+"train/trainC"
path_trainA=datapath+"train/trainC"
path_trainB=datapath+"train/trainD"
path_testA=datapath+"test/testC"
path_testB=datapath+"test/testD"
