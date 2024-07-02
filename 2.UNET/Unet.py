import os
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

lr = 1e-3
batch_size = 2
num_epoch = 100

data_dir ="./BSDS300/images/"
ckpt_dir = "./checkpoint"
result_dir = "./result"
log_dir  = "./log"
task = "denoising"
opts = ["random", 30.0]
ny = 320
nx = 480
nch = 3
nker = 64
network = "unet"

result_dir_train = os.path.join(result_dir, "train")
result_dir_val = os.path.join(result_dir, "val")
result_dir_test = os.path.join(result_dir, "test")
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
    os.mkdir(result_dir_train)
    os.mkdir(result_dir_val)
    os.mkdir(result_dir_test)
    os.mkdir(os.path.join(result_dir_test,"numpy"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr
        
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec4_2 = CBR2d(in_channels=2*512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec3_2 = CBR2d(in_channels=2*256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec2_2 = CBR2d(in_channels=2*128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)
        self.dec1_2 = CBR2d(in_channels=2*64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        

        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 =  self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 =  self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 =  self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 =  self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x


        
## Data Loader
class DataSet(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if  f.startswith('label')]
        lst_input = [f for f in lst_data if  f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label  = lst_label
        self.lst_input  = lst_input

    def __len__(self):
        return len(self.lst_label)
    
    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        inputs = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label/255.0
        inputs = inputs/255.0

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        
        if inputs.ndim == 2:
            inputs = inputs[:, :, np.newaxis]
        
        data = {"input": inputs, "label": label}


        if self.transform:
            data = self.transform(data)

        return data

# Transforms

class ToTensor(object):
    def __call__(self,data):
        label, inputs = data["label"], data["input"]

        label = label.transpose((2, 0 , 1)).astype(np.float32)
        inputs = inputs.transpose((2, 0 , 1)).astype(np.float32)

        data = {"label": torch.from_numpy(label), "input" : torch.from_numpy(inputs)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std= 0.5):
        self.mean = mean
        self.std  = std
    
    def __call__(self, data):
        label, inputs = data["label"], data["input"]

        inputs = (inputs-self.mean)/ self.std

        data = {"label": label, "input":inputs}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, inputs = data["label"], data["input"]

        if np.random.rand()>0.5:
            label = np.fliplr(label)
            inputs = np.fliplr(inputs)
        
        if np.random.rand()>0.5:
            label = np.flipud(label)
            inputs = np.flipud(inputs)

        data  = {"label":label, "input":inputs}

        return data

transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

dataset_train = DataSet(data_dir = os.path.join(data_dir, "train"), transform=transform)
loader_train  = DataLoader(dataset_train, batch_size= batch_size, shuffle=True, num_workers=8)

dataset_val = DataSet(data_dir = os.path.join(data_dir, "val"), transform=transform)
loader_val  = DataLoader(dataset_val, batch_size= batch_size, shuffle=False, num_workers=8)

net = UNet().to(device)
fn_loss  = nn.BCEWithLogitsLoss().to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

num_data_train = len(dataset_train)
num_data_val   = len(dataset_val)

num_batch_train = int(np.ceil(num_data_train/ batch_size))
num_batch_val = int(np.ceil(num_data_val/ batch_size))

fn_tonumpy = lambda x: x.to("cpu").detach().numpy().transpose(0,2,3,1)
fn_denorm  = lambda x, mean, std: (x * std) + mean
fn_class   = lambda x: 1.0 * (x >0.5)

writer_train = SummaryWriter(log_dir = os.path.join(log_dir, "train"))
writer_val = SummaryWriter(log_dir = os.path.join(log_dir, "val"))

def save(ckpt_dir , net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    
    torch.save({"net" : net.state_dict(), "optim": optim.state_dict()}, f"./{ckpt_dir}/model_epoch_{epoch}.pth")

def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst,sort(key=lambda f: int("".join(filter(str,isdigit, f))))

    dict_model  = torch.load(f"./{ckpt_dir}/{ckpt_lst[-1]}")

    net.load_state_dict(dict_model["net"])
    optim.load_state_dict(dict_model["optim"])
    epoch = int(ckpt_lst[-1].split("epoch")[-1].split(".pth")[0])

    return net, optim, epoch

st_epoch = 0

net, optim, epoch = load(ckpt_dir, net, optim)

if __name__ == "__main__":

    for epoch in range(st_epoch + 1, num_epoch +1):
        net.train()
        loss_arr =[]

        for batch, data in enumerate(loader_train, 1):
            label = data["label"].to(device)
            inputs =data["input"].to(device)

            output = net(inputs)

            optim.zero_grad()
            loss = fn_loss(output, label)
            loss.backward()
            optim.step()

            loss_arr += [loss.item()]

            print(f"Train : EPOCH {epoch:04d}/ {num_epoch:04d} | BATCH {batch:04d} / {num_batch_train:04d} | LOSS {np.mean(loss_arr):.4f}")

            label  = fn_tonumpy(label)
            inputs = fn_tonumpy(fn_denorm(inputs, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image("label", label, num_batch_train * (epoch-1) + batch, dataformats="NHWC")
            writer_train.add_image("input", inputs, num_batch_train * (epoch-1) + batch, dataformats="NHWC")
            writer_train.add_image("output", output, num_batch_train * (epoch-1) + batch, dataformats="NHWC")

        writer_train.add_scalar("loss", np.mean(loss_arr), epoch)

        with torch.no_grad():
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                label = data["label"].to(device)
                inputs =data["input"].to(device)

                output = net(inputs)
                loss = fn_loss(output, label)
                loss_arr += [loss.item()]

                print(f"Train : EPOCH {epoch:04d}/ {num_epoch:04d} | BATCH {batch:04d} / {num_batch_val:04d} | LOSS {np.mean(loss_arr):.4f}")

                label  = fn_tonumpy(label)
                inputs = fn_tonumpy(fn_denorm(inputs, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image("label", label, num_batch_val * (epoch-1) + batch, dataformats="NHWC")
                writer_val.add_image("input", inputs, num_batch_val * (epoch-1) + batch, dataformats="NHWC")
                writer_val.add_image("output", output, num_batch_val * (epoch-1) + batch, dataformats="NHWC")

        writer_train.add_scalar("loss", np.mean(loss_arr), epoch)
        save(ckpt_dir, net, optim, epoch)
        

    writer_train.close()
    writer_val.close()
    