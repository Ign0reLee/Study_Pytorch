import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import torch.distributed as dist

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x), None, None

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        return self.conv(x), None, None

class Conv3dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.conv(x)

class AutoencoderKL_MP(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=256, latent_channels=4, dev0=0, dev1=1):
        super().__init__()
        self.dev0 = dev0
        self.dev1 = dev1

        self.encoder = Encoder(in_channels, latent_channels).to(f'cuda:{self.dev0}')
        self.decoder = Decoder(latent_channels, out_channels).to(f'cuda:{self.dev1}')
        self.quant_conv = Conv3dLayer(2 * latent_channels, 2 * latent_channels, 1).to(f'cuda:{self.dev0}')
        self.post_quant_conv = Conv3dLayer(latent_channels, latent_channels, 1).to(f'cuda:{self.dev1}')

    def encode(self, x):
        h, _, _ = self.encoder(x)
        moments = self.quant_conv(h)
        return moments

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec, _, _ = self.decoder(z)
        return dec

    def forward(self, x):
        x = x.to(f'cuda:{self.dev0}')
        moments = self.encode(x)
        z = moments.to(f'cuda:{self.dev1}')
        dec = self.decode(z)
        return dec

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(f'cuda:{self.dev0}')
        y = y.to(f'cuda:{self.dev1}')
        output = self(x)
        loss = nn.functional.mse_loss(output, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index], self.data[index]

    def __len__(self):
        return self.len

class RandomDataModule(pl.LightningDataModule):
    def __init__(self, size, length, batch_size):
        super().__init__()
        self.size = size
        self.length = length
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = RandomDataset(self.size, self.length)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size)

class CustomDDPStrategy(DDPStrategy):
    def setup_environment(self):
        world_size = self.world_size
        rank = self.local_rank
        print("world_size",world_size, "rank",rank)

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank = self.local_rank,
            world_size= self.world_size
        )

        group_0 = [0, 1]
        group_1 = [2, 3]
        if rank in group_0:
            self.process_group = dist.new_group(ranks=group_0)
        elif rank in group_1:
            self.process_group = dist.new_group(ranks=group_1)
        else:
            raise ValueError(f"Rank {rank} is not in any group")

        self.local_rank = rank % 2

        print(f"Global rank {rank} in process group {self.process_group}, local rank {self.local_rank}")

    def configure_ddp(self):
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            process_group=self.process_group
        )

    def setup(self, trainer):
        self.setup_environment()
        self.configure_ddp()
        trainer.model = self.model



if __name__ == "__main__":
    model = AutoencoderKL_MP()
    data_module = RandomDataModule(size=32, length=10000, batch_size=32)

    trainer = Trainer(
        strategy=CustomDDPStrategy(find_unused_parameters=False),
        accelerator="gpu",
        devices=[0, 1],  # Specify all 4 GPUs
        num_nodes=1,
        max_epochs=10,
    )

    trainer.fit(model, data_module)