# GAN for stylized/geometric image synthesis - TRAIN
# Dallas Scott (ds4015)


import os
import logging
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

# weights/config
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
BATCH_SIZE = 4
LR = 2e-4
EPOCHS = 100
L1_WEIGHT = 1.0
PALETTE_HEIGHT = 64
FULL_SIZE      = 256
CHECKPOINT_EPOCH = 79


# load training data
class CombinedDataset(Dataset):
    def __init__(self, root_dir, transform=None, palette_transform=None):
        self.A_dir = os.path.join(root_dir, '../datasets/train_A')
        self.B_dir = os.path.join(root_dir, '../datasets/train_B')
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.files = sorted(
            f for f in os.listdir(self.A_dir)
            if f.lower().endswith(valid_exts)
        )
        self.transform = transform
        self.palette_transform = (
            palette_transform or
            transforms.Compose([
                transforms.Resize((PALETTE_HEIGHT, FULL_SIZE)),
                transforms.ToTensor(),
            ])
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        A = Image.open(os.path.join(self.A_dir, fname)).convert('RGB')
        w, h = A.size
        sketch = A.crop((0, 0, w, h - PALETTE_HEIGHT))
        palette = A.crop((0, h - PALETTE_HEIGHT, w, h))
        if self.transform:
            sketch = self.transform(sketch)
            original = self.transform(
                Image.open(os.path.join(self.B_dir, fname.replace('.png','.jpg')))
                .convert('RGB')
            )
        palette = self.palette_transform(palette)

        return sketch, palette, original

# UNet model architecture (generator) 
class ConditionalUNet(nn.Module):
    def __init__(self, base=64, out_ch=3):
        super().__init__()

        self.pal_embed = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(True),
        )

        # encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, base, 4, 2, 1),
            nn.ReLU(True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base*2, base*2, 4, 2, 1),
            nn.BatchNorm2d(base*2),
            nn.ReLU(True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base*2, base*4, 4, 2, 1), 
            nn.BatchNorm2d(base*4), 
            nn.ReLU(True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base*4, base*8, 4, 2, 1), 
            nn.BatchNorm2d(base*8), 
            nn.ReLU(True)
        )

        # decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(base*8, base*4, 4, 2, 1), 
            nn.BatchNorm2d(base*4),
             nn.ReLU(True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base*8, base*2, 4, 2, 1), 
            nn.BatchNorm2d(base*2), 
            nn.ReLU(True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base*4, base, 4, 2, 1), 
            nn.BatchNorm2d(base), 
            nn.ReLU(True)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base, out_ch, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, sketch, palette):
        B, C, H, W = sketch.shape
        e1 = self.enc1(sketch)
        p0 = self.pal_embed(palette)
        p1 = F.interpolate(p0, size=e1.shape[-2:], mode='bilinear', align_corners=False)
        x2 = torch.cat([e1, p1], dim=1)
        e2 = self.enc2(x2)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.dec4(e4)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        out=self.dec1(d2)
        return out


# PatchGAN discriminator
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ic, oc): return nn.Sequential(
            nn.Conv2d(ic, oc, 4, 2, 1), 
            nn.BatchNorm2d(oc),
            nn.LeakyReLU(0.2, True)
        )
        self.net = nn.Sequential(
            block(3, 64),
            block(64, 128),
            block(128, 256),
            nn.Conv2d(256, 1, 4, 1, 1)
        )
    def forward(self, img):
        return self.net(img)

sobel_x = torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=torch.float32, device=DEVICE)
sobel_y = sobel_x.t()

sobel_x = sobel_x.view(1,1,3,3)
sobel_y = sobel_y.view(1,1,3,3)

def sobel_edges(img):
    gray = img.mean(dim=1, keepdim=True)
    ex = F.conv2d(gray, sobel_x, padding=1)
    ey = F.conv2d(gray, sobel_y, padding=1)
    return torch.sqrt(ex*ex + ey*ey + 1e-6)

# train function
def train():
    logging.basicConfig(level=logging.INFO)
    transform = transforms.Compose([
        transforms.Resize((256,256)), transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)
    ])
    ds = CombinedDataset('.', transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    G = ConditionalUNet().to(DEVICE)
    D = PatchDiscriminator().to(DEVICE)
    optG = optim.Adam(G.parameters(), lr=LR, betas=(0.5,0.999))
    optD = optim.Adam(D.parameters(), lr=LR, betas=(0.5,0.999))
    criterionGAN = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    for epoch in range(EPOCHS):
        for sketch, palette, real in loader:
            sketch, palette, real = [t.to(DEVICE) for t in (sketch,palette,real)]
            
            # discriminator
            optD.zero_grad()
            fake = G(sketch, palette).detach()
            real_pred = D(real)
            fake_pred = D(fake)
            lossD = (criterionGAN(real_pred, torch.ones_like(real_pred)) +
                     criterionGAN(fake_pred, torch.zeros_like(fake_pred))) * 0.5
            lossD.backward(); optD.step()
            
            # generator
            optG.zero_grad()
            fake = G(sketch, palette)
            pred = D(fake)
            
            lossG_gan = criterionGAN(pred, torch.ones_like(pred))
            lossG_l1  = L1_WEIGHT * l1_loss(fake, real)
            tv = torch.mean(torch.abs(fake[:,:,:,1:] - fake[:,:,:,:-1])) \
            + torch.mean(torch.abs(fake[:,:,1:,:] - fake[:,:,:-1,:]))
            
            EDGE_WEIGHT = 5.0
            edge_real = sobel_edges(sketch)
            edge_fake = sobel_edges(fake)

            loss_edge = F.l1_loss(edge_fake, edge_real)

            lossG = lossG_gan + lossG_l1 + 1e-4 * tv + EDGE_WEIGHT * loss_edge
            
            lossG.backward(); optG.step()

        os.makedirs('models', exist_ok=True)
        gen_path = f"models/generator_epoch_{epoch}.pth"
        dis_path = f"models/discriminator_epoch_{epoch}.pth"
        torch.save(G.state_dict(), gen_path)
        torch.save(D.state_dict(), dis_path)
        logging.info(f'Epoch {epoch} done: D={lossD.item():.4f}, G={lossG.item():.4f}')

if __name__=='__main__':
    train()