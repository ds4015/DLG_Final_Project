# GAN for stylized/geometric image synthesis - TEST
# Dallas Scott (ds4015)

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms
import torch.nn.functional as F
from contour_gan import ConditionalUNet

# weights/config
DEVICE           = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
CHECKPOINT_EPOCH = 79
CHECKPOINT       = "../models/main_generator_epoch_99.pth"
INPUT_DIR        = '../aic'
ORIG_DIR         = '../art_institute_drawings'
OUT_DIR          = '../results/aic'
PALETTE_HEIGHT   = 64
FULL_SIZE        = 256


img_tf = transforms.Compose([
    transforms.Resize((FULL_SIZE, FULL_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,)*3, (0.5,)*3),
])

pal_tf = transforms.Compose([
    transforms.Resize((PALETTE_HEIGHT, FULL_SIZE)),
    transforms.ToTensor(),
])

# load model
os.makedirs(OUT_DIR, exist_ok=True)
gen = ConditionalUNet().to(DEVICE)
ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
gen.load_state_dict(ckpt, strict=True)
gen.eval()

# test
for fn in sorted(os.listdir(INPUT_DIR)):
    if not fn.lower().endswith(('.png','.jpg','.jpeg')):
        continue

    im = Image.open(os.path.join(INPUT_DIR, fn)).convert('RGB')
    w, h = im.size
    sketch  = im.crop((0, 0, w, h - PALETTE_HEIGHT))
    palette = im.crop((0, h - PALETTE_HEIGHT, w, h))

    # output as 3-panel grid
    panel1 = sketch.resize((FULL_SIZE, FULL_SIZE))

    sk_t  = img_tf(sketch).unsqueeze(0).to(DEVICE)
    pal_t = pal_tf(palette).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        fake = gen(sk_t, pal_t)

    fake = (fake * 0.5 + 0.5).clamp(0,1)
    fake_np = (fake[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
    pred    = Image.fromarray(fake_np).resize((FULL_SIZE, FULL_SIZE))

    # overlay sketch lines on synth
    lines = sketch.convert('L').resize((FULL_SIZE, FULL_SIZE))
    edges = lines.filter(ImageFilter.FIND_EDGES)
    mask  = edges.point(lambda x: 255 if x>128 else 0)
    mask  = mask.filter(ImageFilter.MinFilter(3))
    pred.paste((0,0,0), mask)

    panel2 = pred

    base, _ = os.path.splitext(fn)
    orig_path = os.path.join(ORIG_DIR, fn)
    if not os.path.exists(orig_path):
        orig_path = os.path.join(ORIG_DIR, base + '.jpg')
    panel3 = Image.open(orig_path).convert('RGB').resize((FULL_SIZE, FULL_SIZE))

    # form grid
    grid = Image.new('RGB', (FULL_SIZE*3, FULL_SIZE))
    for i, pnl in enumerate((panel1, panel2, panel3)):
        grid.paste(pnl, (i*FULL_SIZE, 0))

    out_fn = f"{base}_synthesized.png"
    grid.save(os.path.join(OUT_DIR, out_fn))
    print(f"Saved {out_fn} to {OUT_DIR}")

print("Inference complete.")