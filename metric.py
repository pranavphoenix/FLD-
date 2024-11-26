import os
import argparse, math

## PyTorch
import torch
import torch.utils.data as data

# Torchvision
from torchinfo import summary
from model import ImageFlow
from data import CustomImageDataset, ToTensorTransform

parser = argparse.ArgumentParser()

parser.add_argument("-real", "--real_data_dir", default='/workspace/Spline/code/Data/real/', help = "Path to Real Images Data Directory")
parser.add_argument("-gen", "--gen_data_dir", default='/workspace/Spline/code/Data/noise/', help = "Path to Real Images Data Directory")
parser.add_argument("-batch", "--Batch_size", default=256, help = "Batch Size")
parser.add_argument("-path", "--save_flow_path", default='/workspace/Spline/code/flow.pth', help= "Path to save the trained Flow model")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)


n_flow = ImageFlow()

n_flow.to(device)
print(summary(n_flow, (1, 3, 256, 256), depth = 10))

if os.path.isfile(str(args.save_flow_path)):
    print("Found pretrained n_flow, loading...")
    n_flow.load_state_dict(torch.load(str(args.save_flow_path)))
else:
    print("No saved weights found, use train.py first to train the flow on real images")
    exit()


n_flow.eval()

real_image_folder = str(args.real_data_dir)
gen_image_folder = str(args.gen_data_dir)

transform = ToTensorTransform()

real_dataset = CustomImageDataset(image_folder=real_image_folder, transform=transform)
gen_dataset = CustomImageDataset(image_folder=gen_image_folder, transform=transform)

print(f"No. of Images in Real Data set: {len(real_dataset)}")
print(f"No. of Images in Generated Data set: {len(gen_dataset)}")

# real_dataset = data.Subset(real_dataset, list(range(2000)))
real_data_loader = data.DataLoader(real_dataset, batch_size=int(args.Batch_size), shuffle=False, drop_last=True, pin_memory=True, num_workers=8)
gen_data_loader = data.DataLoader(gen_dataset, batch_size=int(args.Batch_size), shuffle=False, drop_last=True, pin_memory=True, num_workers=8)
# gen_data_loader = data.DataLoader(gen_dataset, batch_size=100, shuffle=False, drop_last=True, pin_memory=True, num_workers=8)

real_log_likelihood = 0
total_count = 0
with torch.no_grad():
    for batch in real_data_loader:
        imgs = batch
        imgs = imgs.to(device)
        real_log_likelihood += n_flow(imgs).sum()
        batch_count = batch.numel()
        total_count += batch_count


    real_log_likelihood = real_log_likelihood / total_count

print(f"Average Log Likelihood of Real Images: {real_log_likelihood}")
print(f"Average Likelihood of Real Images: {math.exp(real_log_likelihood)}")



gen_log_likelihood = 0
total_count = 0

with torch.no_grad():
    for batch in gen_data_loader:
        imgs = batch
        imgs = imgs.to(device)
        gen_log_likelihood += n_flow(imgs).sum()
        batch_count = batch.numel()
        total_count += batch_count


    gen_log_likelihood = gen_log_likelihood / total_count

print(f"Average Log Likelihood of Generated Images: {gen_log_likelihood}")
print(f"Average Likelihood of Generated Images: {math.exp(gen_log_likelihood)}")

FLD_plus =  gen_log_likelihood / real_log_likelihood

FLD_plus = math.exp(FLD_plus)

print(f"FLD+: {FLD_plus:.3f}")