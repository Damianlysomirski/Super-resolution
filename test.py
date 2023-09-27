from utils import psnr
from tqdm import tqdm
from dataset import SR_Dataset
from torch.utils.data import DataLoader
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from models.VDSR import VDSR
from models.SRCNN import SRCNN
from models.ESPCN import ESPCN
from models.Bicubic import Bicubic
from validate import validate
from utils import psnr, plot_psnr_new, plot_loss_new
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize, ToPILImage, GaussianBlur
from train import build_model, define_optimizer, define_criterion
import torchvision
import matplotlib.pyplot as plt
from torchmetrics import StructuralSimilarityIndexMeasure 

BUTTERFLY = './resources/Set5/butterfly.png'

def test_single_image(model, model_name, scale_factor, image):
    model.eval()
    original = Image.open(image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w, h = original.size
    w_mod_3 = w % scale_factor
    h_mod_3 = h % scale_factor
    w = w - w_mod_3
    h = h - h_mod_3
    data_transform = Compose([ToTensor()]) 
    original = original.resize((int(w), int(h)), Image.Resampling.BICUBIC)
    resize_image = original.resize((int(w/scale_factor), int(h/scale_factor)), Image.Resampling.BICUBIC) 
    original = data_transform(original)
    resize_image = data_transform(resize_image).unsqueeze(0).to(device)
    predicted = model(resize_image).squeeze(0)
    bicubic = Bicubic(scale_factor=scale_factor).to(device)
    bicubic = bicubic(resize_image).squeeze(0)

    psnr1 = psnr(original, bicubic)
    psnr2 = psnr(original, predicted)

    ssim1 = StructuralSimilarityIndexMeasure(original, bicubic)
    print(ssim1)
    
    torchvision.utils.save_image(bicubic, './results/bicubic.png')
    torchvision.utils.save_image(predicted, './results/predicted.png')
    torchvision.utils.save_image(original, './results/original.png')
            
    bicubic = Image.open('./results/bicubic.png')
    predicted = Image.open('./results/predicted.png')
    original = Image.open('./results/original.png')     

    #Convert to PIL from Tensor- NOT WORKING
    # predicted = torchvision.transforms.functional.to_pil_image(predicted)
    # bicubic = torchvision.transforms.functional.to_pil_image(bicubic_img)
    # original = torchvision.transforms.functional.to_pil_image(original)
    
    #Plot figure
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    ax[0].imshow(original)
    ax[0].title.set_text("Original Image")
    ax[1].imshow(bicubic)
    ax[1].title.set_text("Bicubic")
    ax[1].set_xlabel('psnr: %f' % psnr1)
    ax[2].imshow(predicted)
    ax[2].title.set_text(model_name)
    ax[2].set_xlabel('psnr: %f' % psnr2)
    plt.show()

    #Dodać jeszcze zapisywanie plotu
                               
"""
Function to build model depending on parsing arguments
"""
def build_model(model, scale_factor, device):
    if (model == "VDSR"):
        model = VDSR(scale_factor).to(device)
        print("Built model VDSR successfully !")
    elif (model == "SRCNN"):
        model = SRCNN(scale_factor).to(device)
        print("Built model SRCNN successfully !")
    elif (model == "ESPCN"):
        model = ESPCN(scale_factor).to(device)
        print("Built model ESPCN successfully !")
    else:
        raise ValueError(
            "Unsupported neural network model, please use 'VDSR', 'SRCNN' or 'ESPCN'.")
    return model

def main () -> None:
    parser = argparse.ArgumentParser(
        prog="Super-resolution testing",
        description="Program allows training 5 different neural networks for SR usage",
        epilog="Example usage: python test.py -m SRCMM -s 3  -p './checkpoints_new/SRCNN_sf_3_epoch_3.pt' ")
    parser.add_argument(
        "-m",
        "--model",
        help="Expected model names: SRCNN, VDSR, ESPCN",
        required=True,
        default="ESPCN"
    )
    parser.add_argument(
    "-s",
    "--scale",
    help="Value of scale factor",
    required=True,
    type=int,
    default=3
    )
    parser.add_argument(
        "-p",
        "--checkpoint_path",
        help="Path to checkpoint for testing",
        default=None,
        required=True
    )

    args = parser.parse_args()

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model
    model = build_model(args.model, args.scale, device)

     # Define loss criterion
    criterion = define_criterion(args.model)

    # Define optimizer and scheduler
    #optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = define_optimizer(args.model, model)

    try:
        #Nie wiem czy tutaj wszystko tak naprawde jest nam potrzebne 
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        scale_factor = checkpoint['model_scale_factor']
        model_name = checkpoint['model_name']
        train_loss = checkpoint['train_loss']
        train_psnr = checkpoint['train_psnr']
        val_loss_dict = checkpoint['val_loss']
        val_psnr_dict = checkpoint['val_psnr']
        best_valid_loss = checkpoint['best_valid_loss']

    except ValueError:
        print("Oops! Passed wrong path to the checkpoint, check it and try again")

    print("Successfully loaded model !")
    plot_loss_new(train_loss, val_loss_dict, str(args.model) + "_scale_factor: " + str(args.scale))
    plot_psnr_new(train_psnr, val_psnr_dict, str(args.model) + "_scale_factor: " + str(args.scale))
    test_single_image(model, args.model, args.scale, BUTTERFLY)

if __name__ == "__main__":
    main()
