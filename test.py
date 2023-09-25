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
from utils import psnr

from train import build_model, define_optimizer, define_criterion

BUTTERFLY = './resources/Set5/butterfly.png'

def test_single_image(model):
    model.eval()
    pass

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
        model == ESPCN(scale_factor).to(device)
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
        val_loss = checkpoint['val_loss']
        val_psnr = checkpoint['val_psnr']
        best_valid_loss = checkpoint['best_valid_loss']

    except ValueError:
        print("Oops! Passed wrong path to the checkpoint, check it and try again")


if __name__ == "__main__":
    main()
