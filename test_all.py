import torch
import argparse
import models
from utils import *
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from train import build_model, define_criterion, define_optimizer
import torchvision
import matplotlib.pyplot as plt
from dataset import SR_Dataset
from torch.utils.data import DataLoader
from validate import validate
from models.Bicubic import Bicubic


#Path's to trained model checkpoints
TRAINED_MODELS = { 2:
{
    "SRCNN": "./trained_models/SRCNN_2",
    "ESPCN": "./trained_models/ESPCN_2",
    "VDSR": "./trained_models/VDSR_2",
    "SRResNet": "./trained_models/SRRESNET_2"
}, 3:
{
    "SRCNN": "./trained_models/SRCNN_3",
    "ESPCN": "./trained_models/ESPCN_3",
    "VDSR": "./trained_models/VDSR_3",
    "SRResNet": "./trained_models/SRRESNET_3"
}, 4:
{
    "SRCNN": "./trained_models/SRCNN_4",
    "ESPCN": "./trained_models/ESPCN_4",
    "VDSR": "./trained_models/VDSR_4",
    "SRResNet": "./trained_models/SRRESNET_4"
}}
    
def test_single(model, model_name, path):
    torch.cuda.empty_cache()
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    data_transform = Compose([ToTensor()])
    img = Image.open(path)
    img = data_transform(img).unsqueeze(0).to(device)
    predicted = model(img)
    file_path = f'./results_all/{model_name}.png'
    torchvision.utils.save_image(predicted, file_path)
    print(f'Prediction for model: {model_name} finished,saved to results_all !')

def main () -> None:
    parser = argparse.ArgumentParser(
        prog="Super-resolution testing",
        description="Program allows testing 4 different neural networks + bicubic for SR usage",
        epilog="Example usage: python test_all.py --s 3  -p './checkpoints_new/SRCNN_sf_3_epoch_3.pt' ")
    parser.add_argument(
    "-s",
    "--scale",
    help="Value of scale factor",
    required=True,
    type=int,
    default=3
    )
    parser.add_argument(
        "-input",
        "--small_image",
        help="Path for low-resolution image",
        required=True
    )

    args = parser.parse_args()
    
    image = args.small_image
    scale = args.scale

    for model_name, checkpoint in TRAINED_MODELS[scale].items():       
        model = torch.load(checkpoint)
        test_single(model, model_name, image)

    bicubic = Bicubic(scale_factor=scale)
    test_single(bicubic, "bicubic", image)

if __name__ == "__main__":
    main()



