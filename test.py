import torch
import argparse
import models
from utils import *
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from train import build_model
import torchvision
import matplotlib.pyplot as plt

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
    bicubic = models.Bicubic(scale_factor=scale_factor).to(device)
    bicubic = bicubic(resize_image).squeeze(0)

    psnr1 = psnr(original, bicubic)
    psnr2 = psnr(original, predicted)

    ssim1 = ssim(original, bicubic)
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

    try:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        scale_factor = checkpoint['model_scale_factor']
        model_name = checkpoint['model_name']
        train_loss = checkpoint['train_loss']
        train_psnr = checkpoint['train_psnr']
        val_loss_dict = checkpoint['val_loss']
        val_psnr_dict = checkpoint['val_psnr']

    except RuntimeError:
        print("!!! ERROR !!!")
        print("Oops! Passed wrong path to the checkpoint, check it and try again !!!")
        print("Check value difference between the given scale_factor in the path and the given one in arg_parse !!!")
        print("!!! ERROR !!!")

    print("Successfully loaded model: " + str(model_name))
    plot_loss_new(train_loss, val_loss_dict, str(args.model) + "_scale_factor: " + str(args.scale))
    plot_psnr_new(train_psnr, val_psnr_dict, str(args.model) + "_scale_factor: " + str(args.scale))
    test_single_image(model, args.model, args.scale, BUTTERFLY)

if __name__ == "__main__":
    main()
