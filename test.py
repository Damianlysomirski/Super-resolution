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
import time
from torchsummary import summary

COMIC = './resources/Set14/comic.png'
BUTTERFLY = './resources/Set5/butterfly.png'
ZEBRA = './resources/Set14/zebra.png'
BABY = './resources/Set5/baby.png'
BIRD = './resources/Set5/bird.png'

def test_set(model, scale_factor, optimizer, criterion, device, set_name):
    model.eval()
    summary(model, (3, 48, 48))
    bicubic = Bicubic(scale_factor=scale_factor).to(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = SR_Dataset(scale_factor=scale_factor, path=f"./resources/{set_name}/", crop_size=0, mode="test")
    test_loader =  DataLoader(dataset=test_dataset, num_workers=0, batch_size=1, shuffle=False)
    _, test_psnr, test_ssim = validate(model, test_loader, optimizer, criterion, device)
    _, bicubic_psnr, bicubic_ssim = validate(bicubic, test_loader, optimizer, criterion, device)
    print(f"Test {set_name} PSNR for model: {test_psnr:.3f}")
    print(f"Test {set_name} PSNR for bicubic: {bicubic_psnr:.3f}")
    print(f"Test {set_name} SSIM for model: {test_ssim:.3f}")
    print(f"Test {set_name} SSIM for bicubic: {bicubic_ssim:.3f}")

def test_image(model, path):
    torch.cuda.empty_cache()
    model.eval()
    device = torch.device("cpu")
    model.to(device)
    data_transform = Compose([ToTensor()])
    img = Image.open(path)
    img = data_transform(img).unsqueeze(0).to(device)
    predicted = model(img)
    torchvision.utils.save_image(predicted, './predict.png')
    

def test_single_image(model, model_name, scale_factor, image):
    model.eval()
    original = Image.open(image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w, h = original.size
    w_mod = w % scale_factor
    h_mod = h % scale_factor
    w = w - w_mod
    h = h - h_mod
    data_transform = Compose([ToTensor()]) 
    original = original.resize((int(w), int(h)), Image.Resampling.BICUBIC)
    resize_image = original.resize((int(w/scale_factor), int(h/scale_factor)), Image.Resampling.BICUBIC) 

    original = data_transform(original).unsqueeze(0)
    resize_image = data_transform(resize_image).unsqueeze(0).to(device)
    predicted = model(resize_image)
    bicubic = models.Bicubic(scale_factor=scale_factor).to(device)
    bicubic = bicubic(resize_image)
    
    psnr1 = psnr(original, bicubic)
    psnr2 = psnr(original, predicted)
    ssim1 = ssim(original, bicubic)
    ssim2 = ssim(original,predicted)
    
    torchvision.utils.save_image(resize_image, './results/resize_image.png')
    torchvision.utils.save_image(bicubic, './results/bicubic.png')
    torchvision.utils.save_image(predicted, './results/predicted.png')
    torchvision.utils.save_image(original, './results/original.png')
            
    bicubic = Image.open('./results/bicubic.png')
    predicted = Image.open('./results/predicted.png')
    original = Image.open('./results/original.png')
    resize_image = Image.open('./results/resize_image.png')

    #Plot comparison 3
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(original)
    ax[0].title.set_text("Obraz oryginalny")

    # inset axes....
    axin1 = ax[0].inset_axes([0.5, 0.5, 0.47, 0.47], xlim=(100, 150), ylim=(200, 250))
    axin1.imshow(original)
    ax[0].indicate_inset_zoom(axin1, edgecolor="blue")
    axin1.axis("off")


    ax[1].imshow(bicubic)
    ax[1].title.set_text("Interpolacja dwusześcienna")
    ax[1].set_xlabel('psnr: %f, ssim: %f' % (psnr1, ssim1))

    # inset axes....
    axin2 = ax[1].inset_axes([0.5, 0.5, 0.47, 0.47], xlim=(100, 150), ylim=(200, 250))
    axin2.imshow(bicubic)
    ax[1].indicate_inset_zoom(axin2, edgecolor="blue")
    axin2.axis("off")


    ax[2].imshow(predicted)
    ax[2].title.set_text(model_name)
    ax[2].set_xlabel('psnr: %f, ssim: %f' % (psnr2, ssim2))

    # inset axes....
    axin3 = ax[2].inset_axes([0.5, 0.5, 0.47, 0.47], xlim=(100, 150), ylim=(200, 250))
    axin3.imshow(predicted)
    ax[2].indicate_inset_zoom(axin3, edgecolor="blue")
    axin3.axis("off")

    plt.savefig("./results/comparison_3_images")

    #Plot comaprision 4
    fig, ax = plt.subplots(2, 2, figsize=(7, 7), sharex=True, sharey=True, constrained_layout = True)
    ax[0, 0].imshow(resize_image)
    ax[0, 0].title.set_text("Obraz niskorozdzielczościowy")
    ax[0, 1].imshow(original)
    ax[0, 1].title.set_text("Obraz oryginalny")
    ax[1, 0].imshow(bicubic)
    ax[1, 0].title.set_text("Interpolacja dwusześcienna")
    ax[1, 0].set_xlabel('psnr: %f, ssim: %f' % (psnr1, ssim1))
    ax[1, 1].imshow(predicted)
    ax[1, 1].title.set_text(model_name)
    ax[1, 1].set_xlabel('psnr: %f, ssim: %f' % (psnr2, ssim2))
    plt.savefig("./results/comparison_4_images")
    
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

    #Defime optimizer

    #Define criterion 
    criterion = define_criterion(args.model)

    # Define optimizer and scheduler
    optimizer = define_optimizer(args.model, model)

    try:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        scale_factor = checkpoint['model_scale_factor']
        model_name = checkpoint['model_name']
        train_loss = checkpoint['train_loss']
        train_psnr = checkpoint['train_psnr']
        val_loss_dict = checkpoint['val_loss']
        val_psnr_dict = checkpoint['val_psnr']
        train_time = checkpoint['train_time']

    except RuntimeError:
        print("!!! ERROR !!!")
        print("Oops! Passed wrong path to the checkpoint, check it and try again !!!")
        print("Check value difference between the given scale_factor in the path and the given one in arg_parse !!!")
        print("!!! ERROR !!!")
    
    print("Successfully loaded model: " + str(model_name))
    print("Total training time: " + time.strftime("%Hh%Mm%Ss", time.gmtime(train_time)))
    plot_loss_new(train_loss, val_loss_dict, str(args.model) + "_scale_factor: " + str(args.scale))
    plot_psnr_new(train_psnr, val_psnr_dict, str(args.model) + "_scale_factor: " + str(args.scale))
    
    test_single_image(model, args.model, args.scale, BIRD)

    test_set(model, scale_factor, optimizer, criterion, device, "Set5")
    test_set(model, scale_factor, optimizer, criterion, device, "Set14")

if __name__ == "__main__":
    main()

