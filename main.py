# from PyQt5.QtWidgets import QApplication
# from test_window import TestWindow
import matplotlib.pyplot as plt
import torch
import time
from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize, ToPILImage, GaussianBlur
from dataset import SR_Dataset, show_pair_of_images
import torch.optim.lr_scheduler as lr_scheduler
from datetime import date
from torch.utils.data import DataLoader
#from models import Bicubic, ESPCN, SRCNN, VDSR
from models.VDSR import VDSR
from models.Bicubic import Bicubic
from models.ESPCN import ESPCN
from PIL import Image
import torch.optim as optim
import torch.nn as nn
import torchvision
from train import train
from validate import validate
from utils import load_checkpoint, plot_psnr, plot_loss, psnr, ssim

# def test_window():
#     app = QApplication([])
#     window = TestWindow()
#     window.show()
#     sys.exit(app.exec_())

def main():
    epochs = 1000
    scale_factor = 3

    #Define datasets
    train_dataset = SR_Dataset(scale_factor=scale_factor, path="./resources/BSDS200/", crop_size=66, mode="train")
    eval_dataset = SR_Dataset(scale_factor=scale_factor, path="./resources/Set5/", crop_size=66, mode="valid")

    #Define dataloaders
    train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=eval_dataset, num_workers=0, batch_size=64, shuffle=False)

    train_loss, val_loss = [], []
    train_psnr, val_psnr = [], []

    start = time.time()

    best_psnr = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ESPCN(scale_factor=scale_factor).to(device)

    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    # optimizer = optim.SGD(model.parameters(),
    #                       lr=1e-2,
    #                       momentum=0.9,
    #                       weight_decay=1e-4)

    #Scaller
    scaler = torch.cuda.amp.GradScaler()

    #Normaly use
    criterion = nn.MSELoss(reduction="mean")

    #Only used for VDSR
    #criterion = nn.MSELoss(reduction="sum")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        train_epoch_loss, train_epoch_psnr = train(model, train_loader, optimizer, criterion, device, scaler)
        val_epoch_loss, val_epoch_psnr = validate(model, valid_loader, optimizer, criterion, device)

        print(f"Train PSNR: {train_epoch_psnr:.3f}")
        print(f"Val PSNR: {val_epoch_psnr:.3f}")

        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)

        train_epoch_psnr > best_psnr

        if (val_epoch_psnr > best_psnr):
            best_psnr = val_epoch_psnr
            today = date.today()

            # dd/mm/YY
            d1 = today.strftime("%d_%m_%Y")
            model_name = model.__class__.__name__
            model_save_name = model_name + "_" + d1 + "_scale_factor_" + str(scale_factor) + "_epochs_" + str(epochs) + "_BSDS200" + ".pt"
            
            #path = F"{model_save_name}" 
            path = "./checkpoints/" + model_save_name

            torch.save({
                    'epoch': epoch,
                    'model_name': model_name,
                    'model_scale_factor': scale_factor,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_epoch_loss': train_epoch_loss,
                    'train_psnr' : train_epoch_psnr,
                    'val_epoch_loss': val_epoch_loss,
                    'val_epoch_psnr': val_epoch_psnr,
                    }, path)
            print("Zapisano nowy checkpoint")

    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes") 

    print("Plot loss...")
    plot_loss(train_loss, val_loss, model_save_name)
    print("Plot psnr...")
    plot_psnr(train_psnr, val_psnr, model_save_name)
    print("Finished !!!")

def test_multiple():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VDSR(scale_factor=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    print("Build VDSR model successfully.")
    checkpoint_path = "./checkpoints/VDSR_23_05_2023_scale_factor_3_epochs_100_DIV2K.pt"
    load_checkpoint(checkpoint_path, model, optimizer)
    print("Loaded model successfully.")
    criterion = nn.MSELoss(reduction="sum")

    '''
    Testowanie modelu na Set5 dataset
    -------------------------------------------------------------------------------------------------------
    '''
    bicubic = Bicubic(scale_factor=3).to(device)
    test_dataset = SR_Dataset(scale_factor=3, path="./resources/Set5/", crop_size=0, mode="test")
    #Remember to use single batch, becouse pytorch requires input tensors to be the same size
    test_loader =  DataLoader(dataset=test_dataset, num_workers=0, batch_size=1, shuffle=False)
    _, test_psnr = validate(model, test_loader, optimizer, criterion, device)
    _, bicubic_psnr = validate(bicubic, test_loader, optimizer, criterion, device)
    print(f"Test Set5 PSNR for model: {test_psnr:.3f}")
    print(f"Test Set5 PSNR for bicubic: {bicubic_psnr:.3f}")
    '''
    -------------------------------------------------------------------------------------------------------
    '''

def test_single_and_compare_3_images(HR_image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESPCN(scale_factor=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    print("Build VDSR model successfully.")
    checkpoint_path = "./checkpoints/ESPCN_scale_factor_3_epochs_20000_BSDS200_adam_scaler.pt"
    load_checkpoint(checkpoint_path, model, optimizer)
    print("Loaded model successfully.")
    criterion = nn.MSELoss(reduction="mean")

    model.eval()
    scale_factor = 3

    data_transform = Compose([ToTensor()]) 
    data_transform_PIL = Compose([ToPILImage()])

    original = Image.open(HR_image_path)

    w, h = original.size

    #TO w przypadku kiedy scale factor = 3, jakoś musi się to zgadzać wtedy
    w_mod_3 = w % scale_factor
    h_mod_3 = h % scale_factor

    w = w - w_mod_3
    h = h - h_mod_3

    #Dodamy cos takiego
    original = original.resize((int(w), int(h)), Image.Resampling.BICUBIC)  

    resize_image = original.resize((int(w/scale_factor), int(h/scale_factor)), Image.Resampling.BICUBIC)  

    original = data_transform(original)

    resize_image = data_transform(resize_image).unsqueeze(0).to(device)
    newHR = model(resize_image)

    bicubic = Bicubic(scale_factor=scale_factor).to(device)
    bicubic_img = bicubic(resize_image)

    torchvision.utils.save_image(newHR, './results/newHR.png')
    torchvision.utils.save_image(bicubic_img, './results/bicubic.png')
    torchvision.utils.save_image(original, './results/original.png')
            
    im1 = Image.open('./results/bicubic.png')
    im2 = Image.open('./results/original.png')
    im3 = Image.open('./results/newHR.png')         

    # Można by dodać jeszcze ssim
    psnr1 = psnr(original, bicubic_img)
    psnr2 = psnr(original, newHR)

    fig, ax = plt.subplots(1, 3, figsize=(15, 10))

    ax[0].imshow(im2)
    ax[0].title.set_text("Original Image")

    ax[1].imshow(im1)
    ax[1].title.set_text("Bicubic")
    ax[1].set_xlabel('psnr: %f' % psnr1)

    today = date.today()
    model_name = model.__class__.__name__
    d1 = today.strftime("%d_%m_%Y")

    ax[2].imshow(im3)
    ax[2].title.set_text(model_name)
    ax[2].set_xlabel('psnr: %f' % psnr2)

    plt.savefig("./results/comparison_3_images_" + model_name + "_" + d1 +"_scale_factor_" + str(scale_factor)  +"_1000_epochs_scaler" + ".png")
    plt.show()
    plt.close()

def test_single_and_compare_4_images(HR_image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VDSR(scale_factor=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    print("Build VDSR model successfully.")
    checkpoint_path = "./checkpoints/VDSR_24_05_2023_scale_factor_3_epochs_300_DIV2K.pt"
    load_checkpoint(checkpoint_path, model, optimizer)
    print("Loaded model successfully.")
    criterion = nn.MSELoss(reduction="sum")

    model.eval()
    scale_factor = 3

    data_transform = Compose([ToTensor()]) 
    data_transform_PIL = Compose([ToPILImage()])

    original = Image.open(HR_image_path)

    w, h = original.size

    #TO w przypadku kiedy scale factor = 3, jakoś musi się to zgadzać wtedy
    w_mod_3 = w % scale_factor
    h_mod_3 = h % scale_factor

    w = w - w_mod_3
    h = h - h_mod_3

    #Dodamy cos takiego
    original = original.resize((int(w), int(h)), Image.Resampling.BICUBIC)  

    resize_image = original.resize((int(w/scale_factor), int(h/scale_factor)), Image.Resampling.BICUBIC)  
    low_resolution = resize_image
    low_resolution = data_transform(low_resolution)

    original = data_transform(original).unsqueeze(0)

    resize_image = data_transform(resize_image).unsqueeze(0).to(device)
    newHR = model(resize_image)

    bicubic = Bicubic(scale_factor=scale_factor).to(device)
    bicubic_img = bicubic(resize_image)

    torchvision.utils.save_image(low_resolution, './results/LRimage.png')
    torchvision.utils.save_image(newHR, './results/newHR.png')
    torchvision.utils.save_image(bicubic_img, './results/bicubic.png')
    torchvision.utils.save_image(original, './results/original.png')
            
    im1 = Image.open('./results/bicubic.png')
    im2 = Image.open('./results/original.png')
    im3 = Image.open('./results/newHR.png')     
    im4 = Image.open('./results/LRimage.png')        

    # Można by dodać jeszcze ssim
    psnr1 = psnr(original, bicubic_img)
    psnr2 = psnr(original, newHR)

    ssim1 = ssim(original, bicubic_img)
    ssim2 = ssim(original, newHR)

    fig, ax = plt.subplots(2, 2, figsize=(7, 7), sharex=True, sharey=True, constrained_layout = True)

    ax[0, 0].imshow(im4)
    ax[0, 0].title.set_text("LR Image")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(im2)
    ax[0, 1].title.set_text("Orignal Image")
    ax[0, 1].axis("off")

    ax[1, 0].imshow(im1)
    ax[1, 0].title.set_text("Bicubic")
    ax[1, 0].set_xlabel('psnr: %f, ssim: %f' % (psnr1, ssim1))

    today = date.today()
    model_name = model.__class__.__name__
    d1 = today.strftime("%d_%m_%Y")

    ax[1, 1].imshow(im3)
    ax[1, 1].title.set_text(model_name)
    ax[1, 1].set_xlabel('psnr: %f, ssim: %f' % (psnr2, ssim2))

    plt.savefig("./results/comparison_4_images" + model_name + "_" + d1 +"_scale_factor_" + str(scale_factor) + "_1000_epochs"  + ".png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    #show_pair_of_images(3, "./resources/Set5/", "train", crop_size=66)
    #main()
    #test_multiple()
    #test_single_and_compare_4_images("./resources/Set5/butterfly.png")
    test_single_and_compare_3_images("./resources/Set5/butterfly.png")