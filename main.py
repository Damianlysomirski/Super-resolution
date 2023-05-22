# from PyQt5.QtWidgets import QApplication
# from test_window import TestWindow

import torch
import time

from dataset import SR_Dataset
import torch.optim.lr_scheduler as lr_scheduler
from datetime import date
from torch.utils.data import DataLoader
#from models import Bicubic, ESPCN, SRCNN, VDSR
from models.VDSR import VDSR

import torch.optim as optim
import torch.nn as nn

from train import train
from validate import validate

# def test_window():
#     app = QApplication([])
#     window = TestWindow()
#     window.show()
#     sys.exit(app.exec_())

def main():
    epochs = 100
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

    model = VDSR(scale_factor=scale_factor).to(device)

    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)

    #Normaly use
    #criterion = nn.MSELoss

    #Only used for VDSR
    criterion = nn.MSELoss(reduction="sum")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        train_epoch_loss, train_epoch_psnr = train(model, train_loader, optimizer, criterion, device)
        val_epoch_loss, val_epoch_psnr = validate(model, train_loader, optimizer, criterion, device)

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
            model_save_name = model_name + "_" + d1 + "_scale_factor_" + str(scale_factor) + "_epochs_" + str(epochs) + ".pt"
            
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

if __name__ == "__main__":
    #show_pair_of_images(3, "./resources/BSDS200/", "valid", "132")
    main()
