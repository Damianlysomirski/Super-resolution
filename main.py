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

# def test_window():
#     app = QApplication([])
#     window = TestWindow()
#     window.show()
#     sys.exit(app.exec_())

def main():
    epochs = 0
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

    







def check_device():
    print(torch.cuda.is_available())
    print("\n")
    print(torch.cuda.current_device())
    print("\n")
    print(torch.cuda.device(0))
    print("\n")
    print(torch.cuda.device_count())
    print("\n")
    print(torch.cuda.get_device_name(0))


if __name__ == "__main__":
    #show_pair_of_images(3, "./resources/BSDS200/", "valid", "132")
    main()