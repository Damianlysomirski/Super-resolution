#from utils import psnr
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

def train(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    
    for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
    #for bi, data in enumerate(dataloader):
        image_data = data[0].to(device)
        label = data[1].to(device)
        
        # Zero grad the optimizer.
        optimizer.zero_grad()
        
        #Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(image_data)
            loss = criterion(outputs, label)
        
        # Backpropagation.
        #loss.backward()
        # Update the parameters.
        #optimizer.step()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Add loss of each item (total items in a batch = batch size).
        running_loss += loss.item()
        
        # Calculate batch psnr (once every `batch_size` iterations).
        batch_psnr =  psnr(label, outputs)
        running_psnr += batch_psnr

        #Plot 1 image from batch
        # output = outputs[0].detach().cpu().numpy() 
        # output = output.transpose(1,2,0) #konwersja tensora na ndarray
        # plt.imshow(output)
        # plt.show()
        
    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/len(dataloader)
    return final_loss, final_psnr


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Super-resolution training",
        description="Program allows training 5 different neural networks for SR usage",
        epilog="Example usage: python train.py -m VDSR -s 3 -e 10", )
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
        "-e",
        "--epochs",
        help="Number of training epochs",
        required=True,
        type=int,
        default=100
    )
    args = parser.parse_args()

    # Define the number of training epochs
    epochs = args.epochs
    print("Defined training epochs: " + str(epochs))

    #Define scale factor
    scale_factor = args.scale
    print("Defined scale_factor: " + str(scale_factor))

    #Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Define model
    model = build_model(args.model, args.scale, device)

    #Define datasets
    train_dataset = SR_Dataset(scale_factor=scale_factor, path="./resources/DIV2K/", crop_size=scale_factor*22, mode="train")
    eval_dataset = SR_Dataset(scale_factor=scale_factor, path="./resources/Set5/", crop_size=scale_factor*22, mode="valid")

    #Define dataloaders
    train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=eval_dataset, num_workers=0, batch_size=64, shuffle=False)

    #Define loss criterion
    criterion = define_criterion(args.model)

    #Define optimizer and scheduler

def build_model(model, scale_factor, device):
    if(model == "VDSR"):
        model = VDSR(scale_factor).to(device)
        print("Built model VDSR successfully !")
    elif(model == "SRCNN"):
        model = SRCNN(scale_factor).to(device)
        print("Built model SRCNN successfully !")
    elif(model == "ESPCN"):
        model == ESPCN(scale_factor).to(device)
        print("Built model ESPCN successfully !")
    else:
        raise ValueError("Unsupported neural network model, please use 'VDSR', 'SRCNN' or 'ESPCN'.")

    return model

def define_criterion(model):
    if(model == "VDSR"):
        criterion = nn.MSELoss(reduction="sum")
    else:
        criterion = nn.MSELoss
    return criterion

def define_optimizer_and_scheduler(model):
    if(model == "VDSR"):
        #TODO
        pass
    elif(model == "SRCNN"):
        #TODO
        pass
    elif(model == "ESPCN"):
        optimizer = optim.SGD(model.parameters(),
                          lr=1e-2,
                          momentum=-0.9,
                          weight_decay=1e-4)

        return optimizer

if __name__ == "__main__":
    main()

