import torch
from utils import psnr
from tqdm import tqdm

def validate(model, dataloader, optimizer, criterion, device):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0

    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        #for bi, data in enumerate(dataloader):
            image_data = data[0].to(device)
            label = data[1].to(device)
            
            outputs = model(image_data)
            loss = criterion(outputs, label)
            # add loss of each item (total items in a batch = batch size) 
            running_loss += loss.item()
            # calculate batch psnr (once every `batch_size` iterations)
            batch_psnr = psnr(label, outputs)
            running_psnr += batch_psnr

    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/len(dataloader)

    return final_loss, final_psnr