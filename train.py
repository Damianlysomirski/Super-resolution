from utils import psnr
from tqdm import tqdm

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    
    for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
    #for bi, data in enumerate(dataloader):
        image_data = data[0].to(device)
        label = data[1].to(device)
        
        # Zero grad the optimizer.
        optimizer.zero_grad()

        outputs = model(image_data)
        loss = criterion(outputs, label)
        # Backpropagation.
        loss.backward()
        # Update the parameters.
        optimizer.step()
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