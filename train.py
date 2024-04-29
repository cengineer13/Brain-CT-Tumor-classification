import os 
import tqdm
import torch
import numpy as np

def training(model, epoch, train_dataloader, criterion, optimizer, device):
    
    model.train()
       
    loss_per_batch = []
    total_acc = 0
    
    print(f"{epoch+1} - Training started ...".upper())
    for images, labels in tqdm.tqdm(train_dataloader):

        images = images.to(device)
        labels = labels.to(device)
        #predict 
        prediction = model(images)
        preds = torch.argmax(prediction, dim=1)
        acc_per_batch = (preds == labels).sum().item()
        total_acc += acc_per_batch #add each batch acc
        
        loss = criterion(prediction, labels)
        loss_per_batch.append(loss.item()) # add each loss of batch 
        optimizer.zero_grad()
        #loss and update weights 
        loss.backward()
        optimizer.step()
        
    epoch_loss = np.mean(loss_per_batch) 
    average_acc = total_acc / len(train_dataloader.dataset)
 
    print(f"Epoch - {epoch+1} ||||| Loss: {epoch_loss:.3f} |||| Accuracy: {average_acc:.3f}")
    print('Training finished !!!!!!!! \n')

    return average_acc, epoch_loss #average acc and loss 
    

def validationing(model, epoch, valid_dataloader, criterion, device): 
    print(f"{epoch+1} - Validation started.....".upper())
    model.eval() 

    loss_per_batch = []
    total_acc = 0
    
    with torch.no_grad():

        for images, labels in tqdm.tqdm(valid_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            #predict 
            prediction = model(images)
            preds = torch.argmax(prediction, dim=1)
            acc_per_batch = (preds== labels).sum().item()
            total_acc += acc_per_batch #add each batch accuracy

            loss = criterion(prediction, labels)
            loss_per_batch.append(loss.item()) #Add each batch loss to one list 
        
        epoch_loss = np.mean(loss_per_batch) 
        average_acc = total_acc / len(valid_dataloader.dataset)
        
        print(f"Epoch - {epoch+1} ||||| Loss: {epoch_loss:.3f} |||| Accuracy: {average_acc:.3f}")
        
    print('Validation finished !!!!!!!!\n')   

    return average_acc, epoch_loss #average acc and loss 
        


def train_validation(model, epochs, train_dl, val_dl, criterion, optimizer, device, save_dir, save_prefix):

    model = model.to(device)
    
    train_acc_list = []
    val_acc_list = []
    train_loss_list = []
    val_loss_list = []
    
    acc = 0
    for epoch in range(epochs):

        train_acc, train_loss = training(model, epoch,  train_dl, criterion, optimizer, device)
        val_acc, val_loss = validationing(model, epoch, val_dl, criterion, device)
        
        #Get metrics for plotting 
        train_acc_list.append(train_acc) #list adds train acc result per epoch
        val_acc_list.append(val_acc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        
        #Save best model according to accuracy
        if val_acc > acc: 
            os.makedirs(save_dir, exist_ok = True)
            acc = val_acc
            best_model = model.state_dict() #whole status of model in dict
            torch.save(best_model, f"{save_dir}/{save_prefix}_best_model.pth")

    return train_acc_list, val_acc_list, train_loss_list, val_loss_list
     
