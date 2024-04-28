from matplotlib import pyplot as plt 
import numpy as np
import torch

def inference(model, test_dataloader, device):
   
    all_imgs, all_gts, all_preds,   = [], [], []
    correct = 0
    for imgs, gts in test_dataloader:
        
        imgs = imgs.to(device)
        gts = gts.to(device)
        preds = torch.argmax(model(imgs),dim=1) 
        acc = (preds== gts).sum().item()
        correct += acc

        #loop through every batch
        for img, gt, pred in zip(imgs, gts, preds): 
            all_imgs.append(torch.squeeze(img.to('cpu')))
            all_gts.append(gt.item())
            all_preds.append(pred.item())

    print(f"{len(test_dataloader.dataset)} rasm test qilinganda {(correct * 100)/len(test_dataloader.dataset):.2f} % foiz aniqlikda topdi")

    return all_imgs, all_gts, all_preds

