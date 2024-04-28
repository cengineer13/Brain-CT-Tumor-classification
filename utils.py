# Import libraries
import os, shutil, torch, random, os, numpy as np
from glob import glob; from tqdm import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tqdm import tqdm; from torchvision import transforms as tfs

class EarlyStopping:
    def __init__(self, metric_to_track = "loss", patience = 5, threshold = 0):

        assert metric_to_track in ["loss", "acc"], "Kuzatadigan metric acc yoki loss bo'lishi kerak!"
        self.metric_to_track, self.patience, self.threshold, self.counter, self.early_stop = metric_to_track, patience, threshold, 0, False
        self.best_value = torch.tensor(float("inf")) if metric_to_track == "loss" else torch.tensor(float("-inf"))
        self.di = {}; self.di[str(self.counter)] = False
        
    def __call__(self, current_value): 
        
        print(f"\n{self.metric_to_track} ni kuzatyapmiz!")
        
        if self.metric_to_track == "loss":
            if current_value > (self.best_value + self.threshold): self.counter += 1
            else: self.best_value = current_value
                
        elif self.metric_to_track == "acc":
            
            if current_value < (self.best_value + self.threshold): self.counter += 1
            else: self.best_value = current_value
            
        for counter, value in self.di.items():
            if int(counter) == self.counter and value == False and int(counter) != 0:
                print(f"{self.metric_to_track} {counter} marta o'zgarmadi!")
        
        self.di[str(self.counter)] = True; self.di[str(self.counter + 1)] = False
                
        if self.counter >= self.patience: 
            print(f"\n{self.metric_to_track} {self.patience} marta o'zgarmaganligi uchun train jarayoni yakunlanmoqda...")
            self.early_stop = True

def visualize(data, rasmlar_soni, qatorlar, cmap = None, klass_nomlari = None):
    
    assert cmap in ["rgb", "gray"], "Rasmni oq-qora yoki rangli ekanini aniqlashtirib bering!"
    if cmap == "rgb": cmap = "viridis"
    
    plt.figure(figsize = (20, 10))
    indekslar = [random.randint(0, len(data) - 1) for _ in range(rasmlar_soni)]
    for idx, indeks in enumerate(indekslar):
        
        im, gt = data[indeks]
        # Start plot
        plt.subplot(qatorlar, rasmlar_soni // qatorlar, idx + 1)
        if cmap:
            plt.imshow(tensor_2_im(im, cmap), cmap=cmap)
        else:
            plt.imshow(tensor_2_im(im))
        plt.axis('off')
        if klass_nomlari is not None:
            plt.title(f"GT -> {klass_nomlari[str(gt)]}")
        else:
            plt.title(f"GT -> {gt}")
            
def visualize_dl(rasmlar, javoblar, rasmlar_soni, qatorlar, bs, reg_turi, cmap = None, klass_nomlari = None):
    
    assert cmap in ["rgb", "gray"], "Rasmni oq-qora yoki rangli ekanini aniqlashtirib bering!"
    if cmap == "rgb": cmap = "viridis"
    
    indekslar = [random.randint(0, bs - 1) for _ in range(rasmlar_soni)]
    plt.figure(figsize = (20, 10))
    
    for idx, indeks in enumerate(indekslar):
        
        im, gt = rasmlar[indeks], javoblar[indeks]
        # Start plot
        plt.subplot(qatorlar, rasmlar_soni // qatorlar, idx + 1)
        if cmap: plt.imshow(tensor_2_im(im, cmap), cmap = cmap)
        else: plt.imshow(tensor_2_im(im))
        plt.axis('off')
        if klass_nomlari is not None:
            plt.title(f"{reg_turi} -> {klass_nomlari[str(gt.cpu().item())]}")
        else:
            plt.title(f"GT -> {gt}")
    plt.show()
            
def data_tekshirish(ds):
    
    data = ds[0]    
    print(f"Dataning birinchi elementining turi: {type(data[0])}")
    print(f"Dataning ikkinchi elementining turi: {type(data[1])}")
    print(f"Dataning birinchi elementining hajmi: {(data[0]).shape}")
    print(f"Dataning birinchi elementidagi piksel qiymatlari: {np.unique(np.array(data[0]))}")
    print(f"Dataning ikkinchi elementi: {data[1]}")
    

def tensor_2_im(t, t_type = "rgb"):
    
    gray_tfs = tfs.Compose([tfs.Normalize(mean = [ 0.], std = [1/0.5]), tfs.Normalize(mean = [-0.5], std = [1])])
    rgb_tfs = tfs.Compose([tfs.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), tfs.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])
    
    invTrans = gray_tfs if t_type == "gray" else rgb_tfs 
    
    return (invTrans(t) * 255).detach().squeeze().cpu().permute(1,2,0).numpy().astype(np.uint8) if t_type == "gray" else (invTrans(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def parametrlar_soni(model): 
    for name, param in model.named_parameters():
        print(f"{name} parametrida {param.numel()} ta parametr bor.")
    print(f"Modelning umumiy parametrlar soni -> {sum(param.numel() for param in model.parameters() if param.requires_grad)} ta.")
    
    
def inference(model, device, test_dl, num_ims, row, cls_names = None):
    
    preds, images, lbls = [], [], []
    for idx, data in enumerate(test_dl):
        im, gt = data
        im, gt = im.to(device), gt.to(device)
        _, pred = torch.max(model(im), dim = 1)
        images.append(im)
        preds.append(pred.item())
        lbls.append(gt.item())
    
    plt.figure(figsize = (20, 10))
    indekslar = [random.randint(0, len(images) - 1) for _ in range(num_ims)]
    for idx, indeks in enumerate(indekslar):
        
        im = images[indeks].squeeze()
        # Start plot
        plt.subplot(row, num_ims // row, idx + 1)
        plt.imshow(tensor_2_im(im), cmap='gray')
        plt.axis('off')
        if cls_names is not None: plt.title(f"GT -> {cls_names[str(lbls[indeks])]} ; Prediction -> {cls_names[str(preds[indeks])]}", color=("green" if {cls_names[str(lbls[indeks])]} == {cls_names[str(preds[indeks])]} else "red"))
        else: plt.title(f"GT -> {gt} ; Prediction -> {pred}")
        
        
def train(model, dataloader, device, loss_fn, optimizer, epoch):
    
    model.train()
    print(f"{epoch + 1}-epoch train jarayoni boshlandi...")
    epoch_loss, epoch_acc, total = 0, 0, 0
    for idx, batch in tqdm(enumerate(dataloader)):
        ims, gts = batch
        ims, gts = ims.to(device), gts.to(device)
        total += ims.shape[0]
        
        preds = model(ims)
        loss = loss_fn(preds, gts)
        _, pred_cls = torch.max(preds.data, dim = 1)
        epoch_acc += (pred_cls == gts).sum().item()
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print(f"{epoch + 1}-epoch train jarayoni natijalari: ")
    print(f"{epoch + 1}-epochdagi train loss     -> {(epoch_loss / len(dataloader)):.3f}")
    print(f"{epoch + 1}-epochdagi train accuracy -> {(epoch_acc / total):.3f}")
    
    return model
    
def validation(model, dataloader, loss_fn, epoch, device, best_acc = 0):
    model.eval()
    with torch.no_grad():
        val_epoch_loss, val_epoch_acc, val_total = 0, 0, 0
        for idx, batch in enumerate(dataloader):
            ims, gts = batch
            ims, gts = ims.to(device), gts.to(device)
            val_total += ims.shape[0]

            preds = model(ims)
            loss = loss_fn(preds, gts)
            _, pred_cls = torch.max(preds.data, dim = 1)
            val_epoch_acc += (pred_cls == gts).sum().item()
            val_epoch_loss += loss.item()
        
        val_acc = val_epoch_acc / val_total
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        print(f"{epoch + 1}-epoch validation jarayoni natijalari: ")
        print(f"{epoch + 1}-epochdagi validation loss     -> {(val_epoch_loss / len(dataloader)):.3f}")
        print(f"{epoch + 1}-epochdagi validation accuracy -> {val_acc:.3f}\n")
        
        if val_acc > best_acc:
            os.makedirs("saved_models", exist_ok=True)
            best_acc = val_acc
            torch.save(model.state_dict(), f"saved_models/best_model.pth")
            
            
def train(model, dataloader, val_dl, device, loss_fn, optimizer, epoch, best_acc = 0):
    
    model.train()
    print(f"{epoch + 1}-epoch train jarayoni boshlandi...")
    epoch_loss, epoch_acc, total = 0, 0, 0
    for idx, batch in tqdm(enumerate(dataloader)):
        ims, gts = batch
        ims, gts = ims.to(device), gts.to(device)
        total += ims.shape[0]
        
        preds = model(ims)
        loss = loss_fn(preds, gts)
        _, pred_cls = torch.max(preds.data, dim = 1)
        epoch_acc += (pred_cls == gts).sum().item()
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print(f"{epoch + 1}-epoch train jarayoni natijalari: ")
    print(f"{epoch + 1}-epochdagi train loss     -> {(epoch_loss / len(dataloader)):.3f}")
    print(f"{epoch + 1}-epochdagi train accuracy -> {(epoch_acc / total):.3f}")
    
    model.eval()
    with torch.no_grad():
        val_epoch_loss, val_epoch_acc, val_total = 0, 0, 0
        for idx, batch in enumerate(val_dl):
            ims, gts = batch
            ims, gts = ims.to(device), gts.to(device)
            val_total += ims.shape[0]

            preds = model(ims)
            loss = loss_fn(preds, gts)
            _, pred_cls = torch.max(preds.data, dim = 1)
            val_epoch_acc += (pred_cls == gts).sum().item()
            val_epoch_loss += loss.item()
        
        val_acc = val_epoch_acc / val_total
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
        print(f"{epoch + 1}-epoch validation jarayoni natijalari: ")
        
        print(f"{epoch + 1}-epochdagi validation loss     -> {(val_epoch_loss / len(val_dl)):.3f}")
        print(f"{epoch + 1}-epochdagi validation accuracy -> {val_acc:.3f}\n")
        
        if val_acc > best_acc:
            os.makedirs("saved_models", exist_ok=True)
            best_acc = val_acc
            torch.save(model.state_dict(), f"saved_models/best_model.pth")

def copy_files(files, dest):
    print(f"{os.path.basename(dest)} dir is created!")
    for file in files: 
        cls_name = os.path.dirname(file).split("/")[-1]
        os.makedirs(f"{dest}/{cls_name}", exist_ok = True)
        shutil.copy(file, f"{dest}/{cls_name}/{os.path.basename(file)}")
    
def split_data(root, im_files = [".jpg", ".png", ".jpeg"], split = [0.5, 0.3, 0.2]):
    
    assert sum(split) == 1.0, "Data split elements' sum must be equal to exactly 1"
    dirs = [f"{root}/train", f"{root}/val", f"{root}/test"]
    for idx, dir in tqdm(enumerate(dirs)): 
        if os.path.isdir(dir): print(f"{os.path.basename(dir)} dir already exists! Deleting..."); shutil.rmtree(dir)
    
    tr_size = split[0]; test_size = split[1] + split[2]

    train, valid = train_test_split(glob(f"{root}/*[{im_file for im_file in im_files}]"), test_size = test_size)
    val, test = train_test_split(valid, test_size = ((test_size - split[1]) / test_size))
    
    # train, valid = train_test_split(glob(f"{root}/*/*[{im_file for im_file in im_files}]"), test_size = test_size)
    # val, test = train_test_split(valid, test_size = ((test_size - split[1]) / test_size))
    
    di = {f"{dirs[0]}": train, f"{dirs[1]}": val, f"{dirs[2]}": test}
    
    for idx, (key, val) in enumerate(di.items()): copy_files(files = val, dest = key)
        
    # copy_files(files = train, dest = dirs[0])
    # copy_files(files = valid, dest = dirs[1])
    # copy_files(files = test, dest = dirs[2])
    
class Plot():
    
    def __init__(self, res):
        
        plt.figure(figsize = (10, 5))
        plt.plot(res["tr_losses"], label = "Train Loss")
        plt.plot(res["val_losses"], label = "Validation Loss")
        plt.title("Loss Learning Curve")
        plt.xlabel("Epochlar")
        plt.ylabel("Loss Qiymati")
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(10, 5))
        plt.plot(res["tr_accs"], label = "Train Accuracy")
        plt.plot(res["val_accs"], label = "Validation Accuracy")
        plt.title("Accuracy Score Learning Curve")
        plt.xlabel("Epochlar")
        plt.ylabel("Accuracy Qiymati")
        plt.legend()
        plt.show()