import os, shutil, glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from PIL import Image
torch.manual_seed(13)

# download dataset
def download_dataset(path_to_download, dataset_name = "brain"): 
    
    assert dataset_name == "brain", f"Iltimos brain nomi bilan kiriting!"
    if dataset_name == "brain": url = "kaggle datasets download -d killa92/brain-ct-tumor-classification-dataset"
    
     # Check if is already exist 
    if os.path.isfile(f"{path_to_download}/{dataset_name}.csv") or os.path.isdir(f"{path_to_download}/{dataset_name}"): 
        print(f"Dataset allaqachon yuklab olingan. {path_to_download}/{dataset_name} papkasini ni tekshiring\n"); 

    # If data doesn't exist in particular folder
    else: 
        ds_name = url.split("/")[-1] 
        # Download the dataset
        print(f"{ds_name} yuklanmoqda...")
        os.system(f"{url} -p {path_to_download}")
        shutil.unpack_archive(f"{path_to_download}/{ds_name}.zip", extract_dir=f"{path_to_download}/{dataset_name}")
        os.remove(f"{path_to_download}/{ds_name}.zip")
        print(f"Tanlangan dataset {path_to_download}/{dataset_name} papkasiga yuklab olindi!\n")
    
    return f"{path_to_download}/{dataset_name}"


#make a Custom dataset class 
class BrainDataset(Dataset): 
    def __init__(self, dataset_path, transformations = None):
        super().__init__()
        self.image_paths = glob.glob(f"{dataset_path}/*/images/*/*")
        self.transformations = transformations
        self.class_names = {} #dict for storing labels 
        class_value = 0
        
        for i, img_path in enumerate(self.image_paths): 
            class_n = self.get_class_name(img_path)
            #print(class_n)
            if class_n not in self.class_names: self.class_names[class_n] = class_value; class_value +=1

    def get_class_name(self, img_path): 
        #path only for brain dataset
        return img_path.split("\\")[-2] #get label according to it's folder 

    def __len__(self): 
        return len(self.image_paths)

    def __getitem__(self, index): 
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        label = self.class_names[self.get_class_name(img_path)] # get ground truth

        if self.transformations is not None: 
            image = self.transformations(image)

        return image, label 


def get_dataloaders(dataset_path, tfs, bs, split = [0.8, 0.1, 0.1]):
    
    ds = BrainDataset(dataset_path=dataset_path, transformations=tfs)
    train_data, valid_data, test_data  = random_split(dataset = ds, lengths = split) 

    print(f"Train data size:{len(train_data)}   |  Valid data size:{len(valid_data)}    |    Test data size: {len(test_data)}\n")

    train_dataloader = DataLoader(dataset = train_data, batch_size=bs, shuffle=True,)
    val_dataloader = DataLoader(dataset = valid_data, batch_size=bs,shuffle=False, )
    test_dataloader = DataLoader(dataset = test_data, batch_size=bs,  shuffle=False, )

    print(f"Train dataloader size:{len(train_dataloader)}   |  Valid dataloader size:{len(val_dataloader)}    |    Test dataloader size: {len(test_dataloader)}\n")

    return train_dataloader, val_dataloader, test_dataloader, ds.class_names_dict


    