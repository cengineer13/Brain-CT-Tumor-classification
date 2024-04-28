import argparse
import os 
import pickle as p
import timm
import torch
import torch.nn as nn
from torchvision import transforms as T
import vis_utils
from dataset import download_dataset, get_dataloaders
from train import train_validation
from inference import inference


def run(args): 
    #CONSONANT VARS
    STD, MEAN = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  #ImagaNet values
    tfs = T.Compose([T.Resize((224,224)), T.ToTensor(),T.Normalize(std=STD, mean=MEAN)])   

    assert args.data_path == "data", "data nomi bilan kiriting"
    # 1 - download dataset
    root_ds = download_dataset(path_to_download = args.data_path, dataset_name = args.dataset_name)
    
    # 2 - get dataloaders 
    tr_dl, val_dl, test_dl, class_names_dict = get_dataloaders(dataset_path=root_ds, tfs=tfs, bs=args.batch_size)
    class_names = list(class_names_dict.keys())
    print(class_names,"\n")
     
    #Save class names 
    os.makedirs(f"{args.model_files}", exist_ok=True)
    with open(f'{args.model_files}/{args.dataset_name}_classnames.pickle', 'wb') as f: p.dump(class_names, f, protocol=p.HIGHEST_PROTOCOL)    

    # 3 - save visualized examples from dataset (not dataloaders)
    for dl, data_type,  in zip([tr_dl, val_dl, test_dl], ['train', 'val', 'test']):
        vis_utils.visualize_ds(ds=dl.dataset, num_images=20, rows=5, cmap='rgb', class_names=class_names, 
                     data_type=data_type, save_folder=args.vis_path)


    # 4 - train and validation process
    model = timm.create_model(model_name=args.model_name, pretrained=True, num_classes=len(class_names))
    loss_fn = nn.CrossEntropyLoss(); learnig_rate = 1e-4; 
    optimizer = torch.optim.Adam(model.parameters(), lr = learnig_rate)
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    print('...................TRAIN JARAYONI BOSHLANDI!.........................\n')
    #start training and validation process and get acc and loss results as list
    train_acc_list, val_acc_list, train_loss_list, val_loss_list = train_validation(model=model, 
                                                        epochs=args.epochs, train_dl=tr_dl, val_dl=val_dl,
                                                        criterion=loss_fn, optimizer=optimizer,
                                                        device=args.device, save_dir=args.model_files, 
                                                        save_prefix=args.dataset_name)
    
    print('...................TRAIN JARAYONI YAKUNLANDI!.........................\n')
    # 5 - Save training and validation acc and loss metric plots to folder
    vis_utils.visualize_metrics(train_acc_list, val_acc_list, train_loss_list, val_loss_list, args.vis_path)

    #6 - Inference part 
    print('...................INFERENCE JARAYONI BOSHLANDI!......................\n')
    model.load_state_dict(torch.load(f"{args.model_files}/{args.dataset_name}_best_model.pth"))

    #this function execute inference and return gt, prediction and image lists for plotting
    imgs_list, gts_list, preds_list = inference(model=model, test_dataloader=test_dl, device=args.device) 

    #Save inference result images to args.vis_path folder
    vis_utils.visualize_inference(imgs_list, gts_list, preds_list, class_names, args.vis_path)
    print('...................INFERENCE JARAYONI YAKUNLANDI!......................\n')

    print('..........................GRAD CAM INFERENCE!..........................\n')
    # 7 - GradCam visualization 
    vis_utils.visualize_gradcam(model, imgs_list, gts_list, preds_list, class_names,
                       num_imgs=20, row=5, img_size=320, save_dir=args.vis_path)
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Cloud quality classification python project")

    #add arguments
    parser.add_argument("-dp", "--data_path", type=str, default="data", help="Datasetni yuklash uchun yo'lak")
    parser.add_argument("-dn", "--dataset_name", type=str, default="cloud", help="Dataset nomi")
    parser.add_argument("-vs", "--vis_path", type=str, default="data/plots", help="Vizualizations graph, plotlarni saqlash uchun yo'lak")
    parser.add_argument("-mn", "--model_name", type=str, default="resnet18", help="Trained bo'lgan timm model nomi")
    parser.add_argument("-ep", "--epochs", type=int, default=10, help="Epochlar soni")
    parser.add_argument("-bs", "--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("-dv", "--device", type=str, default="cuda:0", help="Train qilish qurilmasi GPU yoki CPU")
    parser.add_argument("-mf", "--model_files", type=str, default="data/model_files", 
                        help="Train qilingan model va boshqa parametr fayllar uchun yo'lak")
    args = parser.parse_args()
    
    run(args)