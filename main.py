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

    assert args.data_path == "data", "please insert with the name 'data' "
    # 1 - download dataset
    root_ds = download_dataset(path_to_download = args.data_path, dataset_name = args.dataset_name)

    # 2 - get dataloaders
    #CONSONANT VARS
    STD, MEAN = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  #ImagaNet values
    tfs = T.Compose([T.Resize(args.img_size), T.ToTensor(),T.Normalize(std=STD, mean=MEAN)])    
    tr_dl, val_dl, test_dl, class_names_dict = get_dataloaders(dataset_path=root_ds, tfs=tfs, bs=args.batch_size)
    class_names = list(class_names_dict.keys())
     
    # 3 - save visualized examples from dataset
    for dl, data_type,  in zip([tr_dl, val_dl, test_dl], ['train', 'val', 'test']):
        vis_utils.visualize_ds(ds=dl.dataset, num_images=args.n_imgs, rows=args.rows, cmap=args.cmap, class_names=class_names, 
                     data_type=data_type, save_folder=args.vis_path)

    # 4 - train and validation process
    model = timm.create_model(model_name=args.model_name, pretrained=True, num_classes=len(class_names))
    loss_fn = nn.CrossEntropyLoss(); learnig_rate = args.learning_rate; 
    optimizer = torch.optim.Adam(model.parameters(), lr = learnig_rate)
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    print('...................TRAIN PROCESS STARTED.........................\n')
    #start training and validation process and get acc and loss results as list
    train_acc_list, val_acc_list, train_loss_list, val_loss_list = train_validation(model=model, 
                                                        epochs=args.epochs, train_dl=tr_dl, val_dl=val_dl,
                                                        criterion=loss_fn, optimizer=optimizer,
                                                        device=args.device, save_dir=args.model_files, 
                                                        save_prefix=args.dataset_name)
    
    print('...................TRAIN PROCESS FINISHED!.........................\n')
    # 5 - Save training and validation acc and loss metric plots to folder
    vis_utils.visualize_metrics(train_acc_list, val_acc_list, train_loss_list, val_loss_list, args.vis_path)

    #6 - Inference part 
    print('...................INFERENCE PROCESS STARTED!......................\n')
    model.load_state_dict(torch.load(f"{args.model_files}/{args.dataset_name}_best_model.pth"))

    #this function execute inference and return gt, prediction and image lists for plotting
    imgs_list, gts_list, preds_list = inference(model=model, test_dataloader=test_dl, device=args.device) 

    #Save inference result images to args.vis_path folder
    vis_utils.visualize_inference(imgs_list, gts_list, preds_list, class_names, args.vis_path)
    print('...................INFERENCE PROCESS FINISHED!......................\n')

    print('..........................GRAD CAM INFERENCE!..........................\n')
    # 7 - GradCam visualization 
    vis_utils.visualize_gradcam(model, imgs_list, gts_list, preds_list, class_names,
                       num_imgs=args.n_imgs, row=args.rows, img_size=args.img_size, save_dir=args.vis_path)
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Brain CT Tumor classification python project")

    #add arguments
    parser.add_argument("-dp", "--data_path", type=str, default="data", help="Path to download dataset")
    parser.add_argument("-dn", "--dataset_name", type=str, default="brain", help="Dataset name")
    parser.add_argument("-ims", "--img_size", type=tuple, default=(224,224), help="To be resized size for image")
    parser.add_argument("-n_im", "--n_imgs", type=int, default = 20, help="Number of images for plotting")
    parser.add_argument("-r", "--rows", type=int, default = 5, help="Number of rows for plotting as subplots")
    parser.add_argument("-cm", "--cmap", type=str, default = "rgb", help="COLOR mode for input and output image")
    parser.add_argument("-vs", "--vis_path", type=str, default="data/plots", help="Path for saving vizualizations graphs")
    parser.add_argument("-mn", "--model_name", type=str, default="resnet18", help="Pretrained model name of Timm library")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate for optimization function")
    parser.add_argument("-ep", "--epochs", type=int, default=5, help="The number of epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("-dv", "--device", type=str, default="cuda:0", help="Device to train CPU or GPU")
    parser.add_argument("-mf", "--model_files", type=str, default="data/model_files", 
                        help="Path for trained and saved files")
    args = parser.parse_args()
    
    run(args)