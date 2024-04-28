from torchvision import transforms as T
import numpy as np
import cv2
import random, os
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

def tensor_2_im(tensor, t_type = "rgb"):
    
    gray_tfs = T.Compose([T.Normalize(mean = [ 0.], std = [1/0.5]), T.Normalize(mean = [-0.5], std = [1])])
    rgb_tfs = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])
    
    invTrans = gray_tfs if t_type == "gray" else rgb_tfs 
    
    return (invTrans(tensor) * 255).detach().squeeze().cpu().permute(1,2,0).numpy().astype(np.uint8) if t_type == "gray" else (invTrans(tensor) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

#function for ploting examples from dataset
def visualize_ds(ds, num_images, rows, cmap = None, class_names:list = None, data_type = None, save_folder=None):
    
    assert cmap in ["rgb", "gray"], "Rasmni oq-qora yoki rangli ekanini aniqlashtirib bering!"
    if cmap == "rgb": cmap = "viridis"
    
    plt.figure(figsize = (20, 10))
    indexs = [random.randint(0, len(ds) - 1) for _ in range(num_images)]
    for i, idx in enumerate(indexs):
        
        img, gt = ds[idx]
        # Start plot
        plt.subplot(rows, num_images // rows, i + 1)
        if cmap:
            pass
            plt.imshow(tensor_2_im(img, cmap), cmap=cmap)
        else:
            pass
            plt.imshow(tensor_2_im(img))
        plt.axis('off')
        if class_names is not None:
            plt.title(f"GT -> {class_names[gt]}")
        else:
            plt.title(f"GT -> {gt}")

        #if saving folder is not available
        if not os.path.isdir(save_folder): 
            os.makedirs(f"{save_folder}")
    #plt.show()
    plt.savefig(f"{save_folder}/1-{data_type}_random_examples.png")
    print(f"{data_type} datasetdan namunalar {save_folder} papkasiga yuklandi...")
    print("---------------------------------------------------------------------")
    plt.clf(); plt.close()


#function for ploting training validation results
def visualize_metrics(train_acc_list, val_acc_list, train_loss_list, val_loss_list, save_dir): 
    # Plot the training acc curve
    plt.plot(train_acc_list, label='Training accuracy')
    plt.plot(val_acc_list, label="Validation accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Curve')
    plt.xticks(ticks=np.arange(len(train_acc_list)), labels=[i for i in range(1, len(train_acc_list)+1)])
    plt.legend()
    plt.savefig(f"{save_dir}/2-Training and Validation accuracy metrics.png")
    print(f" Training and Validation accuracy metricslar {save_dir} papkasiga yuklandi...")
    plt.show()

    # Plot the training acc curve
    plt.plot(train_loss_list, label='Training loss', c = 'red')
    plt.plot(val_loss_list, label="Validation loss", c='b')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.xticks(ticks=np.arange(len(train_loss_list)), labels=[i for i in range(1, len(train_loss_list)+1)])
    plt.legend()
    plt.savefig(f"{save_dir}/2-Training and Validation loss metrics.png")
    print(f" Training and Validation loss metricslar {save_dir} papkasiga yuklandi...\n")
    plt.show()
    

#function for ploting inference results
def visualize_inference(rasmlar_list, haqiqiy_list, bashorat_list,  class_names, save_dir):
    random_indexs = [random.randint(0, len(haqiqiy_list) - 1) for _ in range(20)]
    
    plt.figure(figsize=(20, 10))
    for i, idx in enumerate(random_indexs): 
        plt.subplot(4, 5, i+1) 
        tensor_img = rasmlar_list[idx]
        np_image = np.transpose(tensor_img.detach().cpu().numpy())
        np_image = np_image * [0.485, 0.456, 0.406] + [0.229, 0.224, 0.225]
        np_image = np.clip(np_image, 0, 1) * 255  # Clip values to 0-1 and scale to 0-255
        final_img = np_image.astype(np.uint8)  # Convert to uint8 for integer representation

        plt.axis("off")
        equal = class_names[haqiqiy_list[idx]] = class_names[bashorat_list[idx]]
        color = ('green' if equal else 'red')
        plt.title(f"GT:{class_names[haqiqiy_list[idx]]} | Pred:{class_names[bashorat_list[idx]]}", color=color)
        plt.xlabel('ab')
        plt.imshow(final_img)
    plt.savefig(f"{save_dir}/3-Inference_result_examples.png")
    print(f"Inferencedan keyingi natija random example rasmlar {save_dir} papkasiga yuklandi...")

#function for ploting GradCam inference results
def visualize_gradcam(model, img_list, gt_list, pred_list, class_names, num_imgs, row, img_size, save_dir): 
    plt.figure(figsize = (20, 10))
    #get random index between 0 and len(test_dataloder size) for plotting
    indexs = [random.randint(0, len(gt_list)-1) for _ in range(num_imgs)]
    for i, index in enumerate(indexs):
        
        img = img_list[index].squeeze() #extract random index image
        lb = gt_list[index]; pd = pred_list[index]
    
        #GRADCAM tensorlar bn emas arraylar (CPU) bn ishlaydi 
        org_img = tensor_2_im(img) / 255  #Normalize np image into between [0, 1]
        #GradCamPlusPlus classidan obyekt olib unga model va oxirgi fc layerslarni jo'natamiz
        # cam = GradCAMPlusPlus(model=model, target_layers=[model.features[-1]]) #for rexnet_150
        cam = GradCAMPlusPlus(model=model, target_layers=[model.layer4[-1]])
        #cam obyektiga 4D rasm jonatib keyin uni grayscalega convert qilib olamiz. 
        grayscale_cam = cam(input_tensor = img.unsqueeze(0))[0, :]
                                                                    #image-weight - heatmap mask opacity control
        heatmap = show_cam_on_image(img = org_img, mask=grayscale_cam, image_weight=0.4, use_rgb=True)
        
        #Start plot 
        plt.subplot(row, num_imgs // row, i + 1)
        plt.imshow(tensor_2_im(img), cmap = "gray"); plt.axis('off')
        resized_heatmap = cv2.resize(heatmap, (img_size, img_size), interpolation=cv2.INTER_AREA)
        plt.imshow(resized_heatmap, alpha=0.3, cmap='jet'); plt.axis('off')

        color = ("green" if {class_names[lb]} == {class_names[pd]} else "red")
        if class_names: plt.title(f"L: {class_names[lb]} ; B: {class_names[pd]}", color = color)
        else: plt.title(f"L: {lb} ; B: {pd}")

    plt.savefig(f"{save_dir}/4-GradCam_results_examples.png")
    print(f"Inferencedan keyingi GradCam natija rasmlar {save_dir} papkasiga yuklandi...\n")


class SegMetricsPlot():
    def __init__(self, r):

        
        plt.figure(figsize=(8,4))
        plt.plot(r["tr_loss"], label = "Train loss")
        plt.plot(r["val_loss"], label = "Validation loss")
        plt.title("Train and Validation Loss")
        plt.xticks(np.arange(len(r["val_loss"])), [i for i in range(1, len(r["val_loss"])+1)])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.ylim(-0.5, 3)
        plt.legend()
        plt.show()


        plt.figure(figsize=(8,4))
        plt.plot(r["tr_pa"], label = "Train PA")
        plt.plot(r["val_pa"], label = "Validation PA")
        plt.title("Train and Validation PA ")
        plt.xticks(np.arange(len(r["val_pa"])), [i for i in range(1, len(r["val_pa"])+1)])
        plt.xlabel("Epochs")
        plt.ylabel("PA score")
        plt.legend()
        plt.ylim(-0.5, 3)
        plt.show()

        
        plt.figure(figsize=(8,4))
        plt.plot(r["tr_iou"], label = "Train mioU")
        plt.plot(r["val_iou"], label = "Validation mioU")
        plt.title("Train and Validation mIOU ")
        plt.xticks(np.arange(len(r["val_iou"])), [i for i in range(1, len(r["val_iou"])+1)])
        plt.xlabel("epochs")
        plt.ylabel("mIoU score")
        plt.legend()
        plt.show()
plot(result_dict)