import torch, timm, argparse, pickle
from PIL import Image
from torchvision import transforms as T
import streamlit as st

def run(args): 

    with open(f"{args.model_files}/{args.dataset_name}_classnames.pickle", "rb") as f: labels = pickle.load(f)
    num_classes = len(labels)

    STD, MEAN = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  #ImagaNet values
    tfs = T.Compose([T.Resize((224,224)), T.ToTensor(),T.Normalize(std=STD, mean=MEAN)]) 

    #load our best model
    model = load_model(model_name=args.model_name, 
                       model_path=f"{args.model_files}/{args.dataset_name}_best_model.pth", 
                       num_classes=num_classes)
    
    print(f"PreTrained {args.model_name} model succesfully loaded.!")

    #img for prediction
    st.title("Brain CT Tumor classification python project")
    img_path = st.file_uploader("Upload image...")

    #1-parm - model, 2-img_path, 3-transformations, 4-class_names 
    image, label = predict(model, img_path, tfs, labels) if img_path else predict(model, args.test_img, tfs, labels)

    st.write("Uploaded image...")
    st.image(image) 
    st.write(f"Model prediction -> {label.upper()}")
    print(f"Model prediction: {label}")


def load_model(model_name, model_path, num_classes):
    
    m = timm.create_model(model_name=model_name, num_classes=num_classes)
    m.load_state_dict(torch.load(model_path))

    return m.eval()

def predict(model, img_path, tfs, class_names):

    class_nomlari = list(class_names.keys()) if isinstance(class_names, dict) else class_names
    
    #predict 
    img = Image.open(img_path).convert('RGB') 
    tensor_img = tfs(img).unsqueeze(0) #3D img -> 4D as batch for model inputting
    bashorat = model(tensor_img)
    bashorat_class_idx = torch.argmax(bashorat, dim=1).item() #get max prediction index as integer

    return img, class_nomlari[bashorat_class_idx] #rbg img and prediction class


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Brain CT Tumor classification python project DEMO')
    
    parser.add_argument("-mf", "--model_files", type=str, default="data/model_files", help="Path for trained and saved files")
    parser.add_argument("-dn", "--dataset_name", type=str, default="brain", help="Dataset name")
    parser.add_argument("-mn", "--model_name", type=str, default="resnet18", help="Pretrained model name of Timm library")
    parser.add_argument("-ti", "--test_img", default="test_images/Very Low.jpg", help="Path for image to predict unseen image")

    args = parser.parse_args()

    run(args=args)