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
    
    print(f"Train qilingan model {args.model_name} muvaffaqiyatli yuklab olindi.!")

    #img for prediction
    st.title("Cloud quality Classification python project")
    img_path = st.file_uploader("Rasmni yuklang...")

                        #1-parm - model, 2-img_path, 3-transformations, 4-class_names 
    image, label = predict(model, img_path, tfs, labels) if img_path else predict(model, args.test_img, tfs, labels)

    st.write("Yuklangan rasm:...")#text yozadi 
    st.image(image) #rasmni chizazi
    st.write(f"Model bashorati -> {label.upper()}")
    print(f"Bashorat javobi: {label}")


def load_model(model_name, model_path, num_classes):
    
    m = timm.create_model(model_name=model_name, num_classes=num_classes)
    m.load_state_dict(torch.load(model_path))

    return m.eval()

def predict(model, img_path, tfs, class_names):
    #agarda class_names dict bo'lsa 
    class_nomlari = list(class_names.keys()) if isinstance(class_names, dict) else class_names
    
    #predict 
    img = Image.open(img_path).convert('RGB') 
    tensor_img = tfs(img).unsqueeze(0) #3D img -> 4D as batch for model inputting
    bashorat = model(tensor_img)
    bashorat_class_idx = torch.argmax(bashorat, dim=1).item() #get max prediction index as integer

    return img, class_nomlari[bashorat_class_idx] #rbg img and prediction class


if __name__ == "__main__":
    #Parser classdan obyekt olamiz 
    parser = argparse.ArgumentParser(description='Cloud quality Classification python project DEMO')
    
    #Add arguments (- va -- option value oladigon, type - qaysi turni olish )
    parser.add_argument("-mf", "--model_files", type=str, default="data/model_files", 
                        help="Train qilingan model va class name.pickle fayllar uchun yo'lak")
    parser.add_argument("-dn", "--dataset_name", type=str, default="cloud", help="Dataset nomi")
    parser.add_argument("-mn", "--model_name", type=str, default="resnet18", help="AI timm pretrained model nomi")
    parser.add_argument("-mp", "--model_path", type=str, default="data/model_files/cloud_best_model.pth", 
                        help="Path for best saved model")
    parser.add_argument("-ti", "--test_img", default="test_images/Very Low.jpg", help="Path for image to predict unseen image")

    #argumentlarni tasdiqlash parse
    args = parser.parse_args()

    run(args=args)