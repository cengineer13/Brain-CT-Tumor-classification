import torch, timm, argparse, pickle
from PIL import Image
from torchvision import transforms as T
import streamlit as st


#CONSONANT VARIABLES
SAVE_DIR, SAVE_PREFIX = "data/saved_models", "brain"

def run(args): 

    with open(f"{SAVE_DIR}/{SAVE_PREFIX}_classnames.pickle", "rb") as f: labels = pickle.load(f)
    num_classes = len(labels)

    # mean, std, img_size = [0.485, 0.456, 0.406], [0.229, 0.224,0.225], 224
    tfs = T.Compose([T.Resize(224), T.ToTensor()])


    #load our best model
    model = load_model(model_name=args.model_name, model_path=args.model_path, num_classes=num_classes)
    print(f"Train qilingan model {args.model_name} muvaffaqiyatli yuklab olindi.!")

    #img for prediction
    example_im_path = "data/test_images/aneurysm_test.jpeg"
    st.title("Brain desease Classifier")
    img_path = st.file_uploader("Rasmni yuklang...")

                        #1-parm - model, 2-img_path, 3-transformations, 4-class_names 
    image, label = predict(model, img_path, tfs, labels) if img_path else predict(model, example_im_path, tfs, labels)

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
    parser = argparse.ArgumentParser(description='Brain desease classifier DEMO')
    
    #Add arguments (- va -- option value oladigon, type - qaysi turni olish )
    parser.add_argument("-mn", "--model_name", type=str, default="resnet18", help="AI model nomi")
    parser.add_argument("-mp", "--model_path", type=str, default="data/saved_models/brain_best_model.pth", 
                        help="Path for best saved model")
   # parser.add_argument("-imp", "--img_path", default="data/test_rasmlar/brownspot.jpg", help="Path for image")

    #argumentlarni tasdiqlash parse
    args = parser.parse_args()

    run(args=args)