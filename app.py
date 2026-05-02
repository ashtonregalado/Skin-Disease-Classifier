import os
import json
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

from model import build_model


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_classes():
    classes_path = os.path.join("models", "classes.json")
    if os.path.isfile(classes_path):
        with open(classes_path, "r") as f:
            return json.load(f)

    # Try to infer from dataset folder structure
    train_dir = os.path.join("SkinDisease", "train")
    if os.path.isdir(train_dir):
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        os.makedirs("models", exist_ok=True)
        with open(classes_path, "w") as f:
            json.dump(classes, f, indent=2)
        return classes

    raise FileNotFoundError("classes.json not found and SkinDisease/train/ not available to infer classes.")


@st.cache_resource
def load_model_and_device(num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=num_classes)
    model_path = None
    for candidate in ["models/best_model_final.pth", "models/best_model_phaseA.pth"]:
        if os.path.isfile(candidate):
            model_path = candidate
            break
    if model_path is None:
        raise FileNotFoundError("No model file found in models/. Please train or place best_model_final.pth there.")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return preprocess(image)


def predict(image: Image.Image, model, device, classes, topk: int = 3):
    tensor = preprocess_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_probs, top_idx = probs.topk(topk, dim=1)
        top_probs = top_probs.cpu().numpy()[0].tolist()
        top_idx = top_idx.cpu().numpy()[0].tolist()
        top_labels = [classes[i] for i in top_idx]
    return list(zip(top_labels, top_probs))


def main():
    st.set_page_config(page_title="Skin Disease Classifier", layout="centered")
    st.title("Skin Disease Classifier")

    try:
        classes = load_classes()
    except Exception as e:
        st.error(f"Could not load classes: {e}")
        return

    st.write(f"Detected {len(classes)} classes")

    uploaded = st.file_uploader("Upload a skin image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload an image to get a prediction.")
        return

    image = Image.open(uploaded)

    st.image(image, caption="Uploaded image", width=360)

    with st.spinner("Loading model..."):
        try:
            model, device = load_model_and_device(len(classes))
        except Exception as e:
            st.error(f"Model load error: {e}")
            return

    if st.button("Predict"):
        with st.spinner("Running inference..."):
            try:
                results = predict(image, model, device, classes, topk=3)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

        st.success("Prediction complete")
        for label, prob in results:
            st.write(f"- **{label}**: {prob*100:.2f}%")


if __name__ == "__main__":
    main()
