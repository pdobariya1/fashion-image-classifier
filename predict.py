import torch
from collections import OrderedDict
import torchvision.transforms as transforms

from model_architecture import FashionClassifier


# Load label encoders
label_encoders = torch.load("models/label_encoders.pth", weights_only=False)
print("Label encoder loaded successful.")


# Define model architecture using encoded class counts
num_colors = len(label_encoders["baseColour"].classes_)
num_types = len(label_encoders["articleType"].classes_)
num_seasons = len(label_encoders["season"].classes_)
num_genders = len(label_encoders["gender"].classes_)


# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load model
def load_model(model_path="models/fashion_classifier.pth"):
    """Loads the trained model and returns it."""
    model = FashionClassifier(num_colors, num_types, num_seasons, num_genders)
    state_dict = torch.load(model_path, map_location="cpu")
    
    # Remove "module." prefix if trained on multiple GPUs
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model.to("cpu")
    model.eval()
    print("Model loaded on CPU successful.")
    return model



def predict(image, model):
    """Runs inference on an image and returns predictions."""
    input_tensor = transform(image).unsqueeze(0).to("cpu")
    
    with torch.no_grad():
        color_out, type_out, season_out, gender_out = model(input_tensor)
    
    # Convert predictions to class labels
    color_pred = torch.argmax(color_out, dim=1).item()
    type_pred = torch.argmax(type_out, dim=1).item()
    season_pred = torch.argmax(season_out, dim=1).item()
    gender_pred = torch.argmax(gender_out, dim=1).item()
    
    predictions = {
        "Color": label_encoders["baseColour"].inverse_transform([color_pred])[0],
        "Product Type": label_encoders["articleType"].inverse_transform([type_pred])[0],
        "Season": label_encoders["season"].inverse_transform([season_pred])[0],
        "Gender": label_encoders["gender"].inverse_transform([gender_pred])[0]
    }
    return predictions

