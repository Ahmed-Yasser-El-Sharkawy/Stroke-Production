import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import base64
import cv2
import matplotlib.pyplot as plt


def set_background_and_text(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        font-weight: 900;
    }}
    .stSidebar, .css-1d391kg {{
        background-color: #0d2b42 !important;
        color: #ffffff !important;
        font-weight: 900;
    }}
    .stSidebar .css-1v3fvcr, .stSidebar .css-q8sbsg {{
        color: #ffffff !important;
        font-weight: 900;
    }}
    * {{
        color: #ffffff !important;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 900 !important;
    }}
    .stFileUploader {{
        background-color: #0d2b42 !important;
        border: 2px dashed #00b4d8 !important;
        border-radius: 10px;
        color: #ffffff !important;
        font-weight: 900 !important;
    }}
    .stFileUploader label, .stFileUploader div, .stFileUploader span {{
        color: #004080 !important;
        font-weight: 900 !important;
    }}
    header {{
        background-color: #0d2b42 !important;
        color: #ffffff !important;
        font-weight: 900;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    
    
class TextRegionDetector:
    def __init__(self, min_width=10, min_height=10, threshold=220):
        self.min_width = min_width
        self.min_height = min_height
        self.threshold = threshold

    def load_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return image

    def detect_largest_region(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_box = None
        max_area = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if w >= self.min_width and h >= self.min_height and area > max_area:
                max_box = (x, y, w, h)
                max_area = area
        return max_box

    def draw_box(self, image, box, color=(0, 255, 0), thickness=2):
        if box:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        return image

    def process(self, image_path, visualize=False):
        image = self.load_image(image_path)
        box = self.detect_largest_region(image)
        if visualize and box:
            image_with_box = self.draw_box(image.copy(), box)
            self._visualize(image_with_box, image_path)
        return box

    def _visualize(self, image, title=""):
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(os.path.basename(title))
        plt.axis("off")
        plt.show()
    
    def crop_text_region(self, image):
        box = self.detect_largest_region(image)
        if box:
            x, y, w, h = box
            return image[y:y+h, x:x+w]
        return image
    
    
def preprocess_image_MRI(img, size=(224, 224)):
    img = img.convert('L')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

class EnhancedCNN_MRI(nn.Module):
    def __init__(self):
        super(EnhancedCNN_MRI, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No sigmoid — we'll use BCEWithLogitsLoss
        return x

def MRI_UI():
    model = EnhancedCNN_MRI()
    model.load_state_dict(torch.load('MRI/best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    
    set_background_and_text("Images_App/proxy-image.jpeg")

    st.title("Stroke MRI Classifier")
    st.write("Upload an image to classify it as Normal or Stroke.")

    main_class_names = ["Normal", "Stroke"]

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image')
        st.write("Classifying...")

        input_tensor = preprocess_image_MRI(image)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.item()

        predicted_class = 1 if prediction >= 0.5 else 0
        confidence = prediction if predicted_class == 1 else 1 - prediction
        
        st.write(f"**Predicted Class:** {main_class_names[predicted_class]}")
        st.write(f"**Confidence:** {confidence:.4f}")

class EnhancedCNN_CT(nn.Module):
    def __init__(self):
        super(EnhancedCNN_CT, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No sigmoid — we'll use BCEWithLogitsLoss
        return x

class Sub_Class_CNNModel_CT(nn.Module):
    def __init__(self, num_classes=2):
        super(Sub_Class_CNNModel_CT, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return torch.softmax(x, dim=1)

def preprocess_image_CT(img, size=(224, 224)):
    # Convert PIL image to OpenCV format (RGB → BGR)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Initialize and apply the text region detector
    detector = TextRegionDetector()
    cropped = detector.crop_text_region(img_cv)

    # Resize to expected size
    resized = cv2.resize(cropped, size)

    # Convert back to PIL for torchvision transforms
    pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    # Apply transformations (e.g., ToTensor)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(pil_img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor


def preprocess_image_sub_ct(img, size=(224, 224)):
    """
    Takes a PIL image and returns a normalized tensor with batch dimension.
    Assumes input is RGB (3-channel) image.
    """
    if not isinstance(img, Image.Image):
        raise ValueError("Input must be a PIL Image.")
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225] 
    img = img.convert("RGB")  # Ensure 3 channels
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0) 
     
def CT_UI():
    Main_model = EnhancedCNN_CT()
    Main_model.load_state_dict(torch.load('CT/best_model_CT.pth', map_location=torch.device('cpu')))       
    Main_model.eval()
    
    set_background_and_text("Images_App/proxy-image.jpeg")

    st.title("Stroke CT Classifier")
    st.write("Upload an image to classify it as Normal or Stroke.")

    main_class_names = ["Normal", "Stroke"]

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image')
        st.write("Classifying...")

        input_tensor = preprocess_image_CT(image)
        with torch.no_grad():
            output = Main_model(input_tensor)
            Mian_prediction = output.item()

        Mian_predicted_class = 1 if Mian_prediction >= 0.5 else 0
        confidence = Mian_prediction if Mian_predicted_class == 1 else 1 - Mian_prediction
        
        st.write(f"**Predicted Class:** {main_class_names[Mian_predicted_class]}")
        st.write(f"**Confidence:** {confidence:.4f}")
        
        # If the main prediction is "Stroke", run the sub-classifier
        if main_class_names[Mian_predicted_class] == 'Stroke':
            st.write("**Sub-Class Classifier:**")
            sub_model = Sub_Class_CNNModel_CT()
            sub_model.load_state_dict(torch.load('CT/cnn_model_sub_class.pth', map_location=torch.device('cpu')))
            sub_model.eval()
            sub_class_names = ['hemorrhagic', 'ischaemic']
            st.write("Loading Sub-Class Model...")
            st.write("Classifying sub-classes...")
            
            sub_pre_input_tensor = preprocess_image_sub_ct(image)
   
            with torch.no_grad():
                sub_output = sub_model(sub_pre_input_tensor)  
                sub_prediction = torch.argmax(sub_output, dim=1).item()
                confidence = sub_output[0][sub_prediction].item()
                
            with torch.no_grad():
                output = sub_model(input_tensor)  
                sub_prediction = torch.argmax(output, dim=1).item()
                confidence = output[0][sub_prediction].item()


            sub_predicted_class = 1 if sub_prediction >= 0.5 else 0
            confidence = sub_prediction if sub_predicted_class == 1 else 1 - sub_prediction
            
            st.write(f"**Sub-Predicted Class:** {sub_class_names[sub_predicted_class]}")
            st.write(f"**Confidence:** {confidence:.4f}")
            
        

st.set_page_config(
    page_title="Stroke Prediction",
    page_icon="🧠",
    layout="wide"
)

tab = st.sidebar.radio("📚 Types:", [
    "🧠 MRI",  
    "🧠 CT",
])


st.sidebar.image("Images_App/faculty_of_App;ied_Medical_Sciences.jpeg")
st.sidebar.markdown("# Faculty of Applied Health Sciences Technology")
st.sidebar.markdown("## **Supervisor:** Dr. Diana Abbas Al Sherif")
st.sidebar.markdown("**Team Members:**")
st.sidebar.markdown("1. Yousef Mohamed Abdalaal ")
st.sidebar.markdown("2. Nermen Khaled Ahmed")
st.sidebar.markdown("3. Shahd Amin AbdElwhab")
st.sidebar.markdown("1. Yousef Mohamed Abdalaal ")
st.sidebar.markdown("2. Nermen Khaled Ahmed")
st.sidebar.markdown("3. Shahd Amin AbdElwhab")
st.sidebar.markdown("4. Farah Khaled mohamed")
st.sidebar.markdown("5. Manar Mohamed Farid")


st.sidebar.markdown("")

st.sidebar.markdown("**Team Members:**")
st.sidebar.markdown("dr:Yousef Mohamed Abdalaal")
st.sidebar.image("Images_App/Yousef.jpeg")


if tab == "🧠 MRI":
    MRI_UI()
elif tab == "🧠 CT":
    CT_UI()

