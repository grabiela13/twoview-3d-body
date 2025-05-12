import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import pickle
from scipy.io import savemat
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn

# Paths
BASE_DIR = os.path.dirname(__file__)
PCA_DIR = os.path.join(BASE_DIR, "pca_files")
MODEL_PATH = os.path.join(BASE_DIR, "shape_predictor.pth")

# Load PCA data
with open(os.path.join(PCA_DIR, "pca_model.pkl"), "rb") as f:
    pca_model = pickle.load(f)
pca_features = np.load(os.path.join(PCA_DIR, "pca_features.npy"))

#  PyTorch model
class DualStreamWithHeight(nn.Module):
    def __init__(self, output_dim=50):
        super(DualStreamWithHeight, self).__init__()

        # Convolutions for frontal image
        self.conv1_front = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2_front = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_front = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Convolutions for lateral image
        self.conv1_side = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2_side = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3_side = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # FC: 1 unit is added for height
        self.fc1 = nn.Linear(128 * 32 * 32 * 2 + 1, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, front, side, height):
        # Frontal branch
        xf = self.pool(F.relu(self.conv1_front(front)))
        xf = self.pool(F.relu(self.conv2_front(xf)))
        xf = self.pool(F.relu(self.conv3_front(xf)))
        xf = xf.view(xf.size(0), -1)

        # Lateral branch
        xs = self.pool(F.relu(self.conv1_side(side)))
        xs = self.pool(F.relu(self.conv2_side(xs)))
        xs = self.pool(F.relu(self.conv3_side(xs)))
        xs = xs.view(xs.size(0), -1)

        # Concatenate both branchs
        x = torch.cat((xf, xs), dim=1)

        # Concatenate normalized height (shape: [B, 1])
        x = torch.cat((x, height), dim=1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#  Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dim = 50 
model = DualStreamWithHeight(output_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

#  Main function: predicts PCA and saves 3D mesh as .mat
def reconstruct_model(front_img_path, side_img_path, user_height_cm, save_path):
    # Load and transform images
    front_img = cv2.imread(front_img_path, cv2.IMREAD_GRAYSCALE)
    side_img = cv2.imread(side_img_path, cv2.IMREAD_GRAYSCALE)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    front_tensor = transform(front_img).unsqueeze(0).to(device)
    side_tensor = transform(side_img).unsqueeze(0).to(device)

    # Normalize height
    height_norm = torch.tensor([(user_height_cm * 10 - 1300) / (2200 - 1300)], dtype=torch.float32).unsqueeze(0).to(device)

    # Predict PCA
    with torch.no_grad():
        pred_pca_norm = model(front_tensor, side_tensor, height_norm).cpu().numpy()

    # Denormalize 
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(pca_features)
    pred_pca = scaler.inverse_transform(pred_pca_norm.reshape(1, -1)).flatten()

    # Reconstruct 3D mesh
    reconstructed = pca_model.inverse_transform(pred_pca).reshape(-1, 3)

    # Save as .mat file
    savemat(f"{save_path}.mat", {"shape": reconstructed})
    print(f"3D reconstructed shape saved at {save_path}.mat")
