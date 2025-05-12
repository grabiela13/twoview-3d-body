import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet101

# Load DeepLabV3 once globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet101(pretrained=True).to(device).eval()

def extract_silhouette(image_path, save_path):

    # Load and preprocess image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image_rgb.shape[:2]

    # Preprocess for DeepLabV3
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image_rgb).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()

    # Rescale to original size if it was modified internally
    if mask.shape != (original_h, original_w):
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # Convert to binary (person = 15)
    silhouette = (mask == 15).astype(np.uint8) * 255

    # Save result
    cv2.imwrite(save_path, silhouette)
    print(f"Silhouette saved: {save_path}")

    return silhouette