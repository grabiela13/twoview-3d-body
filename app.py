import os
import csv
import cv2
import datetime
from flask import Flask, request, redirect
from preprocessing.silhouette import extract_silhouette
from preprocessing.crop_resize import crop_silhouette, resize_silhouette
from model.predict_3Dmodel import reconstruct_model
import subprocess

import sys

FACES_PATH = os.path.join(os.path.dirname(__file__), "blender", "caesar_faces.npy")

blender_exec = r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"

def run_blender_mesh_creation(mat_path, faces_path, obj_output_path):
    script_path = os.path.join(os.path.dirname(__file__), "blender", "mesh_creation.py")

    subprocess.run([
        blender_exec, "--background", "--python", script_path, "--",
        mat_path, faces_path, obj_output_path
    ], check=True)

def run_blender_rigging(obj_path, export_glb_path, height_m):
    script_path = os.path.join(os.path.dirname(__file__), "blender", "mesh_rigging.py")

    subprocess.run([
        blender_exec, "--background", "--python", script_path, "--",
        obj_path, export_glb_path, str(height_m)
    ], check=True)

# Folder to store all user uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, static_url_path='/uploads', static_folder='uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route for homepage
from flask import render_template

@app.route('/')
def index():
    return render_template('index.html', model_path=None)


# Route to handle form submission
@app.route('/upload', methods=['POST'])
def upload_file():
    # Get uploaded files and form data
    front = request.files['front']
    side = request.files['side']
    gender = request.form['gender']
    height_cm = request.form['height']

    # Create a unique folder for this user using timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"user_{timestamp}"
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder_name)
    os.makedirs(user_folder, exist_ok=True)

    # Save front and side images
    front_path = os.path.join(user_folder, 'front.jpg')
    side_path = os.path.join(user_folder, 'side.jpg')
    front.save(front_path)
    side.save(side_path)

    # Save metadata to CSV file
    metadata_file = os.path.join(user_folder, 'metadata.csv')
    with open(metadata_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['gender', 'height', 'front_image', 'side_image'])
        writer.writerow([gender, height_cm, 'front.jpg', 'side.jpg'])

    print(f"[INFO] Saved user input to {user_folder}")

    # Generate silhouettes
    front_sil_path = os.path.join(user_folder, "front_silhouette.png")
    side_sil_path = os.path.join(user_folder, "side_silhouette.png")

    extract_silhouette(front_path, front_sil_path)
    extract_silhouette(side_path, side_sil_path)

    # Crop and Resize silhouettes
    front_final_path = os.path.join(user_folder, "front_final.png")
    side_final_path = os.path.join(user_folder, "side_final.png")

    front_img = cv2.imread(front_sil_path, cv2.IMREAD_GRAYSCALE)
    front_cropped = crop_silhouette(front_img)
    resize_silhouette(front_cropped, front_final_path)

    side_img = cv2.imread(side_sil_path, cv2.IMREAD_GRAYSCALE)
    side_cropped = crop_silhouette(side_img)
    resize_silhouette(side_cropped, side_final_path)

    # Predict PCA and convert to 3D points
    model_3d_path = os.path.join(user_folder, "reconstructed_points")
    reconstruct_model(front_final_path, side_final_path, float(height_cm), save_path=model_3d_path)

    # 3D points to mesh
    mat_path = model_3d_path + ".mat"
    obj_output_path = model_3d_path + ".obj"
    run_blender_mesh_creation(mat_path, FACES_PATH, obj_output_path)

    # Mesh to rigged mesh
    height_m = float(height_cm) / 100
    rigged_glb_path = os.path.join(user_folder, "reconstructed_mesh.glb")
    run_blender_rigging(obj_output_path, rigged_glb_path, height_m)

    # Redirect with reconstructed shape
    relative_folder = os.path.relpath(user_folder, start=UPLOAD_FOLDER)
    return render_template('index.html', model_path=f"/uploads/{relative_folder}/reconstructed_mesh.glb")


if __name__ == '__main__':
    app.run(debug=True)
