# Automated 3D Human Shape Prediction from Two Photographs

This project allows users to upload a **front** and **side** image of a person, along with height and gender, to automatically generate a realistic 3D human mesh. The resulting model is rigged and exported as a `.glb` file for visualization directly in the browser.

## Authors

Gabriela Ayala, Guillermo Benitez (https://github.com/Guillermelo), Santiago Benitez (https://github.com/Santibenitezl03)
*Universidad Politécnica Taiwan Paraguay*   

Chi-Pin Tsai, Chen-Yu Yuan, Tzung-Han Lin*  
*National Taiwan University of Science and Technology*


## Live Demo

![Demo](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExeXB2OTRzNXVta3lybjlndXUweXc1aXc3bzdrdDhyMGN4NmQwbHhoYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/8OYElLoCJjqR7OKJJm/giphy.gif)

Actual duration of execution is about 1 to 3 minutes. 

---

## Features

* Automatic silhouette extraction using DeepLabV3
* Cropping and resizing to standardize input
* Shape prediction from dual-view images using a PCA-based ML model
* Mesh creation with Blender (run in headless mode)
* Auto-rigging and scaling via Rigify in Blender
* Outputs a `.glb` file viewable directly in the browser

The full pipeline is served via a Flask web app. Upon user upload, the server processes the images and automatically invokes Blender in background (headless) mode using command-line calls (--background --python script.py -- args...). This allows fully automated mesh creation and rigging without any manual interaction.

As the output is a rigged mesh, this can be fully animated. This feature is not yet added to the project, but a quick demonstration can be seen bellow:   
![Animation](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExdnBqem9icGhma2NpNWE2djJxc2NrZTBicG5uM25rNG1jcjI5azJhcSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/O0uoZE85d3yLlADtpm/giphy.gif)

---

## Folder Structure

```
├── app.py                   # Flask app            
├── templates/
│   ├── index.html           # Web UI for uploads
├── blender/
│   ├── mesh_creation.py     # Mesh generation in Blender
│   ├── mesh_rigging.py      # Rigify-based auto-rigging
│   └── caesar_faces.npy     # Face topology for mesh creation
├── preprocessing/
│   ├── silhouette.py        # DeepLabV3 person segmentation
│   └── crop_resize.py       # Cropping and resizing utility
├── model/
│   └── predict_3Dmodel.py   # PCA reconstruction logic 
│   └── shape_predictor.pth  # Trained PyTorch model predicting PCA shape parameters (*)
│   └── pca_files/
│       ├── pca_features.npy
│       └── pca_model.pkl
├── uploads/                 # Auto-created user-specific folders
│   └── user_<timestamp>/
│       ├── input images
│       ├── silhouettes
│       ├── metadata.csv
│       └── output model (.glb)
├── README.md
```
(*) This file is not included due to GitHub’s 100MB limit. It will be available soon via external link.  

---

## Setup Instructions

### 1. Requirements

* Python 3.10+
* Blender 4.2+ with Rigify add-on enabled
* pip packages:

  ```bash
  pip install flask torch torchvision opencv-python numpy scipy
  ```

### 2. Blender Path

Set your Blender executable path in `app.py`:

```python
blender_exec = r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
```

### 3. Run the App

```bash
python app.py
```

Go to `http://localhost:5000` in your browser.

---

## Inputs

* Front photo (`.jpg`, `.png`)
* Side photo (`.jpg`, `.png`)
* Gender (male/female)
* Height (in centimeters)

Note: gender is currently not being used for the shape prediction. Feature added for future improvements. 

---

## Output

* A rigged `.glb` file automatically visualized in-browser.
* Stored under `uploads/user_<timestamp>/reconstructed_mesh.glb`.

---

## Acknowledgments

* DeepLabV3 from PyTorch for person segmentation
* PCA human shape model derived from MPI Human Shape dataset, CAESAR-fitted meshes (http://humanshape.mpi-inf.mpg.de/)
* Blender + Rigify for mesh rigging and export
