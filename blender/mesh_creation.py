import bpy
import sys
import numpy as np
from scipy.io import loadmat

# Cleaning the scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

mat_path = sys.argv[-3]
faces_path = sys.argv[-2]
export_path = sys.argv[-1]

# Load 3D shape from .mat
data = loadmat(mat_path)
points = data['shape']
assert points.shape[1] == 3, "Point array must have shape (N, 3)"

# Load faces indices 
faces = np.load(faces_path)

# Conver to lists
verts_list = points.tolist()
faces_list = faces.tolist()

# Create mesh and object
mesh = bpy.data.meshes.new("BodyMesh")
mesh.from_pydata(verts_list, [], faces_list)
mesh.update()

obj = bpy.data.objects.new("BodyMeshObj", mesh)
bpy.context.collection.objects.link(obj)

# Suavizar y controlar Auto Smooth
#for poly in obj.data.polygons:
#    poly.use_smooth = True

#obj.data.use_auto_smooth = True
#obj.data.auto_smooth_angle = np.radians(90)

#  Exportar como .ply
#bpy.ops.object.select_all(action='DESELECT')
#obj.select_set(True)
#bpy.context.view_layer.objects.active = obj

#bpy.ops.export_mesh.ply(
#    filepath=export_path,
#    use_selection=True,
#    use_normals=True
#)

# Export to .obj
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

bpy.ops.wm.obj_export(
    filepath=export_path,
    export_selected_objects=True,
    export_normals=True,
    export_uv=False,
    export_materials=False,
    forward_axis='NEGATIVE_Z',
    up_axis='Y'
)

# Export to .glb
glb_export_path = export_path.replace(".obj", ".glb")
bpy.ops.export_scene.gltf(
    filepath=glb_export_path,
    export_format='GLB',
    use_selection=True
)