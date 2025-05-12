import bpy
import mathutils
import math
import sys
import os

# Ensure Object Mode is active before making changes
if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')


argv = sys.argv
argv = argv[argv.index("--") + 1:]
model_path = argv[0]
export_path = argv[1]
target_height = float(argv[2])
rig_default_height = target_height

# Clean Scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import Model
bpy.ops.wm.obj_import(filepath=model_path)
print("Model imported.")

# Find the model mesh
model = None
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        model = obj
        break
if model is None:
    raise Exception("Humanoid model not found after import.")
    
    
# Create a new dark gray material
black_mat = bpy.data.materials.new(name="DarkGrayMaterial")
black_mat.use_nodes = True
bsdf = black_mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs['Base Color'].default_value = (0.2, 0.2, 0.2, 1)  # RGBA: dark gray
    bsdf.inputs['Roughness'].default_value = 0.5

# Assign the dark gray material to the model
if model.data.materials:
    model.data.materials[0] = black_mat  # Replace existing material
else:
    model.data.materials.append(black_mat)  # Add new material
    


# Apply initial transformations
bpy.context.view_layer.objects.active = model
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Rotate the model to look at the front 
model.rotation_euler.z += math.radians(45)
bpy.context.view_layer.objects.active = model
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
print("Model rotated to face forward (Y+).")

# Calculate Bounding Box of original Model
bbox = [model.matrix_world @ mathutils.Vector(corner) for corner in model.bound_box]
min_z = min([v.z for v in bbox])
max_z = max([v.z for v in bbox])
mid_x = (max([v.x for v in bbox]) + min([v.x for v in bbox])) / 2
mid_y = (max([v.y for v in bbox]) + min([v.y for v in bbox])) / 2
model_real_height = max_z - min_z
print(f"Actual model height before scaling: {model_real_height:.2f}m")

# Scale the model to be equal to META-RIG

scale_factor = rig_default_height / model_real_height
print(f"Scaling model with factor: {scale_factor:.2f}")

model.scale *= scale_factor
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Recalculate Bounding box of model
bbox = [model.matrix_world @ mathutils.Vector(corner) for corner in model.bound_box]
min_z = min([v.z for v in bbox])
max_z = max([v.z for v in bbox])
mid_x = (max([v.x for v in bbox]) + min([v.x for v in bbox])) / 2
mid_y = (max([v.y for v in bbox]) + min([v.y for v in bbox])) / 2
model_real_height = max_z - min_z
print(f"New height of scaled model: {model_real_height:.2f}m")

# Put the Feet of the model in z = 0
offset_z = -min_z
model.location.z += offset_z
bpy.context.view_layer.objects.active = model
bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
print(f"Model moved down by {offset_z:.3f} units to touch the ground.")

# Create META-RIG
bpy.ops.object.armature_human_metarig_add()
metarig = bpy.context.active_object
metarig.name = "Auto_Humanoid_Rig"

# Scale the metarig to always scale with the mesh
bpy.context.view_layer.objects.active = metarig
bpy.ops.object.mode_set(mode='OBJECT')
metarig.scale *= 0.87
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# Adjust Bones Manually 
bpy.context.view_layer.objects.active = metarig
bpy.ops.object.mode_set(mode='EDIT')
bones = metarig.data.edit_bones

# Edit Mode to adjust bones
bpy.context.view_layer.objects.active = metarig
bpy.ops.object.mode_set(mode='EDIT')
bones = metarig.data.edit_bones

# key vertices indexes ------ Edit this part more to fit more accurately in every mesh
vertex_indices = {
    'head_top': 1180,
    'neck_base': 4039,
    'shoulder.L': 856,
    'shoulder.R': 4129,
    'elbow.L': 2714,
    'elbow.R': 5817,
    'hand.L': 2996,    
    'hand.R': 6117,    
    'hip.L': 387,
    'hip.R': 3576,
    'knee.L': 238,
    'knee.R': 4831,
    'foot.L': 1280,
    'foot.R': 4335,
    'toe.L': 170,            
    'toe.R': 6406,           
    'neck': 66,             
    'pelvis.L': 449,         
    'pelvis.R': 478,         
    'heel.L': 4412,  
    'heel.R': 1281  

}

# Get 3D coordinates of key vertices
coords = {}
for name, index in vertex_indices.items():
    coords[name] = model.matrix_world @ model.data.vertices[index].co

# Position bones using vertices and interpolated vectors

# Calculate key points
pelvis_center = (coords['hip.L'] + coords['hip.R']) / 2
neck_base = coords['neck_base']
spine_points = [pelvis_center.lerp(neck_base, i / 5) for i in range(6)]  # 6 points → 5 bones

# Assign spine bones
spine_names = ["spine", "spine.001", "spine.002", "spine.003", "spine.004"]
for i, name in enumerate(spine_names):
    if name in bones:
        bones[name].head = spine_points[i]
        bones[name].tail = spine_points[i + 1]

# Head and neck using spine.005 as neck and spine.006 as head
if 'spine.005' in bones:
    bones['spine.005'].head = coords['neck_base']
    bones['spine.005'].tail = coords['neck_base'] + mathutils.Vector((0, 0, 0.05))
    bones['spine.005'].parent = bones['spine.004']
    bones['spine.005'].use_connect = True

if 'spine.006' in bones:
    bones['spine.006'].head = bones['spine.005'].tail
    bones['spine.006'].tail = coords['head_top']
    bones['spine.006'].parent = bones['spine.005']
    bones['spine.006'].use_connect = True

# Shoulders
if 'shoulder.L' in bones:
    bones['shoulder.L'].head = coords['shoulder.L']
    bones['shoulder.L'].tail = coords['shoulder.L'] + mathutils.Vector((0.1, 0, 0))
if 'shoulder.R' in bones:
    bones['shoulder.R'].head = coords['shoulder.R']
    bones['shoulder.R'].tail = coords['shoulder.R'] + mathutils.Vector((-0.1, 0, 0))

# Arms
if 'upper_arm.L' in bones:
    bones['upper_arm.L'].head = coords['shoulder.L']
    bones['upper_arm.L'].tail = coords['elbow.L']
if 'upper_arm.R' in bones:
    bones['upper_arm.R'].head = coords['shoulder.R']
    bones['upper_arm.R'].tail = coords['elbow.R']
if 'forearm.L' in bones:
    bones['forearm.L'].head = coords['elbow.L']
    bones['forearm.L'].tail = coords['hand.L']
if 'forearm.R' in bones:
    bones['forearm.R'].head = coords['elbow.R']
    bones['forearm.R'].tail = coords['hand.R']
if 'hand.L' in bones:
    bones['hand.L'].head = coords['hand.L']
    bones['hand.L'].tail = coords['hand.L'] + mathutils.Vector((0.1, 0.05, 0))
if 'hand.R' in bones:
    bones['hand.R'].head = coords['hand.R']
    bones['hand.R'].tail = coords['hand.R'] + mathutils.Vector((-0.1, 0.05, 0))

# Legs for Rigify

# Thigh to shin
if 'thigh.L' in bones:
    bones['thigh.L'].head = coords['hip.L']
    bones['thigh.L'].tail = coords['knee.L']
    bones['thigh.L'].use_connect = False

if 'thigh.R' in bones:
    bones['thigh.R'].head = coords['hip.R']
    bones['thigh.R'].tail = coords['knee.R']
    bones['thigh.R'].use_connect = False

if 'shin.L' in bones:
    bones['shin.L'].head = coords['knee.L']
    bones['shin.L'].tail = coords['foot.L']
    bones['shin.L'].parent = bones['thigh.L']
    bones['shin.L'].use_connect = True

if 'shin.R' in bones:
    bones['shin.R'].head = coords['knee.R']
    bones['shin.R'].tail = coords['foot.R']
    bones['shin.R'].parent = bones['thigh.R']
    bones['shin.R'].use_connect = True

# Rename heel.02.X to heel.X if needed
if 'heel.02.L' in bones and 'heel.L' not in bones:
    bones['heel.02.L'].name = 'heel.L'
if 'heel.02.R' in bones and 'heel.R' not in bones:
    bones['heel.02.R'].name = 'heel.R'

# Foot to toe
if 'foot.L' in bones and 'toe.L' in coords:
    bones['foot.L'].head = coords['foot.L']
    bones['foot.L'].tail = coords['toe.L']
    bones['foot.L'].parent = bones['shin.L']
    bones['foot.L'].use_connect = True

if 'foot.R' in bones and 'toe.R' in coords:
    bones['foot.R'].head = coords['foot.R']
    bones['foot.R'].tail = coords['toe.R']
    bones['foot.R'].parent = bones['shin.R']
    bones['foot.R'].use_connect = True

# Toes
if 'toe.L' in bones and 'toe.L' in coords:
    bones['toe.L'].head = bones['foot.L'].tail
    bones['toe.L'].tail = coords['toe.L'] + mathutils.Vector((0, 0.05, 0))
    bones['toe.L'].parent = bones['foot.L']
    bones['toe.L'].use_connect = True

if 'toe.R' in bones and 'toe.R' in coords:
    bones['toe.R'].head = bones['foot.R'].tail
    bones['toe.R'].tail = coords['toe.R'] + mathutils.Vector((0, 0.05, 0))
    bones['toe.R'].parent = bones['foot.R']
    bones['toe.R'].use_connect = True

# Heels (create new if not present)
if 'heel.L' not in bones:
    heel_L = bones.new('heel.L')
    heel_L.head = coords['heel.L']
    heel_L.tail = coords['heel.L'] + mathutils.Vector((0, 0, -0.05))
    heel_L.parent = bones['foot.L']
    heel_L.use_connect = False

if 'heel.R' not in bones:
    heel_R = bones.new('heel.R')
    heel_R.head = coords['heel.R']
    heel_R.tail = coords['heel.R'] + mathutils.Vector((0, 0, -0.05))
    heel_R.parent = bones['foot.R']
    heel_R.use_connect = False

# Secure Object Mode 
if bpy.context.object and bpy.context.object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

# Allow Rigify to rig only the essential bones 
essential_bones = {
    'spine', 'spine.001', 'spine.002', 'spine.003', 'spine.004','spine.005','spine.006',
    'shoulder.L', 'shoulder.R',
    'upper_arm.L', 'upper_arm.R',
    'forearm.L', 'forearm.R',
    'hand.L', 'hand.R',
    'thigh.L', 'thigh.R',
    'shin.L', 'shin.R',
    'foot.L', 'foot.R',
    'toe.L', 'toe.R',
    'neck', 'head',
    'pelvis.L', 'pelvis.R',
    'heel.L', 'heel.R',
    'heel.02.L', 'heel.02.R'
}

# Switch to pose mode to access rigify_type
bpy.context.view_layer.objects.active = metarig
bpy.ops.object.mode_set(mode='POSE')

# Disable rigify_type for non-essential bones
for bone in metarig.pose.bones:
    if bone.name not in essential_bones:
        if "rigify_type" in bone:
            bone["rigify_type"] = ""

bpy.ops.object.mode_set(mode='OBJECT')
print("Rigify will only rig the essential bones.")

# Select META-RIG and generate final RIG
# Clean selection Manually 
for obj in bpy.context.scene.objects:
    obj.select_set(False)

metarig.select_set(True)
bpy.context.view_layer.objects.active = metarig


# Generate the rig with Rigify
try:
    bpy.ops.pose.rigify_generate()
    print("Rigify: we tried to generate the rig.")
except Exception as e:
    raise Exception("Error generating the rig with Rigify. Make sure the metarig has the minimum required structure.") from e

bpy.context.view_layer.update()

# Look for the new rig generated
rig = None
for obj in bpy.context.scene.objects:
    if obj.type == 'ARMATURE' and obj != metarig and obj.name.startswith("rig"):
        rig = obj
        print(f"Rig found: {rig.name}")
        break

# If not found, try to locate one as a fallback.
if rig is None:
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE' and obj != metarig:
            rig = obj
            print(f"Rig found by fallback: {rig.name}")
            break

if rig is None:
    raise Exception("Generated rig not found.")

# PARENT WITH AUTOMATIC WEIGHTS
# Ensure the model has no old parent
model.select_set(True)
bpy.context.view_layer.objects.active = model
bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

# Select mesh and rig properly
for obj in bpy.context.scene.objects:
    obj.select_set(False)

model.select_set(True)
rig.select_set(True)
bpy.context.view_layer.objects.active = rig

# Apply parent with automatic weights
bpy.ops.object.parent_set(type='ARMATURE_AUTO')
print("Parent with automatic weights applied.")
    
# Erase the META-RIG from scene
# comment this part to recalibrate the joints positions in vertex indices
bpy.data.objects.remove(metarig, do_unlink=True)
print("META-RIG deleted from the scene.")


# Deselect all objects in the scene
for obj in bpy.context.scene.objects:
    obj.select_set(False)

# Select only the model and the generated rig
model.select_set(True)
rig.select_set(True)
bpy.context.view_layer.objects.active = model  # or the rig

# Export to .glb format
bpy.ops.export_scene.gltf(
    filepath=export_path,
    export_format='GLB',  # 'GLTF_SEPARATE' for .gltf + .bin + textures
    export_apply=True,
    use_selection=True,
    export_animations=True,
    export_skins=True,
    export_yup=True,
)

print(f"Exported rigged model to {export_path}")

# Save the Blender file
blend_save_path = os.path.join(os.path.dirname(export_path), "rigged_model.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_save_path)
print(f"File .blend saved at {blend_save_path}")

# Object Mode
bpy.ops.object.mode_set(mode='OBJECT')
