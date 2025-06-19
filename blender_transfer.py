import bpy
import numpy as np
import os
import sys
import math
import random
from mathutils import Vector, Matrix
import struct  # Needed for packing binary data for MDD

# --- MDD Export Helper ---


def export_mdd(filepath, all_vertices_frames):
    """
    Exports vertex animation data to an MDD file.

    Args:
        filepath (str): The path to save the .mdd file.
        all_vertices_frames (list or numpy array): A list or array where each element
                                                  represents a frame. Each frame should
                                                  contain a numpy array of shape (num_vertices, 3)
                                                  with the absolute vertex positions for that frame.
                                                  Ensure the vertex order matches Blender's internal order.
    """
    total_frames = len(all_vertices_frames)
    if total_frames == 0:
        print("Error exporting MDD: No frames provided.")
        return False

    num_vertices = all_vertices_frames[0].shape[0]
    if num_vertices == 0:
        print("Error exporting MDD: No vertices in the first frame.")
        return False

    print(
        f"Exporting MDD: {total_frames} frames, {num_vertices} vertices to {filepath}"
    )

    try:
        with open(filepath, "wb") as f:
            # Write Header
            f.write(struct.pack(">i", total_frames))  # Big-endian integer: total frames
            f.write(
                struct.pack(">i", num_vertices)
            )  # Big-endian integer: points per frame

            # Write frame times (usually just 0.0 to frames-1 for standard playback)
            # Alternatively, could use actual time values if needed, but less common for MDD.
            frame_times = np.arange(total_frames, dtype=">f").astype(
                ">f"
            )  # Big-endian float
            f.write(frame_times.tobytes())

            # Write vertex positions for each frame
            for frame_idx in range(total_frames):
                if all_vertices_frames[frame_idx].shape != (num_vertices, 3):
                    print(
                        f"Error exporting MDD: Frame {frame_idx} has incorrect shape "
                        f"{all_vertices_frames[frame_idx].shape}. Expected ({num_vertices}, 3)."
                    )
                    return False

                # Ensure data is float32 and pack as big-endian floats
                frame_data = all_vertices_frames[frame_idx].astype(np.float32)
                packed_data = struct.pack(f">{num_vertices * 3}f", *frame_data.ravel())
                f.write(packed_data)
                if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
                    print(f"  ...wrote frame {frame_idx+1}/{total_frames}")

        print("MDD export successful.")
        return True
    except IOError as e:
        print(f"Error exporting MDD: Could not write file '{filepath}'. {e}")
        return False
    except Exception as e:
        print(f"Error exporting MDD: An unexpected error occurred. {e}")
        return False


# --- Configuration (Adjust paths as needed) ---
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

object_name_to_copy = "Object_7"
# CUR_ANGLE = 34  # Used for output directory naming, not directly for animation here
# OUTPUT_BASE_DIR = f"/NAS/spa176/papr-retarget/blender/hummingbird/motion_{CUR_ANGLE}/"
EXP_ID = 10  # Used for output directory naming, not directly for animation here
OUTPUT_BASE_DIR = f"/NAS/spa176/papr-retarget/blender/hummingbird/exp_{EXP_ID}/"
# OUTPUT_BLEND_DIR = os.path.join(OUTPUT_BASE_DIR, "blend_output")
OUTPUT_BLEND_DIR = OUTPUT_BASE_DIR
# OUTPUT_CACHE_DIR = os.path.join(OUTPUT_BASE_DIR, "cache")  # Directory for MDD file
OUTPUT_RENDER_DIR = os.path.join(
    OUTPUT_BASE_DIR, "render_frames"
)  # Optional rendering output
OUTPUT_BLEND_FILENAME = f"{object_name_to_copy}_mesh_cache_animation.blend"
OUTPUT_MDD_FILENAME = f"{object_name_to_copy}_deformation.mdd"


# DEFORMATION_DATA_PATH = "/NAS/spa176/papr-retarget/fit_pointcloud_logs/blender_mp_wingL520_wingR520_body1134/exp_1/deformed_pts_smooth.npy"
# DEFORMATION_DATA_PATH = "/NAS/spa176/papr-retarget/fit_pointcloud_logs/blender_mp_wingL96_wingR96_body256/exp_3/deformed_pts_smooth.npy"
DEFORMATION_DATA_PATH = "/NAS/spa176/papr-retarget/fit_pointcloud_logs/blender_mp_wingL96_wingR96_body256/exp_10/deformed_pts_smooth.npy"

VERTEX_INDICES_DIR = "/NAS/spa176/papr-retarget/"
VERTEX_GROUP_NAMES = ["Body", "RightWing", "LeftWing"]

RENDER_ENABLED = False  # Set to True if you still want rendering
# ... other render settings ...

# --- Helper Functions (Keep setup_render_settings, setup_turntable_camera, etc. if RENDER_ENABLED) ---
# Include: find_object, setup_render_settings, setup_turntable_camera, _create_light, randomize_lighting, update_camera_position, clear_animation_data
# (Copy these functions from the previous script version)
RENDER_FRAME_COUNT = 60  # Number of frames for 360-degree view (e.g., 60 frames)
RENDER_RESOLUTION_X = 800
RENDER_RESOLUTION_Y = 800
RENDER_FILE_FORMAT = "JPEG"  # 'PNG', 'JPEG', etc.
VISUALIZE_SKIP = 5  # Step for visualization frames

# --- Helper Functions ---

# (Keep existing helper functions: find_object, setup_render_settings,
#  setup_turntable_camera, _create_light, randomize_lighting, update_camera_position)
# --- Helper Functions ---


def find_object(name, scene=None):
    """Finds an object by name in a specific scene or the current context scene."""
    context_scene = scene if scene else bpy.context.scene
    if name in context_scene.objects:
        return context_scene.objects[name]
    # Check across all data if not found in the scene (less common for typical objects)
    if name in bpy.data.objects:
        print(
            f"Warning: Object '{name}' found in bpy.data but not in scene '{context_scene.name}'. Returning from bpy.data."
        )
        return bpy.data.objects[name]
    return None


def setup_render_settings(output_path):
    """Configures basic render settings."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"  # Or 'CYCLES'
    scene.render.image_settings.file_format = RENDER_FILE_FORMAT
    scene.render.filepath = output_path  # Placeholder, will be set per frame
    scene.render.resolution_x = RENDER_RESOLUTION_X
    scene.render.resolution_y = RENDER_RESOLUTION_Y
    scene.render.resolution_percentage = 100
    # Optional: Set transparent background
    scene.render.film_transparent = False
    scene.eevee.taa_render_samples = 64  # Adjust quality for Eevee
    scene.eevee.taa_samples = 16


def setup_turntable_camera(target_obj):
    """Creates and positions a camera to orbit the target object."""
    # Ensure target_obj is valid
    if target_obj is None:
        print("Error: Cannot setup camera, target object is None.")
        return None, None

    # Create an empty at the object's location to act as pivot
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=target_obj.location)
    pivot = bpy.context.object
    pivot.name = "CameraPivot"

    # Create camera
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "TurntableCamera"

    # Determine distance based on object size (simple bounding box diagonal)
    # Use dimensions after potential scaling is applied
    dims = target_obj.dimensions
    if not any(dims):  # Check if dimensions are zero
        print("Warning: Target object dimensions are zero. Using default distance.")
        distance = 5.0
    else:
        distance = max(dims) * 1.8  # Adjust multiplier for framing
        if distance == 0:  # Handle edge case where max dimension is 0
            distance = 5.0

    # Position camera relative to pivot (move back along Y, up along Z)
    camera.location = (0, -distance, distance * 0.5)

    # Parent camera to pivot
    # Ensure pivot is in the same scene as the camera for parenting
    if camera.name not in bpy.context.scene.objects:
        bpy.context.scene.collection.objects.link(camera)
    if pivot.name not in bpy.context.scene.objects:
        bpy.context.scene.collection.objects.link(pivot)

    camera.parent = pivot

    # Make camera look at the pivot point
    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.target = pivot
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"

    # Make this the active scene camera
    bpy.context.scene.camera = camera

    print(
        f"Turntable camera '{camera.name}' set up orbiting '{pivot.name}' at distance ~{distance:.2f}"
    )
    return pivot, camera


def _create_light(
    name,
    light_type,
    location,
    rotation,
    energy,
    use_shadow=False,
    specular_factor=1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """
    # Ensure we are in the correct context for adding objects
    # bpy.context.view_layer.objects.active = None # Deselect all first

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(
        light_object
    )  # Link to current scene's collection
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object


def randomize_lighting():
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """

    # Clear existing lights IN THE CURRENT SCENE
    # Store lights to delete
    lights_to_delete = [obj for obj in bpy.context.scene.objects if obj.type == "LIGHT"]

    # Deselect all
    bpy.ops.object.select_all(action="DESELECT")

    # Select lights in the current scene
    for light in lights_to_delete:
        light.select_set(True)

    # Delete selected lights if any exist
    if lights_to_delete:
        bpy.ops.object.delete()
        print(
            f"Deleted {len(lights_to_delete)} lights from scene '{bpy.context.scene.name}'."
        )
    else:
        print("No lights found to delete in the current scene.")

    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),
        energy=random.choice([3, 4, 5]),
    )

    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),
        energy=random.choice([2, 3, 4]),
    )

    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),
        energy=random.choice([3, 4, 5]),
    )

    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),
        energy=random.choice([1, 2, 3]),
    )
    print("Created new randomized lighting setup.")

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )


def update_camera_position(obj, camera_pivot):
    """Updates camera pivot based on object's bounding box in world space."""
    if obj is None or camera_pivot is None:
        print("Warning: Cannot update camera position, object or pivot is None.")
        return

    # Calculate world space bounding box corners
    # Ensure object's matrix_world is up-to-date if object moved programmatically
    bpy.context.view_layer.update()

    bbox_corners_local = [Vector(corner) for corner in obj.bound_box]
    bbox_corners_world = [obj.matrix_world @ corner for corner in bbox_corners_local]

    if not bbox_corners_world:
        print("Warning: Could not calculate world bounding box corners.")
        return

    # Calculate new bounding box center
    new_bbox_center = sum(bbox_corners_world, Vector()) / 8.0

    # Update pivot position to new center
    camera_pivot.location = new_bbox_center

    # Calculate max dimension in world space for camera distance
    min_coord = Vector((min(c[i] for c in bbox_corners_world) for i in range(3)))
    max_coord = Vector((max(c[i] for c in bbox_corners_world) for i in range(3)))
    world_dims = max_coord - min_coord

    max_dim = max(world_dims)
    if max_dim <= 0:  # Prevent zero or negative distance
        print(
            "Warning: Calculated max dimension is non-positive. Using default distance adjustment."
        )
        max_dim = 5.0  # Default fallback dimension

    # Optional: adjust camera distance based on the new max dimension
    camera = next(
        (child for child in camera_pivot.children if child.type == "CAMERA"), None
    )
    if camera:
        new_distance = max_dim * 1.8
        # Adjust position relative to the pivot (assuming standard setup: back on Y)
        # Preserve existing camera Z offset relative to pivot if desired, or recalculate
        z_offset_ratio = (
            camera.location.z / abs(camera.location.y)
            if camera.location.y != 0
            else 0.5
        )
        camera.location.y = -new_distance
        camera.location.z = new_distance * z_offset_ratio
        print(f"Updated camera distance based on new max dimension: {new_distance:.2f}")
    else:
        print("Warning: Could not find camera child of pivot to adjust distance.")

    print(f"Updated camera pivot to new center: {new_bbox_center}")


def clear_animation_data(obj):
    """Removes animations, modifiers, and shape keys from an object."""
    print(f"Clearing animation data for object: {obj.name}")
    obj.animation_data_clear()
    if obj.data and hasattr(obj.data, "animation_data_clear"):
        obj.data.animation_data_clear()

    # Remove existing shape keys
    if obj.data and obj.data.shape_keys:
        print("  Removing existing shape keys...")
        # Set Basis key as active before clearing others might be safer sometimes
        # However, shape_key_clear() should handle this.
        # key_blocks = obj.data.shape_keys.key_blocks
        # if "Basis" in key_blocks:
        #      obj.active_shape_key_index = key_blocks.find("Basis")

        # Need to be in object mode to clear shape keys
        if bpy.context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        # Deselect all shape keys first (might help prevent context errors)
        # for sk in obj.data.shape_keys.key_blocks:
        #     sk.value = 0.0 # Reset values? Maybe not needed.

        # Select the object to operate on
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Remove all shape keys
        # This might require the object to be active and selected
        try:
            # Iterate backwards to avoid index issues when removing
            key_blocks = obj.data.shape_keys.key_blocks
            for i in range(len(key_blocks) - 1, -1, -1):
                obj.active_shape_key_index = i  # Select the key
                bpy.ops.object.shape_key_remove(all=False)  # Remove the selected key
            print("  Finished removing shape keys.")
            # Verify shape keys attribute is gone or empty
            if obj.data.shape_keys:
                print(
                    "  Warning: Shape keys attribute still exists after removal attempt."
                )
                # Force remove the shape key data block if needed (use with caution)
                # s_key = obj.data.shape_keys
                # obj.data.shape_keys = None
                # bpy.data.shape_keys.remove(s_key) # This might be risky if shared
                # print("  Attempted forced removal of shape_keys data block.")
            else:
                print("  Shape keys attribute successfully removed.")
        except Exception as e:
            print(f"  Error removing shape keys: {e}. Manual cleanup might be needed.")
            # Attempt setting shape_keys to None as fallback (might not work if data is complex)
            # try:
            #     obj.data.shape_keys = None
            # except:
            #     pass # Ignore if assignment is not allowed

    # Remove Armature modifiers (common source of animation)
    print("  Removing Armature modifiers...")
    for mod in list(obj.modifiers):  # Iterate over a copy
        if mod.type == "ARMATURE":
            print(f"    Removing modifier: {mod.name}")
            obj.modifiers.remove(mod)

    # Optionally remove other constraints or drivers if needed
    # print("  Clearing constraints...")
    # for constraint in list(obj.constraints):
    #     obj.constraints.remove(constraint)


# --- Main Script Logic ---


def main():
    global RENDER_ENABLED
    # --- 0. Preparation ---
    # Ensure output directories exist
    if not os.path.exists(OUTPUT_BLEND_DIR):
        os.makedirs(OUTPUT_BLEND_DIR)
    # if not os.path.exists(OUTPUT_CACHE_DIR):
    #     os.makedirs(OUTPUT_CACHE_DIR)
    if RENDER_ENABLED and not os.path.exists(OUTPUT_RENDER_DIR):
        os.makedirs(OUTPUT_RENDER_DIR)
    print(f"Output Blend Dir: {OUTPUT_BLEND_DIR}")
    # print(f"Output Cache Dir: {OUTPUT_CACHE_DIR}")
    if RENDER_ENABLED:
        print(f"Output Render Dir: {OUTPUT_RENDER_DIR}")

    # --- Define target file paths ---
    output_blend_path = os.path.join(OUTPUT_BLEND_DIR, OUTPUT_BLEND_FILENAME)
    # MDD file will reside in the SAME directory as the blend file
    mdd_filepath = os.path.join(OUTPUT_BLEND_DIR, OUTPUT_MDD_FILENAME)
    print(f"Target Blend File: {output_blend_path}")
    print(f"Target MDD File: {mdd_filepath}")

    # --- 1. Find Original Object ---
    original_scene = bpy.context.scene
    original_obj = find_object(object_name_to_copy, scene=original_scene)
    if original_obj is None or original_obj.type != "MESH":
        print(f"Error: Mesh object '{object_name_to_copy}' not found.")
        return
    print(f"Found original object: '{original_obj.name}'")

    # --- 2. Create New Scene ---
    print("Creating new scene...")
    bpy.ops.scene.new(type="NEW")
    new_scene = bpy.context.scene
    new_scene.name = f"{object_name_to_copy}_CachedScene"
    print(f"Switched to new scene: '{new_scene.name}'")

    # --- 3. Copy Object and Mesh Data ---
    print(f"Copying '{original_obj.name}'...")
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    copied_obj = original_obj.copy()
    if original_obj.data:
        copied_obj.data = original_obj.data.copy()  # Independent mesh data
    new_scene.collection.objects.link(copied_obj)
    bpy.context.view_layer.objects.active = copied_obj
    copied_obj.select_set(True)
    print(f"Copied object '{copied_obj.name}' with mesh '{copied_obj.data.name}'")

    # --- 4. Clear Existing Animations & Reset ---
    clear_animation_data(copied_obj)  # Use the function from previous script
    copied_obj.location = (0, 0, 0)
    copied_obj.rotation_euler = (0, 0, 0)
    copied_obj.scale = (1, 1, 1)
    try:
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    except:  # May fail if object has no geometry initially before cache load?
        print("Warning: Could not set origin initially.")
    print("Cleared animations and reset transforms.")

    # --- 5. Load Deformation Data & Prepare for MDD ---
    mesh = copied_obj.data
    num_vertices = len(mesh.vertices)
    print(f"Mesh has {num_vertices} vertices.")

    if not os.path.exists(DEFORMATION_DATA_PATH):
        print(f"Error: Deformation data file not found: {DEFORMATION_DATA_PATH}")
        return
    try:
        deformed_pts_partial = np.load(
            DEFORMATION_DATA_PATH
        )  # Shape (time, N, 3) where N <= num_vertices
    except Exception as e:
        print(f"Error loading deformation data: {e}")
        return

    total_time_steps = deformed_pts_partial.shape[0]
    num_deformed_vertices = deformed_pts_partial.shape[1]
    print(
        f"Loaded partial deformation data: {total_time_steps} steps for {num_deformed_vertices} vertices."
    )

    # Load the vertex indices corresponding to the partial deformation data
    print("Loading vertex indices for deformed parts...")
    all_part_vertex_indices = []
    try:
        indices_list = []
        for group_name in VERTEX_GROUP_NAMES:
            indices_path = os.path.join(
                VERTEX_INDICES_DIR, f"bird_{group_name}_vertex_indices.npy"
            )
            if not os.path.exists(indices_path):
                print(f"Error: Vertex indices file not found: {indices_path}")
                return
            cur_indices = np.load(indices_path).astype(np.int64)
            indices_list.append(cur_indices)
            print(f"  Loaded {len(cur_indices)} indices for '{group_name}'")
        all_part_vertex_indices = np.concatenate(indices_list, axis=0)
    except Exception as e:
        print(f"Error loading vertex indices: {e}")
        return

    if len(all_part_vertex_indices) != num_deformed_vertices:
        print(
            f"Error: Number of loaded indices ({len(all_part_vertex_indices)}) "
            f"doesn't match number of vertices in deformation data ({num_deformed_vertices})."
        )
        return
    if (
        np.max(all_part_vertex_indices) >= num_vertices
        or np.min(all_part_vertex_indices) < 0
    ):
        print(
            f"Error: Vertex indices out of bounds for mesh with {num_vertices} vertices."
        )
        return

    # --- CRITICAL STEP: Create FULL vertex position array for each frame ---
    print("Constructing full vertex arrays for MDD export...")
    all_vertices_mdd_frames = []
    # Get the basis shape (undeformed state) vertex coordinates
    basis_coords = np.empty((num_vertices, 3), dtype=np.float32)
    mesh.vertices.foreach_get("co", basis_coords.ravel())

    for time_step in range(total_time_steps):
        # Start with the basis coordinates for all vertices
        full_frame_coords = np.copy(basis_coords)
        # Overwrite the positions for the vertices that have deformation data
        # Ensure the order matches: deformed_pts_partial[time_step, i] corresponds to vertex index all_part_vertex_indices[i]
        for i, vert_index in enumerate(all_part_vertex_indices):
            full_frame_coords[vert_index] = deformed_pts_partial[time_step, i]

        all_vertices_mdd_frames.append(full_frame_coords)
        if time_step % 50 == 0 or time_step == total_time_steps - 1:
            print(f"  ...prepared full data for frame {time_step+1}/{total_time_steps}")

    # --- 6. Export MDD File ---
    # MDD file path is now in the same directory as the target blend file
    print(f"Exporting MDD file to: {mdd_filepath}")
    if not export_mdd(mdd_filepath, all_vertices_mdd_frames):
        print("MDD export failed. Aborting.")
        # Optional: Clean up partially created scene/object?
        return

    # --- 7. Save Blend File (First Time) ---
    # This is NECESSARY before setting a relative path in the modifier
    print(
        f"Saving Blender file (first pass) to establish relative path base: {output_blend_path}"
    )
    try:
        bpy.ops.wm.save_as_mainfile(filepath=output_blend_path, check_existing=False)
        print("  Blend file saved successfully.")
    except Exception as e:
        print(f"Error saving Blender file (first pass): {e}")
        print("Cannot proceed with relative path linking.")
        # Optional: Clean up exported MDD file?
        # if os.path.exists(mdd_filepath): os.remove(mdd_filepath)
        return

    # --- 8. Apply Mesh Cache Modifier with Relative Path ---
    print("Applying Mesh Cache modifier with relative path...")
    try:
        mesh_cache_mod = copied_obj.modifiers.new(
            name="DeformationCache", type="MESH_CACHE"
        )
        mesh_cache_mod.cache_format = "MDD"

        # --- Set Relative Path ---
        # Calculate path relative to the *just saved* blend file
        relative_mdd_path = bpy.path.relpath(mdd_filepath)
        mesh_cache_mod.filepath = relative_mdd_path
        # If successful, relative_mdd_path should start with "//"
        print(f"  Set modifier filepath to relative path: {relative_mdd_path}")
        # -------------------------

        mesh_cache_mod.forward_axis = "POS_Y"  # Adjust if needed
        mesh_cache_mod.up_axis = "POS_Z"  # Adjust if needed
        print(f"  Modifier '{mesh_cache_mod.name}' added and configured.")

    except Exception as e:
        print(f"Error adding Mesh Cache modifier: {e}")
        # The blend file is saved, but the modifier setup failed.
        return
    
    # --- 8. Set Scene Frame Range ---
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = total_time_steps - 1
    print(f"Set scene frame range to 0 - {total_time_steps - 1}")

    # --- 9. Optional Rendering Setup ---
    if RENDER_ENABLED:
        print("\n--- Setting up Scene for Rendering ---")
        # (Setup render settings, camera, lighting as before)
        # Note: update_camera_position might be useful here if the overall bounds change significantly
        setup_render_settings(os.path.join(OUTPUT_RENDER_DIR, "frame_"))
        pivot, camera = setup_turntable_camera(copied_obj)
        if pivot and camera:
            randomize_lighting()
            update_camera_position(
                copied_obj, pivot
            )  # Update based on initial state (frame 0)
            # Set world background etc. if needed

    # --- 10. Optional Rendering ---
    if RENDER_ENABLED and pivot and camera:
        print("\n--- Rendering Frames ---")
        # (Render loop as before, potentially using VISUALIZE_SKIP)
        # Make sure to update camera per frame if needed: update_camera_position(copied_obj, pivot) inside loop
        pass  # Add rendering loop code here if needed

    # --- 11. Save New Blend File ---
    output_blend_path = os.path.join(OUTPUT_BLEND_DIR, OUTPUT_BLEND_FILENAME)
    print(f"\n--- Saving New Blend File ---")
    print(f"Saving scene with Mesh Cache modifier to: {output_blend_path}")
    try:
        bpy.ops.wm.save_as_mainfile(filepath=output_blend_path, check_existing=False)
        print("Successfully saved the new Blender file.")
    except Exception as e:
        print(f"Error saving Blender file: {e}")

    print("\n--- Script Finished ---")


# --- Run Main ---
if __name__ == "__main__":
    # Ensure necessary helper functions (find_object, clear_animation_data, etc.)
    # are defined before calling main()
    # --- Placeholder for required helper functions from previous script ---
    def find_object(name, scene=None):
        """Finds an object by name in a specific scene or the current context scene."""
        context_scene = scene if scene else bpy.context.scene
        if name in context_scene.objects:
            return context_scene.objects[name]
        if name in bpy.data.objects:
            print(
                f"Warning: Object '{name}' found in bpy.data but not in scene '{context_scene.name}'. Returning from bpy.data."
            )
            return bpy.data.objects[name]
        return None

    def clear_animation_data(obj):
        """Removes animations, modifiers (Armature), and shape keys from an object."""
        print(f"Clearing animation data for object: {obj.name}")
        obj.animation_data_clear()
        if obj.data and hasattr(obj.data, "animation_data_clear"):
            obj.data.animation_data_clear()
        if obj.data and obj.data.shape_keys:
            print("  Removing existing shape keys...")
            if bpy.context.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            try:
                keys_to_remove = list(obj.data.shape_keys.key_blocks)  # Get keys
                for i in range(len(keys_to_remove) - 1, -1, -1):  # Iterate backwards
                    obj.active_shape_key_index = i  # Select by index
                    print(f"    Removing shape key: {obj.active_shape_key.name}")
                    bpy.ops.object.shape_key_remove(all=False)
                print("  Finished removing shape keys.")
                # Verify (optional)
                if obj.data.shape_keys and obj.data.shape_keys.key_blocks:
                    print(
                        f"  Warning: {len(obj.data.shape_keys.key_blocks)} shape keys remain after clearing attempt."
                    )
                else:
                    print("  Shape keys successfully cleared.")

            except Exception as e:
                print(
                    f"  Error removing shape keys: {e}. Object state: {obj.mode}, Active: {bpy.context.active_object.name if bpy.context.active_object else 'None'}, Selected: {[o.name for o in bpy.context.selected_objects]}"
                )
                # Fallback attempt (may not work reliably)
                # obj.shape_key_clear() # Simpler op, might fail less often?

        print("  Removing Armature modifiers...")
        for mod in list(obj.modifiers):
            if mod.type == "ARMATURE":
                print(f"    Removing modifier: {mod.name}")
                obj.modifiers.remove(mod)

    # Add other helpers if RENDER_ENABLED is True
    # --------------------------------------------------------------------

    main()
