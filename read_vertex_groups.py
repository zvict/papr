import bpy
import numpy as np
import os

# Specify the name of the object containing the mesh
object_name = "Object_7"  # Replace with the actual name of your object
save_dir = "/NAS/spa176/papr-retarget/"

# Get the object
obj = bpy.data.objects.get(object_name)
ordered_group_names = ["Body", "LeftWing", "RightWing"]

if obj and obj.type == "MESH":
    mesh = obj.data
    # save the vertices to a numpy array
    all_vertex_coords = np.empty((len(mesh.vertices), 3), dtype=np.float32)
    mesh.vertices.foreach_get("co", all_vertex_coords.ravel())

    # Save the vertices to a numpy array
    np.save(os.path.join(save_dir, f"bird_vertices.npy"), all_vertex_coords)

    for group_name in ordered_group_names:
        # Find the vertex group by name
        if group_name in obj.vertex_groups:
            vg = obj.vertex_groups.get(group_name)
            count = 0
            part_vertices = []
            vertex_indices = []
            for v in mesh.vertices:
                for group in v.groups:
                    if group.group == vg.index:
                        count += 1
                        part_vertices.append((v.co.x, v.co.y, v.co.z))
                        vertex_indices.append(v.index)
            # Convert to numpy array for further processing if needed
            part_vertices = np.array(part_vertices)
            vertex_indices = np.array(vertex_indices)
            # save the vertex indices to a text file
            np.save(
                os.path.join(save_dir, f"bird_{vg.name}_vertex_indices.npy"),
                vertex_indices,
            )
            # save the vertices to a text file
            np.save(
                os.path.join(save_dir, f"bird_{vg.name}_vertices.npy"),
                part_vertices,
            )
            print(f"Vertex Group '{vg.name}' has {count} vertices.")
        else:
            print(f"Warning: Vertex group '{group_name}' not found in the mesh")

    # # Access vertex groups through the object
    # if obj.vertex_groups:
    #     print(f"Vertex Groups for object: {obj.name}")
    #     for vg in obj.vertex_groups:
    #         if vg.name == "Body" or vg.name == "LeftWing" or vg.name == "RightWing":
    #             count = 0
    #             part_vertices = []
    #             for v in mesh.vertices:
    #                 for group in v.groups:
    #                     if group.group == vg.index:
    #                         count += 1
    #                         part_vertices.append((v.co.x, v.co.y, v.co.z))
    #             # Convert to numpy array for further processing if needed
    #             part_vertices = np.array(part_vertices)
    #             # save the vertices to a text file
    #             np.save(os.path.join(save_dir, f"bird_{vg.name}_vertices.npy"), part_vertices)
    #             print(f"Vertex Group '{vg.name}' has {count} vertices.")
    # else:
    #     print(f"Object '{obj.name}' has no vertex groups.")
else:
    print(f"Object '{object_name}' not found or is not a mesh.")
