# import trimesh
# import numpy as np
# from xml.etree import ElementTree as ET
# from scipy.spatial.transform import Rotation

# def parse_origin(origin_elem):
#     """Parse the origin tag and return position and rotation"""
#     if origin_elem is None:
#         return np.eye(4)
    
#     xyz = origin_elem.get('xyz', '0 0 0').split()
#     rpy = origin_elem.get('rpy', '0 0 0').split()
    
#     xyz = np.array([float(x) for x in xyz])
#     rpy = np.array([float(x) for x in rpy])
    
#     # Create a transformation matrix
#     transform = np.eye(4)
#     transform[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
#     transform[:3, 3] = xyz
    
#     return transform

# def get_joint_transform(joint_elem, joint_transforms):
#     """Recursively get the cumulative joint transform"""
#     parent = joint_elem.find('parent').get('link')
#     origin = parse_origin(joint_elem.find('origin'))
    
#     if parent == 'base_link' or parent == 'root':
#         return origin
    
#     # Find the parent joint
#     for joint in joint_transforms:
#         child = joint.find('child').get('link')
#         if child == parent:
#             parent_transform = get_joint_transform(joint, joint_transforms)
#             return parent_transform @ origin
    
#     return origin

# def urdf_collision_to_trimesh(urdf_file):
#     """Merge all URDF collision geometries into one trimesh"""
    
#     tree = ET.parse(urdf_file)
#     root = tree.getroot()
    
#     meshes = []
#     joints = root.findall('joint')
    
#     # Build a transform dictionary for each link
#     link_transforms = {}
#     link_transforms['base_link'] = np.eye(4)
#     link_transforms['root'] = np.eye(4)
    
#     # Compute the global transform for every link
#     for joint in joints:
#         child_link = joint.find('child').get('link')
#         parent_link = joint.find('parent').get('link')
#         origin = parse_origin(joint.find('origin'))
        
#         if parent_link in link_transforms:
#             link_transforms[child_link] = link_transforms[parent_link] @ origin
    
#     # Iterate over every link
#     for link in root.findall('link'):
#         link_name = link.get('name')
#         collision = link.find('collision')
        
#         if collision is None:
#             continue
        
#         # Get the global transform of the link
#         if link_name in link_transforms:
#             link_transform = link_transforms[link_name]
#         else:
#             link_transform = np.eye(4)
        
#         # Get the local transform of the collision
#         collision_origin = parse_origin(collision.find('origin'))
#         global_transform = link_transform @ collision_origin
        
#         # Get the geometry element
#         geometry = collision.find('geometry')
        
#         if geometry.find('box') is not None:
#             # Create a box mesh
#             size = geometry.find('box').get('size').split()
#             size = [float(s) for s in size]
#             mesh = trimesh.creation.box(size)
            
#         elif geometry.find('cylinder') is not None:
#             # Create a cylinder mesh
#             cylinder = geometry.find('cylinder')
#             radius = float(cylinder.get('radius'))
#             length = float(cylinder.get('length'))
#             mesh = trimesh.creation.cylinder(radius=radius, height=length)
            
#         elif geometry.find('sphere') is not None:
#             # Create a sphere mesh
#             sphere = geometry.find('sphere')
#             radius = float(sphere.get('radius'))
#             mesh = trimesh.creation.icosphere(radius=radius)
            
#         else:
#             continue
        
#         # Apply the transform
#         mesh.apply_transform(global_transform)
#         meshes.append(mesh)
    
#     # Merge all meshes
#     if meshes:
#         combined_mesh = trimesh.util.concatenate(meshes)
#         return combined_mesh
#     else:
#         return None

# # Usage example
# if __name__ == "__main__":
#     # Save your URDF as a file
#     urdf_file = "/home/szy/RDF/RDF/collision_avoidance_example/moma_base.urdf"
    
#     # Generate the combined collision mesh
#     collision_mesh = urdf_collision_to_trimesh(urdf_file)
    
#     if collision_mesh:
#         # Save as STL file
#         collision_mesh.export('collision_combined.stl')
        
#         # Or save as OBJ file
#         collision_mesh.export('collision_combined.obj')
        
#         # Visualize
#         collision_mesh.show()
        
#         print("Merge completed!")
#         print(f"Vertex count: {len(collision_mesh.vertices)}")
#         print(f"Face count: {len(collision_mesh.faces)}")
#     else:
#         print("No collision geometry found")
import trimesh
import xml.etree.ElementTree as ET
import numpy as np

def merge_urdf_visual_meshes(urdf_path, output_mesh_path):
    """
    Merge all visual meshes in a URDF
    """
    # Parse the URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Store all meshes
    meshes = []
    
    # Iterate through all links
    for link in root.findall('.//link'):
        link_name = link.get('name')
        
        # Iterate through all visual elements under the link
        for visual in link.findall('visual'):
            # Get mesh file path
            mesh_elem = visual.find('.//mesh')
            if mesh_elem is not None:
                mesh_filename = mesh_elem.get('filename')
                
                # Get the origin (position and rotation) of the visual
                origin = visual.find('origin')
                if origin is not None:
                    xyz = origin.get('xyz', '0 0 0').split()
                    rpy = origin.get('rpy', '0 0 0').split()
                    xyz = [float(x) for x in xyz]
                    rpy = [float(r) for r in rpy]
                else:
                    xyz = [0, 0, 0]
                    rpy = [0, 0, 0]
                
                # Load mesh
                try:
                    # Handle package:// paths
                    if mesh_filename.startswith('package://'):
                        mesh_filename = mesh_filename.replace('package://', '')
                        # Adjust the path as needed for your environment
                    
                    mesh = trimesh.load(mesh_filename)
                    
                    # Apply transform
                    transform = trimesh.transformations.compose_matrix(
                        translate=xyz,
                        angles=rpy
                    )
                    mesh.apply_transform(transform)
                    
                    meshes.append(mesh)
                    print(f"Loaded: {link_name} - {mesh_filename}")
                    
                except Exception as e:
                    print(f"Failed to load {mesh_filename}: {e}")
    
    # Merge all meshes
    if meshes:
        combined_mesh = trimesh.util.concatenate(meshes)
        combined_mesh.export(output_mesh_path)
        print(f"\nSuccessfully merged {len(meshes)} meshes and saved to: {output_mesh_path}")
        return combined_mesh
    else:
        print("No mesh files found")
        return None

# Example usage
# urdf_file = "<path_to_urdf_file>"
# output_file = "merged_visual.stl"  # or .stl, .ply, etc.
# merge_urdf_visual_meshes(urdf_file, output_file)
