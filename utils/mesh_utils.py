import os
import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation


def convert_to_mesh(scene_or_mesh):
    """Convert a scene or mesh to a single Trimesh object with only vertex and face data.

    Args:
        scene_or_mesh: Input Trimesh scene or mesh object

    Returns:
        Trimesh object or None if the scene is empty
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if not scene_or_mesh.geometry:
            return None
        return trimesh.util.concatenate(tuple(
            trimesh.Trimesh(vertices=geom.vertices, faces=geom.faces)
            for geom in scene_or_mesh.geometry.values()
        ))

    assert isinstance(scene_or_mesh, trimesh.Trimesh), f"Expected Trimesh, got {type(scene_or_mesh)}"
    return scene_or_mesh


def parse_urdf_colors(urdf_path):
    """Extract color information from URDF file for links and global materials.

    Args:
        urdf_path: Path to the URDF file

    Returns:
        Dictionary mapping link names to RGBA color values
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Parse global materials
    global_materials = {
        material.attrib["name"]: [float(c) for c in material.find("color").attrib["rgba"].split()]
        for material in root.findall("material")
        if material.find("color") is not None and "rgba" in material.find("color").attrib
    }

    # Parse link-specific colors
    link_colors = {}
    for link in root.iter("link"):
        link_name = link.attrib["name"]
        visual = link.find("./visual/material")
        if visual is None:
            continue

        color = visual.find("color")
        if color is not None and "rgba" in color.attrib:
            link_colors[link_name] = [float(c) for c in color.attrib["rgba"].split()]
        elif "name" in visual.attrib and visual.attrib["name"] in global_materials:
            link_colors[link_name] = global_materials[visual.attrib["name"]]

    return link_colors


def parse_transform(element):
    """Parse transformation (translation and rotation) from an XML element.

    Args:
        element: XML element containing origin information

    Returns:
        Tuple of (translation vector, rotation matrix)
    """
    origin = element.find("origin")
    translation = np.zeros(3)
    rotation = np.eye(3)

    if origin is not None:
        translation = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
        rotation = Rotation.from_euler("xyz", rpy).as_matrix()

    return translation, rotation


def apply_transformation(mesh, translation, rotation):
    """Apply translation and rotation transformations to a mesh.

    Args:
        mesh: Input Trimesh object
        translation: 3D translation vector
        rotation: 3x3 rotation matrix

    Returns:
        Transformed Trimesh object
    """
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return mesh.apply_transform(transform)


def create_primitive_geometry(geometry, translation, rotation):
    """Create a Trimesh object from primitive geometry with applied transformations.

    Args:
        geometry: XML element containing geometry definition
        translation: 3D translation vector
        rotation: 3x3 rotation matrix

    Returns:
        Transformed Trimesh object

    Raises:
        ValueError: If geometry type is unsupported
    """
    if geometry.tag.endswith("box"):
        size = np.fromstring(geometry.attrib["size"], sep=" ")
        mesh = trimesh.creation.box(extents=size)
    elif geometry.tag.endswith("sphere"):
        radius = float(geometry.attrib["radius"])
        mesh = trimesh.creation.icosphere(radius=radius)
    elif geometry.tag.endswith("cylinder"):
        radius = float(geometry.attrib["radius"])
        length = float(geometry.attrib["length"])
        mesh = trimesh.creation.cylinder(radius=radius, height=length)
    else:
        raise ValueError(f"Unsupported geometry type: {geometry.tag}")

    return apply_transformation(mesh, translation, rotation)


def load_link_meshes(urdf_path, link_names, use_collision=False):
    """Load Trimesh objects for specified links from a URDF file.

    Args:
        urdf_path: Path to the URDF file
        link_names: List of link names to process
        use_collision: If True, use collision geometry instead of visual

    Returns:
        Dictionary mapping link names to their Trimesh objects
    """
    urdf_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    link_colors = parse_urdf_colors(urdf_path)
    geometry_type = "collision" if use_collision else "visual"
    link_meshes = {}

    for link in tree.getroot().findall("link"):
        link_name = link.attrib["name"]
        if link_name not in link_names:
            continue

        meshes = []
        for geom in link.findall(f".//{geometry_type}"):
            geometry = geom.find("geometry")
            translation, rotation = parse_transform(geom)

            try:
                if geometry[0].tag.endswith("mesh"):
                    mesh_path = os.path.join(urdf_dir, geometry[0].attrib["filename"])
                    mesh = convert_to_mesh(trimesh.load(mesh_path))
                else:
                    mesh = create_primitive_geometry(geometry[0], translation, rotation)

                scale = np.fromstring(geometry[0].attrib.get("scale", "1 1 1"), sep=" ")
                mesh.apply_scale(scale)
                meshes.append(apply_transformation(mesh, translation, rotation))

            except Exception as e:
                print(f"Failed to load geometry for {link_name}: {str(e)}")
                continue

        if not meshes:
            continue

        # Combine multiple meshes if present
        final_mesh = convert_to_mesh(trimesh.Scene(meshes)) if len(meshes) > 1 else meshes[0]

        # Apply color if available
        if link_name in link_colors:
            final_mesh.visual.face_colors = np.array(link_colors[link_name])

        link_meshes[link_name] = final_mesh

    return link_meshes
