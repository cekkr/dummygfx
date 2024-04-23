def process_obj_file(input_file, output_file):
    vertices = []  # To store vertex coordinates
    triangles = []  # To store triangles

    # Read the OBJ file
    with open(input_file, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex definition
                parts = line.split()
                # Add vertex tuple (convert each part to float)
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith('f '):  # Face definition
                parts = line.split()
                face_vertices = [int(p.split('/')[0]) for p in parts[1:]]  # Get vertex indices, ignore texture/normal
                # Assume the face is a triangle or decompose polygons into triangles
                for i in range(1, len(face_vertices) - 1):
                    triangles.append((face_vertices[0], face_vertices[i], face_vertices[i + 1]))

    # Write output file
    with open(output_file, 'w') as file:
        for tri in triangles:
            for idx in tri:
                v = vertices[idx - 1]  # OBJ indices are 1-based
                file.write(f"{v[0]},{v[1]},{v[2]} ")
            file.write("\n")


# Example usage:
input_path = 'complexModel.obj'
output_path = 'complexModel.txt'
process_obj_file(input_path, output_path)
