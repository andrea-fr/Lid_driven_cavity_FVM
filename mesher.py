import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import os
import pandas as pd


def edge_normal_outward(p1, p2, centroid):
    edge_vec = p2 - p1 
    normal = np.array([edge_vec[1], -edge_vec[0]])
    length = np.linalg.norm(edge_vec)
    normal = normal / np.linalg.norm(normal)
    midpoint = 0.5 * (p1 + p2)
    if np.dot(midpoint - centroid, normal) < 0:
        normal = -normal
    return normal, length, midpoint


lx=1
ly=1
nx=100 #Insert an even number to allow for comparison with the results of Ghia et al.
ny=nx
x_bottom=np.linspace(0,lx,nx)
y_bottom=np.zeros(nx)
grid_space=x_bottom[1]

##############################
# STRUCTURED MESH
##############################
coord_points = [np.vstack((x_bottom, y_bottom)).T] 
for i in range (nx-1):
    y_bottom=np.ones(nx)*grid_space*(i+1)
    coord_points.append(np.vstack((x_bottom, y_bottom)).T)

quad_points=[] #to list of tuples
for ylayers in coord_points:
    for point in ylayers:
        quad_points.append(point.tolist())

Nx = len(x_bottom)
quad_cells = []

for i in range(Nx-1):          # between layer k e k+1
    for j in range(Nx-1):
        n0 = i * Nx + j
        n1 = i * Nx + j + 1
        n2 = (i + 1) * Nx + j + 1
        n3 = (i + 1) * Nx + j  
        quad_cells.append([n0, n1, n2, n3])

global_points=np.array(quad_points)

#####################################
# MESH FEATURES 
####################################
cells = []
# 1) Prism layer cells (quad)
for idx,quad in enumerate(quad_cells):
    quad_np = global_points[quad] 
    x, y = quad_np[:,0], quad_np[:,1]
    area = 0.5*np.abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))
    cx = np.sum((x+np.roll(x,-1))*(x*np.roll(y,-1)-np.roll(x,-1)*y))/(6*area)
    cy = np.sum((y+np.roll(y,-1))*(x*np.roll(y,-1)-np.roll(x,-1)*y))/(6*area)
    cells.append({
        "id": len(cells),
        "nodes": quad,
        "centroid": np.array([cx, cy]),
        "area": area,
        "faces": [],
        "type": "quad"
    })

edges = {}
for cell_id, cell in enumerate(cells):
    nodes = cell["nodes"]
    n_nodes = len(nodes)
    for i in range(n_nodes):
        n1 = nodes[i]
        n2 = nodes[(i+1)%n_nodes]
        edge = tuple(sorted((n1,n2)))
        if edge not in edges:
            edges[edge] = [cell_id]
        else:
            edges[edge].append(cell_id)

# ==========================================================
# FACES + NEIGHBORS
# ==========================================================

boundary_type=""
for edge_nodes, attached_cells in edges.items():
    n1, n2 = edge_nodes
    p1, p2 = global_points[[n1,n2]]
    if len(attached_cells) == 2:
        c1, c2 = attached_cells
        internal = True
    else:
        c1 = attached_cells[0]
        c2 = None
        internal = False
    centroid1 = cells[c1]["centroid"]
    normal1, length, midpoint = edge_normal_outward(p1,p2,centroid1)
    d = np.linalg.norm(cells[c2]["centroid"] - centroid1) if internal else None
    if internal==False:
        if global_points[n1][1]==0 and global_points[n2][1]==0: #bottom side
            boundary_type="fixed_wall"
        elif global_points[n1][0]==lx and global_points[n2][0]==lx: #right side
            boundary_type="fixed_wall"
        elif global_points[n1][1]==ly and global_points[n2][1]==ly: #top side
            boundary_type="moving_wall"
        elif global_points[n1][0]==0 and global_points[n2][0]==0: #left side
            boundary_type="fixed_wall"
    else:
        boundary_type=""
    face1 = {
        "nodes": (n1,n2),
        "neighbor": c2,
        "normal": normal1,
        "length": length,
        "center": midpoint,
        "distance": d,
        "boundary": boundary_type
    }
    cells[c1]["faces"].append(face1)
    if internal:
        centroid2 = cells[c2]["centroid"]
        normal2, _, _ = edge_normal_outward(p1,p2,centroid2)
        face2 = {
            "nodes": (n1,n2),
            "neighbor": c1,
            "normal": normal2,
            "length": length,
            "center": midpoint,
            "distance": d,
            "boundary": None
        }
        cells[c2]["faces"].append(face2)


# ==========================================================
# EXPORT EXCEL
# ==========================================================

cells_data = []
for cell in cells:
    nodes = cell["nodes"] 
    n_nodes = len(nodes)
    cx, cy = cell["centroid"] 
    
    nodes_padded = list(nodes) + [None]*(4-n_nodes)
    
    cells_data.append([
        cell["id"],
        *nodes_padded,          
        cell["area"],
        cx, cy,
        len(cell["faces"]),
        cell["type"]
    ])

df_cells = pd.DataFrame(cells_data, columns=[
    "cell_id",
    "node1", "node2", "node3", "node4",
    "area",
    "centroid_x", "centroid_y",
    "num_faces",
    "type"
])

edges_data = []
for edge_nodes, attached_cells in edges.items():
    n1, n2 = edge_nodes
    if len(attached_cells) == 2:
        c1, c2 = attached_cells
    else:
        c1, c2 = attached_cells[0], None
    edges_data.append([n1, n2, c1, c2])

df_edges = pd.DataFrame(edges_data, columns=[
    "node1", "node2",
    "cell1", "cell2"
])

faces_data = []
for cell in cells:
    for face in cell["faces"]: 
        n1, n2 = face["nodes"] 
        nx, ny = face["normal"] 
        cx, cy = face["center"] 
        faces_data.append([
            cell["id"],
            n1, n2,
            face["neighbor"],
            nx, ny,
            face["length"],
            cx, cy,
            face["distance"],
            face["boundary"]
        ])

df_faces = pd.DataFrame(faces_data, columns=[
    "cell_id",
    "node1", "node2",
    "neighbor",
    "normal_x", "normal_y",
    "length",
    "center_x", "center_y",
    "distance",
    "boundary_type"
])



file_path = os.path.dirname(os.path.abspath(__file__))
excel_file = os.path.join(file_path, "mesh_data.xlsx")

with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
    df_cells.to_excel(writer, sheet_name="Cells", index=False)
    df_edges.to_excel(writer, sheet_name="Edges", index=False)
    df_faces.to_excel(writer, sheet_name="Faces", index=False)


# ==========================================================
# PLOT
# ==========================================================
plt.figure(figsize=(8,6))
ax = plt.gca()
ax.set_aspect('equal')

patches_list = []
colors = []

for cell in cells:
    nodes = cell["nodes"]
    coords = np.array([global_points[n] for n in nodes if n is not None])
    polygon = patches.Polygon(coords, closed=True)
    patches_list.append(polygon)

pc = PatchCollection(patches_list, facecolor=colors, edgecolor='k', linewidths=0.5, alpha=0.5)
ax.add_collection(pc)

# Print IDs
"""for cell in cells:
    cx, cy = cell["centroid"]
    plt.text(cx, cy, str(cell["id"]), fontsize=8, ha='center', va='center', color='red')"""

plt.scatter(global_points[:,0], global_points[:,1], s=2, color='black') 
plt.title("Mesh")
plt.xlabel("X")
plt.ylabel("Y")
plt.show() 


def edge_lengths(coords):
    return [np.linalg.norm(coords[i] - coords[(i+1)%len(coords)]) for i in range(len(coords))]

def triangle_angles(coords):
    a = np.linalg.norm(coords[1]-coords[0])
    b = np.linalg.norm(coords[2]-coords[1])
    c = np.linalg.norm(coords[0]-coords[2])
    angles = [
        np.degrees(np.arccos((b**2 + c**2 - a**2)/(2*b*c))),
        np.degrees(np.arccos((a**2 + c**2 - b**2)/(2*a*c))),
        np.degrees(np.arccos((a**2 + b**2 - c**2)/(2*a*b)))
    ]
    return angles

def quadrilateral_angles(coords):
    angles = triangle_angles([coords[0], coords[1], coords[2]]) + triangle_angles([coords[0], coords[2], coords[3]])
    return angles


################## SAVE DATA FOR OTHER SCRIPTS ####################
file_path = os.path.dirname(os.path.abspath(__file__))
npz_file = os.path.join(file_path, "mesh_data.npz")

np.savez(npz_file,
        points=global_points,
        cells=cells,
        edges=edges)