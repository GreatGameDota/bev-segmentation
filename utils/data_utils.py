import matplotlib
import numpy as np
import open3d as o3d
import yaml

from kitti360scripts.helpers.annotation import Annotation3D, Annotation3DPly, global2local
from kitti360scripts.helpers.labels import id2label, labels, Label
'''
Author: GreatGameDota https://gihub.com/GreatGameDota
Copyright 2022

Modified from: https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/viewer/kitti360Viewer3D.py
'''

def getColor(idx):
    cmap_length = 9
    cmap = matplotlib.cm.get_cmap('Set1')

    if idx==0:
        return np.array([0,0,0])
    return np.asarray(cmap(idx % cmap_length)[:3])*255.

def assignColor(globalIds, gtType='semantic'):
    if not isinstance(globalIds, (np.ndarray, np.generic)):
        globalIds = np.array(globalIds)[None]
    color = np.zeros((globalIds.size, 3))
    for uid in np.unique(globalIds):
        semanticId, instanceId = global2local(uid)
        if gtType=='semantic':
            color[globalIds==uid] = id2label[semanticId].color
        elif instanceId>0:
            color[globalIds==uid] = getColor(instanceId)
        else:
            color[globalIds==uid] = (96,96,96) # stuff objects in instance mode
    color = color.astype(np.float)/255.0
    return color

def assignColorLocal(localIds, gtType='semantic'):
    color = np.zeros((localIds.size, 3))
    for uid in np.unique(localIds):
        semanticId = uid
        if gtType=='semantic':
            color[localIds==uid] = id2label[semanticId].color
        elif instanceId>0:
            color[localIds==uid] = getColor(instanceId)
        else:
            color[localIds==uid] = (96,96,96) # stuff objects in instance mode
    color = color.astype(np.float)/255.0
    return color

def loadBoundingBoxes(annotation3D):
    bboxes,bboxes_window,colors = [],[],[]
    for globalId,v in annotation3D.objects.items():
        # skip dynamic objects
        if len(v)>1:
            continue
        for obj in v.values():
            lines=np.array(obj.lines)
            vertices=obj.vertices
            faces=obj.faces
            # transform=obj.transform
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(obj.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(obj.faces)
            color = assignColor(globalId, 'semantic')
            semanticId, instanceId = global2local(globalId)
            mesh.paint_uniform_color(color.flatten())
            mesh.compute_vertex_normals()
            colors.append(color)
            bboxes.append( mesh )
            bboxes_window.append([obj.start_frame, obj.end_frame])
    return bboxes, bboxes_window, colors

def loadBoundingBoxesDynamic(annotation3D):
    bboxes,bboxes_window,colors,times = [],[],[],[]
    for globalId,v in annotation3D.objects.items():
        if len(v)>1:
            bboxes_,bboxes_window_,colors_,times_=[],[],[],[]
            for obj in v.values():
                lines=np.array(obj.lines)
                vertices=obj.vertices
                faces=obj.faces
                # transform=obj.transform
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(obj.vertices)
                mesh.triangles = o3d.utility.Vector3iVector(obj.faces)
                color = assignColor(globalId, 'semantic')
                semanticId, instanceId = global2local(globalId)
                mesh.paint_uniform_color(color.flatten())
                mesh.compute_vertex_normals()
                colors_.append(color)
                bboxes_.append( mesh )
                bboxes_window_.append([obj.start_frame, obj.end_frame])
                times_.append(obj.timestamp)
                
            colors.append(colors_)
            bboxes.append(bboxes_)
            bboxes_window.append(bboxes_window_)
            times.append(times_)
    return bboxes, bboxes_window, colors, times

def remapColors(pts, path, dtype='points'):
    with open(path) as f:
        remap = yaml.load(f, Loader=yaml.FullLoader)[dtype]
    
    for i,pt in enumerate(pts):
        try:
            pts[i] = remap[tuple(pt)]
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            pts[i] = [0,0,0]
    return pts