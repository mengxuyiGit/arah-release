import numpy as np
import cv2
from ipdb import set_trace as st

from im2mesh.utils.utils import save_verts

def _update_extrinsics(
        extrinsics, 
        angle, 
        trans=None, 
        rotate_axis='y'):
    r""" Uptate camera extrinsics when rotating it around a standard axis.

    Args:
        - extrinsics: Array (3, 3)
        - angle: Float
        - trans: Array (3, )
        - rotate_axis: String

    Returns:
        - Array (3, 3)
    """
    E = extrinsics
    inv_E = np.linalg.inv(E) #C2W

    camrot = inv_E[:3, :3]
    campos = inv_E[:3, 3]
    if trans is not None:
        # print("trans.shape", trans.shape, "campos.shape", campos.shape)
        campos -= trans

    rot_y_axis = camrot.T[1, 1]
    if rot_y_axis < 0.:
        angle = -angle
    
    rotate_coord = {
        'x': 0, 'y': 1, 'z':2
    }
    grot_vec = np.array([0., 0., 0.])

    grot_vec[rotate_coord[rotate_axis]] = angle

    grot_mtx = cv2.Rodrigues(grot_vec)[0].astype('float32')
    # print("grot_mtx.shape:",grot_mtx.shape)

    rot_campos = grot_mtx.dot(campos) 
    rot_camrot = grot_mtx.dot(camrot)
    # print("after grot_mtx.dot(campos) shape:",rot_camrot.shape)

    if trans is not None:
        rot_campos += trans
    
    new_E = np.identity(4)
    new_E[:3, :3] = rot_camrot.T
    new_E[:3, 3] = -rot_camrot.T.dot(rot_campos)

    return new_E[:3, :3], new_E[:3, 3] # R,T

def _update_extrinsics_with_chatGPT(extrinsics, target_location, frame_idx, views, levels=3):
    # print("target_location in chat",target_location)
    # save_verts(target_location[None,...], 'cam_lookat.npy', 7)
    # exit(0)
# def simulate_camera_rotation(target_location, up_dir, views, stepsize, circle_fixed_start, circle_fixed_end, rotation_mode):

    # if target_location.ndim != extrinsics.ndim:
    #     target_location = target_location[None,...] # ndim=(3,) -> (1,3)
    
    circle_fixed_start = (-0.5*np.pi, 0)
    circle_fixed_end = (0.5*np.pi, 2*np.pi)
    up_dir = np.array([0,0,1])
    cam_to_target = np.linalg.inv(extrinsics)[:3, 3]-target_location
    radius = (cam_to_target**2).sum()**0.5
    # print(f"Radius    :{radius}")

    position_noise = False
    
    phi = np.linspace(circle_fixed_start[0], circle_fixed_end[0], levels)[((levels*frame_idx)//views)]
    # phi = np.linspace(circle_fixed_start[0], circle_fixed_end[0], views)[frame_idx%views]
    # phi = circle_fixed_start[0]
    # phi = 0

    theta = np.linspace(circle_fixed_start[1], circle_fixed_end[1], views//levels)[np.rint(frame_idx%(views//levels)).astype(int)]
    # theta = np.linspace(circle_fixed_start[1], circle_fixed_end[1], views)[frame_idx%views]
    # theta=0
    # print(f"Phi:{phi}, Theta:{theta}\n")
    
    if position_noise:
        phi = 0.9*phi + 0.2*np.random.rand()*phi
        # theta = 0.8*theta + 0.2*np.random.rand()*theta
    
    # print("target_location in chat",target_location)
    x = target_location[0] + np.cos(phi) * np.cos(theta)*radius
    y = target_location[1] + np.cos(phi) * np.sin(theta)*radius
    z = target_location[2] - np.sin(phi)*radius 
    # print("x,y,z shape", -.shape, y.shape, z.shape)
    cam_on_axis = False
    # if cam_on_axis:
    #     radius = radius
    #     assert np.unique(target_location)==0
    #     x = radius if frame_idx%3==0 else 1e-6
    #     # x = 0
    #     y = radius if frame_idx%3==1 else 1e-6
    #     z = radius if frame_idx%3==2 else 1e-6

    cam_location = np.column_stack((x, y, z))
    # print("cam location", cam_location)
    cam_lookat = target_location - cam_location
    
    ######### plot the ray from cam location to target location ###########
    cam_lookat_ray = np.linspace(target_location, cam_location, 10)
    ######### plot the ray from cam location to target location ###########
    
    
    cam_lookat = cam_lookat / np.linalg.norm(cam_lookat, axis=1, keepdims=True)
    cam_right = np.cross(up_dir, cam_lookat)
    cam_right = cam_right / np.linalg.norm(cam_right, axis=1, keepdims=True)
    cam_up = np.cross(cam_lookat, cam_right)
    cam_up = cam_up / np.linalg.norm(cam_up, axis=1, keepdims=True)
    # print("cam_right, cam_up, cam_lookat", cam_right.shape, cam_up.shape, cam_lookat.shape)
    rotation_matrix = np.stack((cam_right, cam_up, cam_lookat), axis=2)
    
    # print(rotation_matrix.shape)
    rotation = rotation_matrix
    # rotation = np.linalg.inv(rotation_matrix)
    # print("Camera location: ", cam_location.shape, "Camera rotation: ", rotation.shape)
    # print(rotation.shape)
    # return rotation[0], cam_location[0] # directly return R,T in arah. Below is humannerf
    # return rotation[0], cam_location[0], cam_lookat_ray
    
    # save_cam_json = True
    #     if save_cam_json:
    #         if outfile = os.path.join('minimal_shape','original_cam_locs.npy')
    # if os.path.exists(outfile):
    #     os.remove(outfile)

    new_c2w = np.identity(4)
    new_c2w[:3, :3] = rotation
    new_c2w[:3, 3] = cam_location
    # print(new_c2w)
    new_E = np.linalg.inv(new_c2w)
    return new_E[:3, :3], new_E[:3, 3], cam_lookat_ray # R,T

    # return new_E


def rotate_camera_by_frame_idx(
        extrinsics, 
        frame_idx, 
        trans=None,
        rotate_axis='y',
        period=420,
        inv_angle=False):
    r""" Get camera extrinsics based on frame index and rotation period.

    Args:
        - extrinsics: Array (3, 3)
        - frame_idx: Integer
        - trans: Array (3, )
        - rotate_axis: String
        - period: Integer
        - inv_angle: Boolean (clockwise/counterclockwise)

    Returns:
        - Array (3, 3)
    """

    # print("rotate_axis:", rotate_axis)

    angle = 2 * np.pi * (frame_idx / period)
    if inv_angle:
        angle = -angle
    
    # print("angle:", angle)
    chatGPT_rotation=True
    if chatGPT_rotation:
        levels = 11
        # save_verts(trans[None,...], 'cam_lookat.npy', 6)
        # exit(0)
        # print("target_location before chat", trans)
        return _update_extrinsics_with_chatGPT(extrinsics, trans, frame_idx, period, levels)

    return _update_extrinsics(
                extrinsics, angle, trans, rotate_axis)