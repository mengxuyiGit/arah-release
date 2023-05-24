import numpy as np
import os
import glob
import torch
import argparse
from ipdb import set_trace as st

from im2mesh import config
from im2mesh.utils.root_finding_utils import (
                    forward_skinning,
                    unnormalize_canonical_points,
                    normalize_canonical_points
                )
from im2mesh.metaavatar_render.models.skinning_model import SkinningModel
from im2mesh.utils.utils_mobile import load_obj_mesh, replace_obj_mesh_vertices
from im2mesh.utils.utils import get_02v_bone_transforms
# from im2mesh.metaavatar_render.config import get_skinning_model

def get_skinning_model(cfg, dim=3, init_weights=True):
    ''' Returns Skinning Model instances.

    Args:
        cfg (yaml config): yaml config object
        dim (int): points dimension
        init_weights (bool): whether to initialize the weights for the skinning network with pre-trained model (MetaAvatar)
    '''
    decoder = get_skinning_decoder(cfg, dim=dim)
    st()

    optim_skinning_net_path = cfg['model']['skinning_net2']
    if init_weights and optim_skinning_net_path is not None:
        ckpt = torch.load(optim_skinning_net_path, map_location='cpu')

        skinning_decoder_fwd_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():
            if k.startswith('module'):
                k = k[7:]

            if k.startswith('skinning_decoder_fwd'):
                skinning_decoder_fwd_state_dict[k[21:]] = v

        decoder.load_state_dict(skinning_decoder_fwd_state_dict, strict=False)

    skinning_model = SkinningModel(skinning_decoder_fwd=decoder)

    return skinning_model

#################### Arguments ####################
parser = argparse.ArgumentParser(
    description='Test function that renders images without quantitative evaluation.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--mesh_path_2', type=str, help='Path to config file.')
parser.add_argument('--subject_dir', type=str, default=None, help='Path to config file.') # e.g., 
# command example: 

if  __name__ == '__main__':
    args = parser.parse_args()
    
    ################### load config dict ###################
    cfg = config.load_config(args.config, 'configs/default.yaml')
    num_workers = args.num_workers

    # Shorthands
    out_dir = cfg['training']['out_dir']
    batch_size = cfg['training']['batch_size']

    os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
    device = torch.device('cuda:0')

    # Overwrite configuration
    cfg['data']['test_views'] = args.test_views.split(',')
    # cfg['data']['dataset'] = 'zju_mocap_odp' # FIXME: why original repo use overwriting here?
    # cfg['data']['path'] = 'data/odp'
    cfg['data']['test_subsampling_rate'] = args.subsampling_rate
    cfg['data']['test_start_frame'] = args.start_frame
    cfg['data']['test_end_frame'] = args.end_frame
    cfg['data']['pose_dir'] = args.pose_dir
    
    ##### get skinning model ######
    init_weights = True
    skinning_model = get_skinning_model(cfg, init_weights=init_weights)
    #################################################################
    
    
    ################### load mesh ###################
    mesh_path_2 = args.mesh_path_2
    mesh_path_w_face = args.mesh_path_2 
    
    vertices, _ = load_obj_mesh(mesh_path_2)
    pts = torch.from_numpy(vertices).to(device).reshape(1, -1, 3) #.to(rays_o.device).type(rays_o.dtype).reshape(1, -1, 3)
    print(f"Load mesh from {mesh_path_2}, of {pts.shape} vertices")
    mesh_path_save = mesh_path_2.replace(".obj", "_reverse_yz.obj")
    #################################################################
    

    ################### load bone transformations ###################
    subject_dir = args.subject_dir
    if subject_dir is not None:
        model_files = sorted(glob.glob(os.path.join(subject_dir, 'models/*.npz')))
        # data_path = self.data[idx]['model_file']
        # for data_path in model_files:
        data_path = model_files[0]
        model_dict = np.load(data_path)
        
        bone_transforms = model_dict['bone_transforms'].astype(np.float32)
    else:
        bone_transforms_02v = get_02v_bone_transforms(Jtr, self.rot45p, self.rot45n)
        # the above 02v can be replaces with any transforms that controlled by the 24 joints
        bone_transforms = bone_transforms_02v  
    st() # check bone transforms
    #################################################################
    
    
    ################### forward smpl lbs ###################
    ### humannerf
    # mv_output = self._sample_motion_fields_forward_warp(
    #                 pts=pts,
    #                 motion_scale_Rs=motion_scale_Rs[0], 
    #                 motion_Ts=motion_Ts[0], 
    #                 motion_weights_vol=motion_weights_vol,
    #                 cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
    #                 cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
    #                 output_list=['x_skel', 'fg_likelihood_mask'])

    # cnl_pts = mv_output['x_skel'].reshape(-1,3)
    
    ### arah
    points_hat = pts
    batch_size = 1
    
    loc = torch.zeros(batch_size, 1, 3, device=device, dtype=torch.float32)
    points_lbs, _ = forward_skinning(points_hat,
                        loc=loc,
                        sc_factor=sc_factor,
                        coord_min=coord_min,
                        coord_max=coord_max,
                        center=center,
                        skinning_model=skinning_model,
                        vol_feat=vol_feat,
                        bone_transforms=inputs['bone_transforms'],
                        return_w=False
                    )
    #################################################################


    mesh_path_save = mesh_path_2.replace(".obj", "_x_skel_hn.obj")
    save_obj_mesh(mesh_path_save, cnl_pts)
    print(f"Saved mesh to {mesh_path_save}, of {cnl_pts.shape} cnl vertices")

    mesh_path_replace = mesh_path_w_face.replace('.obj', '_driven_mesh_f2.obj')
    replace_obj_mesh_vertices(load_mesh_path=mesh_path_w_face, save_mesh_path=mesh_path_replace, new_vertices=cnl_pts, reversed_yz=True)
    print(f"Replaced mesh to {mesh_path_replace}, of {cnl_pts.shape} driven vertices")

    exit(0)
