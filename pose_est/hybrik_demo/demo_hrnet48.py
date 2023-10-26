import argparse
import os
import sys
import os.path as osp
import subprocess
sys.path.append('./')
import pickle
import cv2
import time	
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPLCam
from hybrik.utils.render import SMPLRenderer
from hybrik.utils.render_pytorch3d import render_mesh
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d, vis_smpl_3d

det_transform = T.Compose([T.ToTensor()])
torch.set_grad_enabled(False)

# Index pairs representing connections between joints
bones_jts_29 = np.array([    
    [0, 3],	    # pelvis -> spine1              # Color 1 (verde) [self.JOINT_NAMES.index('pelvis'),self.JOINT_NAMES.index('spine1')
    [3, 6],	    # spine1 -> spine2
    [6, 9],   	# spine2 -> spine3
    [9, 12],	# spine3 -> neck
    [12, 15],	# neck -> jaw
    # [15, 24],	# jaw -> head               
    [0, 1],   	# pelvis -> left_hip        # Color 2.1 (azul)
    [1, 4],	    # left_hip -> left_knee
    [4, 7],	    # left_knee -> left_ankle
    [7, 10],	# left_ankle -> left_foot
    # [10, 27],	# left_foot -> left_bigtoe
    [9, 13],	# spine3 -> left_collar     # Color 2.2 (azul)
    [13, 16],	# left_collar -> left_shoulder
    [16, 18],	# left_shoulder -> left_elbow
    [18, 20],	# left_elbow -> left_wrist
    [20, 22],	# left_wrist -> left_thumb  
    # [22, 25],   # left_thumb -> left_middle
    [0, 2],	    # pelvis -> right_hip           # Color 3.1 (vermelho)
    [2, 5],	    # right_hip -> right_knee
    [5, 8],	    # right_knee -> right_ankle
    [8, 11],	# right_ankle -> right_foot
    # [11, 28],	# right_foot -> right_bigtoe
    [9, 14],	# spine3 -> right_collar    # Color 3.2 (vermelho)
    [14, 17],	# right_collar -> right_shoulder
    [17, 19],	# right_shoulder -> right_elbow
    [19, 21],	# right_elbow -> right_wrist
    [21, 23],	# right_wrist -> right_thumb 
    # [23, 26],	# right_thumb -> right_middle    
])
colors = ['green'] * 5 + ['blue'] * 9 + ['red'] * 9

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

def cxcywh2xyxy(bbox):

    cx, cy, w, h = bbox
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    
    return [x1, y1, x2, y2]


def images_to_video(img_dir, out_path, img_fmt="%06d.jpg", fps=30, crf=25, verbose=True):
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    FFMPEG_PATH = '/usr/bin/ffmpeg' if osp.exists('/usr/bin/ffmpeg') else 'ffmpeg'
    cmd = [FFMPEG_PATH, '-y', '-r', f'{fps}', '-f', 'image2', '-start_number', '1',
            '-i', f'{img_dir}/{img_fmt}', '-vcodec', 'libx264', '-crf', f'{crf}', '-pix_fmt', 'yuv420p', out_path]
    if not verbose:
        cmd += ['-hide_banner', '-loglevel', 'error']
    subprocess.run(cmd)

parser = argparse.ArgumentParser(description='HybrIK Demo')

parser.add_argument('--gpu',
                    help='gpu',
                    default=0,
                    type=int)
parser.add_argument('--multi',
                    help='multi-person',
                    default=0,
                    type=int)
parser.add_argument('--img_folder',
                    help='images path ',
                    default='out/glamr_static/workout_5s/pose_est/frames',
                    type=str)
parser.add_argument('--out_dir',
                    help='output folder',
                    default='out/glamr_static/workout_5s/pose_est',
                    type=str)
parser.add_argument('--MPT_method',
                    help='strongsort, deepocsort, ocsort, bytetrack, botsort or sort (original)',
                    default='ocsort',
                    type=str)
parser.add_argument('--person_detection_method',
                    help='strongsort, deepocsort, ocsort, bytetrack, botsort or fasterrcnn_resnet50 (original)',
                    default='ocsort',
                    type=str)
parser.add_argument('--high_performance', 
                    help='dont plot 2D poses and dont reconstruct mesh', 
                    default=False)

opt = parser.parse_args()


cfg_file = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml'
# cfg_file = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_wo_pw3d.yaml'
CKPT = './pretrained_models/hybrik_hrnet48_w3dpw.pth'
# CKPT = './pretrained_models/hybrik_hrnet48_wo3dpw.pth'
cfg = update_config(cfg_file)

bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

transformation = SimpleTransform3DSMPLCam(
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])

hybrik_model = builder.build_sppe(cfg.MODEL)

print(f'Loading model from {CKPT}...')
save_dict = torch.load(CKPT, map_location='cpu')
if type(save_dict) == dict:
    model_dict = save_dict['model']
    hybrik_model.load_state_dict(model_dict)
else:
    hybrik_model.load_state_dict(save_dict)

hybrik_model.cuda(opt.gpu)
hybrik_model.eval()

res_images_path = os.path.join(opt.out_dir, 'res_images')
res_2D_poses_images_path = os.path.join(opt.out_dir, 'res_2D_poses_images')
res_3D_poses_images_path = os.path.join(opt.out_dir, 'res_3D_poses_images')

os.makedirs(res_images_path, exist_ok=True)
os.makedirs(res_2D_poses_images_path, exist_ok=True)
os.makedirs(res_3D_poses_images_path, exist_ok=True)


files = os.listdir(f'{opt.img_folder}')
files.sort()

img_path_list = []
for file in tqdm(files):
    if not os.path.isdir(file) and file[-4:] in ['.jpg', '.png']:
        img_path = os.path.join(opt.img_folder, file)
        img_path_list.append(img_path)

if opt.multi:

    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(opt.gpu))    
        print(f'-> {torch.cuda.get_device_name(0)} available!\nRunning inference on GPU :) ')
    else:
        device =  torch.device('cpu') 
        print(f"-> NVIDIA not available! -> Running inference on CPU :'(")

    if opt.MPT_method=="sort":
        # load multi-person tracking model
        # mot = MPT(
        #     device=device,
        #     batch_size=4,
        #     display=False,
        #     detection_threshold=0.7,
        #     detector_type='yolo',
        #     output_format='dict',
        #     yolo_img_size=416,
        # )
        # print('\n### Run MPT...')
        # tracking_results = mot(opt.img_folder)
        # offset_frames=0
        print("Sort não implementado, escolher outro método (e.g., ocsort)")
    else:           # strongsort, deepocsort, ocsort, bytetrack ou botsort 
        # from src.MPT.posprocess_human_track import convert_track_info
        track_pkl_file = os.path.join(opt.out_dir, 'track')
        tracking_results = pickle.load(open(f'{track_pkl_file}/mpt.pkl', 'rb')) 
        if opt.MPT_method=='strongsort':
            offset_frames = 2                           # Required to initialize the model: https://github.com/mikel-brostrom/yolo_tracking/issues/379      https://github.com/mikel-brostrom/yolo_tracking/blob/8885642c9d049c933c6e1df1d05478dab4a0c37c/deep_sort/configs/deep_sort.yaml#L6        
            img_1 = cv2.imread(img_path_list[0])        # See file: src/yolo_tracking/boxmot/strongsort/configs/strongsort.yaml
            img_2 = cv2.imread(img_path_list[1])
            text = f'Model initialization ({opt.MPT_method}): frame 1'
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            textX = (img_1.shape[1] - textsize[0]) / 2
            textY = (img_1.shape[0]/5 - textsize[1])
            cv2.putText(img_1, text, (int(textX), int(textY)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img_2, f'Model initialization ({opt.MPT_method}): frame 2', (int(textX), int(textY)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(img_1, text, (int(textX+2), int(textY+2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img_2, f'Model initialization ({opt.MPT_method}): frame 2', (int(textX+2), int(textY+2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            save_dirs=[res_images_path,res_2D_poses_images_path]    # res_3D_poses_images_path
            for dir in save_dirs:
                img_1_path = os.path.join(dir, '000000.jpg')    
                img_2_path = os.path.join(dir, '000001.jpg')    
                cv2.imwrite(img_1_path, img_1)
                cv2.imwrite(img_2_path, img_2)
        else:
            offset_frames = 0

    detection_all = defaultdict(dict)
    for person_id in tracking_results:  # Cycle through all detected people
        frames_ids = tracking_results[person_id]['frames']
        for idx in range(len(frames_ids)):      # Cycle through all frames that detect the same person -> Conversion from cx,cy,w,h to x1,y1,x2,y2
            frames_id = frames_ids[idx]
            cx, cy, w, h = tracking_results[person_id]['bbox'][idx]
            x1, y1, x2, y2 = max(0, cx-w//2), max(0, cy-h//2), cx+w//2, cy+h//2
            detection_all[frames_id][person_id-1] = [x1, y1, x2, y2]                # detection_all -> frame -> person ID -> person bbox

    out_dict = defaultdict(lambda: defaultdict(list))   # Dictionary of lists (index must be int)
    bbox_exist = defaultdict(list)          # Unique dictionary ("index" is a string -> dictionary)
    bboxes = defaultdict(list)
    poses_dict = defaultdict(lambda: defaultdict(list))

    time_oper_by_person = []
    averages_times_by_frame = []
    cnt_num_persons=0 # number of people in a given frame

    # initialize
    for person_id in tracking_results:
        bbox_exist[person_id-1] = [0 for _ in range(len(img_path_list))]

    #####################

    print('\n### Run HybrIK (HRnet-48) multi person...')
    for frame_idx in tqdm(range(len(img_path_list)-offset_frames)):

        frame_idx+=offset_frames
        img_path = img_path_list[frame_idx]
        # dirname = os.path.dirname(img_path)
        # basename = os.path.basename(img_path)

        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = input_image.copy()
        image_pose_2d = input_image.copy()
        
        if frame_idx in detection_all:
            # For each detected person, starting from 0,1,...
            for idx in detection_all[frame_idx]:
                tight_bbox = detection_all[frame_idx][idx]
                bbox_exist[idx][frame_idx] = 1.0
                cnt_num_persons+=1  # increases the number of people detected
                
                # Run HybrIK
                pose_input, bbox, img_center = transformation.test_transform(input_image, tight_bbox)   #  Used to prepare HybrIK model input
                pose_input = pose_input.to(opt.gpu)[None, :, :, :]
                start_timer = time.time()
                pose_output = hybrik_model(                                                                 # ModelOutput = namedtuple(typename='ModelOutput',
                    pose_input, flip_test=True,                                                                                      # field_names=['pred_shape', 'pred_theta_mats', 'pred_phi', 'pred_delta_shape', 'pred_leaf',
                    bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),                              # 'pred_uvd_jts', 'pred_xyz_jts_29', 'pred_xyz_jts_24', 'pred_xyz_jts_24_struct',
                    img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float()                               # 'pred_xyz_jts_17', 'pred_vertices', 'maxvals', 'cam_scale', 'cam_trans', 'cam_root',
                )    
                time_oper_by_person.append((time.time()-start_timer))                                                                                                                    # 'uvd_heatmap', 'transl', 'img_feat']    
                uv_3D_29 = pose_output.pred_uvd_jts.reshape(29, 3)
                uv_29 = uv_3D_29[:, :2]         # RReshape to exclusively obtain the coordinates (x,y) of the 29 rows (joints) of the pose_output.pred_uvd_jts matrix that contained 29 rows and 3 columns (x,y,z)
                
                # Convert poses to image and save to figure
                
                # Visualization
                img_size = (image.shape[0], image.shape[1])
                focal = np.array([1000, 1000])
                bbox_xywh = xyxy2xywh(bbox)
                # princpt = [bbox_xywh[0], bbox_xywh[1]]
                princpt = [img_center[0], img_center[1]]
                # SMPL Render from main point, faces and focal point of the camera
                renderer = SMPLRenderer(faces=hybrik_model.smpl.faces,
                                        img_size=img_size, focal=focal,
                                        princpt=princpt)
                transl = pose_output.transl.detach()
                transl_camsys = transl.clone()
                transl_camsys = transl_camsys * 256 / bbox_xywh[2]  # Attention here, new transl_camsys nomenclature
                transl = pose_output.transl.detach().cpu().numpy().squeeze()
                transl[2] = transl[2] * 256 / bbox_xywh[2]

                # vis 3d   
                # res_path = os.path.join(res_3D_poses_images_path, f'{frame_idx:06d}.jpg')
                pts_3D = uv_3D_29 * bbox_xywh[2]
                pts_3D[:, 0] = pts_3D[:, 0] + bbox_xywh[0]    # shift x from bbox
                pts_3D[:, 1] = pts_3D[:, 1] + bbox_xywh[1]    # shift y from bbox
                # draw_3D_skeleton(pts_3D, res_path, bones = bones_jts_29, colors = colors, show_image = True)

                # vis 2d
                pts = uv_29 * bbox_xywh[2]  
                pts[:, 0] = pts[:, 0] + bbox_xywh[0]    # shiftar x da bbox
                pts[:, 1] = pts[:, 1] + bbox_xywh[1]    # shiftar y da bbox

                bboxes[idx].append(np.array(bbox_xywh))

                image_pose_2d, tamanho_texto, espessura = vis_2d(image_pose_2d, tight_bbox, pts, idx, 
                                                                 bones = bones_jts_29, extended = True)

                # Generate image with smpl rendering of human in 2D image
                image = vis_smpl_3d(
                    pose_output, image, cam_root=transl, bbox_xywh=bbox_xywh,
                    f=focal, c=princpt, renderer=renderer, color_id=idx)
                cv2.putText(image, f'{idx}', (int(pts[24][0])-5, int(pts[24][1]) - 15), cv2.FONT_HERSHEY_SIMPLEX, tamanho_texto, (0, 0, 0), espessura)
                cv2.putText(image, f'{idx}', (int(pts[24][0])-4, int(pts[24][1]) - 14), cv2.FONT_HERSHEY_SIMPLEX, tamanho_texto, (255, 255, 255), espessura)

                new_princpt = np.array([image.shape[1], image.shape[0]]) * 0.5
                transl[:2] += (np.array(princpt) - new_princpt) * transl[2] / np.array(focal) 
                princpt = new_princpt

                # save to dict
                K = np.eye(3)
                K[[0, 1], [0, 1]] = focal
                K[:2, 2] = princpt
                out_dict[idx]['smpl_pose_quat_wroot'].append(pose_output.pred_theta_mats[0].cpu().numpy().reshape(-1, 4))       # Joint matrices in quartinions and not rotation matrices
                out_dict[idx]['smpl_beta'].append(pose_output.pred_shape[0].cpu().numpy())
                out_dict[idx]['root_trans'].append(transl)
                out_dict[idx]['kp_2d'].append(pts.cpu().numpy())
                out_dict[idx]['cam_K'].append(K.astype(np.float32))
                # Added to save data rendered by hybrik
                out_dict[idx]['kp_3d'].append(pts_3D.cpu().numpy())    # KEYPOINTS 3D (pixels size)
                # out_dict[idx]['uv_3D_29'].append(uv_3D_29.cpu().numpy())    # KEYPOINTS 3D (referencial da camara)
                # out_dict[idx]['heatmaps'].append(pose_output.uvd_heatmap.cpu().numpy())
                # out_dict[idx]['renderer_hybrik_faces'].append(renderer.faces)
                # out_dict[idx]['renderer_hybrik_focal'].append(renderer.focal)
                # out_dict[idx]['renderer_hybrik_h'].append(renderer.h)
                # out_dict[idx]['renderer_hybrik_w'].append(renderer.w)
                # out_dict[idx]['renderer_hybrik_princpt'].append(renderer.princpt)
                # out_dict[idx]['renderer_hybrik_vertices_mesh'].append(pose_output.pred_vertices.detach().cpu().numpy().squeeze())   # 6890 vertices generated in the hybrik smpl mesh (ready to use in vis_smpl_3d() and replace the first line of code in that function)
            
            averages_times_by_frame.append((sum(time_oper_by_person) / cnt_num_persons)) # append of the average time per frame (already considering the number of people detected)
            cnt_num_persons = 0     # resets number of people counter
            time_oper_by_person = [] # resets the timer for each person
            
        image_vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_pose_2d = cv2.cvtColor(image_pose_2d, cv2.COLOR_RGB2BGR)

        res_path = os.path.join(res_images_path, f'{frame_idx+1:06d}.jpg')
        cv2.imwrite(res_path, image_vis)
        res_path = os.path.join(res_2D_poses_images_path, f'{frame_idx+1:06d}.jpg')
        cv2.imwrite(res_path, image_pose_2d)

    if len(averages_times_by_frame) > 1:
        averages_times_by_frame[0] = averages_times_by_frame[1]
    average_time_by_frame = sum(averages_times_by_frame) / (len(img_path_list)-offset_frames) 
    
    mot_bboxes = defaultdict(dict)
    for idx in bbox_exist:
        mot_bboxes[idx]['id'] = idx
        mot_bboxes[idx]['bbox'] = np.stack(bboxes[idx]),
        mot_bboxes[idx]['exist'] = np.array(bbox_exist[idx])
        
        find = np.where(mot_bboxes[idx]['exist'])[0]
        mot_bboxes[idx]['id'] = idx
        mot_bboxes[idx]['start'] = find[0]
        mot_bboxes[idx]['end'] = find[-1]
        mot_bboxes[idx]['num_frames'] = mot_bboxes[idx]['exist'].sum()
        mot_bboxes[idx]['exist_frames'] = find
        
    for idx, pose_dict in out_dict.items():
        for key in pose_dict.keys():
            pose_dict[key] = np.stack(pose_dict[key])
        pose_dict['frames'] = mot_bboxes[idx]['exist_frames']   # out_dict[idx]['frames']
        pose_dict['frame2ind'] = {f: i for i, f in enumerate(pose_dict['frames'])}  # out_dict[idx]['frame2ind']
        pose_dict['bboxes_dict'] = mot_bboxes[idx]

        #####################

else:
    # load detection model
    if opt.person_detection_method=="fasterrcnn_resnet50":
        det_model = fasterrcnn_resnet50_fpn(pretrained=True)
        det_model.cuda(opt.gpu)
        det_model.eval()
    print('\n### Run HybrIK (HRnet-48) single person...')

    prev_box = None
    # renderer = None
    out_dict = defaultdict(lambda: defaultdict(list))
    idx = 0   # single person id

    # frame_idx = 0

    bbox_exist = []             
    bboxes = []
    time_oper = []
    # smpl_faces = torch.from_numpy(hybrik_model.smpl.faces.astype(np.int32))

    if not opt.person_detection_method=="fasterrcnn_resnet50":  
        track_pkl_file = os.path.join(opt.out_dir, 'track') 
        tracking_results = pickle.load(open(f'{track_pkl_file}/mpt.pkl', 'rb'))

    for frame_idx,img_path in enumerate(tqdm(img_path_list)):

        with torch.no_grad():
            input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            if opt.person_detection_method=="fasterrcnn_resnet50":
                # dirname = os.path.dirname(img_path)
                # basename = os.path.basename(img_path)
                # Run Detection
                det_input = det_transform(input_image).to(opt.gpu)
                det_output = det_model([det_input])[0]

                if prev_box is None:
                    tight_bbox = get_one_box(det_output)  # xyxy
                    if tight_bbox is None:
                        continue
                else:
                    tight_bbox = get_max_iou_box(det_output, prev_box)  # xyxy

                if tight_bbox is None:
                    bbox_exist.append(0.0)
                    continue
                else:
                    bbox_exist.append(1.0)
            else: 
                # track_pkl_file = os.path.join(opt.out_dir, 'track')
                # tracking_results = pickle.load(open(f'{track_pkl_file}/mpt.pkl', 'rb'))
                if frame_idx in tracking_results[1]['frames']:
                    bbox_idx = np.where(np.array(tracking_results[1]['frames']) == frame_idx)[0]
                    tight_bbox = cxcywh2xyxy(tracking_results[1]['bbox'][bbox_idx[0]])
                    bbox_exist.append(1.0)
                else:
                    bbox_exist.append(0.0)
                    tight_bbox= None
                    res_path = os.path.join(res_2D_poses_images_path, f'{frame_idx+1:06d}.jpg')
                    cv2.imwrite(res_path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
                    res_path = os.path.join(res_images_path, f'{frame_idx+1:06d}.jpg')
                    cv2.imwrite(res_path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
                    continue

            prev_box = tight_bbox

            # Run HybrIK
            # bbox: [x1, y1, x2, y2]
            pose_input, bbox, img_center = transformation.test_transform(input_image, tight_bbox)
            pose_input = pose_input.to(opt.gpu)[None, :, :, :]                                          #  pose_output with structure defined at the end of def forward(self, x, flip_item=None, flip_output=False, **kwargs)
            start_timer = time.time()
            pose_output = hybrik_model(                                                                 # ModelOutput = namedtuple(typename='ModelOutput',
                pose_input, flip_test=True,                                                                                      # field_names=['pred_shape', 'pred_theta_mats', 'pred_phi', 'pred_delta_shape', 'pred_leaf',
                bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),                              # 'pred_uvd_jts', 'pred_xyz_jts_29', 'pred_xyz_jts_24', 'pred_xyz_jts_24_struct',
                img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float()                               # 'pred_xyz_jts_17', 'pred_vertices', 'maxvals', 'cam_scale', 'cam_trans', 'cam_root',
            )
            time_oper.append((time.time()-start_timer))                                                                                                                    # 'uvd_heatmap', 'transl', 'img_feat']    
            uv_3D_29 = pose_output.pred_uvd_jts.reshape(29, 3)
            uv_29 = uv_3D_29[:, :2]
            
            # Convert poses to image and save to figure

            # Visualization
            image = input_image.copy()
            image_pose_2d = input_image.copy()
            img_size = (image.shape[0], image.shape[1])
            focal = np.array([1000, 1000])
            bbox_xywh = xyxy2xywh(bbox)
            # princpt = [bbox_xywh[0], bbox_xywh[1]]
            princpt = [img_center[0], img_center[1]]
            # focal = focal / 256 * bbox_xywh[2]
            # focal_length=((2 * focal_length / min(height, width), 2 * focal_length / min(height, width)),),
            transl = pose_output.transl.detach()
            transl_camsys = transl.clone()
            transl_camsys = transl_camsys * 256 / bbox_xywh[2]  # Attention here, new transl_camsys nomenclature

            # SMPL Render from main point, faces and camera focal length
            renderer = SMPLRenderer(faces=hybrik_model.smpl.faces,
                                    img_size=img_size, focal=focal,
                                    princpt=princpt)

            transl = pose_output.transl.detach().cpu().numpy().squeeze()
            transl[2] = transl[2] * 256 / bbox_xywh[2]

            # frame_idx += 1

            pts = uv_29 * bbox_xywh[2]
            pts[:, 0] = pts[:, 0] + bbox_xywh[0]    # shift x from bbox
            pts[:, 1] = pts[:, 1] + bbox_xywh[1]    # shift y from bbox

            bboxes.append(np.array(bbox_xywh))
            bbox_img, tamanho_texto, espessura = vis_2d(image, tight_bbox, pts, idx, bones = bones_jts_29, extended=False)       
            bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)                                    
            res_path = os.path.join(res_2D_poses_images_path, f'{frame_idx+1:06d}.jpg')
            cv2.imwrite(res_path, bbox_img)

            image_vis = vis_smpl_3d(
                pose_output, image_pose_2d, cam_root=transl, bbox_xywh=bbox_xywh,
                f=focal, c=princpt, renderer=renderer)
            cv2.putText(image_vis, f'{idx}', (int(pts[24][0])-15, int(pts[24][1]) - 15), cv2.FONT_HERSHEY_SIMPLEX, tamanho_texto, (0, 0, 0), espessura)
            cv2.putText(image_vis, f'{idx}', (int(pts[24][0])-14, int(pts[24][1]) - 14), cv2.FONT_HERSHEY_SIMPLEX, tamanho_texto, (255, 255, 255), espessura)
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
            res_path = os.path.join(res_images_path, f'{frame_idx+1:06d}.jpg')
            cv2.imwrite(res_path, image_vis)

            # vertices = pose_output.pred_vertices.detach()
            # vis 3d   
            # res_path = os.path.join(res_3D_poses_images_path, f'{frame_idx:06d}.jpg')
            pts_3D = uv_3D_29 * bbox_xywh[2]
            pts_3D[:, 0] = pts_3D[:, 0] + bbox_xywh[0]    # shift x from bbox
            pts_3D[:, 1] = pts_3D[:, 1] + bbox_xywh[1]    # shift y from bbox
            # draw_3D_skeleton(pts_3D, bones = bones_jts_29, colors = colors, save_path = res_path, show_image = False)

            new_princpt = np.array([image.shape[1], image.shape[0]]) * 0.5
            transl[:2] += (np.array(princpt) - new_princpt) * transl[2] / np.array(focal) 
            princpt = new_princpt

            # save to dict
            K = np.eye(3)
            K[[0, 1], [0, 1]] = focal
            K[:2, 2] = princpt
            out_dict[idx]['smpl_pose_quat_wroot'].append(pose_output.pred_theta_mats[0].cpu().numpy().reshape(-1, 4))   # QUATERNIONS HYBRIK
            out_dict[idx]['smpl_beta'].append(pose_output.pred_shape[0].cpu().numpy())  # BETA HYBRIK
            out_dict[idx]['root_trans'].append(transl)          # ROOT TRANSLATION GLAMR
            out_dict[idx]['kp_2d'].append(pts.cpu().numpy())    # KEYPOINTS 2D
            out_dict[idx]['cam_K'].append(K.astype(np.float32)) # INTRINSECOS GLAMR
            out_dict[idx]['kp_3d'].append(pts_3D.cpu().numpy())    # KEYPOINTS 3D (pixel size)
            # out_dict[idx]['uv_3D_29'].append(uv_3D_29.cpu().numpy())    # KEYPOINTS 3D (referencial da camara)
            out_dict[idx]['heatmaps'].append(pose_output.uvd_heatmap.cpu().numpy())
            out_dict[idx]['maxvals'].append(pose_output.maxvals.cpu().numpy())   
    mot_bboxes = {
        0: {
            'id': idx,
            'bbox': np.stack(bboxes),
            'exist': np.array(bbox_exist),
        }
    }
    # mot_bboxes[0]['bbox'][numero_box] # tem coordenadas do centro (x,y,w,h) -> de bbox_xywh
    find = np.where(mot_bboxes[idx]['exist'])[0]
    mot_bboxes[idx]['id'] = idx
    mot_bboxes[idx]['start'] = find[idx]
    mot_bboxes[idx]['end'] = find[-1]
    mot_bboxes[idx]['num_frames'] = mot_bboxes[idx]['exist'].sum()
    mot_bboxes[idx]['exist_frames'] = find
    for idx, pose_dict in out_dict.items():
        for key in pose_dict.keys():
            pose_dict[key] = np.stack(pose_dict[key])
        pose_dict['frames'] = mot_bboxes[idx]['exist_frames']
        pose_dict['frame2ind'] = {f: i for i, f in enumerate(pose_dict['frames'])}
        pose_dict['bboxes_dict'] = mot_bboxes[idx]

    if len(time_oper) > 1:
        time_oper[0] = time_oper[1]
    average_time_by_frame = sum(time_oper) / len(img_path_list) 

# average_time_by_person= sum(time_oper) / len(img_path_list) 
print(f"Average running time by frame: {average_time_by_frame:.4f} seconds ({1/average_time_by_frame:.2f} fps)")

new_dict = dict()
for k in sorted(out_dict.keys()):   # out_dict.keys() has people identifiers (basketball scenario -> out_dict.keys() = [1, 0, 2])
    v = out_dict[k]     # Information by person ID (output dictionary)
    new_dict[k] = dict()    # Dictionary within a dictionary
    for ck, cv in v.items():    # ck = key word(ids -> 0, 1, 2, ... number of persons) and cv = value assigned to key ('cam_K', 'bboxes_dict, 'frames')
        new_dict[k][ck] = cv    
pickle.dump(new_dict, open(f'{opt.out_dir}/pose.pkl', 'wb'))  

images_to_video(res_images_path, f'{opt.out_dir}/render.mp4', img_fmt='%06d.jpg')
images_to_video(res_2D_poses_images_path, f'{opt.out_dir}/render_2D_pose.mp4', img_fmt='%06d.jpg')
# images_to_video(res_3D_poses_images_path, f'{opt.out_dir}/render_3D_pose.mp4', img_fmt='%06d.jpg')
# shutil.rmtree(f'{opt.out_dir}/res_images')
