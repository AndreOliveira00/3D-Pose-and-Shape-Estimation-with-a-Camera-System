"""Image demo script."""
import argparse
import os
import sys
import os.path as osp
import subprocess
sys.path.append('./')
import cv2
import numpy as np
import torch
import shutil
import pickle

from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.utils.presets import SimpleTransform3DSMPL
from hybrik.utils.render import SMPLRenderer
from hybrik.utils.vis import get_max_iou_box, get_one_box, vis_2d, vis_smpl_3d, draw_3D_skeleton
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
from collections import defaultdict


torch.set_grad_enabled(False)
det_transform = T.Compose([T.ToTensor()])

# Pares de índices representando as conexões entre as articulações
bones_jts_29 = np.array([    
    [0, 3],	    # pelvis -> spine1              # Cor 1 (verde) [self.JOINT_NAMES.index('pelvis'),self.JOINT_NAMES.index('spine1')
    [3, 6],	    # spine1 -> spine2
    [6, 9],   	# spine2 -> spine3
    [9, 12],	# spine3 -> neck
    [12, 15],	# neck -> jaw
    # [15, 24],	# jaw -> head               
    [0, 1],   	# pelvis -> left_hip        # Cor 2.1 (azul)
    [1, 4],	    # left_hip -> left_knee
    [4, 7],	    # left_knee -> left_ankle
    [7, 10],	# left_ankle -> left_foot
    # [10, 27],	# left_foot -> left_bigtoe
    [9, 13],	# spine3 -> left_collar     # Cor 2.2 (azul)
    [13, 16],	# left_collar -> left_shoulder
    [16, 18],	# left_shoulder -> left_elbow
    [18, 20],	# left_elbow -> left_wrist
    [20, 22],	# left_wrist -> left_thumb  
    # [22, 25],   # left_thumb -> left_middle
    [0, 2],	    # pelvis -> right_hip           # Cor 3.1 (vermelho)
    [2, 5],	    # right_hip -> right_knee
    [5, 8],	    # right_knee -> right_ankle
    [8, 11],	# right_ankle -> right_foot
    # [11, 28],	# right_foot -> right_bigtoe
    [9, 14],	# spine3 -> right_collar    # Cor 3.2 (vermelho)
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
    cmd = [FFMPEG_PATH, '-y', '-r', f'{fps}', '-f', 'image2', '-start_number', '0',
            '-i', f'{img_dir}/{img_fmt}', '-vcodec', 'libx264', '-crf', f'{crf}', '-pix_fmt', 'yuv420p', out_path]
    if not verbose:
        cmd += ['-hide_banner', '-loglevel', 'error']
    subprocess.run(cmd)


parser = argparse.ArgumentParser(description='HybrIK Demo')
CKPT = 'pretrained_w_cam.pth'       # Chamada ao pretrained weights (pretrained_w_cam.pth) -> Usado na ResNet-34 (ver abaixo em cfg_file)
cfg_file = 'configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix.yaml'
# CKPT = 'pretrained_weights/hybrik_hrnet48_w3dpw.pth'
# cfg_file = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml'
cfg = update_config(cfg_file)


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
                    default='/home/andre/Documents/Projects/GLAMR/out/glamr_static/workout_5s/pose_est/frames',
                    type=str)
parser.add_argument('--out_dir',
                    help='output folder',
                    default='/home/andre/Documents/Projects/GLAMR/out/glamr_static/workout_5s/pose_est',
                    type=str)
parser.add_argument('--MPT_method',
                    help='strongsort, deepocsort, ocsort, bytetrack, botsort or sort (original)',
                    default='ocsort',
                    type=str)
parser.add_argument('--person_detection_method',
                    help='strongsort, deepocsort, ocsort, bytetrack, botsort or fasterrcnn_resnet50 (original)',
                    default='ocsort',
                    type=str)
opt = parser.parse_args()


bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
bbox_3d_shape = [item*1e-3 for item in bbox_3d_shape]
dummpy_set = edict({
    'joint_pairs_17': None,
    'joint_pairs_24': None,
    'joint_pairs_29': None,
    'bbox_3d_shape': bbox_3d_shape
})

transformation = SimpleTransform3DSMPL(     # Transformação necessária para preparar a entrada do modelo Hybrik (pré-processamento)
    dummpy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
    color_factor=cfg.DATASET.COLOR_FACTOR,
    occlusion=cfg.DATASET.OCCLUSION,
    input_size=cfg.MODEL.IMAGE_SIZE,    #256x256
    output_size=cfg.MODEL.HEATMAP_SIZE,
    depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
    bbox_3d_shape=bbox_3d_shape,
    rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
    train=False, add_dpg=False,
    loss_type=cfg.LOSS['TYPE'])

hybrik_model = builder.build_sppe(cfg.MODEL)

print(f'Loading model from {CKPT}...')
hybrik_model.load_state_dict(torch.load(CKPT, map_location='cpu'), strict=False)
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

    from multi_person_tracker import MPT
    
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(opt.gpu))  
    else:
        device =  torch.device('cpu')
    
    # file_txt = "frames.txt"
    # method="sort"
    # opt.MPT_method = "strongsort"

    if opt.MPT_method=="sort":
        # load multi-person tracking model
        mot = MPT(
            device=device,
            batch_size=4,
            display=False,
            detection_threshold=0.7,
            detector_type='yolo',
            output_format='dict',
            yolo_img_size=416,
        )
        print('\n### Run MPT...')
        tracking_results = mot(opt.img_folder)
        offset_frames=0
    else:           # strongsort, deepocsort, ocsort, bytetrack ou botsort 
        # from src.MPT.posprocess_human_track import convert_track_info
        track_pkl_file = os.path.join(opt.out_dir, 'track')
        tracking_results = pickle.load(open(f'{track_pkl_file}/mpt.pkl', 'rb')) 
        if opt.MPT_method=='strongsort':
            offset_frames = 2                           # Necessário para inicializar o modelo: https://github.com/mikel-brostrom/yolo_tracking/issues/379      https://github.com/mikel-brostrom/yolo_tracking/blob/8885642c9d049c933c6e1df1d05478dab4a0c37c/deep_sort/configs/deep_sort.yaml#L6        
            img_1 = cv2.imread(img_path_list[0])        # Ver file: src/yolo_tracking/boxmot/strongsort/configs/strongsort.yaml
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
    
        

        # Root_GLAMR = os.path.dirname(os.getcwd())          # Root GLAMR
        # func_dir = os.path.join(Root_GLAMR, 'src', 'MPT', 'posprocess_human_track.py')  

    detection_all = defaultdict(dict)
    for person_id in tracking_results:  # Percorrer todas as pessoas detetadas
        frames_ids = tracking_results[person_id]['frames']
        for idx in range(len(frames_ids)):      # Percorrer todos os frames que deteta a mesma pessoa -> Conversão de cx,cy,w,h para x1,y1,x2,y2
            frames_id = frames_ids[idx]
            cx, cy, w, h = tracking_results[person_id]['bbox'][idx]
            x1, y1, x2, y2 = max(0, cx-w//2), max(0, cy-h//2), cx+w//2, cy+h//2
            detection_all[frames_id][person_id-1] = [x1, y1, x2, y2]                # detection_all -> frame -> ID da pessoa -> bbox da pessoa 

    out_dict = defaultdict(lambda: defaultdict(list))   # Dicionário de listas (index deve ser int)
    bbox_exist = defaultdict(list)          # Dicionário exclusivo ("index" é uma string -> dicionário)
    bboxes = defaultdict(list)
    poses_dict = defaultdict(lambda: defaultdict(list))

    # initialize
    for person_id in tracking_results:
        bbox_exist[person_id-1] = [0 for _ in range(len(img_path_list))]

    print('\n### Run HybrIK...')
    for frame_idx in tqdm(range(len(img_path_list)-offset_frames)):

        frame_idx+=offset_frames
        img_path = img_path_list[frame_idx]
        dirname = os.path.dirname(img_path)
        basename = os.path.basename(img_path)

        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = input_image.copy()
        image_pose_2d = input_image.copy()
        
        if frame_idx in detection_all:
            # For each detected person, starting from 0,1,...
            for idx in detection_all[frame_idx]:
                tight_bbox = detection_all[frame_idx][idx]
                bbox_exist[idx][frame_idx] = 1.0
                
                # Run HybrIK
                pose_input, bbox = transformation.test_transform(img_path, tight_bbox)  #  Usada para preparar a entrada do modelo HybrIK
                pose_input = pose_input.to(opt.gpu)[None, :, :, :]
                pose_output = hybrik_model(pose_input)
                uv_3D_29 = pose_output.pred_uvd_jts.reshape(29, 3)
                uv_29 = uv_3D_29[:, :2]         # Reshape para obter exclusivamente as coordenandas (x,y) das 29 linhas (articulações) da matriz pose_output.pred_uvd_jts que continha 29 linhas e 3 colunas (x,y,z)
                
                # Visualization
                img_size = (image.shape[0], image.shape[1])
                focal = np.array([1000, 1000])
                bbox_xywh = xyxy2xywh(bbox)
                princpt = [bbox_xywh[0], bbox_xywh[1]]
                # SMPL Render a partir de principal point, faces e focal da camera
                renderer = SMPLRenderer(faces=hybrik_model.smpl.faces,
                                        img_size=img_size, focal=focal,
                                        princpt=princpt)

                transl = pose_output.transl.detach().cpu().numpy().squeeze()
                transl[2] = transl[2] * 256 / bbox_xywh[2]
                # Gerar imagem com renderização smpl do humano na imagem 2D
                image = vis_smpl_3d(
                    pose_output, image, cam_root=transl,
                    f=focal, c=princpt, renderer=renderer)
                cv2.putText(image, f'{idx}', (int(tight_bbox[0]), int(tight_bbox[1]) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                cv2.putText(image, f'{idx}', (int(tight_bbox[0]) + 1, int(tight_bbox[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                
                # vis 3d   
                # res_path = os.path.join(res_3D_poses_images_path, f'{frame_idx:06d}.jpg')
                pts_3D = uv_3D_29 * bbox_xywh[2]
                pts_3D[:, 0] = pts_3D[:, 0] + bbox_xywh[0]    # shiftar x da bbox
                pts_3D[:, 1] = pts_3D[:, 1] + bbox_xywh[1]    # shiftar y da bbox
                # draw_3D_skeleton(pts_3D, res_path, bones = bones_jts_29, colors = colors, show_image = True)


                # vis 2d
                pts = uv_29 * bbox_xywh[2]
                pts[:, 0] = pts[:, 0] + bbox_xywh[0]
                pts[:, 1] = pts[:, 1] + bbox_xywh[1]

                bboxes[idx].append(np.array(bbox_xywh))

                # bboxes.append(np.array(bbox_xywh))
                image_pose_2d = vis_2d(image_pose_2d, tight_bbox, pts, idx, bones = bones_jts_29, extended = True)

                new_princpt = np.array([image.shape[1], image.shape[0]]) * 0.5
                transl[:2] += (np.array(princpt) - new_princpt) * transl[2] / np.array(focal) 
                princpt = new_princpt

                # save to dict
                K = np.eye(3)
                K[[0, 1], [0, 1]] = focal
                K[:2, 2] = princpt
                out_dict[idx]['smpl_pose_quat_wroot'].append(pose_output.pred_theta_mats[0].cpu().numpy().reshape(-1, 4))
                out_dict[idx]['smpl_beta'].append(pose_output.pred_shape[0].cpu().numpy())
                out_dict[idx]['root_trans'].append(transl)
                out_dict[idx]['kp_2d'].append(pts.cpu().numpy())
                out_dict[idx]['cam_K'].append(K.astype(np.float32))
                # Adicionado para guardar dados renderizados pelo hybrik
                out_dict[idx]['kp_3d'].append(pts_3D.cpu().numpy())    # KEYPOINTS 3D
                # out_dict[idx]['heatmaps'].append(pose_output.uvd_heatmap.cpu().numpy())
                # out_dict[idx]['renderer_hybrik_faces'].append(renderer.faces)
                # out_dict[idx]['renderer_hybrik_focal'].append(renderer.focal)
                # out_dict[idx]['renderer_hybrik_h'].append(renderer.h)
                # out_dict[idx]['renderer_hybrik_w'].append(renderer.w)
                # out_dict[idx]['renderer_hybrik_princpt'].append(renderer.princpt)
                # out_dict[idx]['renderer_hybrik_vertices_mesh'].append(pose_output.pred_vertices.detach().cpu().numpy().squeeze())   # 6890 veritices gerados na malha smpl do hybrik (pronto a utilizar em vis_smpl_3d() e substituir a primeira linha de código dessa função)

        image_vis = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_pose_2d = cv2.cvtColor(image_pose_2d, cv2.COLOR_RGB2BGR)

        res_path = os.path.join(res_images_path, f'{frame_idx:06d}.jpg')
        cv2.imwrite(res_path, image_vis)
        res_path = os.path.join(res_2D_poses_images_path, f'{frame_idx:06d}.jpg')
        cv2.imwrite(res_path, image_pose_2d)

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

else:
    # load detection model
    if opt.person_detection_method=="fasterrcnn_resnet50":
        det_model = fasterrcnn_resnet50_fpn(pretrained=True)
        det_model.cuda(opt.gpu)
        det_model.eval()

    print('\n### Run HybrIK...')

    prev_box = None

    out_dict = defaultdict(lambda: defaultdict(list))
    idx = 0   # single person id

    frame_idx = 0

    bbox_exist = []
    bboxes = []
    
    
    
    
    for i,img_path in enumerate(tqdm(img_path_list)):
        input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if opt.person_detection_method=="fasterrcnn_resnet50":
            # basename = os.path.basename(img_path)
            # dirname = os.path.dirname(img_path)
            # Run Detection
            det_input = det_transform(input_image).to(opt.gpu)
            det_output = det_model([det_input])[0]

            if prev_box is None:
                tight_bbox = get_one_box(det_output)  # xyxy
            else:
                tight_bbox = get_max_iou_box(det_output, prev_box)  # xyxy

            if tight_bbox is None:
                bbox_exist.append(0.0)
                continue
            else:
                bbox_exist.append(1.0)
        else: 
            track_pkl_file = os.path.join(opt.out_dir, 'track')
            tracking_results = pickle.load(open(f'{track_pkl_file}/mpt.pkl', 'rb'))
            if i in tracking_results[1]['frames']:
                tight_bbox = cxcywh2xyxy(tracking_results[1]['bbox'][i])
                bbox_exist.append(1.0)
            else:
                bbox_exist.append(0.0)
                continue

        prev_box = tight_bbox

        # Run HybrIK
        pose_input, bbox = transformation.test_transform(img_path, tight_bbox)
        pose_input = pose_input.to(opt.gpu)[None, :, :, :]              # pose_output com estrutura definida no final de def forward(self, x, flip_item=None, flip_output=False, **kwargs)
        pose_output = hybrik_model(pose_input)                          # ModelOutput = namedtuple(typename='ModelOutput',
        uv_3D_29 = pose_output.pred_uvd_jts.reshape(29, 3)                                  #      field_names=['pred_shape', 'pred_theta_mats', 'pred_phi', 'pred_delta_shape', 'pred_leaf',
        # uv_29 = pose_output.pred_uvd_jts.reshape(29, 3)[:, :2]                            #            'pred_uvd_jts', 'pred_xyz_jts_29', 'pred_xyz_jts_24', 'pred_xyz_jts_24_struct',
        uv_29 = uv_3D_29[:, :2]                                                             #            'pred_xyz_jts_17', 'pred_vertices', 'maxvals', 'cam_scale', 'cam_trans', 'cam_root',
                                                                                            #            'uvd_heatmap', 'transl', 'img_feat']                                                                                                                     
        # Converter poses para a imagem e guardar em figura
        
        # Visualization
        image = input_image.copy()
        img_size = (image.shape[0], image.shape[1])
        focal = np.array([1000, 1000])
        bbox_xywh = xyxy2xywh(bbox)             # Calculo do ponto central (x,y)
        princpt = [bbox_xywh[0], bbox_xywh[1]]

        renderer = SMPLRenderer(faces=hybrik_model.smpl.faces,
                                img_size=img_size, focal=focal,
                                princpt=princpt)

        transl = pose_output.transl.detach().cpu().numpy().squeeze()
        transl[2] = transl[2] * 256 / bbox_xywh[2]

        image_vis = vis_smpl_3d(
            pose_output, image, cam_root=transl,
            f=focal, c=princpt, renderer=renderer)

        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)  
        # import matplotlib.pyplot as plt
        # plt.imshow(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))

        frame_idx += 1
        res_path = os.path.join(res_images_path, f'{frame_idx:06d}.jpg')
        cv2.imwrite(res_path, image_vis)
        
        # vis 3d   
        # res_path = os.path.join(res_3D_poses_images_path, f'{frame_idx:06d}.jpg')
        pts_3D = uv_3D_29 * bbox_xywh[2]
        pts_3D[:, 0] = pts_3D[:, 0] + bbox_xywh[0]    # shiftar x da bbox
        pts_3D[:, 1] = pts_3D[:, 1] + bbox_xywh[1]    # shiftar y da bbox
        # draw_3D_skeleton(pts_3D, bones = bones_jts_29, colors = colors, save_path = res_path, show_image = False)
        # Armazenar pts_3D no dicionário que já criei (poses_dict)? Para depois desenhar um gráfico 3D no final do ciclo for (quando tiver analisado e guardado 
        #                                                                                                                     a informação de todas as pessoas)
        
        # vis 2d
        pts = uv_29 * bbox_xywh[2]
        pts[:, 0] = pts[:, 0] + bbox_xywh[0]    # shiftar x da bbox
        pts[:, 1] = pts[:, 1] + bbox_xywh[1]    # shiftar y da bbox

        bboxes.append(np.array(bbox_xywh))
        bbox_img = vis_2d(image, tight_bbox,pts, bones = bones_jts_29, extended=True)       # VERIFICAR SE DESENHA BEM PARA AS VÁRIAS PESSOAS!!!
        bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
        res_path = os.path.join(res_2D_poses_images_path, f'{frame_idx:06d}.jpg')
        cv2.imwrite(res_path, bbox_img)

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
        out_dict[idx]['kp_3d'].append(pts_3D.cpu().numpy())    # KEYPOINTS 3D
        # out_dict[idx]['heatmaps'].append(pose_output.uvd_heatmap.cpu().numpy())
        
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


new_dict = dict()
for k in sorted(out_dict.keys()):   # out_dict.keys() possui os identificadores de pessoas (cenário basketball -> out_dict.keys() = [1, 0, 2])
    v = out_dict[k]     # Informação por person ID (dicionário output)
    new_dict[k] = dict()    # Dicionário dentro de dicionário
    for ck, cv in v.items():    # ck = key word(ids -> 0, 1, 2, ... número de pessoas) e cv = valor atribuido à key ('cam_K', 'bboxes_dict, 'frames')
        new_dict[k][ck] = cv    
pickle.dump(new_dict, open(f'{opt.out_dir}/pose.pkl', 'wb'))  

images_to_video(res_images_path, f'{opt.out_dir}/render.mp4', img_fmt='%06d.jpg')
images_to_video(res_2D_poses_images_path, f'{opt.out_dir}/render_2D_pose.mp4', img_fmt='%06d.jpg')
# images_to_video(res_3D_poses_images_path, f'{opt.out_dir}/render_3D_pose.mp4', img_fmt='%06d.jpg')
# shutil.rmtree(f'{opt.out_dir}/res_images')
