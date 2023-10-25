import os
import pickle
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import drawGaussian, drawGaussian_multi, draw_2D_heatmaps, images_to_video, draw_3D_skeleton_multi, draw_3D_skeleton, resize_images

# pose_est_dir="out/glamr_static/basketball/pose_est"
pose_est_dir="out/glamr_static/WalkingAround_Joe_0_10/pose_est_hybrik_hrnet48"
# pose_est_dir="out/glamr_dynamic/courtyard_basketball_00_0_415/pose_est_hybrik_hrnet48"

# background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
# frame_with_box_poses = cv2.imread('out/glamr_static/basketball/pose_est/res_2D_poses_images/000001.jpg')    # Código para determinar este esqueleto 2D na imagem??
# frame_with_box_poses = cv2.cvtColor(frame_with_box_poses, cv2.COLOR_RGB2BGR)

sigma=2
amplitude=1
time_between_images = 0.5
multi_person = False 
plot_2D_heatmaps = False           # drawGaussian Heatmaps (res_2D_heatmaps, done single, done multi person?)    # One frame
plot_3D_skeleton = True         # draw_3D_skeleton (done single, multi person done)            # All frames
plot_multi_heatmaps = False        # draw_2D_heatmaps (uma imagem com 29 imagens -> todos os heatmaps separados)(done single, multi person done)    # One frame
image2video = False
frame_id = 1                # De 1 a ...    
    

frame = str(frame_id).zfill(6) + ".jpg"
frame_num = os.path.splitext(os.path.basename(frame))[0]
img_path = f'{pose_est_dir}/frames/{frame}'

dict_file = f'{pose_est_dir}/pose.pkl'                                  # Poses do Hybrik 
# dict_file = f'{pose_est_dir}/pose_with_3D_witout_hm_300_frames.pkl'   # Poses do Hybrik com SORT tracker 
# dict_file = f'{pose_est_dir}/pose_hm_10_frames.pkl'                   # Tem heatmaps de 10 frames
hybrik_dict = pickle.load(open(dict_file, 'rb'))

heatmaps_output_path = f'{pose_est_dir}/res_2D_heatmaps'
res_3D_poses_images_path = f'{pose_est_dir}/res_3D_poses_images'
os.makedirs(heatmaps_output_path, exist_ok=True)
os.makedirs(res_3D_poses_images_path, exist_ok=True)

# print(out_dict.items())

if multi_person:
    new_hybrik_dict = dict()
    for k in sorted(hybrik_dict.keys()):
        new_hybrik_dict[k] = dict()
        v = hybrik_dict[k]
        for ck, cv in v.items():
            if ck == 'frames' or ck == 'frame2ind' or (ck == 'bboxes_dict' and plot_3D_skeleton):
                new_hybrik_dict[k][ck] = cv
            elif ck == 'kp_3d' or ck == 'kp_2d':            # Keypoints 2D / # Keypoints 3D 
                new_hybrik_dict[k][ck] = torch.tensor(cv)
            elif ck == 'heatmaps' and (plot_2D_heatmaps or plot_multi_heatmaps):      
                new_hybrik_dict[k][ck] = torch.tensor(cv)
else: 
    if plot_2D_heatmaps or plot_multi_heatmaps:
        heatmaps=hybrik_dict[0]['heatmaps']   
        heatmaps = torch.tensor(heatmaps)            
    kp_3D_29 = hybrik_dict[0]['kp_3d']             # Keypoints 3D          
    kp_2D_29 = hybrik_dict[0]['kp_2d']             # Keypoints 2D  
    # hm_stack=out_dict['hm_stack']               # Não preciso 
    # tight_bbox = out_dict['tight_bbox']         # Não preciso 
    # bbox_xywh = hybrik_dict['bbox_xywh']           # Preciso e já tenho (se já tiver pts não preciso)
    frames = hybrik_dict[0]['frames']
    # frame2ind = hybrik_dict[0]['frame2ind']

    kp_3D_29 = torch.tensor(kp_3D_29)
    kp_2D_29 = torch.tensor(kp_2D_29)
    # hm_stack = torch.tensor(hm_stack)



# Sacar dict para multi person (gravar heatmap (apenas alguns frames) e poses 3D (sempre))
# Implementar tudo isto em multi person   

if multi_person:        
    if plot_multi_heatmaps:    # One frame
        res_path = f'{heatmaps_output_path}/frame_{frame_num}/2D_heatmaps'
        for human_id in sorted(new_hybrik_dict.keys()): 
            draw_2D_heatmaps(new_hybrik_dict[human_id]['heatmaps'][frame_id-1], human_id=human_id, mode='mean', save_path = res_path)   

    if plot_2D_heatmaps:       # One frame
        # all_joints e 1-Channel_joints pessoas numa só figura. # uni_joints pessoas em diretórios separados 
        drawGaussian_multi(img_path, frame_id, new_hybrik_dict, sigma=sigma, amplitude=amplitude, fig=1, calc_mode = 'normal', mode='mean', time=time_between_images, one_channel= True, save_path=heatmaps_output_path)
        # drawGaussian_multi(img_path, frame_id, new_hybrik_dict, sigma=sigma, amplitude=amplitude, fig=1, calc_mode = 'uni', mode='mean', time=time_between_images, one_channel= True, save_path=heatmaps_output_path)

    if plot_3D_skeleton:    # All frames
        init_frame=124
        range_frames=5
        draw_3D_skeleton_multi(new_hybrik_dict, init_frame=init_frame, range_frames=range_frames, save_path = res_3D_poses_images_path, show_image = True)
        # init_frame=265
        # draw_3D_skeleton_multi(new_hybrik_dict, init_frame=init_frame, range_frames=range_frames, save_path = res_3D_poses_images_path, show_image = True)
        # init_frame=5
        # draw_3D_skeleton_multi(new_hybrik_dict, init_frame=init_frame, range_frames=range_frames, save_path = res_3D_poses_images_path, show_image = True)
        # init_frame=285
        # draw_3D_skeleton_multi(new_hybrik_dict, init_frame=init_frame, range_frames=range_frames, save_path = res_3D_poses_images_path, show_image = True)
        # init_frame=295
        # draw_3D_skeleton_multi(new_hybrik_dict, init_frame=init_frame, range_frames=range_frames, save_path = res_3D_poses_images_path, show_image = True)
        # init_frame=225
        # draw_3D_skeleton_multi(new_hybrik_dict, init_frame=init_frame, range_frames=range_frames, save_path = res_3D_poses_images_path, show_image = True)

else:
    if plot_multi_heatmaps:
        res_path = f'{heatmaps_output_path}/frame_{frame_num}/2D_heatmaps'
        draw_2D_heatmaps(heatmaps[frame_id-1], human_id=0, mode='mean', save_path = res_path)   

    if plot_2D_heatmaps:
        drawGaussian(img_path, kp_2D_29[frame_id-1], heatmaps[frame_id-1], sigma=sigma, amplitude=amplitude, fig=1, calc_mode = 'normal', mode='mean', time=time_between_images, one_channel= True, save_path=heatmaps_output_path)
        # drawGaussian(img_path, kp_2D_29[frame_id-1], heatmaps[frame_id-1], sigma=sigma, amplitude=amplitude, fig=1, calc_mode = 'uni', mode='mean', time=time_between_images, one_channel= False, save_path=heatmaps_output_path)

    if plot_3D_skeleton:
        for frame_idx in tqdm(frames):
            res_path = os.path.join(res_3D_poses_images_path, f'{frame_idx+1:06d}.jpg')
            draw_3D_skeleton(kp_3D_29[frame_idx], show_extend_pose_skeleton= False, save_path = res_path, show_image = True)

if image2video and plot_3D_skeleton:
    # if plot_3D_skeleton:
        # img_dir = res_3D_poses_images_path
        # resized_img_dir = res_3D_poses_images_path
        # target_height = 960  # Altura par
        # resize_images(img_dir, resized_img_dir, target_height)
        images_to_video(res_3D_poses_images_path, f'{pose_est_dir}/render_3D_pose.mp4', img_fmt='%06d.jpg', fps=30)







           