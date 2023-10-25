import os, sys
sys.path.append(os.path.join(os.getcwd()))
import os.path as osp
import glob
import torch
torch.cuda.empty_cache()
import numpy as np
import pickle
import cv2 as cv
import shutil
import argparse
from lib.utils.log_utils import create_logger
from lib.utils.vis import get_video_num_fr, get_video_fps, hstack_video_arr, get_video_width_height, video_to_images
from global_recon.utils.config import Config
from global_recon.models import model_dict
from global_recon.vis.vis_grecon import GReconVisualizer
from global_recon.vis.vis_cfg import demo_seq_render_specs as seq_render_specs
from pose_est.run_pose_est_demo import run_pose_est_on_video
# from src.MPT.posprocess_human_track import convert_track_info, find_folder_with_condition, reorganize_track_info, append_track_info, check_image_folder


# mode = "dynamic_single"
# mode = "static_single"
# mode = "dynamic_multi"
# mode = "static_multi"
mode = "glamr_3dpw"

video2images = True # False para não converter video para imagens no run_pose_est_on_video() antes do hybrik
high_performance = True # Não faz reconstrução da mesh nem plot das poses 2D no hybrik e inclui contador para inference time

MOT_settings = {
    "multi_person_tracking_method": "ocsort",    #deepocsort, strongsort, botsort  
    # multi_person_tracking_method = "sort"
    "single_person_detection_method": "ocsort", 
    # "single_person_detection_method": "fasterrcnn_resnet50", # Outro qualquer utiliza o processo de multi_person_tracking_method
    "YOLO_model": "yolov8x.pt",
    "ReID_model": "osnet_ain_x1_0_msmt17.pt",   # osnetx1_0_dukemtcereid.pt
    "iou_value": 0.6,
    "conf_value": 0.3
}
  
parser = argparse.ArgumentParser()
if mode=="static_single":
    parser.add_argument('--cfg', default='glamr_static')
    parser.add_argument('--multi', action='store_true', default=False)
    # parser.add_argument('--video_path', default='assets/static/basketball.mp4')
    # parser.add_argument('--out_dir', default='out/glamr_static/basketball')
    # parser.add_argument('--video_path', default='assets/static/workout.mp4')
    # parser.add_argument('--out_dir', default='out/glamr_static/workout')
    parser.add_argument('--video_path', default='assets/static/WalkingAround_Joe.mp4')
    parser.add_argument('--out_dir', default='out/glamr_static/WalkingAround_Joe')
elif mode == "static_multi":
    parser.add_argument('--cfg', default='glamr_static_multi')
    parser.add_argument('--multi', action='store_true', default=True)
    parser.add_argument('--video_path', default='assets/static/basketball.mp4')
    parser.add_argument('--out_dir', default='out/glamr_static_multi/basketball')
elif mode == "dynamic_single":
    parser.add_argument('--multi', action='store_true', default=False)
    parser.add_argument('--cfg', default='glamr_dynamic')
    parser.add_argument('--video_path', default='assets/dynamic/running.mp4')
    parser.add_argument('--out_dir', default='out/glamr_dynamic/running')
elif mode == "dynamic_multi":
    parser.add_argument('--cfg', default='glamr_3dpw')
    parser.add_argument('--multi', action='store_true', default=True)
    parser.add_argument('--video_path', default='assets/dynamic/Havoc_Ladies.mp4')
    parser.add_argument('--out_dir', default='out/glamr_dynamic/Havoc_Ladies')
elif mode == "glamr_3dpw":
    parser.add_argument('--cfg', default='glamr_3dpw')
    parser.add_argument('--video_path', default='assets/dynamic/Havoc_Ladies.mp4')
    parser.add_argument('--out_dir', default='out/glamr_dynamic/Havoc_Ladies')
    # parser.add_argument('--video_path', default='assets/static/basketball.mp4')
    # parser.add_argument('--out_dir', default='out/glamr_static/basketball')
    parser.add_argument('--multi', action='store_true', default=True)

parser.add_argument('--network_type', default='hrnet48') # network_type: hrnet48    # resnet34 is the old backbone from Hybrik  
parser.add_argument('--pose_est_dir', default=None) # path to save video frames (set below)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--cached', type=int, default=1)                
parser.add_argument('--vis', action='store_true', default=False)            # output meshs after inference
parser.add_argument('--vis_cam', action='store_true', default=False)        # if true plot meshs after inference on picture
parser.add_argument('--save_video', action='store_true', default=True)      # save videos 
parser.add_argument('--render_mode',help='save video with shapes -> "shape" , poses -> "pose" or shape and poses -> "shape+pose', default='shape+pose')                    # save video with shape, poses or shape and poses prev 
parser.add_argument('--save_side_by_side', action='store_true', default=False)      # Guardar os vídeos lado a lado (Hybrik, e glamr camera e glamr world)
parser.add_argument('--rend_fps', type=int, default=30)
args = parser.parse_args()
  
cached = int(args.cached)
rend_fps=int(args.rend_fps)
cfg = Config(args.cfg, out_dir=args.out_dir)
if torch.cuda.is_available() and args.gpu >= 0:
    device = torch.device('cuda', index=args.gpu)
    torch.cuda.set_device(args.gpu)     
    print(f'-> {torch.cuda.get_device_name(0)} available!\nRunning inference on GPU :) ')
else:
    device = torch.device('cpu')
    print(f"-> NVIDIA not available! -> Running inference on CPU :'(")

cfg.save_yml_file(f'{args.out_dir}/config.yml')
log = create_logger(f'{cfg.log_dir}/log.txt')
grecon_path = f'{args.out_dir}/grecon_from_{args.network_type}'          # generative reconstruction
render_path = f'{args.out_dir}/grecon_videos_from_{args.network_type}'   # generative reconstruction videos
os.makedirs(grecon_path, exist_ok=True)
os.makedirs(render_path, exist_ok=True)

if args.pose_est_dir is None:
    pose_est_dir = f'{args.out_dir}/pose_est_{cfg.grecon_model_specs["est_type"]}_{args.network_type}'
else:
    pose_est_dir = args.pose_est_dir

if args.render_mode=='shape+pose':
    render_mode='shape+pose'
else:
    # assert args.render_mode in {'shape', 'pose'}
    render_mode = args.render_mode if args.render_mode in {'shape', 'pose'} else 'shape'

print(f'-------------------- Args/settings: -------------------- \n \
    cfg: {args.cfg}\n \
    video_path: {args.video_path}\n \
    out_dir: {args.out_dir}\n \
    pose_est_dir: {pose_est_dir}\n \
    grecon_path: {grecon_path}\n \
    render_path: {render_path}\n \
    log_path: {log}\n \
    seed: {args.seed}\n \
    gpu: {args.gpu}\n \
    cached: {args.cached}\n \
    multi: {args.multi}\n \
    vis: {args.vis}\n \
    vis_cam: {args.vis_cam}\n \
    save_video: {args.save_video}\n \
    rend_fps: {args.rend_fps}\n')   

if args.pose_est_dir is None:
    # pose_est_dir = f'{args.out_dir}/pose_est_{cfg.grecon_model_specs["est_type"]}_{args.network_type}'
    log.info(f"running {cfg.grecon_model_specs['est_type']} pose estimation on {args.video_path}...")   # image_dir abaixo usado para prever as poses num diretório já com imagens obtidas de vídeo        
    run_pose_est_on_video(args.video_path, pose_est_dir, cfg.grecon_model_specs['est_type'], args.network_type, MOT_settings, image_dir=None, cached_pose=cached, gpu_index=args.gpu, multi=args.multi, high_performance=high_performance, video2images=video2images)
else:
    pose_est_dir = args.pose_est_dir
pose_est_model_name = {'hybrik': 'HybrIK'}[cfg.grecon_model_specs['est_type']]


# global recon model
grecon_model = model_dict[cfg.grecon_model_name](cfg, device, log)
#Leitura da "semente" para incialização do gerador de números do NumPy (np.random.randint() e np.random.rand()) e PyTorch (torch.randn()). Ao fornecer a mesma semente, garantimos que a sequência de números aleatórios gerada seja a mesma em diferentes partes do código.
seed = args.seed    
np.random.seed(args.seed)
torch.manual_seed(args.seed)

log.info(f'running global reconstruction on {args.video_path}, seed: {seed}')
seq_name = osp.splitext(osp.basename(args.video_path))[0]
pose_est_file = f'{pose_est_dir}/pose.pkl'                  # Poses from Hybrik
shape_est_video_hybrik = f'{pose_est_dir}/render.mp4'       # The poses in this video are generated by HybrIK, the final shape is on the path defined by the variable video_glamr_cam
pose_est_video_hybrik = f'{pose_est_dir}/render_2d_pose.mp4'
img_w, img_h = get_video_width_height(args.video_path)
num_fr = get_video_num_fr(args.video_path)
fps = get_video_fps(args.video_path) 
img_w = 1080 if img_w == 0 else img_w
img_h = 1920 if img_h == 0 else img_h
fps = args.rend_fps if fps==0 else fps

# main
out_file_GLAMR = f'{grecon_path}/{seq_name}_GLAMR_seed{seed}.pkl'

if cached and osp.exists(out_file_GLAMR):                 # load determined poses from GLAMR if already exist.
    out_dict = pickle.load(open(out_file_GLAMR, 'rb'))
else:                                               # Otherwise, determine a new dict
    est_dict = pickle.load(open(pose_est_file, 'rb'))
    in_dict = {'est': est_dict, 'gt': dict(), 'gt_meta': dict(), 'seq_name': seq_name}
    # global optimization
    out_dict = grecon_model.optimize(in_dict)
    pickle.dump(out_dict, open(out_file_GLAMR, 'wb'))     # Write a new dict for the pkl file with GLAMR poses

if (args.vis and args.vis_cam) or args.save_video:      
    frame_dir = f'{pose_est_dir}/frames'
    if len(glob.glob(f'{frame_dir}/*.jpg')) != out_dict['meta']['num_fr']:      # Checks if the frames are already extracted
        log.info(f'generating frames from {args.video_path}...')
        video_to_images(args.video_path, frame_dir, fps=fps, verbose=False)

# visualization
if args.vis:
    render_specs = seq_render_specs.get(seq_name, dict())   # Camera focus and position
    if args.vis_cam:
        visualizer = GReconVisualizer(out_dict, coord='cam_in_world', verbose=False, background_img_dir=frame_dir)
        visualizer.show_animation(window_size=(img_w, img_h), show_axes=False)
    else:
        render_specs = seq_render_specs.get(seq_name, seq_render_specs['default'])
        visualizer = GReconVisualizer(out_dict, coord='world', verbose=False, show_camera=True,  
                                      render_cam_pos=render_specs.get('cam_pos', None), render_cam_focus=render_specs.get('cam_focus', None), show_center=True)
        visualizer.show_animation(window_size=(1920, 1080), frame_mode='fps', fps=fps, repeat=True, show_axes=True)

# save video
if args.save_video: # Camera position and configuration for scene rendering. 'cam_focus' -> set of coordinates (x, y, z) that represent the camera's focus point. Location where the camera is directed or the object of interest in the scene. 'cam_pos' -> set of coordinates (x, y, z) that represent the position of the camera in the scene. They indicate the location of the camera in relation to the global coordinate system.
    render_specs = seq_render_specs.get(seq_name, seq_render_specs['default'])  # The get method returns the value corresponding to the key "seq_name" in the dictionary, if it exists. Otherwise, it returns the value associated with the 'default' key.
    video_glamr_world_shape = f'{render_path}/{seq_name}_GLAMR_seed{seed}_world(shape)_{rend_fps}fps.mp4'
    video_glamr_world_poses = f'{render_path}/{seq_name}_GLAMR_seed{seed}_world(poses)_{rend_fps}fps.mp4'
    video_hybrik_world = f'{render_path}/{seq_name}_{pose_est_model_name}_seed{seed}_world_{rend_fps}fps.mp4'    # Hybrik's pose in pose_est_video_hybrik's path
    video_glamr_cam = f'{render_path}/{seq_name}_GLAMR_seed{seed}_cam_{rend_fps}fps.mp4'
    video_all = f'{render_path}/{seq_name}_seed{seed}_all_{rend_fps}fps.mp4'

    if render_mode == 'shape+pose' or render_mode == 'shape':
        log.info(f'-> saving world animation (shape) for {seq_name}')
        visualizer = GReconVisualizer(out_dict, coord='world', verbose=False, show_gt_pose=False, show_est_pose=True, show_smpl=True, show_skeleton=False, show_camera=True, align_pose=False,
                                    render_cam_pos=render_specs.get('cam_pos', None), render_cam_focus=render_specs.get('cam_focus', None), show_axes=True, show_center=True)   # Adjust zoom in def init_camera(self): -> self.pl.camera.zoom(0.5) 
        visualizer.save_animation_as_video(video_glamr_world_shape, window_size=render_specs.get('wsize', (int(1.5 * max(img_w,img_h)), int(0.75 * max(img_w,img_h)))), fps=rend_fps, cleanup=True, crf=5)
    
    if render_mode == 'shape+pose' or render_mode == 'pose':
        log.info(f'-> saving world animation (poses) for {seq_name}')
        visualizer = GReconVisualizer(out_dict, coord='world', verbose=False, show_gt_pose=False, show_est_pose=True, show_smpl=False, show_skeleton=True, show_camera=True, align_pose=False,
                                    render_cam_pos=render_specs.get('cam_pos', None), render_cam_focus=render_specs.get('cam_focus', None), show_axes=True, show_center=True)   # Adjust zoom in def init_camera(self): -> self.pl.camera.zoom(0.5) 
        visualizer.save_animation_as_video(video_glamr_world_poses, window_size=render_specs.get('wsize', (int(1.5 * max(img_w,img_h)), int(0.75 * max(img_w,img_h)))), fps=rend_fps, cleanup=True, crf=5)

    log.info(f'-> saving cam animation for {seq_name}')
    visualizer = GReconVisualizer(out_dict, coord='cam_in_world', verbose=False, background_img_dir=frame_dir)
    visualizer.save_animation_as_video(video_glamr_cam, window_size=(img_w, img_h), fps=rend_fps, cleanup=True)
    
    if args.save_side_by_side:
        log.info(f'-> saving side-by-side animation for {seq_name}')
        hstack_video_arr([shape_est_video_hybrik, video_glamr_cam, video_glamr_world_shape], video_all, text_arr=[pose_est_model_name + ' (Cam)', 'GLAMR (Cam)', 'GLAMR (World)'], text_color='blue', text_size=img_h // 16, verbose=False)

print("We are done!")