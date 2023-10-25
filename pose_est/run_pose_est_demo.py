import os, sys
sys.path.append(os.path.join(os.getcwd()))
import os.path as osp
import subprocess
import glob
import numpy as np
import argparse
import yaml
import pickle
import shutil
from lib.utils.vis import video_to_images
from src.MPT.posprocess_human_track import convert_track_info, find_folder_with_condition, reorganize_track_info, append_track_info, check_image_folder


def run_pose_est_on_video(video_file, output_dir, pose_est_model, network_type, MOT_settings, image_dir=None, bbox_file=None, cached_pose=True, gpu_index=0, multi=False, high_performance=False, video2images=True):
    
    if cached_pose and (osp.exists(f'{output_dir}/pose.pkl') or osp.exists(f'{os.path.dirname(output_dir)}/pose_est/pose.pkl')):
        return

    if image_dir is None:
        image_folder = osp.join(output_dir, 'frames')
        os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)
        # if len(glob.glob(f'{image_folder}/*.jpg')) != 300:              
        if video2images:
            video_to_images(video_file, image_folder, fps=30)
    else:
        # image_folder = image_dir
        image_folder = osp.join(output_dir, 'frames')
        shutil.rmtree(image_folder) if osp.exists(image_folder) else None
        shutil.copytree(image_dir, image_folder)
        # os.makedirs(image_folder, exist_ok=True)

    check_image_folder(image_folder)           # Verificar se os frames começam em 000001.jpg
        
    conda_path = os.environ["CONDA_PREFIX"].split('/envs')[0]

    root_dir = os.path.join(output_dir, 'track')
    os.makedirs(root_dir, exist_ok=True)
    file = "frames.txt"

    filter_enabled = False 
    
    if pose_est_model == 'hybrik':
        tracking_method = MOT_settings['multi_person_tracking_method'] if MOT_settings['multi_person_tracking_method'] is not None else "ocsort"
        person_detection_method = MOT_settings['single_person_detection_method'] if MOT_settings['single_person_detection_method'] is not None else "ocsort"
        # Criação do dicionário MPT (.pkl file) 
        # tracking_method = "sort"
        if tracking_method!="sort" and person_detection_method!= "fasterrcnn_resnet50" and not osp.exists(f'{root_dir}/mpt.pkl'):   # else faz sort  (agora faz strongsort)
            # Verifica se é exequível criar mpt.pkl pois MPT já foi executado
            YOLO_model = MOT_settings['YOLO_model'] if MOT_settings['YOLO_model'] is not None else "yolov8x.pt"
            ReID_model = MOT_settings['ReID_model'] if MOT_settings['ReID_model'] is not None else "osnet_ain_x1_0_msmt17.pt"
            iou_value = MOT_settings['iou_value'] if MOT_settings['iou_value'] is not None else 0.3
            conf_value = MOT_settings['conf_value'] if MOT_settings['conf_value'] is not None else 0.3
            selected_folder=None
            selected_folder = find_folder_with_condition(root_dir)                                                                    
            if selected_folder is not None:                     
                print(f'\nSaving MPT file to {root_dir}/mpt.pkl')           # Converte dados do traking para criar dicionário em .pkl                       
                track_dict = convert_track_info(root_dir,file)
                pickle.dump(track_dict, open(f'{root_dir}/mpt.pkl', 'wb')) 
                # Usar mpt.pkl e filtrar por pessoa nesta zona
            else:                                                           # Ou cria novos dados para posteriormente criar o mpt.pkl           # mais reID models em: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
                cmd = f'{conda_path}/envs/mot/bin/python src/yolo_tracking/examples/track.py --yolo-model {YOLO_model} --classes 0 --reid-model {ReID_model} --source {image_folder} --conf {conf_value} --iou {iou_value} --save --save-txt --tracking-method {tracking_method} --project {root_dir}'
                print(f"\nRunning MPT ({YOLO_model}) with command:\n-> {cmd}")
                subprocess.run(cmd.split(' '))
                track_dict = convert_track_info(root_dir,file)
                pickle.dump(track_dict, open(f'{root_dir}/mpt.pkl', 'wb'))
                pickle.dump(track_dict, open(f'{root_dir}/mpt_original.pkl', 'wb'))
            if filter_enabled:
                # Usar mpt.pkl e filtrar por pessoa nesta zona
                humans_ids_to_keep = [1,3]
                ### track_dict = pickle.load(open(f'{root_dir}/mpt.pkl', 'rb'))
                dict_humans_track_remove = dict()
                # downtown_runForBus_00
                # dict_humans_track_remove[2] = np.array([398, 399, 400, 401, 402, 435, 437, 438, 441, 442, 443, 444, 445, 446, 547, 548, 549, 559, 560, 561, 562, 563, 564, 565])
                # dict_humans_track_remove[3] = np.array([488, 489, 490, 491, 492, 563, 574])
                # courtyard_basketball_00_0_415
                # dict_humans_track_remove[2] = np.array([139, 140, 141, 142, 145])
                # dict_humans_track_remove[5] = np.array([147, 148, 165, 188, 189, 190, 191, 192, 193])
                # courtyard_basketball_01_0_200_cut_images_516x1350
                # Ladies 1790_2127
                # dict_humans_track_remove[1] = np.array([101,102,103])
                # dict_humans_track_remove[2] = np.array([293,294])
                # dict_humans_track_remove[3] = np.array([29, 30, 31, 32, 33, 34, 73])
                # dict_humans_track_remove[4] = np.array([44,45,46,47,48, 49, 50, 289, 290, 294])
                # dict_humans_track_remove[5] = np.array([121,122])
                # dict_humans_track_remove[9] = np.array([249,250,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284])
                # dict_humans_track_remove[10] = np.array([261,262,263,264,272,273,274,275,276,277,278,279,280])
                # Ladies 1750_2127
                humans_ids_to_keep = [1,2,3,4,5,7,8]
                dict_humans_track_remove[2] = np.array([141,143,144])
                dict_humans_track_remove[1] = np.array([333,334])
                dict_humans_track_remove[3] = np.array([69, 70, 71, 72, 73, 74, 76])
                dict_humans_track_remove[4] = np.array([83, 84, 85, 86, 87, 88, 329, 330, 334])
                dict_humans_track_remove[5] = np.array([161,162])
                dict_humans_track_remove[7] = np.array([289,290,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,316,317,323,324])
                dict_humans_track_remove[8] = np.array([301,302,303,304,309,311,315,318,319,320])
                # humans_ids_to_keep = [1,2,3,4,5]
                # new_track_dict2 = append_track_info(new_track_dict, 2, 5, root_dir=None)
                # new_track_dict3 = append_track_info(new_track_dict2, 3, 4, root_dir=None)
                # humans_ids_to_keep = [1,2,3,6,7]
                # new_track_dict4 = reorganize_track_info(new_track_dict3, root_dir, humans_ids_to_keep) 
                # pickle.dump(new_track_dict4, open(f'{root_dir}/mpt.pkl', 'wb'))
                # downtown_sitOnStairs_00
                dict_humans_track_remove[1] = np.array([372,379,856,857,859,860,861])
                dict_humans_track_remove[3] = np.array([864,865,866,867,868,869])
                # dict_humans_track_remove[1] = np.array([71,72,73,74,75,76,77])
                print(f'Filter MPT file for human {humans_ids_to_keep}: {root_dir}/mpt.pkl')
                new_track_dict = reorganize_track_info(track_dict, root_dir, humans_ids_to_keep, dict_humans_track_remove) 
                # new_track_dict2 = append_track_info(new_track_dict, 1, 3, root_dir=None)
                # new_track_dict2 = append_track_info(new_track_dict2, 2, 4, root_dir=root_dir)
                # pickle.dump(new_track_dict, open(f'{root_dir}/mpt_1,2.pkl', 'wb'))
                # pickle.dump(new_track_dict, open(f'{root_dir}/mpt.pkl', 'wb'))
        if person_detection_method!= "fasterrcnn_resnet50" and not multi and not osp.exists(f'{root_dir}/mpt_original.pkl') and osp.exists(f'{root_dir}/mpt.pkl'):
            humans_ids_to_keep = [1]
            track_dict = pickle.load(open(f'{root_dir}/mpt.pkl', 'rb'))
            pickle.dump(track_dict, open(f'{root_dir}/mpt_original.pkl', 'wb')) 
            print(f'\Filter MPT file for human {humans_ids_to_keep}: {root_dir}/mpt.pkl')
            new_track_dict = reorganize_track_info(track_dict, root_dir, humans_ids_to_keep)    # Guardar novo dicionário organizado e plot de novo tracking
            pickle.dump(new_track_dict, open(f'{root_dir}/mpt.pkl', 'wb')) 
        # Run HybrIK module
        # if network_type == 'resnet34':
        #     if bbox_file is None:
        #         cmd = f'{conda_path}/envs/hybrik/bin/python ../pose_est/hybrik_demo/demo.py --img_folder {osp.abspath(image_folder)} --MPT_method {tracking_method} --person_detection_method {person_detection_method} --out_dir {osp.abspath(output_dir)} --gpu {gpu_index} --multi {1 if multi else 0}'
        #     else:
        #         cmd = f'{conda_path}/envs/hybrik/bin/python ../pose_est/hybrik_demo/demo_dataset.py --img_folder {osp.abspath(image_folder)} --out_dir {osp.abspath(output_dir)} --bbox_file {osp.abspath(bbox_file)} --gpu {gpu_index}'
        #     print(f"\nRunning pose estimation with {pose_est_model.upper()}: {network_type} with command:\n-> {cmd}")
        #     subprocess.run(cmd.split(' '), cwd='./HybrIK')
        elif network_type == 'hrnet48':
            if bbox_file is None:
                cmd = f'{conda_path}/envs/hybrikx2/bin/python ../HybrIK_X/scripts/demo_hrnet48.py --img_folder {osp.abspath(image_folder)} --MPT_method {tracking_method} --person_detection_method {person_detection_method} --out_dir {osp.abspath(output_dir)} --gpu {gpu_index} --multi {1 if multi else 0} --high_performance {high_performance}'
            else:
                print("Bbox is not None...")
            print(f"\nRunning pose estimation with {pose_est_model.upper()}: {network_type} with command:\n-> {cmd}")
            subprocess.run(cmd.split(' '), cwd='./HybrIK_X')
        else:
            print("No HybrIK module available!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default=None, help="path to video file or a directory that contains video files")
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--pose_est_model', default='hybrik')
    parser.add_argument('--pose_est_cfg', default=None)
    parser.add_argument('--seq_range', default=None)
    parser.add_argument('--glob_pattern', default='*')
    parser.add_argument('--render', action='store_true', default=True)
    parser.add_argument('--cached_pose', action='store_true', default=False)
    parser.add_argument('--cached_video', action='store_true', default=False)
    parser.add_argument('--merge_all_ids', action='store_true', default=True)
    parser.add_argument('--cleanup', action='store_true', default=True)
    args = parser.parse_args()

    video_path, output_path = args.video_path, args.output_path
    os.makedirs(output_path, exist_ok=True)
    yaml.safe_dump(args.__dict__, open(f'{output_path}/args.yml', 'w'))

    """ single file """
    if osp.isfile(video_path):
        if osp.splitext(video_path)[1] != '.mp4':
            raise ValueError('Unsupported video file format!')
        print(f'estimating pose for {video_path}')
        output_dir = osp.join(output_path, osp.splitext(osp.basename(video_path))[0])

        run_pose_est_on_video(video_path, output_dir, args.pose_est_model, cached_pose=args.cached_pose)

    else:
        files = sorted(glob.glob(f'{video_path}/{args.glob_pattern}.mp4'))
        seq_names = [os.path.splitext(os.path.basename(x))[0] for x in files]
        if args.seq_range is not None:
            seq_range = [int(x) for x in args.seq_range.split('-')]
            seq_range = np.arange(seq_range[0], seq_range[1])
        else:
            seq_range = np.arange(len(seq_names))

        for sind in seq_range:
            seq_name = seq_names[sind]
            print(f'{sind}/{len(seq_names)} estimating pose for {seq_name}')
            seq_video_path = f'{video_path}/{seq_name}.mp4'
            output_dir = f'{output_path}/{seq_name}'

            run_pose_est_on_video(seq_video_path, output_dir, args.pose_est_model, cached_pose=args.cached_pose)
