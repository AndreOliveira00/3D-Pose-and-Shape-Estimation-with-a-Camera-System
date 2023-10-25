from tqdm import tqdm
import numpy as np
import os
from posprocess_human_track import convert_track_info, reorganize_track_info, plot_new_track_info, append_track_info, check_image_folder
import pickle

# filename = '/home/andre/Documents/Projects/GLAMR/datasets/3DPW/processed_v1/pose/downtown_crossStreets_00.pkl'
# with open(filename, 'rb') as f:
#     data = pickle.load(f, encoding='latin1')

# for idx,i in enumerate(data['person_data'][0]['visible']):
#     if i==0:
#         print(f"frame: {idx}")

# root_dir = '/home/andre/Documents/Projects/GLAMR/out/glamr_dynamic/Havoc_Ladies_1790_1922/pose_est_hybrik_hrnet48/track' 
# track_dict = pickle.load(open(f'{root_dir}/mpt.pkl', 'rb'))
# humans_ids_to_keep = [1,3,91,135]
# plot_new_track_info(track_dict, root_dir, humans_ids_to_keep)

CONVERT_DICT = False
CLEAN_IDS = True
APPEND_IDS = False
# root_dir = '/home/andre/Documents/Projects/GLAMR/out/glamr_static_multi/basketball/pose_est/track'
# root_dir = '/home/andre/Documents/Projects/GLAMR/out/glamr_static/basketball/pose_est/track' 
# root_dir= '/home/andre/Documents/Projects/GLAMR/out/glamr_dynamic/downtown_runForBus_00/pose_est/track (copy)'
root_dir = '/home/andre/Documents/Projects/GLAMR/out/glamr_dynamic/courtyard_basketball_00_0_415/pose_est_hybrik_hrnet48_relatorio/track/test'
# root_dir = '/home/andre/Documents/Projects/GLAMR/out/3dpw/downtown_crossStreets_00/pose_est/track' 
file = "downtown_crossStreets_00.txt"

frames_dir = f'{os.path.dirname(root_dir)}/frames'
check_image_folder (frames_dir)
print("done")
# import cv2
# import matplotlib.pyplot as plt 
# image_file='/home/andre/Documents/Projects/GLAMR/out/glamr_static_multi/basketball/pose_est/res_2D_poses_images/image-000002.jpg'
# image = cv2.imread(image_file) 
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# plt.imshow(image)

##################################################################################
# Converter dicionário
if CONVERT_DICT:
    track_dict = convert_track_info(root_dir,file)
    pickle.dump(track_dict, open(f'{root_dir}/mpt.pkl', 'wb')) 
    pickle.dump(track_dict, open(f'{root_dir}/mpt_original.pkl', 'wb'))  
##################################################################################

##################################################################################
# Limpar IDs de pessoas não pretendidas 
if CLEAN_IDS:
    # root_dir = '/home/andre/Documents/Projects/GLAMR/out/glamr_dynamic/Havoc_Ladies_1790_1922/pose_est_hybrik_hrnet48/track' 

    # Load do dicionário
    track_dict = pickle.load(open(f'{root_dir}/mpt_original.pkl', 'rb'))
    ### track_dict = pickle.load(open(f'{root_dir}/mpt [2,3,67].pkl', 'rb'))
    ### pickle.dump(track_dict, open(f'{root_dir}/mpt_original.pkl', 'wb')) # Guardar original
    # humans_ids_to_keep = [1, 2, 55] # IDs de humanos a manter
    humans_ids_to_keep = [1,3,91,135]
    #### del track_dict[2]

    dict_humans_track_remove = dict()
    # dict_humans_track_remove[1] = np.array([100,101,102,103])
    # dict_humans_track_remove[3] = np.array([28, 29, 30, 31, 32, 33, 34,73])
    # dict_humans_track_remove[4] = np.array([44,45,46,47,48])
    # dict_humans_track_remove[5] = np.array([121,122])
    dict_humans_track_remove[1] = np.array([325,328,329,330,331,332,336,337,338,339,340,341,348,349,350,351,352,354])
    dict_humans_track_remove[91] = np.array([520,521,525])
    dict_humans_track_remove[135] = np.array([505,508,509,510,511,523])
    new_track_dict = reorganize_track_info(track_dict, root_dir, humans_ids_to_keep, dict_humans_track_remove)

    # Dicionário id: ([frames]) a remover o track -> preencher com frames que quero remover
    # dict_humans_track_remove = dict()
    # dict_humans_track_remove[2] = np.array([398, 399, 400, 401, 402, 435, 437, 438, 441, 442, 443, 444, 445, 446, 547, 548, 549, 559, 560, 561, 562, 563, 564, 565])
    # dict_humans_track_remove[3] = np.array([488, 489, 490, 491, 492, 563, 574])

    # Reorganizar as keys do dicionário
    # new_track_dict = reorganize_track_info(track_dict, root_dir, humans_ids_to_keep, dict_humans_track_remove=None)    # Guardar novo dicionário organizado e plot de novo tracking
    pickle.dump(new_track_dict, open(f'{root_dir}/mpt.pkl', 'wb'))      

    # Load novo dicionário organizado e plot de novo tracking (implementado em reorganize_track_info())
    # # new_track_dict = pickle.load(open(f'{root_dir}/mpt.pkl', 'rb'))
    # new_track_info_images_path = f'{root_dir}/images_only_with_{humans_ids_to_keep}_humans_IDs'
    # plot_new_track_info(new_track_dict, root_dir, humans_ids_to_keep)

# novo_track_dict[2] = novo_track_dict.pop(4)   # Passar a key 4 para a posição 2    
##################################################################################


# sort_mpt_dict = pickle.load(open('out/glamr_static_multi/basketball/pose_est/track/sort_mpt.pkl', 'rb'))
# mpt_dict = pickle.load(open('out/glamr_static_multi/basketball/pose_est/track/mpt.pkl', 'rb'))


if APPEND_IDS:
    # root_dir = '/home/andre/Documents/Projects/GLAMR/out/glamr_dynamic/Havoc_Ladies_1790_1922/pose_est_hybrik_hrnet48/track'
    track_dict = pickle.load(open(f'{root_dir}/mpt.pkl', 'rb'))
    # pickle.dump(track_dict, open(f'{root_dir}/mpt_1_2_3_4_5.pkl', 'wb')) 
    new_track_dict = append_track_info(track_dict, 3, 4, root_dir=None)     # Informação do 4 guardadas no 3
    new_track_dict = append_track_info(new_track_dict, 1, 5, root_dir=root_dir) # Informação do 5 guardadas no 1
    pickle.dump(new_track_dict, open(f'{root_dir}/mpt_1,2,3.pkl', 'wb'))
    pickle.dump(new_track_dict, open(f'{root_dir}/mpt.pkl', 'wb'))
print("We are done!!")
