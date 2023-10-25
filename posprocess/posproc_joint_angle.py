import os
import pickle
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
from utils import quaternion_to_angle_axis
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from utils import joints_name_29
from tqdm import tqdm
# from scipy.ndimage.filters import gaussian_filter1d

# root_dir = "/home/andre/Documents/Projects/GLAMR/out/glamr_dynamic/Havoc_Ladies_1790_1922"
# root_dir = "/home/andre/Documents/Projects/GLAMR/out/glamr_dynamic/courtyard_basketball_01_0_200_cut_images_516x1350"
root_dir = "/home/andre/Documents/Projects/GLAMR/out/glamr_dynamic/downtown_runForBus_01_425_560"
# pose_est_dir="out/glamr_static/basketball/pose_est"
pose_est_dir = f"{root_dir}/pose_est_hybrik_hrnet48"
grecon_dir = f"{root_dir}/grecon_from_hrnet48" 
save_path = f"{root_dir}/angular_variation_of_joints"
os.makedirs(save_path, exist_ok=True)
    
print("Loading dictionaries...")
hybrik_dict_file = f'{pose_est_dir}/pose.pkl'                                  # Poses do Hybrik 
glamr_dict_file = f'{grecon_dir}/downtown_runForBus_01_425_560_GLAMR_seed1.pkl'
hybrik_dict = pickle.load(open(hybrik_dict_file, 'rb'))
glamr_dict = pickle.load(open(glamr_dict_file, 'rb'))
human_id = 1

joint_index = 4         # [:, 1:] remove a articulação 0, logo este valor não pode ser inferior a 1!!!
if joint_index<1:
    print(" -> Joint index must be between [1,23]!")
    sys.exit()

smpl_pose_wroot = quaternion_to_angle_axis(torch.tensor(hybrik_dict[human_id][f'smpl_pose_quat_wroot'], device='cuda')).cpu().numpy()
hybrik_angles_3R = smpl_pose_wroot[:, 1:]                               # Remove root joint das matrizes com angulos das juntas do modelo SMPL (inicialmente (frames,24,3, após [:, 1:]) fica (frames,23,3), logo, smpl_pose_wroot[0][1] = smpl_pose_wroot[:, 1:][0][0]
glamr_angles_3R = glamr_dict['person_data'][human_id]['smpl_pose'].reshape(-1,23,3)     # Rechape a (frames, 23, 3)

frames = range(glamr_dict['person_data'][human_id]['visible'].shape[0])

hybrik_angles_3R_x = np.zeros(max(frames)+1, dtype = np.float32)
glamr_angles_3R_x = glamr_angles_3R[:, joint_index-1, 0]
# hybrik_angles_3R_x = hybrik_angles_3R[:, joint_index, 0]

# Preenchimento do dicionário hybrik_dict[human_id]['frame2ind'] com valores np.nan na falha da deteção
frame2ind_original = hybrik_dict[human_id]['frame2ind'] 
frame2ind_filled = {}
for frame in frames:  # Itera pelos frames até o último frame com identificador
    if frame in frame2ind_original:
        frame2ind_filled[frame] = frame2ind_original[frame]  # Copia o valor existente
    else:
        frame2ind_filled[frame] = np.nan  # Define como np.nan
hybrik_dict[human_id]['frame2ind'] = frame2ind_filled

# Preenchimento do hybrik_angles_3R_x comnp.nan quando tiver sido detetada um frame inferido (falha na deteção do HybrIK)
i=0
for i in frames:
    id = hybrik_dict[human_id]['frame2ind'][i]
    if glamr_dict['person_data'][human_id]['invis_frames'][i] == True:
        hybrik_angles_3R_x[i]=np.nan
    else:
        hybrik_angles_3R_x[i]=hybrik_angles_3R[id, joint_index-1, 0]

# Interpolação para falha de frames no HybrIK
nan_array = np.full(len(hybrik_angles_3R_x), np.nan)    # vai conter valores de interpolação
interpolated_hybrik_angles_3R_x = hybrik_angles_3R_x.copy()
nan_true_false = np.isnan(hybrik_angles_3R_x)
nan_indices = np.where(nan_true_false)[0]

if np.any(nan_indices):
    print(f"Starting interpolation in frames {nan_indices}") 

for i, nan_index in enumerate(tqdm(nan_true_false)):
    if nan_index:
        next_value_index = i + 1
        while next_value_index < len(hybrik_angles_3R_x) and np.isnan(hybrik_angles_3R_x[next_value_index]):
            next_value_index += 1
        if next_value_index < len(hybrik_angles_3R_x):
            interpolated_hybrik_angles_3R_x[i] = np.interp(i, [i - 1, next_value_index], [interpolated_hybrik_angles_3R_x[i - 1], hybrik_angles_3R_x[next_value_index]])
            nan_array[i] = interpolated_hybrik_angles_3R_x[i]
            nan_array[i-1] = interpolated_hybrik_angles_3R_x[i-1]
            nan_array[i+1] = interpolated_hybrik_angles_3R_x[i+1]

# Suavização dos valores dos angulos
window_size_glamr = 2
# glamr_smoothed_angles = np.convolve(glamr_angles_3R_x, np.ones(window_size_glamr) / window_size_glamr, mode='same')
glamr_smoothed_angles = uniform_filter1d(glamr_angles_3R_x, window_size_glamr)
# glamr_smoothed_angles2 = gaussian_filter1d(glamr_angles_3R_x, sigma=2)
# glamr_smoothed_angles=glamr_angles_3R_x

window_size_hybrik = 2
# hybrik_smoothed_angles = uniform_filter1d(hybrik_angles_3R_x, window_size_hybrik)
# hybrik_smoothed_angles = np.convolve(hybrik_angles_3R_x, np.ones(window_size_hybrik) / window_size_hybrik, mode='same')
# nan_array_smoothed_angles = np.convolve(nan_array, np.ones(window_size_hybrik) / window_size_hybrik, mode='same')

plt.plot(frames, glamr_smoothed_angles, '-', label='GLAMR')  # Plota o gráfico dos ângulos da articulação 4 do primeiro dicionário
plt.plot(frames, hybrik_angles_3R_x, '-', label='HybrIK', color="#ff7f0e")  # Plota o gráfico dos ângulos da articulação
plt.plot(frames, nan_array, 'o-', label='Linear Interp.', color='red')

plt.xlabel('Frames')  # Define o rótulo do eixo x
plt.ylabel('Joint Angle (x-axis)')  # Define o rótulo do eixo y
plt.title(f'Articulation {joint_index} ({joints_name_29[joint_index]}): Angle over Frames')  # Define o título do gráfico
plt.legend()  # Adiciona a legenda para diferenciar as linhas de dados

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.pause(0.5)

image_path = f"{save_path}/Articulation {joint_index} ({joints_name_29[joint_index]})_human{human_id}.png"
print(f"Image saved in directory:\n{image_path}")
plt.savefig(image_path)

plt.show()  # Mostra o gráfico




print("We are done!")
