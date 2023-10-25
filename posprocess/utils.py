from tqdm import tqdm
import time	
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import os.path as osp
import subprocess
import enum
import warnings


joints_name_29 = (
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb',            # 23
        'head', 'left_middle', 'right_middle',  # 26
        'left_bigtoe', 'right_bigtoe'           # 28
    )

# Pares de índices representando as conexões entre as articulações
bones_jts_29 = np.array([               
    [0, 1],   	# pelvis -> left_hip        # Cor 2.1 (azul)
    [1, 4],	    # left_hip -> left_knee
    [4, 7],	    # left_knee -> left_ankle
    [7, 10],	# left_ankle -> left_foot
    # [10, 27],	# left_foot -> left_bigtoe
    [9, 13],	# spine3 -> left_collar     # Cor 2.2 (purple)
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
    [9, 14],	# spine3 -> right_collar    # Cor 3.2 (orange)
    [14, 17],	# right_collar -> right_shoulder
    [17, 19],	# right_shoulder -> right_elbow
    [19, 21],	# right_elbow -> right_wrist
    [21, 23],	# right_wrist -> right_thumb 
    # [23, 26],	# right_thumb -> right_middle    
    [0, 3],	    # pelvis -> spine1              # Cor 1 (verde) [self.JOINT_NAMES.index('pelvis'),self.JOINT_NAMES.index('spine1')
    [3, 6],	    # spine1 -> spine2
    [6, 9],   	# spine2 -> spine3
    [9, 12],	# spine3 -> neck               # Cor 4 (amarelo)
    [12, 15],	# neck -> jaw
    # [15, 24],	# jaw -> head    
])

colors = ['blue'] * 4 + ['purple'] * 5 + ['red'] * 4 + ['orange'] * 5 + ['yellow'] * 3 + ['green'] * 2


def drawGaussian_multi(img_path, frame_id, hybrik_dict, sigma, amplitude, fig, calc_mode='uni', mode='mean', time=2, one_channel= True, save_path=None):
    ''' Plot Gaussian Heatmaps

        Parameters
        ----------
        img_path : inicial image (3-Ch)
        pts : torch.tensor Jx3
            The pose skeleton in (X, Y) format
        heatmaps : torch.tensor 1xJx64x64x64
            Heatmaps of all joints
        sigma :
            Sigma value of gaussian distribution (only if calc_mode='gauss') 
        amplitude
            Amplitude value of gaussian distribution (only if calc_mode='gauss') 
        fig :
            Initial figure enumeration 
        calc_mode
            'normal' -> intensities of heatmaps used directly (heatmaps saved individually)
            'uni' -> intensities of heatmaps used directly (heatmaps saved in one image)
            'gauss' -> Intensities subject to a new Gaussian distribution as a function of position in the final image (heatmaps saved in one image)
        mode :
            'mean -> Heatmaps reduced to two dimensions (64x64) by mean value along z (first dimension)
            'max' -> Heatmaps reduced to two dimensions (64x64) by the maximum value along z (first dimension)
        time :
            Time between images (only if calc_mode='uni') 
        one_channel :
            Enable plot one channel mode 
        save_path :
            Path to save images, optional if desired
    '''
    img = cv2.imread(img_path)
    img_num = os.path.splitext(os.path.basename(img_path))[0]
    save_path = f'{save_path}/frame_{img_num}'
    os.makedirs(save_path, exist_ok=True)

    if save_path is not None:
        if calc_mode == 'uni':
            uni_path = f'{save_path}/uni_joints'
            os.makedirs(uni_path, exist_ok=True)
        elif calc_mode == 'normal' or calc_mode == 'gauss':
            normal_gauss_path = f'{save_path}/3-Channel_all_joints'
            os.makedirs(normal_gauss_path, exist_ok=True)
        if one_channel:
            one_channel_path = f'{save_path}/1-Channel_all-joints'
            os.makedirs(one_channel_path, exist_ok=True)

    num_joints = hybrik_dict[0]['heatmaps'].shape[2]
    amplitude_init = 1          # só gauss calc_mode
    time_offset=0.3

    height = img.shape[0]
    width = img.shape[1]
    sub_width = 64      # self.height_dim, self.width_dim
    sub_height = 64

    # Inicializar heatmaps_person_dict que deverá conter as imagens de heatmaps_plot_all_joints (dicionário será tanto maior quando mais pessoas houver envolvidas no cenário)
    # if one_channel or calc_mode == 'gauss' or calc_mode == 'normal':
    #     heatmaps_person_dict = {}
    #     for i in range(max(hybrik_dict.keys()) + 1):
    #         heatmaps_person_dict[i] = []  
    heatmaps_plot_all_joints_all_humans = np.zeros((height, width)) # Imagem com todos os heatmaps de todas as pessoas (inicializado uma única vez)

    for human_id in sorted(hybrik_dict.keys()):     # Ciclo por pessoas
        heatmaps_plot_all_joints = np.zeros((height, width))        # Imagem com todos os heatmaps de uma só pessoa (inicializado por pessoa)
        heatmaps_plot_uni = np.zeros((height, width))               # Imagem com 1 heatmap por pessoa (inicializado por articulação e por pessoa)
        if calc_mode=='uni':
            print(f"Starting plot (human {human_id}) heatmaps with delay of {time} seconds between images...")

        for idx_joint in tqdm(range(num_joints)):  # Ciclo por joints
        # for idx_joint in range(num_joints):
            # Extrair o heatmap 3D para a articulação escolhida
            # heatmap = heatmaps[0, idx_joint, :, :, :]       # torch.Size([64, 64, 64]) -> self.depth_dim, self.height_dim, self.width_dim
            heatmap = hybrik_dict[human_id]['heatmaps'][frame_id][0][idx_joint]
            
            # Calcular os valores máximos/médios do heatmap para cada articulação mantendo a dimensão 64x64
            if mode == 'max':
                heatmap = torch.max(heatmap, dim=0)[0]      # torch.Size([64, 64])  # máximo ao longo de z (primeira dimensão)
            elif mode == 'mean':
                heatmap = torch.mean(heatmap, dim=0)        # torch.Size([64, 64])

            # Converter o tensor do heatmap para um array NumPy
            heatmap = heatmap.cpu().numpy()


            # Normalizar    # torch.Size([64, 64])
            heatmap_normalized = heatmap * amplitude / np.max(heatmap)      # valores entre 0 e 1
            # heatmap_normalized = heatmap
            # indice_maximo = np.unravel_index(np.argmax(heatmap_normalized), heatmap_normalized.shape)       # np.where(heatmap == heatmap.max())
            # indice_min = np.unravel_index(np.argmin(heatmap_normalized), heatmap_normalized.shape)          # np.where(heatmap == heatmap.min())

            joint_x = to_numpy(hybrik_dict[human_id]['kp_2d'][frame_id])[idx_joint][0]
            joint_y = to_numpy(hybrik_dict[human_id]['kp_2d'][frame_id])[idx_joint][1]

            start_x = max(0, joint_x - sub_width // 2)
            end_x = min(width, joint_x + sub_width // 2 + 1)
            start_y = max(0, joint_y - sub_height // 2)
            end_y = min(height, joint_y + sub_height // 2 + 1)

            i=j=0
            if calc_mode == 'uni':                              # A cada articulação
                heatmaps_plot_uni = np.zeros((height, width))
                
            for x in range(int(start_x), int(end_x)-1):         # A cada articulação
                j=0
                for y in range(int(start_y), int(end_y)-1):
                    if calc_mode == 'gauss':
                        dist_x = x - joint_x
                        dist_y = y - joint_y
                        amplitude = amplitude_init * heatmap_normalized[j, i] / (sigma * np.sqrt(2 * np.pi))
                        exponent = -((dist_x) ** 2 + (dist_y) ** 2) / (2 * sigma ** 2)
                        heatmaps_plot_all_joints[y, x] = heatmaps_plot_all_joints[y, x]+ amplitude * np.exp(exponent)
                        heatmaps_plot_all_joints_all_humans[y, x] = heatmaps_plot_all_joints_all_humans[y, x] + heatmap_normalized[j, i]
                    elif calc_mode == 'normal':
                        heatmaps_plot_all_joints[y, x] = heatmaps_plot_all_joints[y, x] + heatmap_normalized[j, i]
                        heatmaps_plot_all_joints_all_humans[y, x] = heatmaps_plot_all_joints_all_humans[y, x] + heatmap_normalized[j, i]
                        # heatmaps_plot_all_joints[y, x] = mean_min if heatmaps_plot_all_joints[y, x]<mean_min else heatmaps_plot_all_joints[y, x]
                    elif calc_mode == 'uni':
                        heatmaps_plot_uni[y, x] = heatmap_normalized[j, i]
                        if one_channel:
                            heatmaps_plot_all_joints[y, x] = heatmaps_plot_all_joints[y, x] + heatmap_normalized[j, i]
                            heatmaps_plot_all_joints_all_humans[y, x] = heatmaps_plot_all_joints_all_humans[y, x] + heatmap_normalized[j, i]
                    j+=1
                i+=1        

            # Save and show images 
            if calc_mode == 'uni':  # Criar diretório (dentro de uni_joints) para cada pessoa e dentro as guardar a imagem de cada joint  
                plt.figure(fig)
                plt.title(f"Heatmap Joint {idx_joint} ({joints_name_29[idx_joint]})")
                smoothed_heatmap = cv2.GaussianBlur(heatmaps_plot_uni, (0, 0), sigmaX=12, sigmaY=12)
                # smoothed_heatmap = cv2.resize(smoothed_heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
                heatmapshow = None
                heatmapshow = cv2.normalize(smoothed_heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
                super_imposed_img = cv2.addWeighted(heatmapshow, 0.4, img, 0.55, 0)
                plt.imshow(cv2.cvtColor(super_imposed_img, cv2.COLOR_RGB2BGR))
                temp_path = f'{uni_path}/human-{human_id}'
                os.makedirs(temp_path, exist_ok=True)
                cv2.imwrite(f'{temp_path}/joint_{idx_joint} ({joints_name_29[idx_joint]}).jpg', super_imposed_img)
                time_offset = time-0.001 if (time-time_offset)<=0 else time_offset
                plt.pause(time-time_offset)

        # if one_channel or calc_mode == 'gauss' or calc_mode == 'normal':
        #     heatmaps_person_dict[human_id].append(heatmaps_plot_all_joints[:,:])       # heatmaps_person_dict[0][0] primeira imagem, primeira pessoa 
        if one_channel:
            fig +=1
            smoothed_heatmap = cv2.GaussianBlur(heatmaps_plot_all_joints, (0, 0), sigmaX=4, sigmaY=4)
            # smoothed_heatmap = cv2.resize(smoothed_heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
            heatmapshow = None
            heatmapshow = cv2.normalize(smoothed_heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            plt.figure(fig)
            plt.title(f"Heatmaps Representation 1-Channel (Human {human_id})")
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            plt.pause(0.01)
            plt.imshow(heatmapshow, cmap='hot')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.colorbar()
            plt.pause(0.01)
            plt.show(block=False)
            plt.savefig(f'{one_channel_path}/1-Channel_human-{human_id}.jpg')
            

        # Terminou leitura de todas as articulações
        if (calc_mode == 'normal') or (calc_mode == 'gauss'):    # Guardar (dentro de normal_gauss_path) a imagem de todas as joints de todas as pessoas 
            fig +=1
            # smoothed_heatmap = cv2.GaussianBlur(heatmaps_plot_all_joints, (7, 7), sigmaX=3, sigmaY=0)    # Pós processamento às intensidades dos heatmaps (filtro gaussiano -> suavização)
            smoothed_heatmap = cv2.GaussianBlur(heatmaps_plot_all_joints, (0, 0), sigmaX=7, sigmaY=7)
            # smoothed_heatmap = cv2.resize(smoothed_heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
            heatmapshow = None
            heatmapshow = cv2.normalize(smoothed_heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
            super_imposed_img = cv2.addWeighted(heatmapshow, 0.4, img, 0.55, 0)
            plt.figure(fig)
            plt.title(f"Human Body Joints (Heatmap Representation) 3-Channel")
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            plt.imshow(cv2.cvtColor(super_imposed_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'{normal_gauss_path}/3-Channel_human-{human_id}.jpg', super_imposed_img)
        # heatmaps_person_dict[0][0].shape = (720,1280) -> Para gravar cada pessoa e heatmaps_plot_all_joints_all_humans[y, x] -> com todas os heatmaps de todas as pessoas (gravar como All)
        
        plt.show()

    # Plot all joints all humans
    if (calc_mode == 'normal') or (calc_mode == 'gauss'):    # Guardar (dentro de normal_gauss_path) a imagem de todas as joints de todas as pessoas 
        fig +=1
        # smoothed_heatmap = cv2.GaussianBlur(heatmaps_plot_all_joints_all_humans, (7, 7), sigmaX=3, sigmaY=0)    # Pós processamento às intensidades dos heatmaps (filtro gaussiano -> suavização)
        smoothed_heatmap = cv2.GaussianBlur(heatmaps_plot_all_joints_all_humans, (0, 0), sigmaX=7, sigmaY=7)
        # smoothed_heatmap = cv2.resize(smoothed_heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        heatmapshow = None
        heatmapshow = cv2.normalize(smoothed_heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        super_imposed_img = cv2.addWeighted(heatmapshow, 0.4, img, 0.55, 0)
        plt.figure(fig)
        plt.title(f"Human Body Joints (Heatmap Representation) 3-Channel")
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.imshow(cv2.cvtColor(super_imposed_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'{normal_gauss_path}/3-Channel_all-humans.jpg', super_imposed_img)

    if one_channel:
        fig +=1
        smoothed_heatmap = cv2.GaussianBlur(heatmaps_plot_all_joints_all_humans, (0, 0), sigmaX=4, sigmaY=4)
        # smoothed_heatmap = cv2.resize(smoothed_heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        heatmapshow = None
        heatmapshow = cv2.normalize(smoothed_heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        plt.figure(fig)
        plt.title(f"Heatmaps Representation 1-Channel")
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.imshow(heatmapshow, cmap='hot')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        plt.show(block=False)
        plt.pause(0.01)
        plt.savefig(f'{one_channel_path}/1-Channel_all-humans.jpg')


def drawGaussian(img_path, pts, heatmaps, sigma, amplitude, fig, calc_mode='uni', mode='mean', time=2, one_channel= True, save_path=None):
    ''' Plot Gaussian Heatmaps

        Parameters
        ----------
        img_path : inicial image (3-Ch)
        pts : torch.tensor Jx3
            The pose skeleton in (X, Y) format
        heatmaps : torch.tensor 1xJx64x64x64
            Heatmaps of all joints
        sigma :
            Sigma value of gaussian distribution (only if calc_mode='gauss') 
        amplitude
            Amplitude value of gaussian distribution (only if calc_mode='gauss') 
        fig :
            Initial figure enumeration 
        calc_mode
            'normal' -> intensities of heatmaps used directly (heatmaps saved individually)
            'uni' -> intensities of heatmaps used directly (heatmaps saved in one image)
            'gauss' -> Intensities subject to a new Gaussian distribution as a function of position in the final image (heatmaps saved in one image)
        mode :
            'mean -> Heatmaps reduced to two dimensions (64x64) by mean value along z (first dimension)
            'max' -> Heatmaps reduced to two dimensions (64x64) by the maximum value along z (first dimension)
        time :
            Time between images (only if calc_mode='uni') 
        one_channel :
            Enable plot one channel mode 
        save_path :
            Path to save images, optional if desired
    '''
    img = cv2.imread(img_path)
    img_num = os.path.splitext(os.path.basename(img_path))[0]
    save_path = f'{save_path}/frame_{img_num}'
    os.makedirs(save_path, exist_ok=True)
    if save_path is not None:
        if calc_mode == 'uni':
            uni_path = f'{save_path}/uni_joints'
            os.makedirs(uni_path, exist_ok=True)
        elif calc_mode == 'normal' or calc_mode == 'gauss':
            normal_gauss_path = f'{save_path}/3-Channel_all_joints'
            os.makedirs(normal_gauss_path, exist_ok=True)
        if one_channel:
            one_channel_path = f'{save_path}/1-Channel_all-joints'
            os.makedirs(one_channel_path, exist_ok=True)

    num_joints = heatmaps.shape[1]
    amplitude_init = 1          # só gauss calc_mode
    time_offset=0.3

    height = img.shape[0]
    width = img.shape[1]
    sub_width = 64      # self.height_dim, self.width_dim
    sub_height = 64

    heatmaps_plot_all_joints = np.zeros((height, width))
    heatmaps_plot_uni = np.zeros((height, width))

    # max_value = 0
    # mean_max = 0
    # mean_min = 0
    # min_value = float('inf')
    # value = float('inf')
    # max_array=[]
    # min_array=[]
    # if calc_mode=='uni':
    #     print(f"Starting plot heatmaps with delay of {time} seconds between images...")

    # for idx_joint in range(num_joints):
    #     heatmap = heatmaps[0, idx_joint, :, :, :]
    #     heatmap = torch.mean(heatmap, dim=0)
        # max_array.append(np.max(heatmap.cpu().numpy()))
        # min_array.append(np.min(heatmap.cpu().numpy()))
        # max_value = np.max([max_value, np.max(heatmap.cpu().numpy())])
        # value = np.min(heatmap.cpu().numpy())
        # min_value = value  if ((value < min_value) and value!=0) else min_value
        # print(max_value)
        # print(min_value)
    
    # mean_max = np.mean(max_array)
    # mean_min = np.mean(min_array)
    print("Processing heatmaps...")
    for idx_joint in tqdm(range(num_joints)):
    # for idx_joint in range(num_joints):
        # Extrair o heatmap 3D para a articulação escolhida
        # heatmap = heatmaps[0, idx_joint, :, :, :]       # torch.Size([64, 64, 64]) -> self.depth_dim, self.height_dim, self.width_dim
        heatmap = heatmaps[0,idx_joint, :, :, :]
        
        # Calcular os valores máximos/médios do heatmap para cada articulação mantendo a dimensão 64x64
        if mode == 'max':
            heatmap = torch.max(heatmap, dim=0)[0]      # torch.Size([64, 64])  # máximo ao longo de z (primeira dimensão)
        elif mode == 'mean':
            heatmap = torch.mean(heatmap, dim=0)        # torch.Size([64, 64])

        # Converter o tensor do heatmap para um array NumPy
        heatmap = heatmap.cpu().numpy()


        # Normalizar    # torch.Size([64, 64])
        heatmap_normalized = heatmap * amplitude / np.max(heatmap)      # valores entre 0 e 1
        # heatmap_normalized = heatmap
        # indice_maximo = np.unravel_index(np.argmax(heatmap_normalized), heatmap_normalized.shape)       # np.where(heatmap == heatmap.max())
        # indice_min = np.unravel_index(np.argmin(heatmap_normalized), heatmap_normalized.shape)          # np.where(heatmap == heatmap.min())

        joint_x = to_numpy(pts)[idx_joint][0]
        joint_y = to_numpy(pts)[idx_joint][1]

        start_x = max(0, joint_x - sub_width // 2)
        end_x = min(width, joint_x + sub_width // 2 + 1)
        start_y = max(0, joint_y - sub_height // 2)
        end_y = min(height, joint_y + sub_height // 2 + 1)

        i=j=0
        if calc_mode == 'uni':
            heatmaps_plot_uni = np.zeros((height, width))
        for x in range(int(start_x), int(end_x)-1):
            j=0
            for y in range(int(start_y), int(end_y)-1):
                if calc_mode == 'gauss':
                    dist_x = x - joint_x
                    dist_y = y - joint_y
                    amplitude = amplitude_init * heatmap_normalized[j, i] / (sigma * np.sqrt(2 * np.pi))
                    exponent = -((dist_x) ** 2 + (dist_y) ** 2) / (2 * sigma ** 2)
                    heatmaps_plot_all_joints[y, x] = heatmaps_plot_all_joints[y, x]+ amplitude * np.exp(exponent)
                elif calc_mode == 'normal':
                    heatmaps_plot_all_joints[y, x] = heatmaps_plot_all_joints[y, x] + heatmap_normalized[j, i]
                    # heatmaps_plot_all_joints[y, x] = mean_min if heatmaps_plot_all_joints[y, x]<mean_min else heatmaps_plot_all_joints[y, x]
                elif calc_mode == 'uni':
                    heatmaps_plot_uni[y, x] = heatmap_normalized[j, i]
                    heatmaps_plot_all_joints[y, x] = heatmaps_plot_all_joints[y, x] + heatmap_normalized[j, i]
                j+=1
            i+=1
        if calc_mode == 'uni':  # Criar diretório (dentro de uni_joints) para cada pessoa e dentro as guardar a imagem de cada joint  
            plt.figure(fig)
            plt.title(f"Heatmap Joint {idx_joint}")
            smoothed_heatmap = cv2.GaussianBlur(heatmaps_plot_uni, (0, 0), sigmaX=12, sigmaY=12)
            # smoothed_heatmap = cv2.resize(smoothed_heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
            heatmapshow = None
            heatmapshow = cv2.normalize(smoothed_heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
            super_imposed_img = cv2.addWeighted(heatmapshow, 0.4, img, 0.55, 0)
            plt.imshow(cv2.cvtColor(super_imposed_img, cv2.COLOR_RGB2BGR))
            temp_path = f'{uni_path}/human-0'
            os.makedirs(temp_path, exist_ok=True)
            cv2.imwrite(f'{temp_path}/joint_{idx_joint} ({joints_name_29[idx_joint]}).jpg', super_imposed_img)
            plt.pause(time-time_offset)

    # Terminou leitura de todas as articulações
      
    if (calc_mode == 'normal') or (calc_mode == 'gauss'):    # Guardar (dentro de normal_gauss_path) a imagem de todas as joints de todas as pessoas 
        # smoothed_heatmap = cv2.GaussianBlur(heatmaps_plot_all_joints, (7, 7), sigmaX=3, sigmaY=0)    # Pós processamento às intensidades dos heatmaps (filtro gaussiano -> suavização)
        smoothed_heatmap = cv2.GaussianBlur(heatmaps_plot_all_joints, (0, 0), sigmaX=13, sigmaY=13)
        # smoothed_heatmap = cv2.resize(smoothed_heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        heatmapshow = None
        heatmapshow = cv2.normalize(smoothed_heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        super_imposed_img = cv2.addWeighted(heatmapshow, 0.55, img, 0.6, 0)
        plt.figure(fig)
        plt.title(f"Human Body Joints (Heatmap Representation) 3-Channel")
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.imshow(cv2.cvtColor(super_imposed_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'{normal_gauss_path}/3-Channel_human-0.jpg', super_imposed_img)
            
    if one_channel:
        fig +=1
        smoothed_heatmap = cv2.GaussianBlur(heatmaps_plot_all_joints, (0, 0), sigmaX=10, sigmaY=10)
        # smoothed_heatmap = cv2.resize(smoothed_heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        heatmapshow = None
        heatmapshow = cv2.normalize(smoothed_heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        plt.figure(fig)
        plt.title(f"Heatmaps Representation 1-Channel")
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.imshow(heatmapshow, cmap='hot')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar()
        plt.savefig(f'{one_channel_path}/1-Channel_human-0.jpg')
    plt.show()


def images_to_video(img_dir, out_path, img_fmt="%06d.jpg", fps=30, crf=25, verbose=True):

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    FFMPEG_PATH = '/usr/bin/ffmpeg' if osp.exists('/usr/bin/ffmpeg') else 'ffmpeg'
    cmd = [FFMPEG_PATH, '-y', '-r', f'{fps}', '-f', 'image2', '-start_number', '0',
            '-i', f'{img_dir}/{img_fmt}', '-vcodec', 'libx264', '-crf', f'{crf}', '-pix_fmt', 'yuv420p', out_path]
    if not verbose:
        cmd += ['-hide_banner', '-loglevel', 'error']
    subprocess.run(cmd)


def resize_images(img_dir, output_dir, target_height):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(img_dir):
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        ratio = target_height / height
        new_height = target_height
        new_width = int(width * ratio)
        resized_img = cv2.resize(img, (new_width, new_height))
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, resized_img) 

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def draw_2D_heatmaps(heatmaps,human_id=0, mode='mean', save_path=None):
        
        Amplitude = 1
        # Número de subplots = articulações
        num_subplots = heatmaps.shape[1]
        # num_subplots=3

        # Verificar se há pelo menos uma articulação para plotar
        if num_subplots > 0: 
            # Calcular o número de linhas e colunas para a grade de subplots
            num_lines = (num_subplots - 1) // 5 + 1
            num_columns = min(num_subplots, 5)

            # Criar a figura com a grade de subplots
            # plt.figure(1)
            fig, axs = plt.subplots(num_lines, num_columns, figsize=(15, 10), dpi=75)
            fig.tight_layout()  # Garantir espaçamento igual entre gráficos

            # Loop sobre as articulações até o número mínimo entre o número de subplots e o número de articulações
            print(f"Plotting heatmaps (human {human_id})...\n")
            for idx_joint in tqdm(range(num_subplots)):
                # Extrair o heatmap 3D para a articulação escolhida
                heatmap = heatmaps[0, idx_joint, :, :, :]
                
                # Calcular os valores máximos/médios do heatmap para cada articulação mantendo a dimensão 64x64
                if mode == 'max':
                    heatmap = torch.max(heatmap, dim=0)[0]
                elif mode == 'mean':
                    heatmap = torch.mean(heatmap, dim=0)

                # Converter o tensor do heatmap para um array NumPy
                heatmap = heatmap.cpu().numpy()

                # Normalizar
                heatmaps_normalized = heatmap * Amplitude / np.max(heatmap)

                # Calcular o índice da linha e da coluna do subplot atual
                linha = idx_joint // num_columns
                coluna = idx_joint % num_columns

                # Plot do heatmap 2D no subplot correspondente
                ax = axs[linha, coluna]
                im = ax.imshow(heatmaps_normalized, cmap='hot')
                ax.set_title(f"Joint {idx_joint} ({joints_name_29[idx_joint]})")
                plt.colorbar(im, ax=ax)  # Adicionar a barra de cores
                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
                # mng.window.state('zoomed')
            # plt.ion()
            plt.show(block=False)
            plt.pause(0.01)

            # Remover subplots vazios, se houver
            if num_subplots < num_lines * num_columns:
                for idx_joint in range(num_subplots, num_lines * num_columns):
                    linha = idx_joint // num_columns
                    coluna = idx_joint % num_columns
                    fig.delaxes(axs[linha, coluna])
        else:
            print("Error! There are no joints to plot!.\n")  
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            plt.pause(0.01)
            plt.savefig(f'{save_path}/human-{human_id}.jpg')
        # Exibir a figura
        


def to_torch(ndarray):
    # numpy.ndarray => torch.Tensor
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def to_numpy(tensor):
    # torch.Tensor => numpy.ndarray
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def draw_3D_skeleton_multi(hybrik_dict, init_frame=0, range_frames=50, save_path=None, show_image=False):       # camera coordinates (atualmente em pixel coordinates da imagem)
        
    if hybrik_dict[0]['kp_3d'].shape[1] < 29:
        print(f"Only {hybrik_dict[0]['kp_3d'].shape[1]} joints were predicted")
    if init_frame>2:
        flag_first_plot = True

    # Determinar nº de frames máximo
    end_frames = init_frame + range_frames
    max_frames=0
    for human_id in sorted(hybrik_dict.keys()):
        max_frames = max(hybrik_dict[1]['frames']) if max_frames < max(hybrik_dict[1]['frames']) else max_frames 
    
    end_frames = max_frames+1 if (init_frame + range_frames)>(max_frames+1) else (init_frame + range_frames)

    plt.clf()
    for frame_idx in tqdm(range(init_frame,end_frames)):   # 0 299
        res_path = os.path.join(save_path, f'{frame_idx:06d}.jpg')

        # clear figure
        ax = plt.axes(projection='3d')	
        for human_id in sorted(hybrik_dict.keys()):

            # if frame_idx in hybrik_dict[human_id]['frame2ind']: 
            if frame_idx in hybrik_dict[human_id]['frames']:
                pose_3D_idx = hybrik_dict[human_id]['frame2ind'][frame_idx]
                extend_pose_skeleton = hybrik_dict[human_id]['kp_3d'][pose_3D_idx]

                extend_pose_skeleton = extend_pose_skeleton.cpu().numpy()
                # extend kinematic tree
                if extend_pose_skeleton.shape[0] > 28:  #self.num_joints-1:
                    extend_pose_skeleton[-5:] = extend_pose_skeleton[0]     # Anular as articulações adicionais (coloca-las na posição da root joint)

                # Coordenadas das articulações
                x = [coord[0] for coord in extend_pose_skeleton]
                y = [coord[1] for coord in extend_pose_skeleton]
                z = [coord[2] for coord in extend_pose_skeleton]

                
                ax.scatter3D(x, y, z, marker='o', color='brown',s=25)
                for i, bone in enumerate(bones_jts_29):
                    x_bone = [x[bone[0]], x[bone[1]]]
                    y_bone = [y[bone[0]], y[bone[1]]]
                    z_bone = [z[bone[0]], z[bone[1]]]
                    color = colors[i % len(colors)]
                    ax.plot(x_bone, y_bone, z_bone, color=color,linewidth=3)
                ax.text(x[15]+20, y[15]-20, z[15], f'{human_id}', fontsize=14, fontweight='bold')   
                
            else:
                pass
                # print(f"Human {human_id} poses are missing in frame {frame_idx}.")    # Fica com os 3D kp anteriores, mas não importa porque neste else não vou dar plot de nada

        ax.view_init(elev=255,azim=90,roll=180,vertical_axis='z')
        # plt.pause(0.05)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel("$X-axis~(pixel~space)$")
        ax.set_ylabel("$Y-axis~(pixel~space)$")
        ax.set_zlabel("$Z-axis~(depth)$")
        if frame_idx<3 or flag_first_plot:          # Se for precio refazer este bloco de código com análise do máximo e mínimo de cada eixo, guardar iterativamente e atualizar com esse valor (cenário tende a aumentar em todos os sentidos naturalmente)
            x_lim = (ax.get_xlim()[0]*0.9,ax.get_xlim()[1]*1.05)
            y_lim = (ax.get_ylim()[0]*0.5,ax.get_ylim()[1]*0.99)
            z_lim = (ax.get_zlim()[0]*1,ax.get_zlim()[1]*1)
            flag_first_plot = False
        ax.set_xlim(x_lim[0]*0.5,x_lim[1]*1.2)
        ax.set_ylim(y_lim[0]*1.5,y_lim[1])
        ax.set_zlim(z_lim[0],z_lim[1])
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.pause(0.01)
        # plt.axis('off')  # Desabilitar os eixos
        plt.title('3D Human Pose - HybrIK')
        # plt.show()
        plt.pause(0.01)
        # if save_path is not None and init_frame>2:
        if save_path is not None:
            plt.savefig(res_path)
        if not show_image:
            plt.close()  


def draw_3D_skeleton(extend_pose_skeleton, show_extend_pose_skeleton=False, save_path=None, show_image=False):       # camera coordinates (atualmente em pixel coordinates da imagem)
        
        if extend_pose_skeleton.shape[0] < 29:
            print(f"Only {extend_pose_skeleton.shape[0]} joints were predicted")
        # if not show_extend_pose_skeleton:
        #     if extend_pose_skeleton.shape[0] > (24):  #self.num_joints:
        #         pose_skeleton = extend_pose_skeleton[:24]
        # else:
        #     pose_skeleton = extend_pose_skeleton

        extend_pose_skeleton = extend_pose_skeleton.cpu().numpy()

        # extend kinematic tree
        if extend_pose_skeleton.shape[0] > 28 and not show_extend_pose_skeleton:  #self.num_joints-1:
            extend_pose_skeleton[-5:] = extend_pose_skeleton[0]     # Anular as articulações adicionais (coloca-las na posição da root joint)
        
        x = [coord[0] for coord in extend_pose_skeleton]
        y = [coord[1] for coord in extend_pose_skeleton]
        z = [coord[2] for coord in extend_pose_skeleton]

        plt.clf()	# clear figure
        ax = plt.axes(projection='3d')
        ax.scatter3D(x, y, z, marker='o', color='brown',s=25)
        for i, bone in enumerate(bones_jts_29):
            x_bone = [x[bone[0]], x[bone[1]]]
            y_bone = [y[bone[0]], y[bone[1]]]
            z_bone = [z[bone[0]], z[bone[1]]]
            color = colors[i % len(colors)]
            ax.plot(x_bone, y_bone, z_bone, color=color,linewidth=3)
        ax.view_init(elev=-90,azim=90,roll=180,vertical_axis='z')
        # plt.pause(0.05)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel("$X-axis~(pixel~space)$")
        ax.set_ylabel("$Y-axis~(pixel~space)$")
        ax.set_zlabel("$Z-axis~(depth)$")
        ax.set_xlim(ax.get_xlim()[0]*0.95,ax.get_xlim()[1]*1.05)
        ax.set_ylim(ax.get_ylim()[0]*0.95,ax.get_ylim()[1]*1)
        ax.set_zlim(ax.get_zlim()[0]*0.9,ax.get_zlim()[1]*1)
        # plt.axis('off')  # Desabilitar os eixos
        plt.title('3D Human Pose - HybrIK')
        # mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        # plt.show()
        plt.pause(0.02)
        if save_path is not None:
            plt.savefig(save_path)
        if not show_image:
            plt.close()


def torch_safe_atan2(y, x, eps: float = 1e-6):
    y = y.clone()
    y[(y.abs() < eps) & (x.abs() < eps)] += eps
    return torch.atan2(y, x)

def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    denominator = denominator.clone()
    denominator[denominator.abs() < eps] += eps
    return numerator / denominator

class QuaternionCoeffOrder(enum.Enum):
    XYZW = 'xyzw'
    WXYZ = 'wxyz'

def quaternion_to_angle_axis(
    quaternion: torch.Tensor, eps: float = 1.0e-6, order: QuaternionCoeffOrder = QuaternionCoeffOrder.WXYZ
) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    The quaternion should be in (x, y, z, w) or (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion: tensor with quaternions.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3
    """

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape Nx4 or 4. Got {quaternion.shape}")

    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(f"order must be one of {QuaternionCoeffOrder.__members__.keys()}")

    if order == QuaternionCoeffOrder.XYZW:
        warnings.warn(
            "`XYZW` quaternion coefficient order is deprecated and"
            " will be removed after > 0.6. "
            "Please use `QuaternionCoeffOrder.WXYZ` instead."
        )
    # unpack input and compute conversion
    q1: torch.Tensor = torch.tensor([])
    q2: torch.Tensor = torch.tensor([])
    q3: torch.Tensor = torch.tensor([])
    cos_theta: torch.Tensor = torch.tensor([])

    if order == QuaternionCoeffOrder.XYZW:
        q1 = quaternion[..., 0]
        q2 = quaternion[..., 1]
        q3 = quaternion[..., 2]
        cos_theta = quaternion[..., 3]
    else:
        cos_theta = quaternion[..., 0]
        q1 = quaternion[..., 1]
        q2 = quaternion[..., 2]
        q3 = quaternion[..., 3]

    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt((sin_squared_theta).clamp_min(eps))
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch_safe_atan2(-sin_theta, -cos_theta), torch_safe_atan2(sin_theta, cos_theta)
    )

    k_pos: torch.Tensor = safe_zero_division(two_theta, sin_theta, eps)
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis