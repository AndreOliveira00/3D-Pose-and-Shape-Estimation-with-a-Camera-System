from tqdm import tqdm
import numpy as np
import os
import pickle
import cv2

# root_dir = 'out/glamr_static/basketball/pose_est/track' 
# file = "frames.txt"

def find_folder_with_condition(root_dir):

    ''' Procurar diretório (exp) pretendido que verifique as seguintes condições:\n
            -> Número de ficheiros .txt = número de imagens em '/frames'+1 

        Parameters
        ----------
        root_dir: path para o diretório onde contem a informação do tracking '.../track'

        Returns
        -------
        folder_path ou None: None se não encontrar o diretório que verifique a condição
    '''

    for i in range(1,1000):  
        if i == 1:
            folder_name = "exp/labels"
        else:
            folder_name = f"exp{i}/labels"
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        file_count = len([filename for filename in os.listdir(folder_path) if filename.endswith('.txt')])
        frames_dir = f'{os.path.dirname(root_dir)}/frames'
        if file_count == len([filename for filename in os.listdir(frames_dir) if filename.endswith('.jpg')])+1:
            return folder_path
    return None


def convert_track_info(root_dir,file):    

    ''' Processamento do file (frames.txt) que contem a informação do tracking das pessoas identificadas nas imagens

        Parameters
        ----------
        root_dir: path para o diretório onde contem a informação do tracking '.../track'
        file: Ficheiro .txt onde normalmente é armazenada a informação do tracking (.../track/exp/labels/frames.txt)

        Returns
        -------
        track_dict: dicionário com os IDs e respetiva informação associada ('bbox' e 'frames') detetados no MOT, sem qualquer filtragem 
    '''

    selected_folder = find_folder_with_condition(root_dir)
    if selected_folder is not None:
        print(f"Folder with condition found: {selected_folder}")
    else:
        print("No folder with condition found!")

    track_dict = {}
    
    file = os.path.join(selected_folder, file)
    with open(file, "r") as file:
        lines = file.readlines()

    for line in tqdm(lines):
        data = line.strip().split()
        frame_idx = int(data[0])
        person_id = int(data[1])
        x, y, width, height = map(int, data[2:6])
        
        cx = x + width / 2
        cy = y + height / 2
        
        bbox = np.array([cx, cy, width, height])

        if person_id not in track_dict:
            track_dict[person_id] = {"bbox": np.empty((0, 4)), "frames": np.empty((0,), dtype=int)}
        
        track_dict[person_id]["bbox"] = np.vstack((track_dict[person_id]["bbox"], bbox))
        track_dict[person_id]["frames"] = np.append(track_dict[person_id]["frames"], frame_idx-1)
    return track_dict

def remove_humans_track_from_dict (track_dict, dict_humans_track_remove):
    ''' Processamento do track_dict que contem a informação do tracking das pessoas identificadas nas imagens
        Remove em track_dict a informação disponibilizada em dict_humans_track_remove

        Parameters
        ----------
        dict_humans_track_remove: Dicionário com ids e frames a remover o track -> id: lista de frames
        file: Ficheiro .txt onde normalmente é armazenada a informação do tracking (.../track/exp/labels/frames.txt)

        Returns
        -------
        track_dict: dicionário com os IDs e respetiva informação associada ('bbox' e 'frames') detetados no MOT, com primeira filtragem 
    '''
    # offset = 1 # se frames começarem em image_00000.jpg caso contrário é 0
    for key, arr in dict_humans_track_remove.items():
        dict_humans_track_remove[key] = np.subtract(arr, 1)

    for human_id, frames_to_remove in dict_humans_track_remove.items():
        if human_id in track_dict:
            for frame in frames_to_remove:
                if frame in track_dict[human_id]['frames']:
                    index = np.where(track_dict[human_id]['frames']==frame)[0][0]
                    track_dict[human_id]['frames'] = np.delete(track_dict[human_id]['frames'],index)
                    track_dict[human_id]['bbox'] = np.delete(track_dict[human_id]['bbox'], index, axis=0)

    return track_dict

def reorganize_track_info(track_dict, root_dir, humans_ids_to_keep, dict_humans_track_remove=None):
    
    ''' Reorganiza o dicionário determinado na fase posterior ao convert_track_info e traking de pessoas (yolov8x + ocsort)\n
        Plot dos frames iniciais com novos ids, devidamente organizados

        Parameters
        ----------
        track_dict: dicionário original tipicamente guardado como mpt.pkl
        root_dir: path para o diretório onde contem a informação do tracking '.../track'
        humans_ids_to_keep: Lista com a informação dos IDs a manter (construida manualmente após análise das imagens geradas no processo de tracking) 
        dict_humans_track_remove: Dicionário com ids e frames a remover o track -> id: lista de frames 

        Returns
        -------
        new_track_dict: dicionário com os IDs pretendidos, devidamente organizado (1,...)
    '''

    pickle.dump(track_dict, open(f'{root_dir}/mpt_original.pkl', 'wb'))

    # Criar um novo dicionário excluindo as human_ids indesejadas
    track_dict_intermediate = {human_id: value for human_id, value in track_dict.items() if human_id in humans_ids_to_keep}

    if dict_humans_track_remove is not None:
        track_dict_intermediate = remove_humans_track_from_dict (track_dict_intermediate, dict_humans_track_remove)

    new_track_dict = {}
    for i, (human_id, value) in enumerate(track_dict_intermediate.items(), start=1):
        new_track_dict[i] = value

    # for key, value in track_dict.items():
    #     if key not in humans_ids_to_remove:
    #         new_track_dict[key] = value

    plot_new_track_info(new_track_dict, root_dir, humans_ids_to_keep)

    return new_track_dict

def get_image_path_list(frames_dir):

    ''' Dado um diretório retorna uma lista com os frames nela presentes 

        Parameters
        ----------
        frames_dir: diretório com os frames 

        Returns
        -------
        img_path_list: lista com os frames nela presentes 
    '''
    frames = os.listdir(frames_dir)
    frames.sort()

    img_path_list = []
    for frame in frames:
        if not os.path.isdir(frame) and frame[-4:] in ['.jpg', '.png']:
            img_path = os.path.join(frames_dir, frame)
            img_path_list.append(img_path)

    return img_path_list


def plot_new_track_info(track_dict, root_dir, humans_ids_to_keep):

    ''' Desenho das bboxes no de um dado dicionário (tipicamente o novo dicionário)\n
        Deve ser executado posteriormente a  reorganize_track_info

        Parameters
        ----------
        track_dict: dicionário original tipicamente guardado como mpt.pkl
        root_dir: path para o diretório onde contem a informação do tracking '.../track'
        humans_ids_to_keep: Lista com a informação dos IDs a manter (construida manualmente após análise das imagens geradas no processo de tracking)  
        
    '''

    new_track_info_images_path = f'{root_dir}/images_only_with_{humans_ids_to_keep}_humans_IDs'
    os.makedirs(new_track_info_images_path, exist_ok=True)	# (166, 189, 219) -> Azul do hybrik	# (120, 198, 121) -> outro verde  

    # colors = [(51, 180, 0), (142, 149, 242), (230, 179, 179), 
    #          (255, 165, 0), (255, 255, 255), (128, 0, 128), (0, 255, 255), 
    #          (0, 0, 255), (255, 0, 0), (255, 255, 0), (0,0,0)]
    # colors = ['dark_green', 'light_purple', 'light_pink','orange', 'light_white' 'cyan', 'dark_blue', 'purple', 'red', 'yellow', 'black']
    colors = [(51, 180, 0), (142, 149, 242), (230, 179, 179), 
              (255, 165, 0), (247, 237, 220), (0, 255, 255), (0, 0, 255), 
              (128, 0, 128), (255, 0, 0), (227, 204, 52), (23, 21, 21)]
    id_for_color = {}
    proporcao_espessura = 0.0015
    proporcao_tamanho_texto = 0.0001 

    frames_dir = f'{os.path.dirname(root_dir)}/frames'
    img_path_list = get_image_path_list(frames_dir)

    print("Saving images with new tracking values...")
    for frame_index, img_path in enumerate(tqdm(img_path_list)):
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        height, width = img.shape[:2]
        espessura = max(int(width * proporcao_espessura), 1)
        tamanho_texto = max(width * proporcao_tamanho_texto, 0.9)

        for human_id, human_data in track_dict.items():
            if frame_index in human_data['frames']:
                bbox_index = np.where(np.array(human_data['frames']) == frame_index)[0]
                if bbox_index.size > 0:
                    bbox_index = bbox_index[0]
                    bbox = human_data['bbox'][bbox_index]

                    # Extrair informações da bounding box
                    cx, cy, w, h = bbox
                    
                    if human_id not in id_for_color:
                        id_for_color[human_id] = len(id_for_color) % len(colors)
                    color_index = id_for_color[human_id]
                    color = colors[color_index][::-1]       # Inverter a ordem dos elementos (RGB para BGR)

                    # Desenhar a bounding box e ID do humano na imagem
                    cv2.rectangle(img, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), color, espessura+3)
                    text = f"ID: {human_id} person"
                    (largura, altura), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, tamanho_texto+0.001, espessura+1)
                    x1 = int(cx - w / 2)
                    y1 = int(cy - h / 2 - altura - 10)
                    x2 = x1 + largura + 10
                    y2 = y1 + altura + 10
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                    # tamanho_texto=0.9
                    # cv2.putText(img, f"ID: {human_id} person", (int(cx-w/2), int(cy-h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255, 255, 255), 3)
                    cv2.putText(img, text, (x1 + 5, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, tamanho_texto, (255, 255, 255), espessura+1)

        # Salvar a imagem com as bounding boxes e IDs dos humanos
        output_path = os.path.join(new_track_info_images_path, os.path.basename(img_path))
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img)
    


def append_track_info(track_dict, person_id1, person_id2, root_dir=None):
    ''' Concatenação de IDs para eventual correção do método sort escolhido\n

        Parameters
        ----------
        track_dict: dicionário original tipicamente guardado como mpt.pkl
        person_id1: ID da pessoa que deverá conter a informação de person_id2
        person_id2: ID da pessoa que deverá conter informação a concatenar em person_id1 (person_id2 é removido de track_dict)  
        
    '''
    # Verificar se as pessoas com os IDs fornecidos existem no dicionário
    if person_id1 in track_dict and person_id2 in track_dict:       # 72 adicionado apos 26 e antes de 48
        # Obter os frames e bboxes da pessoa person_id2
        frames2 = track_dict[person_id2]['frames']
        bbox2 = track_dict[person_id2]['bbox']
        
        # Concatenar os frames e bboxes da pessoa person_id2 com a pessoa person_id1
        track_dict[person_id1]['frames'] = np.concatenate((track_dict[person_id1]['frames'], frames2))
        track_dict[person_id1]['bbox'] = np.concatenate((track_dict[person_id1]['bbox'], bbox2))

        # Remover a pessoa person_id2 do dicionário
        del track_dict[person_id2]
        # return track_dict

        if root_dir is not None:
            humans_ids_to_keep = list(track_dict.keys()) 
            plot_new_track_info(track_dict, root_dir, humans_ids_to_keep)

        return track_dict
    else:
        print("IDs de pessoas inválidos.")


def check_image_folder(frames_dir):
    # frames_dir = f'{os.path.dirname(root_dir)}/frames'
    frames_list = sorted(os.listdir(frames_dir))
    first_image_name = frames_list[0]
    # image_number = int(first_image_name[9:-4])
    # offset = 1 if image_number==0 else 0                  # offset = 1 # se frames começarem em image_00000.jpg caso contrário é 0

    if first_image_name != '000001.jpg':
        print("Renaming images...")
        for i, image_name in enumerate(tqdm(frames_list)):
            new_image_name = f'{i+1:06d}.jpg'
            old_image_path = os.path.join(frames_dir, image_name)
            new_image_path = os.path.join(frames_dir, new_image_name)
            os.rename(old_image_path, new_image_path)
            frames_list[i] = new_image_name
    else:
        print("Frame dir ok!")