from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.getcwd()))
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from lib.utils.vis import video_to_images
from preprocess.images2video import images_to_video

def verificar_mesma_largura(diretorio, largura_referencia, jump):
    lista_arquivos = os.listdir(diretorio)
    if len(lista_arquivos) == 0:
        print("O diretório está vazio.")
        return

    for arquivo in lista_arquivos[1:]:
        caminho_arquivo = os.path.join(diretorio, arquivo)
        if os.path.isfile(caminho_arquivo):
            imagem = Image.open(caminho_arquivo)
            # print(imagem.width)
            if imagem.width != largura_referencia:
                print(f"As imagens não têm a mesma width. -> {os.path.basename(caminho_arquivo)}")
                return

    print("Todas as imagens têm a mesma width.")

def cut_images(source_directory, x_init, y_init, width, height, destination_directory=None):
    if destination_directory is None:
        destination_directory=source_directory
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    
    for arquivo in tqdm(sorted(os.listdir(source_directory))):
        caminho_arquivo_origem = os.path.join(source_directory, arquivo)
        if os.path.isfile(caminho_arquivo_origem):
            imagem = Image.open(caminho_arquivo_origem)
            largura_imagem, altura_imagem = imagem.size
            x_final = min(x + width, largura_imagem)
            y_final = min(y + height, altura_imagem)
            cut_image = imagem.crop((x, y, x_final, y_final))
            nome_arquivo_destino = os.path.join(destination_directory, arquivo)
            cut_image.save(nome_arquivo_destino)
            
def cut_images_as_dyn(source_directory, destination_directory, x, y, width, height):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)
    flag_up = True

    for arquivo in tqdm(sorted(os.listdir(source_directory))):
        caminho_arquivo_origem = os.path.join(source_directory, arquivo)
        if os.path.isfile(caminho_arquivo_origem):
            imagem = Image.open(caminho_arquivo_origem)
            largura_imagem, altura_imagem = imagem.size

            x_final = min(x + width, largura_imagem)
            y_final = min(y + height, altura_imagem)

            regiao_cortada = imagem.crop((x, y, x_final, y_final))

            nome_arquivo_destino = os.path.join(destination_directory, arquivo)
            regiao_cortada.save(nome_arquivo_destino)

            # Atualiza o valor de x para a próxima iteração
            if flag_up and x + width < largura_imagem:
                x += jump             # width / jump deve ser interior senão imagens ficam com diferentes resoluções
            else:
                if x + width > width+60:
                    x -= jump
                    flag_up = False
                else:
                    flag_up = True



# Diretório das imagens originais
source_directory = 'assets/dynamic/courtyard_basketball_01_0_469'

# Coordenadas de início do corte (ponto x, y)
x = 320
y = 240

# Dimensões do corte (width, height)
width = 516 # must be divisible by 2
height = 1350

# width / jump deve ser interior senão imagens ficam com diferentes resoluções
jump=12

# Diretório de destino para as imagens cortadas
destination_directory = f'{source_directory}/cut_images_{width}x{height}'

# Chama a função para cortar as imagens
cut_images_as_dyn(source_directory, destination_directory, x, y, width, height)

verificar_mesma_largura(destination_directory, width, jump)


# x = 560
# y = 0
# width = 1360 # must be divisible by 2
# height = 1010
# source_directory = "out/glamr_static/WalkingAround_Joe_1s/pose_est_hybrik_hrnet48/frames"
# destination_directory = "assets/static/WalkingAround_Joe_0_10"
# cut_images(source_directory,x,y,width,height,destination_directory)

# frame_dir = destination_directory
# vid_out_file =  "assets/static/WalkingAround_Joe_0_10.mp4"
# init_frame = 1
# images_to_video(frame_dir, vid_out_file, init_frame, fps=30, verbose=False)

print("We are done!")
