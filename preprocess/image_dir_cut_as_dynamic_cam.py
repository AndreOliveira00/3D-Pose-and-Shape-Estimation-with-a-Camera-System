from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.getcwd()))
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from lib.utils.vis import video_to_images
from preprocess.images2video import images_to_video

# video_file = "/home/andre/Downloads/videos/House Dance Choreo by MaMSoN.mp4"
# image_folder = "/home/andre/Downloads/videos/House Dance Choreo by MaMSoN"
# video_to_images(video_file, image_folder, fps=30)

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
                print(f"As imagens não têm a mesma largura. -> {os.path.basename(caminho_arquivo)}")
                return

    print("Todas as imagens têm a mesma largura.")

def cut_images(diretorio_origem, x_init, y_init, largura, altura, diretorio_destino=None):
    if diretorio_destino is None:
        diretorio_destino=diretorio_origem
    if not os.path.exists(diretorio_destino):
        os.makedirs(diretorio_destino)
    
    for arquivo in tqdm(sorted(os.listdir(diretorio_origem))):
        caminho_arquivo_origem = os.path.join(diretorio_origem, arquivo)
        if os.path.isfile(caminho_arquivo_origem):
            imagem = Image.open(caminho_arquivo_origem)
            largura_imagem, altura_imagem = imagem.size
            x_final = min(x + largura, largura_imagem)
            y_final = min(y + altura, altura_imagem)
            cut_image = imagem.crop((x, y, x_final, y_final))
            nome_arquivo_destino = os.path.join(diretorio_destino, arquivo)
            cut_image.save(nome_arquivo_destino)
            
def cut_images_as_dyn(diretorio_origem, diretorio_destino, x, y, largura, altura):
    if not os.path.exists(diretorio_destino):
        os.makedirs(diretorio_destino)
    flag_up = True

    for arquivo in tqdm(sorted(os.listdir(diretorio_origem))):
        caminho_arquivo_origem = os.path.join(diretorio_origem, arquivo)
        if os.path.isfile(caminho_arquivo_origem):
            imagem = Image.open(caminho_arquivo_origem)
            largura_imagem, altura_imagem = imagem.size

            x_final = min(x + largura, largura_imagem)
            y_final = min(y + altura, altura_imagem)

            regiao_cortada = imagem.crop((x, y, x_final, y_final))

            nome_arquivo_destino = os.path.join(diretorio_destino, arquivo)
            regiao_cortada.save(nome_arquivo_destino)

            # Atualiza o valor de x para a próxima iteração
            if flag_up and x + largura < largura_imagem:
                x += jump             # largura / jump deve ser interior senão imagens ficam com diferentes resoluções
            else:
                if x + largura > largura+60:
                    x -= jump
                    flag_up = False
                else:
                    flag_up = True



# Diretório das imagens originais
diretorio_origem = 'assets/dynamic/courtyard_basketball_01_0_469'

# Coordenadas de início do corte (ponto x, y)
x = 320
y = 240

# Dimensões do corte (largura, altura)
largura = 516 # must be divisible by 2
altura = 1350

# largura / jump deve ser interior senão imagens ficam com diferentes resoluções
jump=12

# Diretório de destino para as imagens cortadas
diretorio_destino = f'{diretorio_origem}/cut_images_{largura}x{altura}'

# Chama a função para cortar as imagens
cut_images_as_dyn(diretorio_origem, diretorio_destino, x, y, largura, altura)

verificar_mesma_largura(diretorio_destino, largura, jump)


# x = 560
# y = 0
# largura = 1360 # must be divisible by 2
# altura = 1010
# diretorio_origem = "/home/andre/Documents/Projects/GLAMR/out/glamr_static/WalkingAround_Joe_1s/pose_est_hybrik_hrnet48/frames"
# diretorio_destino = "/home/andre/Documents/Projects/GLAMR/assets/static/WalkingAround_Joe_0_10"
# cut_images(diretorio_origem,x,y,largura,altura,diretorio_destino)

# frame_dir = diretorio_destino
# vid_out_file =  "/home/andre/Documents/Projects/GLAMR/assets/static/WalkingAround_Joe_0_10.mp4"
# init_frame = 1
# images_to_video(frame_dir, vid_out_file, init_frame, fps=30, verbose=False)

print("We are done!")
