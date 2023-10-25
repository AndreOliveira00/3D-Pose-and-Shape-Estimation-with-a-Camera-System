import os
import os.path as osp
import subprocess
from PIL import Image

FFMPEG_PATH = '/usr/bin/ffmpeg' if osp.exists('/usr/bin/ffmpeg') else 'ffmpeg'
font_files = {
    'Windows': 'C:/Windows/Fonts/arial.ttf',
    'Linux': '/usr/share/fonts/truetype/lato/Lato-Regular.ttf',
    'Darwin': '/System/Library/Fonts/Supplemental/Arial.ttf'
}

def processar_imagem(imagem):
    largura, altura = imagem.size
    flag_corte = False

    if largura % 2 != 0:
        # Se a largura não for divisível por 2, corta 1 pixel da direita
        imagem = imagem.crop((0, 0, largura - 1, altura))
        largura=largura-1
        flag_corte=True

    if altura % 2 != 0:
        # Se a altura não for divisível por 2, corta 1 pixel de baixo
        imagem = imagem.crop((0, 0, largura, altura - 1))
        altura=altura-1
        flag_corte=True

    return imagem, flag_corte

def processar_imagens_no_diretorio(diretorio):
    for nome_arquivo in os.listdir(diretorio):
        if nome_arquivo.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            caminho_completo = os.path.join(diretorio, nome_arquivo)
            
            try:
                imagem = Image.open(caminho_completo)
                imagem_processada,flag_corte = processar_imagem(imagem)
                if flag_corte:
                    imagem_processada.save(caminho_completo)
                    print(f"Imagem cortada para dimensões pares: {caminho_completo}")
            except Exception as e:
                print(f"Erro ao processar {caminho_completo}: {str(e)}")

def images_to_video(img_dir, out_path, init_frame, img_fmt="%06d.jpg", fps=30, crf=25, verbose=True):
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cmd = [FFMPEG_PATH, '-y', '-r', f'{fps}', '-f', 'image2', '-start_number', f'{init_frame}',
            '-i', f'{img_dir}/{img_fmt}', '-vcodec', 'libx264', '-crf', f'{crf}', '-pix_fmt', 'yuv420p', out_path]
    if not verbose:
        cmd += ['-hide_banner', '-loglevel', 'error']
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise Exception('Something went wrong during images_to_video!')

frame_dir = "/home/andre/Documents/Projects/GLAMR/out/glamr_static/bow/grecon_images/20230901-162333"
vid_out_file =  "/home/andre/Documents/Projects/GLAMR/out/glamr_static/bow/grecon_videos_from_hrnet48/forpresentation.mp4"
init_frame = 0

processar_imagens_no_diretorio(frame_dir)

images_to_video(frame_dir, vid_out_file, init_frame, fps=30, verbose=False)
print("We are done!")


