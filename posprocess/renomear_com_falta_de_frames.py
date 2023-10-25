import os
import shutil

# Caminho para o diretório de origem e destino
diretorio_origem = "forvideo"
diretorio_destino = "forvideo/forvideo_ren"

# Garantir que o diretório de destino exista
if not os.path.exists(diretorio_destino):
    os.mkdir(diretorio_destino)

# Lista de imagens ordenadas
imagens_ordenadas = sorted([f for f in os.listdir(diretorio_origem) if f.endswith(".jpg")])

# Variável para acompanhar o próximo número na sequência
prox_numero = 1

# Renomear as imagens
for imagem_atual in imagens_ordenadas:
    nome_antigo = os.path.join(diretorio_origem, imagem_atual)
    nome_novo = f"{prox_numero:06d}.jpg"  # Formato com 6 dígitos
    nome_novo_caminho = os.path.join(diretorio_destino, nome_novo)
    
    # Verificar se o novo nome já existe
    while os.path.exists(nome_novo_caminho):
        prox_numero += 1
        nome_novo = f"{prox_numero:06d}.jpg"
        nome_novo_caminho = os.path.join(diretorio_destino, nome_novo)
    
    # Renomear a imagem
    shutil.copy(nome_antigo, nome_novo_caminho)
    prox_numero += 1

print("Renomeação concluída.")

