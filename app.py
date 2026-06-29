import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np

# MONKEY PATCHING
original_dense_init = tf.keras.layers.Dense.__init__

def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)

tf.keras.layers.Dense.__init__ = patched_dense_init

# 1. Configuração visual básica do site
st.set_page_config(page_title="Classificador de Flores", page_icon="🌻")
st.title("Descubra que flor é essa 🌻")
st.write("Faça o upload de uma foto e nossa Inteligência Artificial dirá qual é a espécie!")

# 2. Carregando o modelo salvo
@st.cache_resource
def carregar_modelo():
    return tf.keras.models.load_model('melhor_modelo_flores.keras', compile=False)

modelo = carregar_modelo()

# 3. Lista com os nomes das flores
nomes_flores = [
    'Pink Primrose', 'Hard-Leaved Pocket Orchid', 'Canterbury Bells','Sweet Pea',
    'English Marigold','Tiger Lily', 'Moon Orchid','Bird Of Paradise','Monkshood',
    'Globe Thistle', 'Snapdragon','ColtS Foot','King Protea','Spear Thistle', 'Yellow Iris',
    'Globe-Flower','Purple Coneflower','Peruvian Lily','Balloon Flower','Giant White Arum Lily',
    'Fire Lily','Pincushion Flower','Fritillary','Red Ginger',
    'Grape Hyacinth','Corn Poppy','Prince Of Wales Feathers',
    'Stemless Gentian','Artichoke','Sweet William','Carnation',
    'Garden Phlox','Love In The Mist','Mexican Aster',
    'Alpine Sea Holly','Ruby-Lipped Cattleya','Cape Flower',
    'Great Masterwort','Siam Tulip','Lenten Rose','Barbeton Daisy',
    'Daffodil','Sword Lily','Poinsettia','Bolero Deep Blue','Wallflower','Marigold',
    'Buttercup','Oxeye Daisy','Common Dandelion','Petunia','Wild Pansy','Primula',
    'Sunflower','Pelargonium','Bishop Of Llandaff','Gaura','Geranium','Orange Dahlia',
    'Pink-Yellow Dahlia?','Cautleya Spicata','Japanese Anemone','Black-Eyed Susan','Silverbush',
    'Californian Poppy','Osteospermum','Spring Crocus','Bearded Iris','Windflower','Tree Poppy',
    'Gazania','Azalea','Water Lily','Rose','Thorn Apple','Morning Glory','Passion Flower','Lotus',
    'Toad Lily','Anthurium','Frangipani','Clematis','Hibiscus','Columbine','Desert-Rose','Tree Mallow',
    'Magnolia','Cyclamen','Watercress','Canna Lily','Hippeastrum','Bee Balm','Ball Moss',
    'Foxglove','Bougainvillea','Camellia','Mallow','Mexican Petunia','Bromelia','Blanket Flower','Trumpet Creeper','Blackberry Lily'
]


st.warning("⚠️ **Aviso:** Se a flor que você está enviando não estiver na nossa lista de espécies conhecidas, o resultado sairá errado e a Inteligência Artificial tentará adivinhar a flor mais parecida com ela.")

with st.expander("Ver lista das 102 flores suportadas"):
    flores_ordenadas = sorted(nomes_flores)
    st.write(" O modelo foi treinado exclusivamente para reconhecer estas espécies:")
    st.write(" • " + "\n • ".join(flores_ordenadas))

st.divider() # Linha de separação visual

arquivo_foto = st.file_uploader("Escolha a foto da flor", type=["jpg", "jpeg", "png"])

if arquivo_foto is not None:
    imagem = Image.open(arquivo_foto)
    st.image(imagem, caption='Sua Flor', use_container_width=True)

    st.write("🧠 Analisando...")

    # 5. Pré-processamento
    img_redimensionada = imagem.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img_redimensionada)
    img_array = tf.expand_dims(img_array, 0)
    img_array = preprocess_input(img_array)

    # 6. A Predição
    previsoes = modelo.predict(img_array)
    indice_classe = np.argmax(previsoes[0])
    certeza = np.max(previsoes[0]) * 100

    flor_predita = nomes_flores[indice_classe].title()

    # 7. Lógica de Resposta Baseada na Certeza
    if certeza >= 99.9: # Usamos 99.9 porque é muito raro dar 100% cravado matematicamente
        st.success(f"🎉 Sua flor é **{flor_predita}**")
    elif certeza >= 75.0:
        st.success(f"✅ Sua flor com quase certeza é **{flor_predita}**")
    elif certeza >= 50.0:
        st.warning(f"🤔 Sua flor talvez seja: **{flor_predita}**")
    else:
        st.error(f"🤷‍♂️ Não temos certeza, mas achamos que é **{flor_predita}**")

    st.info(f"**Certeza matemática do Modelo:** {certeza:.2f}%")