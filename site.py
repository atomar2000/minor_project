import streamlit as st
from PIL import Image
import numpy as np
import os, shutil
import tensorflow as tf

load_model = tf.keras.models.load_model('./pix2pix/model')
folder = './uploaded_images'
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 100

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def ex_image_test(image_file):
    print('load_ex started')
    input_image = load_ex(image_file)
    print('load_ex completed')
    input_image = resize_ex(input_image, IMG_HEIGHT, IMG_WIDTH)
    print('resize completed')
    input_image = normalize_ex(input_image)
    print('normalize completed')
    return input_image

def normalize_ex(input_image):
    input_image = (input_image / 127.5) - 1
    return input_image

def resize_ex(input_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return input_image

def load_ex(image_file):
    image = tf.io.read_file(image_file)
    image= tf.image.decode_jpeg(image)
    input_image = tf.cast(image, tf.float32)
    return input_image

def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

def predictThis(ex_dataset):
        for example_input in ex_dataset.take(1):
            prediction = load_model(example_input, training=True)

        image = (prediction[0]+1)*127.5;
        image  = image/255;
        image = np.array(image)
        print(image)
        return image


st.title("pix2pix GAN ")
st.subheader("Upload a map image to convert to sattelite image")
st.spinner("Testing spinner")

uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.')
    st.write("")
    im1 = image.save('./uploaded_images/up_img.jpg')
    ex_dataset = tf.data.Dataset.list_files("./uploaded_images/" + '*.jpg')
    ex_dataset = ex_dataset.map(ex_image_test)
    ex_dataset = ex_dataset.batch(BATCH_SIZE)
    if st.button('predict'):
        st.write("prediting..")
        pred = predictThis(ex_dataset)
        st.image(pred, caption='Predited Image', use_column_width=True)
        clear_folder(folder)
