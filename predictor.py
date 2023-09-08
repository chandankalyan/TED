# importing libraries
import tensorflow as tf
from tensorflow.keras.layers import StringLookup
import numpy as np
from tensorflow import keras
import os

# required to get the vocabulary
text_files = open("char/characters.txt", "r").readlines()
characters = list()
for lines in text_files:
    characters.append(lines.split("\n")[0])

char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


# function to resize and pad the input image
def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check the amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image


# required shape of the image
image_width = 128
image_height = 32


# function to read and preprocess the image
def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


# function to decode the prediction made
max_len = 21
res = str()


def decode_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


# loading the image
img_path = "/Users/amogh/Downloads/test2.jpeg"
test_img = preprocess_image(img_path)
test_img = tf.reshape(test_img, (1, 128, 32, 1))

# loading the model
model = tf.keras.models.load_model("model.h5")

# making the prediction
prediction = model.predict(test_img)

# decoding the prediction
decoded = decode_predictions(prediction)

# specifying the path to store the prediction
filename = "output/prediction.txt"

# writing the prediction into a .txt file
with open(filename, "w") as f:
    f.write(decoded[0])

os.remove(img_path)