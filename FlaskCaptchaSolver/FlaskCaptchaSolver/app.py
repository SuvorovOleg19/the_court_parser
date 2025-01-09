import os

# Устанавливаем бэкенд Keras
os.environ["KERAS_BACKEND"] = "tensorflow"

import json

import numpy as np
from flask import Flask, request
from math import sqrt
import cv2
import tensorflow as tf
import keras
from keras import ops
from keras import layers
import random

# Размеры изображения
IMG_WIDTH = 100
IMG_HEIGHT = 50

max_length = 5

app = Flask(__name__)



with open("./char/char_to_num.json") as f:
    char_to_num_config = json.load(f)

with open("./char/num_to_char.json") as f:
    num_to_char_config = json.load(f)

char_to_num = layers.StringLookup.from_config(char_to_num_config)
num_to_char = layers.StringLookup.from_config(num_to_char_config)

def encode_single_sample(img_path):
    """Обработка изображения для использования в модели."""
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = ops.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = ops.transpose(img, axes=[1, 0, 2])
    return img

def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    """Вычисление CTC потерь."""
    label_length = ops.cast(ops.squeeze(label_length, axis=-1), dtype="int32")
    input_length = ops.cast(ops.squeeze(input_length, axis=-1), dtype="int32")
    sparse_labels = ops.cast(
        ctc_label_dense_to_sparse(y_true, label_length), dtype="int32"
    )

    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())

    return ops.expand_dims(
        tf.compat.v1.nn.ctc_loss(
            inputs=y_pred, labels=sparse_labels, sequence_length=input_length
        ),
        1,
    )

def ctc_label_dense_to_sparse(labels, label_lengths):
    """Конвертация меток в разреженный формат."""
    label_shape = ops.shape(labels)
    num_batches_tns = ops.stack([label_shape[0]])
    max_num_labels_tns = ops.stack([label_shape[1]])

    def range_less_than(old_input, current_input):
        return ops.expand_dims(ops.arange(ops.shape(old_input)[1]), 0) < tf.fill(
            max_num_labels_tns, current_input
        )

    init = ops.cast(tf.fill([1, label_shape[1]], 0), dtype="bool")
    dense_mask = tf.compat.v1.scan(
        range_less_than, label_lengths, initializer=init, parallel_iterations=1
    )
    dense_mask = dense_mask[:, 0, :]

    label_array = ops.reshape(
        ops.tile(ops.arange(0, label_shape[1]), num_batches_tns), label_shape
    )
    label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)

    batch_array = ops.transpose(
        ops.reshape(
            ops.tile(ops.arange(0, label_shape[0]), max_num_labels_tns),
            tf.reverse(label_shape, [0]),
        )
    )
    batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
    indices = ops.transpose(
        ops.reshape(ops.concatenate([batch_ind, label_ind], axis=0), [2, -1])
    )

    vals_sparse = tf.compat.v1.gather_nd(labels, indices)

    return tf.SparseTensor(
        ops.cast(indices, dtype="int64"),
        vals_sparse,
        ops.cast(label_shape, dtype="int64")
    )

class CTCLayer(layers.Layer):
    """Слой для расчета CTC потерь."""
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(ops.shape(y_true)[1], dtype="int64")

        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

# Загрузка модели
model = keras.models.load_model("./models/model.h5", custom_objects={'CTCLayer': CTCLayer})
prediction_model = keras.models.Model(
    model.input[0], model.get_layer(name="dense2").output
)

def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    input_shape = ops.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())
    input_length = ops.cast(input_length, dtype="int32")

    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length
        )
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    return decoded_dense, log_prob


def decode_batch_predictions(pred):
    """Декодирование предсказаний модели."""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]

    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def calculate_color_difference(color_1, color_2):
    """Вычисление цветового расстояния между двумя RGB цветами."""
    color_1 = np.array(color_1)
    color_2 = np.array(color_2)
    return np.linalg.norm(color_1 - color_2)

def preprocess_image(image_path):
    """Подготовка изображения к обработке: удаление шума на основе цвета."""
    image = cv2.imread(image_path)
    target_color = np.array([153, 102, 0])
    threshold = 90

    # Итерируемся по всем пикселям изображения
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            current_color = image[row, col]
            if calculate_color_difference(current_color, target_color) >= threshold:
                image[row, col] = [255, 255, 255]

    cv2.imwrite(image_path, image)


@app.route('/captcha', methods=["POST"])
def handle_captch():
    filename = "input_image.png"
    uploaded_file = request.files['file']
    
    if uploaded_file.filename != "":
        uploaded_file.save(filename)

    preprocess_image(filename)
    prediction = prediction_model.predict([tf.reshape(encode_single_sample(filename), [1, IMG_WIDTH, IMG_HEIGHT, 1])])
    prediction_texts = decode_batch_predictions(prediction)
    return prediction_texts[0]

if __name__ == "__main__":
    HOST = os.environ.get("SERVER_HOST", 'localhost')
    PORT = 5555
    app.run(HOST, PORT)