import os
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import re
import Snake
model_name = ""


def gpu(turn):  # enable of disable GPU backend
    if turn:
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024 * 6))])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass


def inp_out():  # returns all "good" routs from the log file
    inp = np.array([])
    out = np.array([])
    buffer_i = np.array([])
    buffer_o = np.array([])
    log_txt = open("snake_log.txt", mode="r")
    for line in log_txt:
        regex = re.findall(r'[-+]?\d*\.?\d+', line)
        if len(regex) == Snake.data_len:  # main data
            for r in regex:
                buffer_i = np.append(buffer_i, float(r))
        elif len(regex) == 1:  # output
            buffer_o = np.append(buffer_o, int(regex[0]))
        elif line == "\n":
            buffer_i = buffer_i.reshape(int(buffer_i.shape[0] / Snake.data_len), Snake.data_len)
            inp = np.append(inp, buffer_i)
            out = np.append(out, buffer_o)
            buffer_i = np.array([])
            buffer_o = np.array([])
        else:
            buffer_i = np.array([])
            buffer_o = np.array([])
    if len(inp) > 0:
        inp = inp.reshape(int(inp.shape[0] / Snake.data_len), Snake.data_len)
    return inp, out


def add_to_log(message, mode="a"):  # write something to the log file
    log_txt = open("snake_log.txt", mode)
    log_txt.write(message)
    log_txt.close()
    # print(message)


def add_to_meta(message, mode="a"):  # write something to the meta file
    meta_txt = open(model_name + "/meta.txt", mode)
    meta_txt.write(message)
    meta_txt.close()


def get_from_meta(number_or_string):  # get info from the meta file
    log_txt = open(model_name + "/meta.txt", "r")
    if number_or_string:
        data = [int(re.findall(r'\d+', line)[-1]) for line in log_txt]
    else:
        data = [line for line in log_txt]
    log_txt.close()
    return data


def config_parser():
    data = []
    if os.path.exists("config.txt"):
        config = open("config.txt")
        for line in config:
            if line.__contains__("model_name"):
                data.append(line.split("=")[1].replace(" ", "").replace("\n", ""))
            elif line.__contains__("rewrite_model"):
                if line.__contains__("True"):
                    data.append(True)
                else:
                    data.append(False)
            elif line.__contains__("GPU_usage"):
                if line.__contains__("True"):
                    data.append(True)
                else:
                    data.append(False)
            elif line.__contains__("snake_color"):
                data.append(tuple(list(map(int, re.findall(r'\d+', line)))))
            elif line.__contains__("snack_color"):
                data.append(tuple(list(map(int, re.findall(r'\d+', line)))))
            elif line.__contains__("possible_steps_const"):
                data.append(int(re.findall(r'\d+', line)[0]))
            elif line.__contains__("learning_rate"):
                data.append(int(re.findall(r'\d+', line)[0]))
            elif line.__contains__("epochs"):
                data.append(int(re.findall(r'\d+', line)[0]))
            elif line.__contains__("iterations_limit"):
                data.append(int(re.findall(r'\d+', line)[0]))
        config.close()
        return data
    else:
        raise IOError("config.txt can\'t be found!")


def get_snake():  # get the NN model
    model = tf.keras.Sequential([
        keras.layers.Conv1D(16, 3, activation='relu', input_shape=(Snake.data_len, 1),
                            kernel_initializer=keras.initializers.random_normal),
        keras.layers.Conv1D(32, 3, activation='relu', kernel_initializer=keras.initializers.random_normal),
        keras.layers.Conv1D(64, 3, activation='relu', kernel_initializer=keras.initializers.random_normal),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu', kernel_initializer=keras.initializers.random_normal),
        keras.layers.Dense(32, activation='relu', kernel_initializer=keras.initializers.random_normal),
        keras.layers.Dense(16, activation='relu', kernel_initializer=keras.initializers.random_normal),
        keras.layers.Dense(4, activation='softmax', kernel_initializer=keras.initializers.random_normal)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
