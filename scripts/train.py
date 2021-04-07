# -*- coding: utf-8 -*-
'''
  @author: jeffzhengye
  @contact: yezheng@scuec.edu.cn
  @file: train.py
  @time: 2021/2/17 17:44
  @desc:
 '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Activation, Permute, Reshape, Lambda, Dot
from tensorflow.python.layers.core import Dense

sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import tensorflow as tf

# whether to use gpu
tf.config.set_soft_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print('cannot set memory growth, use default')
        print(e)

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import *
from tensorflow.keras.metrics import AUC
from nir.modeling.layers import EsmmMLPLayer
from nir.utils.args_utils import ArgsParser, load_config, merge_config
from nir.utils.tf_utils import get_lr, get_layers, get_optimizer, get_loss, get_callbacks_from_config
from nir.data.ir_datasets import get_keras_train_input

# from nir.modeling.inputs_builder import build_din, build_inputs

FLAGS = ArgsParser().parse_args()
config = load_config(FLAGS.config)
merge_config(FLAGS.opt)

global_config = config['Global']
run_name = config['Global']['run_name']
dataset_name = config['Datasets']['active']
train_file = config['Datasets'][dataset_name]['train_qrel_format'].format(**global_config)
train_file_histogram = config['Datasets'][dataset_name]['train_histogram']
test_file = config['Datasets'][dataset_name]['test_qrel_format'].format(**global_config)
test_file_histogram = config['Datasets'][dataset_name]['test_histogram']

query_term_maxlen = config['Global']['query_term_maxlen']
hist_size = config['Global']['hist_size']
num_layers = config['Global']['num_layers']
hidden_sizes = config['Global']['hidden_sizes']

batch_size = config['Train']['dataset']['batch_size']
shuffle = config['Train']['dataset']['shuffle']

# model paras
model_name = config['Models']['active']
embed_size = config['Models'][model_name]['embed_size']
min_by = config['Models'][model_name]['min_by']
feature_dim = config['Models'][model_name]['feature_dim']  # used to hash features
share_embedding = config['Models'][model_name]['share_embedding']  # bool if share embedding.
mlp_ctr = config['Models'][model_name]['mlp_ctr']
mlp_cvr = config['Models'][model_name]['mlp_cvr']
activation = get_layers(config['Models'][model_name]['activation'])
batch_norm = config['Models'][model_name]['batch_norm']
dropouts = config['Models'][model_name]['dropouts']

save_freq = 'epoch'
# save_freq = 10

# Global
lr = get_lr(config)
epochs = config['Global']['epochs']
opt = get_optimizer(config)
name_tag = config['Global']['additional_tag']
# loss paras
loss_name = config['Loss'][model_name]['name']
loss = get_loss(loss_name)


def get_prefix(paras=None):
    prefix_formatter = "{model}_lr{lr}_bn{bn}_shuffle{shuffle}_dp{dropout}_{loss_name}_{loss_weight}_share{share_embedding}_e{embedding_size}_{activation}_{optimizer}"
    if paras:
        f_dict = paras
    else:
        f_dict = {
            'model': model_name + '-xd-fc_l' + '-'.join([str(d) for d in mlp_cvr]) + '_r' + '-'.join(
                [str(d) for d in mlp_ctr]),
            'lr': lr,
            'bn': int(batch_norm),
            'dropout': dropouts[0] if dropouts else '0',
            'share_embedding': int(share_embedding),
            'embedding_size': embed_size,
            'activation': activation if isinstance(activation, str) else type(activation).__name__,
            'optimizer': type(opt).__name__,
            'shuffle': int(shuffle)
        }
    if name_tag:
        prefix_formatter += "_" + name_tag
    return prefix_formatter.format_map(f_dict) + "_{epoch}"


prefix = get_prefix()

checkpoint_dir = os.path.join(config['Global']['checkpoint'], prefix)
result_dir = config['Global']['results_dir']
tensor_board_dir = os.path.join(result_dir, prefix + "_tensor_board")
csv_logger_dir = os.path.join(result_dir, prefix + '_csv_log.log')

# call back function
checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss', verbose=1, save_best_only=False, mode='min',
                             save_freq=save_freq)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=1, min_lr=0.0001, verbose=1)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1, mode='auto')
csv_logger = CSVLogger(csv_logger_dir)
callbacks = [reduce_lr, checkpoint, earlystopping, csv_logger]

query = Input(name='query', shape=(query_term_maxlen, 1))
# -> the histogram (2d array: every query gets 1d histogram
doc = Input(name='doc', shape=(query_term_maxlen, hist_size))


def build_inputs():
    return {
        'query': query,
        'doc': doc
    }


def get_drmm(input_gi, input_histogram):
    z = input_histogram
    for i in range(num_layers):
        z = Dense(hidden_sizes[i], activation='tanh')(z)

    z = layers.Flatten(z)

    # term gating network for idf weighting strategy.  @TODO  the first strategy of Term Vector (TV)
    q_w = Dense(1, use_bias=False)(input_gi)  # what is that doing here ??
    q_w = layers.Flatten(q_w)
    q_w = layers.Softmax(q_w)

    #
    # combination of softmax(query term idf) and feed forward result per query term
    #
    out_ = Dot(axes=[1, 1])([z, q_w])
    return out_


def train_graph():
    inputs = build_inputs()
    output = get_drmm(inputs['query'], inputs['doc'])
    model = Model(inputs=inputs, outputs=output)
    model.summary()

    model.compile(optimizer=opt, loss=loss)
    print('start fitting', prefix)

    train_input, train_labels = get_keras_train_input(train_file, train_file_histogram)

    model.fit(x=train_input, y=train_labels, callbacks=callbacks, batch_size=batch_size,
              verbose=1, workers=10, epochs=epochs, shuffle=False)


train_graph()
