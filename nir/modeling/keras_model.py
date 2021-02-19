import keras
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Activation, merge, Lambda, Permute
from keras.layers import Reshape, Dot
from keras.activations import softmax

query_term_maxlen = 5
hist_size = 30
num_layers = 2
hidden_sizes = [5, 1]

initializer_fc = keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=11)
initializer_gate = keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=11)


#
# returns the raw keras model object
#
def build_keras_model():

    #
    # input layers (query and doc)
    # 输入层（查询和文档）
    #

    # -> the query idf input (1d array of float32)
    # 查询idf输入
    query = Input(name='query', shape=(query_term_maxlen,1))

    # -> the histogram (2d array: every query gets 1d histogram
    doc = Input(name='doc', shape=(query_term_maxlen, hist_size))

    #
    # the histogram handling part (feed forward network)
    # 直方图处理部分(前馈网络)
    #

    z = doc
    for i in range(num_layers):
        z = Dense(hidden_sizes[i], kernel_initializer=initializer_fc)(z)
        z = Activation('tanh')(z)

    z = Permute((2, 1))(z)
    z = Reshape((query_term_maxlen,))(z)

    #
    # the query term idf part
    # 查询词idf部分
    #

    q_w = Dense(1, kernel_initializer=initializer_gate, use_bias=False)(query)
    q_w = Lambda(lambda x: softmax(x, axis=1), output_shape=(query_term_maxlen,))(q_w)
    q_w = Reshape((query_term_maxlen,))(q_w) # isn't that redundant ??

    #
    # combination of softmax(query term idf) and feed forward result per query term
    # softmax(查询词idf)和每个查询词的前馈结果的组合
    #
    out_ = Dot(axes=[1, 1])([z, q_w])

    model = Model(inputs=[query, doc], outputs=[out_])

    return model
