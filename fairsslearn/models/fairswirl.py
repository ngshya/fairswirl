import os

from tensorflow.compat.v1 import set_random_seed as tf_seed
from tensorflow.random import set_seed as tf_seed_2
from random import seed as rnd_seed
from numpy.random import seed as np_seed

from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Layer, \
    BatchNormalization, ReLU, Activation
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow import custom_gradient, identity

from numpy import max


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #


@custom_gradient
def grad_reverse(x):
    y = identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad


class GradReverse(Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #


def build_models(
    input_size,
    ae_encoder_size,
    ae_embedder_size, 
    ae_decoder_size,
    cl_encoder_size,
    n_classes, 
    di_encoder_size,
    s_levels,
    w_rec,
    w_cla,
    w_dis,
    optimizer="Adam",
    learning_rate=0.001,
):

    dict_layers = {}
    dict_models = {}

    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #

    dict_layers["input"] = Input(
        shape=(input_size, ), 
        name="input"
    )

    dict_layers["s"] = Input(
        shape=(s_levels, ), 
        name="s"
    )

    for j, k in enumerate(ae_encoder_size):
        if j == 0:
            dict_layers["ae_encoder_"+str(j+1)] = \
            ReLU()(BatchNormalization()(Dense(
                units=k,
                activation="linear",
                name="ae_encoder_"+str(j+1)
            )(Dropout(rate=0.2)(dict_layers["input"]))))
        else:
            dict_layers["ae_encoder_"+str(j+1)] =  \
            ReLU()(BatchNormalization()(Dense(
                units=k,
                activation="linear",
                name="ae_encoder_"+str(j+1)
            )(Dropout(rate=0.2)(dict_layers["ae_encoder_"+str(j)]))))           

    dict_layers["embedder"] =  \
        BatchNormalization()(Dense(
        units=ae_embedder_size,
        activation=None,
        name="embedder"
    )(Dropout(rate=0.2)(dict_layers["ae_encoder_"+str(len(ae_encoder_size))])))

    for j, k in enumerate(ae_decoder_size):
        if j == 0:
            dict_layers["ae_decoder_"+str(j+1)] =  \
            ReLU()(BatchNormalization()(Dense(
                units=k,
                activation="linear",
                name="ae_decoder_"+str(j+1)
            )(
                Concatenate()([Dropout(rate=0.2)(dict_layers["embedder"]), dict_layers["s"]])
            )))
        else:
            dict_layers["ae_decoder_"+str(j+1)] =  \
            ReLU()(BatchNormalization()(Dense(
                units=k,
                activation="linear",
                name="ae_decoder_"+str(j+1)
            )(
                Dropout(rate=0.2)(dict_layers["ae_decoder_"+str(j)])
            )))            

    dict_layers["ae_reconstructor"] =  \
        Activation("sigmoid")(BatchNormalization()(Dense(
        units=input_size,
        activation="linear",
        name="ae_reconstructor"
    )(
        Dropout(rate=0.2)(dict_layers["ae_decoder_"+str(len(ae_decoder_size))])
    )))

    for j, k in enumerate(cl_encoder_size):
        if j == 0:
            dict_layers["cl_encoder_"+str(j+1)] =  \
            ReLU()(BatchNormalization()(Dense(
                units=k,
                activation="linear",
                name="cl_encoder_"+str(j+1)
            )(
                Dropout(rate=0.2)(dict_layers["embedder"])
            )))
        else:
            dict_layers["cl_encoder_"+str(j+1)] =  \
            ReLU()(BatchNormalization()(Dense(
                units=k,
                activation="linear",
                name="cl_encoder_"+str(j+1)
            )(
                Dropout(rate=0.2)(dict_layers["cl_encoder_"+str(j)])
            )))            

    dict_layers["classifier"] = Dense(
        units=n_classes, 
        activation="softmax", 
        name="classifier"
    )(
        Dropout(rate=0.2)(dict_layers["cl_encoder_"+str(len(cl_encoder_size))])
    )

    for j, k in enumerate(di_encoder_size):
        if j == 0:
            dict_layers["di_encoder_"+str(j+1)] =  \
            ReLU()(BatchNormalization()(Dense(
                units=k,
                activation="linear",
                name="di_encoder_"+str(j+1)
            )(
                GradReverse()(Dropout(rate=0.2)(dict_layers["embedder"]))
            )))
        else:
            dict_layers["di_encoder_"+str(j+1)] =  \
            ReLU()(BatchNormalization()(Dense(
                units=k,
                activation="linear",
                name="di_encoder_"+str(j+1)
            )(
                Dropout(rate=0.2)(dict_layers["di_encoder_"+str(j)])
            )))

    dict_layers["discriminator"] = Dense(
        units=s_levels, 
        activation="softmax", 
        name="discriminator"
    )(
        Dropout(rate=0.2)(dict_layers["di_encoder_"+str(len(di_encoder_size))])
    )

    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #
    
    dict_models["embedder"] = Model(
        inputs=dict_layers["input"],
        outputs=dict_layers["embedder"],
    )

    dict_models["reconstructor"] = Model(
        inputs=[dict_layers["input"], dict_layers["s"]],
        outputs=dict_layers["ae_reconstructor"],
    )

    dict_models["classifier"] = Model(
        inputs=dict_layers["input"],
        outputs=dict_layers["classifier"],
    )

    dict_models["discriminator"] = Model(
        inputs=dict_layers["input"],
        outputs=dict_layers["discriminator"],
    )

    dict_models["rec+cla+dis"] = Model(
        inputs=[dict_layers["input"], dict_layers["s"]],
        outputs=[
            dict_layers["ae_reconstructor"],
            dict_layers["classifier"],
            dict_layers["discriminator"],
        ],
    )

    dict_models["rec+dis"] = Model(
        inputs=[dict_layers["input"], dict_layers["s"]],
        outputs=[
            dict_layers["ae_reconstructor"],
            dict_layers["discriminator"],
        ],
    )

    dict_models["everything"] = Model(
        inputs=[dict_layers["input"], dict_layers["s"]],
        outputs=[
            dict_layers["ae_reconstructor"],
            dict_layers["classifier"],
            dict_layers["discriminator"],
            dict_layers["embedder"],
        ],
    )

    if optimizer == "Adam":
        optimizer = Adam(learning_rate=learning_rate)

    dict_models["rec+cla+dis"].compile(
        optimizer=optimizer, 
        loss=['mse', 'categorical_crossentropy', 'categorical_crossentropy'], 
        loss_weights=[w_rec, w_cla, w_dis], 
        metrics=[["mse"], ["accuracy"], ["accuracy"]]
    )

    dict_models["rec+dis"].compile(
        optimizer=optimizer, 
        loss=['mse', 'categorical_crossentropy'], 
        loss_weights=[w_rec, w_dis], 
        metrics=[["mse"], ["accuracy"]]
    )

    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #

    return dict_layers, dict_models


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #


class FairSwiRL: 

    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #

    def __init__(
        self,
        input_size,
        n_classes, 
        s_levels,
        ae_encoder_size=None,
        ae_embedder_size=None, 
        ae_decoder_size=None,
        cl_encoder_size=None,
        di_encoder_size=None,
        w_rec=0.1,
        w_cla=0.1,
        w_dis=0.1,
        optimizer="Adam",
        learning_rate=0.001,
        seed=None
    ):

        # seed

        if seed is not None:
            self.seed = seed
        else:
            self.seed = 1102
        np_seed(self.seed)
        rnd_seed(self.seed)
        tf_seed(self.seed)
        tf_seed_2(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        
        # default values

        if ae_encoder_size is None:
            ae_encoder_size = [max((int(0.88 * input_size), 2))]
        self.ae_encoder_size = ae_encoder_size

        if ae_embedder_size is None:
            ae_embedder_size = max((int(0.66 * input_size), 1))
        self.ae_embedder_size = ae_embedder_size

        if ae_decoder_size is None:
            ae_decoder_size = ae_encoder_size
        self.ae_decoder_size = ae_decoder_size

        if cl_encoder_size is None:
            cl_encoder_size = [ae_embedder_size]
        self.cl_encoder_size = cl_encoder_size

        if di_encoder_size is None:
            di_encoder_size = [ae_embedder_size]
        self.di_encoder_size = di_encoder_size

        # other parameters

        self.input_size = input_size
        self.n_classes = n_classes
        self.s_levels = s_levels

        self.w_rec = w_rec
        self.w_cla = w_cla
        self.w_dis = w_dis
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        # build layers and models

        self.layers, self.models = build_models(
            input_size=self.input_size,
            ae_encoder_size=self.ae_encoder_size,
            ae_embedder_size=self.ae_embedder_size, 
            ae_decoder_size=self.ae_decoder_size,
            cl_encoder_size=self.cl_encoder_size,  
            n_classes=self.n_classes, 
            di_encoder_size=self.di_encoder_size,
            s_levels=self.s_levels,
            w_rec=self.w_rec,
            w_cla=self.w_cla,
            w_dis=self.w_dis,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
        )

        # monitoring metrics

        self.metrics = []

        # initial weights

        self.models_initial_weights = {}
        for k in self.models.keys():
            self.models_initial_weights[k] = self.models[k].get_weights()        

    # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-* #

    def fit(
        self, 
        Xl, 
        yl,
        sl,
        Xu,
        su,
        batch_size, 
        epochs, 
        Xt=None,
        yt=None,
        st=None,
    ):

        np_seed(self.seed)
        rnd_seed(self.seed)
        tf_seed(self.seed)
        tf_seed_2(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)

        for e in range(epochs):

            if (Xu is not None) and (su is not None):

                history_1 = self.models["rec+dis"].fit(
                    x=[Xu, su], 
                    y=[Xu, su], 
                    batch_size=batch_size, 
                    epochs=1, 
                    verbose=0, 
                    validation_data=([Xt, st], [Xt, st])
                )

            history_2 = self.models["rec+cla+dis"].fit(
                x=[Xl, sl], 
                y=[Xl, yl, sl], 
                batch_size=batch_size, 
                epochs=1, 
                verbose=0, 
                validation_data=([Xt, st], [Xt, yt, st])
            )

            if (e==0) or (e==epochs-1) or (e%10 == 9):
                dict_tmp = history_2.history
                for k in dict_tmp.keys():
                    dict_tmp[k] = dict_tmp[k][0]
                dict_tmp["epoch"] = e+1
                dict_tmp["model"] = 2
                self.metrics.append(dict_tmp)
