# Original implementation: https://github.com/yevgeni-integrate-ai/VFAE
# adapted to the semi-supervised setting

import os

from tensorflow.compat.v1 import set_random_seed as tf_seed
from tensorflow.random import set_seed as tf_seed_2
from random import seed as rnd_seed
from numpy.random import seed as np_seed

import tensorflow as tf
import numpy as np


def gen_weights_biases(shape):
    return tf.Variable(
        tf.random.normal(shape, stddev=tf.sqrt(0.5 / float(shape[0])))
    )


def activate(x, activation):
    if (activation == 'relu'):
        return tf.nn.relu(x)
    elif (activation == 'sigmoid'):
        return tf.nn.sigmoid(x)
    elif (activation == 'tanh'):
        return tf.nn.tanh(x)
    elif (activation == 'softmax'):
        return tf.nn.softmax(x)
    elif (activation == 'linear'):
        return x


def get_batch(list_arrays, batch_size, index_shuffled, b):
    return [x[index_shuffled[b*batch_size:(b+1)*batch_size]] \
        for x in list_arrays]


def MLP(x_in, weights, bias, activation, epsilon):
    hidden = activate(
        tf.matmul(x_in,weights['hid'])+bias['hid'],activation['hid']
    )
    mu = activate(
        tf.matmul(hidden,weights['mu'])+bias['mu'],activation['mu']
    )
    log_sigma = activate(
        tf.matmul(
            hidden,
            weights['log_sigma']
        ) + bias['log_sigma'],
        activation['log_sigma']
    )
    return mu + tf.exp(log_sigma / 2) * epsilon, mu, log_sigma


def KL(mu1, log_sigma_sq1, mu2=0., log_sigma_sq2=0.):
    return 0.5 * tf.reduce_sum(
        log_sigma_sq2 - log_sigma_sq1 - 1 + \
        (tf.exp(log_sigma_sq1)+tf.pow(mu1-mu2,2))/tf.exp(log_sigma_sq2), 
        axis=1
    )


def fast_MMD(x1, x2, params):
    inner_difference = tf.reduce_mean(
        psi(x1, params), axis=0) - tf.reduce_mean(psi(x2, params),
        axis=0
    )
    return tf.tensordot(inner_difference, inner_difference, axes=1)


def psi(x,params):
    W = tf.Variable(
        tf.random.normal(
            [params['enc1']['out_dim'],params['D']], 
            stddev=tf.sqrt(0.5 / float(params['enc1']['out_dim'])),
            dtype=tf.float32
        )
    )
    b = tf.Variable(tf.random.uniform([params['D']],0,2*np.pi,dtype=tf.float32))
    
    return tf.pow(2./params['D'],0.5) \
        * tf.cos(tf.pow(2./params['gamma'],0.5) * tf.matmul(x,W) + b)


def initialize_params(
    dims,
    N_epochs=1000,
    print_freq=100,
    batch_size=100,
    lr=1e-3,
    alpha=1.,
    beta=1.,
    D=500,
    gamma=1.
):
    params = {
        'enc1':{
            'in_dim':dims['x']+dims['s'],
            'hid_dim':dims['enc1_hid'],
            'out_dim':dims['z1'],
            'act':{
                'hid':'relu',
                'mu':'linear',
                'log_sigma':'linear'
            }
        },   
        'enc2':{
            'in_dim':dims['z1']+1,
            'hid_dim':dims['enc2_hid'],
            'out_dim':dims['z2'],
            'act':{
                'hid':'relu',
                'mu':'linear',
                'log_sigma':'linear'
            }
        },
        'dec1':{
            'in_dim':dims['z2']+1,
            'hid_dim':dims['dec1_hid'],
            'out_dim':dims['z1'],
            'act':{
                'hid':'relu',
                'mu':'linear',
                'log_sigma':'linear'
            }
        },
        'dec2':{
            'in_dim':dims['z1']+dims['s'],
            'hid_dim':dims['dec2_hid'],
            'out_dim':dims['x']+dims['s'],
            'act':{
                'hid':'relu',
                'mu':'sigmoid',
                'log_sigma':'sigmoid',
                'bernoulli':'sigmoid'
            }
        },
        'us':{
            'in_dim':dims['z1'],
            'hid_dim':dims['us_hid'],
            'out_dim':dims['y_cat'],
            'act':{
                'hid': 'relu',
                'out': 'softmax',
                'mu': 'sigmoid', # <-- required
                'log_sigma': 'sigmoid', # <-- required
            }
        },
        'N_epochs':N_epochs,
        'print_frequency':print_freq,
        'batch_size':batch_size,
        'lr':lr,
        'alpha':alpha,
        'beta':beta,
        'D':D,
        'gamma':gamma
    }
    return params


def initialize_weights_biases(params):
    weights = {
        'enc1':{
            'hid': gen_weights_biases(
                [params['enc1']['in_dim'],params['enc1']['hid_dim']]
            ),
            'mu': gen_weights_biases(
                [params['enc1']['hid_dim'],params['enc1']['out_dim']]
            ),
            'log_sigma': gen_weights_biases(
                [params['enc1']['hid_dim'],params['enc1']['out_dim']]
            )
        },
        'enc2':{
            'hid': gen_weights_biases(
                [params['enc2']['in_dim'],params['enc2']['hid_dim']]
            ),
            'mu': gen_weights_biases(
                [params['enc2']['hid_dim'],params['enc2']['out_dim']]
            ),
            'log_sigma': gen_weights_biases(
                [params['enc2']['hid_dim'],params['enc2']['out_dim']]
            )
        },
        'dec1':{
            'hid': gen_weights_biases(
                [params['dec1']['in_dim'],params['dec1']['hid_dim']]
            ),
            'mu': gen_weights_biases(
                [params['dec1']['hid_dim'],params['dec1']['out_dim']]
            ),
            'log_sigma': gen_weights_biases(
                [params['dec1']['hid_dim'],params['dec1']['out_dim']]
            )
        },
        'dec2':{
            'hid': gen_weights_biases(
                [params['dec2']['in_dim'],params['dec2']['hid_dim']]
            ),
            'mu': gen_weights_biases(
                [params['dec2']['hid_dim'],params['dec2']['out_dim']]
            ),
            'log_sigma': gen_weights_biases(
                [params['dec2']['hid_dim'],params['dec2']['out_dim']]
            )
        },
        'us':{
            'hid': gen_weights_biases(
                [params['us']['in_dim'],params['us']['hid_dim']]
            ),
            'mu': gen_weights_biases(
                [params['us']['hid_dim'],params['us']['out_dim']]
            ),
            'log_sigma': gen_weights_biases(
                [params['us']['hid_dim'],params['us']['out_dim']]
            )
        }       
    }

    bias = {
        'enc1':{
            'hid': gen_weights_biases([params['enc1']['hid_dim']]),
            'mu': gen_weights_biases([params['enc1']['out_dim']]),
            'log_sigma': gen_weights_biases([params['enc1']['out_dim']])
        },
        'enc2':{
            'hid': gen_weights_biases([params['enc2']['hid_dim']]),
            'mu': gen_weights_biases([params['enc2']['out_dim']]),
            'log_sigma': gen_weights_biases([params['enc2']['out_dim']])
        },
        'dec1':{
            'hid': gen_weights_biases([params['dec1']['hid_dim']]),
            'mu': gen_weights_biases([params['dec1']['out_dim']]),
            'log_sigma': gen_weights_biases([params['dec1']['out_dim']])
        },
        'dec2':{
            'hid': gen_weights_biases([params['dec2']['hid_dim']]),
            'mu': gen_weights_biases([params['dec2']['out_dim']]),
            'log_sigma': gen_weights_biases([params['dec2']['out_dim']])
        },
        'us':{
            'hid': gen_weights_biases([params['us']['hid_dim']]),
            'mu': gen_weights_biases([params['us']['out_dim']]),
            'log_sigma': gen_weights_biases([params['us']['out_dim']])
        }
    }
    
    return weights, bias


class VFAE:

    def __init__(
        self, 
        z1=20,
        z2=20,
        enc1_hid=50,
        enc2_hid=50,
        dec1_hid=50,
        dec2_hid=50,
        us_hid=50,
        lr=1e-3,
        alpha=1.0,
        beta=1.0,
        D=100,
        gamma=1.0,
        seed=None,
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

        self.z1 = z1
        self.z2 = z2
        self.enc1_hid = enc1_hid
        self.enc2_hid = enc2_hid
        self.dec1_hid = dec1_hid
        self.dec2_hid = dec2_hid
        self.us_hid = us_hid
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.D=100
        self.gamma = gamma

        
    def fit(self, X_train, y_train, s_train, epochs, batch_size):

        tf.compat.v1.disable_eager_execution()

        # seed

        np_seed(self.seed)
        rnd_seed(self.seed)
        tf_seed(self.seed)
        tf_seed_2(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)

        self.idx_del = []
        for j in range(X_train.shape[1]):
            if (sum(X_train[:, j] == s_train) == X_train.shape[0]):
                self.idx_del.append(j)
        X_train = np.delete(X_train, self.idx_del, axis=1)

        # initialization

        self.dims = {
            'x': X_train.shape[1],
            'y_cat': len(np.unique(y_train[~np.isnan(y_train)])),
            's': 1,
            'z1': self.z1,
            'z2': self.z2,
            'enc1_hid': self.enc1_hid,
            'enc2_hid': self.enc2_hid,
            'dec1_hid': self.dec1_hid,
            'dec2_hid': self.dec2_hid,
            'us_hid': self.us_hid,
        }

        self.params = initialize_params(
            dims = self.dims,
            N_epochs = epochs, 
            lr = self.lr,
            print_freq = 10,
            batch_size = batch_size,
            alpha = self.alpha,
            beta = self.beta,
            D = self.D,
            gamma = self.gamma,
        )

        self.weights, self.bias = initialize_weights_biases(self.params)

        # network

        self.x = tf.compat.v1.placeholder(
            tf.float32, 
            shape=[None, self.dims['x']],
            name='x'
        )
        self.s = tf.compat.v1.placeholder(
            tf.float32,
            shape=[None, self.dims['s']],
            name='s'
        )
        self.y = tf.compat.v1.placeholder(
            tf.float32,
            shape=[None, 1], 
            name='y'
        )

        # z1 enc

        epsilon0 = tf.random.normal(
            [self.params['enc1']['out_dim']], 
            dtype=tf.float32,
            name='epsilon0'
        )
        self.z1_enc, z1_enc_mu, z1_enc_log_sigma = MLP(
            tf.concat([self.x, self.s], axis=1),
            self.weights['enc1'],
            self.bias['enc1'],
            self.params['enc1']['act'],
            epsilon0
        )

        # imputed y

        epsilon4 = tf.zeros(
            [self.params['us']['out_dim']], 
            dtype=tf.float32,
            name='epsilon4'
        )
        self.y_us, us_mu, us_log_sigma = MLP(
            self.z1_enc,
            self.weights['us'],
            self.bias['us'],
            self.params['us']['act'],
            epsilon4
        )

        bool_not_nan = ~tf.math.is_nan(self.y) # for the semi-supervised case   

        self.y_imputed = tf.where(
            bool_not_nan, 
            self.y,
            tf.cast(tf.expand_dims(tf.math.argmax(self.y_us, 1), axis=1), tf.float32)
        )

        # z2 enc

        epsilon1 = tf.random.normal(
            [self.params['enc2']['out_dim']], 
            dtype=tf.float32,
            name='epsilon1'
        )
        z2_enc, z2_enc_mu, z2_enc_log_sigma = MLP(
            tf.concat([
                self.z1_enc, 
                self.y_imputed
            ],axis=1),
            self.weights['enc2'],
            self.bias['enc2'],
            self.params['enc2']['act'],
            epsilon1
        )

        # z1 dec

        epsilon2 = tf.random.normal(
            [self.params['dec1']['out_dim']], 
            dtype=tf.float32,
            name='epsilon2'
        )
        z1_dec, z1_dec_mu, z1_dec_log_sigma = MLP(
            tf.concat([z2_enc, self.y_imputed], axis=1),
            self.weights['dec1'],
            self.bias['dec1'],
            self.params['dec1']['act'],
            epsilon2
        )

        # x out

        epsilon3 = tf.zeros(
            [self.params['dec2']['out_dim']], 
            dtype=tf.float32,
            name='epsilon3'
        )
        self.x_out = MLP(
            tf.concat([z1_dec, self.s],axis=1),
            self.weights['dec2'],
            self.bias['dec2'],
            self.params['dec2']['act'],
            epsilon3
        )[0]

        KL_z1 = KL(z1_enc_mu,z1_enc_log_sigma,z1_dec_mu,z1_dec_log_sigma)
        KL_z2 = KL(z2_enc_mu,z2_enc_log_sigma)
        KL_us = KL(us_mu,us_log_sigma)

        LH_x = tf.reduce_sum(
            tf.concat([self.x, self.s],axis=1) * \
            tf.math.log(1e-10+self.x_out) + \
            (1 - tf.concat([self.x, self.s],axis=1)) * \
            tf.math.log(1e-10+1 - self.x_out), 
            axis=1
        )

        index = tf.range(tf.shape(self.y)[0])
        idx = tf.stack(
            [tf.boolean_mask(index, bool_not_nan[:, 0])[:, tf.newaxis], 
            tf.cast(tf.boolean_mask(self.y, bool_not_nan[:, 0]),tf.int32)], 
            axis=-1
        )
        LH_y = tf.reduce_sum(
            tf.math.log(1e-10+tf.gather_nd(self.y_us, idx)),
            axis=1
        )


        MMD_x1 = tf.boolean_mask(
            self.z1_enc,
            tf.tile(tf.cast(self.s,tf.bool), [1,tf.shape(self.z1_enc)[1]])
        )
        MMD_x2 = tf.boolean_mask(
            self.z1_enc,
            tf.tile(tf.cast(1-self.s,tf.bool),[1,tf.shape(self.z1_enc)[1]])
        )
        MMD = fast_MMD(
            tf.reshape(
                MMD_x1,
                [
                    tf.cast(
                        tf.shape(MMD_x1)[0]/tf.shape(self.z1_enc)[1],
                        tf.int32
                    ),
                    tf.shape(self.z1_enc)[1]
                ]
            ),
            tf.reshape(
                MMD_x2,
                [
                    tf.cast(
                        tf.shape(MMD_x2)[0]/tf.shape(self.z1_enc)[1],
                        tf.int32
                    ),
                    tf.shape(self.z1_enc)[1]
                ]
            ),
            self.params
        )

        tot_kl_z1 = tf.reduce_mean(KL_z1)
        tot_kl_z2 = tf.reduce_mean(KL_z2)
        tot_kl_us = tf.reduce_mean(KL_us)
        tot_lh_x = tf.reduce_mean(LH_x)
        tot_lh_y = self.params['alpha'] * tf.reduce_mean(LH_y)
        tot_mmd = self.params['beta'] * MMD

        loss = - ( 
            - tot_kl_z1 \
            - tot_kl_z2 \
            - tot_kl_us \
            + tot_lh_x \
            - tot_lh_y \
            - tot_mmd 
        )

        optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self.params['lr']
        )

        train = optimizer.minimize(loss)

        self.sess = tf.compat.v1.Session()

        self.sess.run(tf.compat.v1.global_variables_initializer())

        index_shuffled = np.arange(X_train.shape[0])
        np.random.shuffle(index_shuffled)

        N_batches = int(
            float(X_train.shape[0])/float(self.params['batch_size'])
        )
    
        for i in range(self.params['N_epochs']):

            for b in range(N_batches):

                batch_X, batch_Y, batch_s = get_batch(
                    [X_train, y_train, s_train], 
                    self.params['batch_size'],
                    index_shuffled,
                    b
                )

                batch_dict = {
                    self.x: batch_X, 
                    self.y: np.expand_dims(batch_Y, axis=1), 
                    self.s: np.expand_dims(batch_s, axis=1),
                }
                full_dict = {
                    self.x: X_train, 
                    self.y: np.expand_dims(y_train, axis=1), 
                    self.s: np.expand_dims(s_train, axis=1),
                }

                loss_batch = self.sess.run(loss, feed_dict=batch_dict)
                if (not np.isnan(loss_batch)) and np.isfinite(loss_batch):
                    self.sess.run(train, feed_dict=batch_dict)

            if (i % self.params['print_frequency'] == 0 \
                or i == self.params['N_epochs']-1):

                print("Epoch %s: batch loss = %s and global loss = %s"%(i,
                        self.sess.run(loss,feed_dict=batch_dict),
                        self.sess.run(loss,feed_dict=full_dict)))
                print("kl_z1 = %s, kl_z2 = %s, kl_us = %s, lh_x  = %s, lh_y = %s, mmd = %s" \
                    % self.sess.run(
                        (
                            tot_kl_z1,
                            tot_kl_z2,
                            tot_kl_us,
                            tot_lh_x,
                            tot_lh_y,
                            tot_mmd 
                        ),
                        feed_dict=full_dict))


    def predict_emb(self, X, s):
        X = np.delete(X, self.idx_del, axis=1)
        return self.sess.run(
            self.z1_enc, 
            feed_dict={
                self.x: X, 
                self.y: [[1.0]]*X.shape[0], 
                self.s: np.expand_dims(s, axis=1),
            }
        )
