import argparse
import glob

import cv2
import numpy as np
from six.moves import range
from dataloader import *
from tensorpack import *
from tensorpack.utils import logger
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.python.training import moving_averages
from tensorflow.keras.layers import UpSampling2D
from tensorpack.tfutils.export import ModelExporter
from GAN_sg import GANTrainer, GANModelDesc
import os
import sys
import six
from datetime import datetime
import random
import math

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

seed = 56
utils.fix_rng_seed(seed)
tf.set_random_seed(seed)

DIS_SCALE = 3
SHAPE = 256
NF = 64  # channel size
M_F = None  # feature size, set later
M_S = 64  # number of points sampled along each ray
Lx = 30
Ld = 12
N_DIS = 4
N_SAMPLE = 2
N_SCALE = 3
N_RES = 4
STYLE_DIM = 8
STYLE_DIM_z2 = 8
n_upsampling = 5
chs = NF * 8
gauss_std = 0.05
LR = 1e-4
enable_argscope_for_module(tf.layers)


def SmartInit(obj, ignore_mismatch=False):
    """
    Create a :class:`SessionInit` to be loaded to a session,
    automatically from any supported objects, with some smart heuristics.
    The object can be:

    + A TF checkpoint
    + A dict of numpy arrays
    + A npz file, to be interpreted as a dict
    + An empty string or None, in which case the sessinit will be a no-op
    + A list of supported objects, to be initialized one by one

    Args:
        obj: a supported object
        ignore_mismatch (bool): ignore failures when the value and the
            variable does not match in their shapes.
            If False, it will throw exception on such errors.
            If True, it will only print a warning.

    Returns:
        SessionInit:
    """
    if not obj:
        return JustCurrentSession()
    if isinstance(obj, list):
        return ChainInit([SmartInit(x, ignore_mismatch=ignore_mismatch) for x in obj])
    if isinstance(obj, six.string_types):
        obj = os.path.expanduser(obj)
        if obj.endswith(".npy") or obj.endswith(".npz"):
            assert tf.gfile.Exists(obj), "File {} does not exist!".format(obj)
            filename = obj
            logger.info("Loading dictionary from {} ...".format(filename))
            if filename.endswith('.npy'):
                obj = np.load(filename, encoding='latin1').item()
            elif filename.endswith('.npz'):
                obj = dict(np.load(filename))
        elif len(tf.gfile.Glob(obj + "*")):
            # Assume to be a TF checkpoint.
            # A TF checkpoint must be a prefix of an actual file.
            return (SaverRestoreRelaxed if ignore_mismatch else SaverRestore)(obj)
        else:
            raise ValueError("Invalid argument to SmartInit: " + obj)

    if isinstance(obj, dict):
        return DictRestore(obj)
    raise ValueError("Invalid argument to SmartInit: " + type(obj))


def tpad(x, pad, mode='CONSTANT', name=None):
    return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode=mode)


def INLReLU(x, name='IN'):
    with tf.variable_scope(name):
        x = InstanceNorm('in', x)
        x = tf.nn.leaky_relu(x)
    return x


def kl(mu1, sigma1, mu2, sigma2):

    return (tf.log(tf.linalg.det(sigma2) / tf.linalg.det(sigma1)) - 3. +
            tf.linalg.trace(tf.matmul(tf.linalg.inv(sigma2), sigma1)) +
            tf.squeeze(tf.matmul(tf.matmul((mu1 - mu2), tf.linalg.inv(sigma2)),
                                 tf.transpose(mu1 - mu2, (0, 1, 3, 2))))) * 0.5


def kl_div(mu1, sigma1, mu2, sigma2):
    kl_est = tf.zeros(tf.shape(mu1)[0], np.float32)
    # mu1 = tf.cast(mu1, tf.float32)
    # mu2 = tf.cast(mu2, tf.float32)
    # sigma1 = tf.cast(sigma1, tf.float32)
    # sigma2 = tf.cast(sigma2, tf.float32)
    for i in range(mu1.shape[1]):
        mu_a = tf.expand_dims(mu1[:, i, :, :], 1)
        sigma_a = tf.expand_dims(sigma1[:, i, :, :], 1)
        divs = kl(mu_a, sigma_a, mu2, sigma2)
        kl_est += tf.clip_by_value(tf.reduce_min(divs, 1), clip_value_min=0., clip_value_max=5.)

    return tf.reduce_mean(kl_est)


def get_rotation(rotx, roty, rotz):
    """
    Gets a rotation matrix.

    :param rotx: rotation around x-axis
    :param roty: rotation around y-axis
    :param rotz: rotation around z-axis

    :return: a rotation matrix
    """
    # rotx = tf.cast(rotx, tf.float64)
    # roty = tf.cast(roty, tf.float64)
    # rotz = tf.cast(rotz, tf.float64)
    rotX = rotx * np.pi / 180
    rotY = roty * np.pi / 180
    rotZ = rotz * np.pi / 180
    zr = tf.zeros_like(rotY)
    ons = tf.ones_like(rotY)

    RX = tf.stack([tf.stack([ons, zr, zr], axis=-1), tf.stack([zr, tf.cos(rotX), -tf.sin(rotX)], axis=-1),
                   tf.stack([zr, tf.sin(rotX), tf.cos(rotX)], axis=-1)], axis=-1)
    RY = tf.stack([tf.stack([tf.cos(rotY), zr, tf.sin(rotY)], axis=-1), tf.stack([zr, ons, zr], axis=-1),
                   tf.stack([-tf.sin(rotY), zr, tf.cos(rotY)], axis=-1)], axis=-1)
    RZ = tf.stack([tf.stack([tf.cos(rotZ), -tf.sin(rotZ), zr], axis=-1),
                   tf.stack([tf.sin(rotZ), tf.cos(rotZ), zr], axis=-1),
                   tf.stack([zr, zr, ons], axis=-1)], axis=-1)

    # Composed rotation matrix with (RX,RY,RZ)
    R = tf.matmul(tf.matmul(RX, RY), RZ)

    return R


def apply_rotation(mus, sigma, rotx, roty, rotz):
    """
    Rotate gaussians

    :param mus: [batch_size, nb_landmarks, 1, 3]. Parameter mu.
    :param sigma: [batch_size, nb_landmarks, 3, 3]. Parameter sigma.
    :param rotx: Rotation along x axis.
    :param roty: Rotation along y axis.
    :param rotz: Rotation along z axis.
    :return: the rotated mu's and sigma's.
    """

    assert mus is not None
    assert sigma is not None

    R = get_rotation(rotx, roty, rotz)

    R = R * tf.ones_like(sigma)
    sigma = tf.linalg.matmul(tf.linalg.matmul(R, sigma), R, transpose_b=True)
    # mus = tf.cast(mus, tf.float64)
    mus = tf.transpose(tf.linalg.matmul(R, tf.transpose(mus, [0, 1, 3, 2])), [0, 1, 3, 2])

    return mus, sigma


def get_density(mus, sigmas, points):
    """
    Query the volume density for each location

    :param mus: [batch_size, nb_landmarks, 1, 3]. Parameter mu.
    :param sigmas: [batch_size, nb_landmarks, 3, 3]. Parameter sigma.
    :param points: [batch_size, ray_batch_size, M_S, 3]. 3D locations.
    :return: [batch_size, ray_batch_size, M_S, nb_landmarks]. Volume densities.
    """

    points = tf.expand_dims(points, axis=3)
    points = tf.expand_dims(points, axis=3)
    mus = tf.expand_dims(mus, axis=1)
    mus = tf.expand_dims(mus, axis=1)
    sigmas = tf.expand_dims(sigmas, axis=1)
    sigmas = tf.expand_dims(sigmas, axis=1)
    densities = tf.exp(tf.matmul(tf.matmul((points - mus), tf.linalg.inv(sigmas)),
                                 tf.transpose(-(points - mus), [0, 1, 2, 3, 5, 4])))
    densities = tf.squeeze(densities, [4, 5])
    # densities = tf.reduce_sum(tf.squeeze(densities, [0, 4, 5]), axis=-1)

    return densities


def numerical_interg(sigmas, fs, M_S=M_S, ray_len=2., include_feature=True):
    """
    Conduct numerical integration along each ray to get the final feature f
    if include_feature is True, or final density if False, assuming that
    points are sampled uniformly.

    :param sigmas: [batch_size, ray_batch_size, M_S, nb_landmarks]. Volume densities.
    :param fs: [batch_size, ray_batch_size, M_S, M_F] or None. Features.
    :param M_S: Number of points sampled.
    :param ray_len: the length of each ray.
    :param include_feature: Boolean. Calculates final feature if True, otherwise
            caculates density.

    :return: [batch_size, ray_batch_size, M_F]. Final features.
            or [batch_size, ray_batch_size, nb_landmarks]. Final densities.
    """

    assert fs is None or include_feature, "Missing feature vectors!"

    # thetas = tf.reduce_sum(tf.square(xs[:, :, 1:, :] - xs[:, :, :-1, :]), axis=-1)
    sigmas = tf.cast(sigmas, tf.float32)
    if include_feature:
        sigmas = tf.reduce_sum(sigmas, axis=-1)
        sigmas = tf.expand_dims(sigmas, axis=-1)
    thetas = tf.zeros_like(sigmas) + (ray_len / M_S)
    alphas = 1 - tf.exp(-tf.multiply(sigmas, thetas))  # calculate alpha values

    # calculate transmittance
    output_list = []
    output_list.append(tf.zeros_like(sigmas[:, :, 0, :]))
    output_list.append(1 - alphas[:, :, 0, :])
    for i in range(2, M_S):
        output_list.append(tf.multiply(output_list[-1], 1 - alphas[:, :, i - 1, :]))
    ts = tf.stack(output_list, axis=-2)

    if include_feature:
        fs = tf.cast(fs, tf.float32)
        output = tf.reduce_sum(tf.multiply(alphas * ts, fs), axis=-2)
    else:
        output = tf.reduce_sum(alphas * ts, axis=-2)

    return output


def point_sampling(rez=16, M_S=64, ray_len=2., f=1.):
    """
    Cast rays in the canonical pose and sample points uniformly along each ray.
    It is assumed that the object is within the range [-1, 1]^3 of the world
    coordinate system, camera position is at (0, 0, 1 + f), and the look direction
    is (0, 0, -1).

    :param rez: the resolution of the rendered image
    :param M_S: number of points sampled along each ray
    :param ray_len: the length of each ray
    :param f: focus
    :return: [rez * rez, M_S, 3], Sampled point coordinates;
            [rez * rez, 1, 3], viewing directions.
    """

    X, Y = np.mgrid[-1:1:complex(0, rez), -1:1:complex(0, rez)]
    starts = np.vstack((np.vstack((X.flatten(), Y.flatten())), np.zeros_like(X.flatten()) + 1)).T
    starts = tf.cast(starts, tf.float32)
    focus = np.array((0, 0, 1. + f), np.float32)
    direction = tf.linalg.l2_normalize(starts - focus, axis=-1)
    steps = direction * ray_len / M_S

    starts = tf.expand_dims(starts, axis=1)
    steps = tf.expand_dims(steps, axis=1)
    points = starts + steps * tf.reshape(np.arange(M_S).astype(np.float32), [1, -1, 1])
    direction = tf.expand_dims(direction, axis=-2)

    return points, direction


def positional_encoding(tensor, L=60):
    """
    Performs element-wise positional encoding on an input tensor.

    :param tensor: input
    :param L: the number of frequency octaves
    :return: a positional encoded tensor
    """

    def encoding(t, L=L):
        output_list = []
        for i in range(L):
            output_list.append(tf.sin(2**i*t*math.pi))
            output_list.append(tf.cos(2 ** i * t * math.pi))
        return tf.stack(output_list, axis=-1)

    output = tf.map_fn(fn=encoding, elems=tensor)
    # flatten the last two dimensions
    # print(tf.concat((tf.shape(output)[:-2], np.array([L*2*3])), axis=-1))
    output = tf.reshape(output, tf.concat((tf.shape(output)[:-2], np.array([L*2*3])), axis=-1))

    return output


def img2mask(img):
    m = tf.reduce_mean(img, axis=-1)
    m = tf.expand_dims(m, axis=-1)
    m = tf.cast(tf.math.greater(m, -1.), tf.float32)

    return m


def z_sample(mean, logvar):
    eps = tf.random_normal(tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)

    return mean + tf.exp(logvar * 0.5) * eps


class CheckNumerics(Callback):
    """
    When triggered, check variables in the graph for NaN and Inf.
    Raise exceptions if such an error is found.
    """

    def _setup_graph(self):
        vars = tf.trainable_variables()
        ops = [tf.check_numerics(v, "CheckNumerics['{}']".format(v.op.name)).op for v in vars]
        self._check_op = tf.group(*ops)

    def _before_run(self, _):
        self._check_op.run()


class Model(GANModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 3), 'img'),
                tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 1), 'template'),
                tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 1), 'mask'),
                tf.placeholder(tf.float32, (None, 1, 1, 4), 'bbx'),
                tf.placeholder(tf.float32, (None, 1, 1, STYLE_DIM), 'z1'),
                tf.placeholder(tf.float32, (None, 1, 1, STYLE_DIM_z2), 'z2'),
                ]

    @auto_reuse_variable_scope
    def get_features(self, input, nb_layers=8, hidden_d=128, output_d=M_F, name='get_features'):
        # channel = pow(2, N_SAMPLE) * NF
        with tf.variable_scope(name), \
             argscope([tf.layers.dense],
                      kernel_initializer=tf.keras.initializers.VarianceScaling()):

            x = tf.layers.dense(input, hidden_d, activation=tf.nn.relu)
            for _ in range(nb_layers - 1):
                x = tf.layers.dense(x, hidden_d, activation=tf.nn.relu)
            x = tf.layers.dense(x, output_d, activation=tf.nn.sigmoid)

        return x

    @auto_reuse_variable_scope
    def get_musigma(self, nb_blocks=4):
        with argscope([Conv2D, Conv2DTranspose], activation=INLReLU):
            x0 = tf.get_variable('cann_lmks', shape=[1, NF * 4], dtype=tf.float32, trainable=True,
                                 initializer=tf.ones_initializer)
            x = tf.layers.dense(x0, NF * 4, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)

            mus = tf.layers.dense(x, nb_landmarks * 3, activation=tf.nn.tanh)
            mus = tf.reshape(mus, [-1, nb_landmarks, 1, 3])
            # mus = mus[:, :, None, :]

            v0 = tf.layers.dense(x, nb_landmarks * 3, activation=tf.nn.tanh)
            v0 = tf.reshape(v0, [-1, nb_landmarks, 3])
            v1 = tf.layers.dense(x, nb_landmarks * 3, activation=tf.nn.tanh)
            v1 = tf.reshape(v1, [-1, nb_landmarks, 3])

            v0 = tf.math.l2_normalize(v0, axis=-1)
            v1 = tf.linalg.cross(v0, v1)
            v1 = tf.math.l2_normalize(v1, axis=-1)
            v2 = tf.linalg.cross(v1, v0)
            v0 = tf.reshape(v0, [-1, nb_landmarks, 3, 1])
            v1 = tf.reshape(v1, [-1, nb_landmarks, 3, 1])
            v2 = tf.reshape(v2, [-1, nb_landmarks, 3, 1])

            V = tf.concat([v0, v2, v1], axis=-1)
            u = tf.layers.dense(x, nb_landmarks * 3, activation=tf.nn.sigmoid) * 0.5 + 1e-2
            u = tf.reshape(u, [-1, nb_landmarks, 3])
            U = tf.linalg.diag(u)
            sigma = tf.linalg.matmul(tf.linalg.matmul(V, U), V, transpose_b=True)

        return mus, sigma

    @auto_reuse_variable_scope
    def transform_mu_sigma(self, x, cannmu, cannsigma, nb_blocks=4):
        assert x is not None
        with tf.variable_scope('rotscale'):
            # cannsigma = tf.cast(cannsigma, tf.float64)
            x = tf.squeeze(x, axis=[1, 2])
            for k in range(nb_blocks):
                x = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)
                # musigmak = musigma[:, k]
                # x = Model.adaln(x, musigmak, 'adaln%d'%k, NF*8)
            x0 = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)
            x0 = tf.layers.dense(x0, NF * 4, activation=tf.nn.leaky_relu)

            x1 = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)
            x1 = tf.layers.dense(x1, NF * 4, activation=tf.nn.leaky_relu)

            x2 = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)
            x2 = tf.layers.dense(x2, NF * 4, activation=tf.nn.leaky_relu)

            x3 = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)
            x3 = tf.layers.dense(x3, NF * 4, activation=tf.nn.leaky_relu)

            mu_t = tf.layers.dense(x0, nb_landmarks * 3, activation=tf.nn.tanh) * 0.1
            mu_t = tf.reshape(mu_t, [-1, nb_landmarks, 1, 3])
            # mu_t = mu_t[:, :, None, :]
            mus = cannmu * tf.ones_like(mu_t) + mu_t

            S, U, V = tf.linalg.svd(cannsigma, full_matrices=True)
            S = tf.linalg.diag(tf.sqrt(S))
            Sscale = tf.layers.dense(x1, nb_landmarks * 3, activation=tf.nn.tanh) * 0.05 + 1.
            Sscale = tf.reshape(Sscale, [-1, nb_landmarks, 3])
            # sfty = tf.linalg.diag(tf.ones_like(Sscale)*0.01)
            Sscale = tf.linalg.diag(Sscale)
            # Sscale = tf.cast(Sscale, tf.float64)
            Sscale = S * Sscale  # + sfty

            sigrotx = tf.layers.dense(x2, nb_landmarks, activation=tf.nn.tanh) * np.pi / 10.
            sigroty = tf.layers.dense(x2, nb_landmarks, activation=tf.nn.tanh) * np.pi / 10.
            sigrotz = tf.layers.dense(x2, nb_landmarks, activation=tf.nn.tanh) * np.pi / 10.
            zr = tf.zeros_like(sigrotz)
            os = tf.ones_like(sigrotz)
            RX = tf.stack([tf.stack([os, zr, zr], axis=-1), tf.stack([zr, tf.cos(sigrotx), -tf.sin(sigrotx)], axis=-1),
                           tf.stack([zr, tf.sin(sigrotx), tf.cos(sigrotx)], axis=-1)], axis=-1)
            RY = tf.stack([tf.stack([tf.cos(sigroty), zr, tf.sin(sigroty)], axis=-1), tf.stack([zr, os, zr], axis=-1),
                           tf.stack([-tf.sin(sigroty), zr, tf.cos(sigroty)], axis=-1)], axis=-1)
            RZ = tf.stack([tf.stack([tf.cos(sigrotz), -tf.sin(sigrotz), zr], axis=-1),
                           tf.stack([tf.sin(sigrotz), tf.cos(sigrotz), zr], axis=-1),
                           tf.stack([zr, zr, os], axis=-1)], axis=-1)

            # Composed rotation matrix with (RX,RY,RZ)
            R = tf.matmul(tf.matmul(RX, RY), RZ)
            # R = tf.cast(R, tf.float64)
            newU = tf.matmul(R, U * tf.ones_like(R))
            sigmas = tf.matmul(tf.matmul(newU, tf.matmul(Sscale, Sscale)), newU, transpose_b=True)

            theta = tf.layers.dense(x3, 1, activation=tf.nn.tanh) * 180.
            # theta = tf.concat([theta]*nb_landmarks, axis=-1)

        return mus, sigmas, theta

    @auto_reuse_variable_scope
    def encoder(self, img, mask):
        assert img is not None
        assert mask is not None
        with argscope([Conv2D, Conv2DTranspose], activation=INLReLU):
            x1 = Conv2D('mask_conv0_0', mask, NF, 7, strides=2, activation=INLReLU)
            x1 = Conv2D('mask_conv0_1', x1, NF, 3, strides=1, activation=INLReLU)

            x2 = Conv2D('img_conv0_0', img, NF, 7, strides=2, activation=INLReLU)
            x2 = Conv2D('img_conv0_1', x2, NF, 3, strides=1, activation=INLReLU)

            dim_list = [2, 2, 4, 4, 8]
            for i in range(5):
                x1 = Conv2D('mask_conv{}_0'.format(str(i + 1)),
                            x1, NF * dim_list[i], 3, strides=2, activation=INLReLU)
                x1 = Conv2D('mask_conv{}_1'.format(str(i + 1)),
                            x1, NF * dim_list[i], 3, strides=1, activation=INLReLU)

                x2 = Conv2D('img_conv{}_0'.format(str(i + 1)),
                            x2, NF * dim_list[i], 3, strides=2, activation=INLReLU)
                x2 = Conv2D('img_conv{}_1'.format(str(i + 1)),
                            x2, NF * dim_list[i], 3, strides=1, activation=INLReLU)

            x1 = tf.nn.max_pool(x1, [1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID')
            x1 = tf.reshape(x1, [-1, NF * 8])

            x2 = tf.nn.max_pool(x2, [1, 4, 4, 1], strides=[1, 1, 1, 1], padding='VALID')
            x2 = tf.reshape(x2, [-1, NF * 8])

            z_p = tf.layers.dense(x1, STYLE_DIM_z2 * 2)
            z_p = tf.reshape(z_p, [-1, 1, 1, STYLE_DIM_z2 * 2], name='z_p')

            z_f = tf.layers.dense(x2, STYLE_DIM_z2 * 2)
            z_f = tf.reshape(z_f, [-1, 1, 1, STYLE_DIM_z2 * 2], name='z_f')

        return z_p, z_f

    @auto_reuse_variable_scope
    def get_zm(self, img):
        assert img is not None
        with argscope([Conv2D, Conv2DTranspose], activation=INLReLU):
            x = Conv2D('conv0_0', img, NF, 7, strides=2, activation=INLReLU)

            x = Conv2D('conv1_0', x, NF * 2, 2, strides=2, activation=INLReLU)

            x = Conv2D('conv2_0', x, NF * 4, 2, strides=2, activation=INLReLU)

            x = Conv2D('conv3_0', x, NF * 4, 2, strides=2, activation=INLReLU)

            x = Conv2D('conv4_0', x, NF * 8, 2, strides=2, activation=INLReLU)
            # x = Conv2D('conv4_1', x, NF * 4, 3, strides=1, activation=INLReLU)
            #
            # x = Conv2D('conv5_0', x, NF*8, 3, strides=2, activation=INLReLU)
            # x = Conv2D('conv5_1', x, NF * 8, 3, strides=1, activation=INLReLU)
            x = tf.nn.max_pool(x, [1, 4, 4, 1], strides=[1, 2, 2, 1], padding='VALID')
            sz = x.shape.as_list()
            x = tf.reshape(x, [-1, sz[1] * sz[2] * sz[3]])
            mean = tf.layers.dense(x, STYLE_DIM_z2, name='fcmean')
            mean = tf.reshape(mean, [-1, 1, 1, STYLE_DIM_z2])
            var = tf.layers.dense(x, STYLE_DIM_z2, name='fcvar')
            var = tf.reshape(var, [-1, 1, 1, STYLE_DIM_z2])

        return mean, var

    @auto_reuse_variable_scope
    def volume_renderer(self,
                        mus,
                        sigmas,
                        z_f,
                        include_feature=True,
                        rez=16,
                        M_S=64,
                        ray_len=2.,
                        f=1.,
                        rotx=0.,
                        roty=0.,
                        rotz=0.,
                        Lx=Lx,
                        Ld=Ld,
                        feat_dim=M_F):

        # cast rays and sample points
        points, directions = point_sampling(rez=rez, M_S=M_S, ray_len=ray_len, f=f)

        # get rotation matrix
        R = get_rotation(rotx, roty, rotz)

        # rotation points and viewing directions
        points = tf.matmul(points, R)
        directions = tf.matmul(directions, R)

        # duplicate points and directions into correct shapes
        points = tf.expand_dims(points, axis=0)
        directions = tf.expand_dims(directions, axis=0)
        points = tf.tile(points, (tf.shape(z_f)[0], 1, 1, 1))
        directions = tf.tile(directions, (tf.shape(z_f)[0], 1, points.shape[-2], 1))

        # get volume densities
        densities = get_density(mus, sigmas, points)

        if include_feature:
            if use_pencoding:
                # apply positional encodings
                points = positional_encoding(points, L=Lx)
                directions = positional_encoding(directions, L=Ld)

            # stack z_f, points, directions together
            z_f = tf.tile(z_f, (1, points.shape[1], points.shape[2], 1))
            net_input = tf.concat([z_f, points, directions], axis=-1)

            # pass into the MLPs to get feature vectors
            features = self.get_features(net_input, output_d=feat_dim)

            # get feature maps
            feature_maps = numerical_interg(sigmas=densities, fs=features)
            feature_maps = tf.reshape(feature_maps, [-1, rez, rez, feat_dim])

        else:
            # get feature maps
            feature_maps = numerical_interg(sigmas=densities, fs=None, include_feature=False)
            feature_maps = tf.reshape(feature_maps, [-1, rez, rez, nb_landmarks])
            for _ in range(int(np.log2(SHAPE // rez))):
                feature_maps = UpSampling2D(interpolation='bilinear')(feature_maps)

        feature_maps = tf.transpose(feature_maps, (0, 2, 1, 3))
        return feature_maps

    @auto_reuse_variable_scope
    def generator(self, pe, chan=3):
        with argscope([Conv2D, Conv2DTranspose], activation=INLReLU):
            szs = pe[0].shape.as_list()
            in0 = pe[0]
            x = Conv2DTranspose('deconv0_0', in0, NF * 4, 3, strides=1)
            x = Conv2DTranspose('deconv0_1', x, NF * 4, 3, strides=2)
            in1 = tf.concat([x, pe[1]], axis=-1)
            x = Conv2DTranspose('deconv1_0', in1, NF * 2, 3, strides=1)
            x = Conv2DTranspose('deconv1_1', x, NF * 2, 3, strides=2)
            in2 = tf.concat([x, pe[2]], axis=-1)
            x = Conv2DTranspose('deconv2_0', in2, NF, 3, strides=1)
            x = Conv2DTranspose('deconv2_1', x, NF, 3, strides=2)
            in3 = tf.concat([x, pe[3]], axis=-1)
            x = tf.pad(in3, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
            x = Conv2D('convlast', x, chan, 7, padding='VALID', activation=tf.tanh)
        return x

    @auto_reuse_variable_scope
    def neural_renderer(self, f_img, input_dim=128, output_dim=3, min_feat=32, img_size=256):
        """
        Implements a neural rendering operator

        :param f_img: input feature image
        :param input_dim: the number of features in the input
        :param output_dim: the number of features in the output
        :param min_feat: the minimum feature during convolution
        :param img_size: output image size

        :return: output image
        """
        rgb = Conv2D('conv0_nr_rgb', f_img, output_dim, 3, padding='SAME')
        nb_blocks = int(math.log2(img_size) - 4)

        for i in range(nb_blocks):
            f_img = UpSampling2D(interpolation='nearest')(f_img)
            f_img = Conv2D('conv{}_nr_f'.format(str(i)),
                           f_img, max(input_dim // (2 ** (i + 1)), min_feat), 3,
                           activation=tf.nn.leaky_relu, padding='SAME')
            add_rgb = Conv2D('conv{}_nr_rgb'.format(str(i+1)),
                             f_img, output_dim, 3, padding='SAME')
            rgb = UpSampling2D(interpolation='bilinear')(rgb)
            rgb = add_rgb + rgb

        output = tf.nn.sigmoid(rgb)

        return output

    @auto_reuse_variable_scope
    def sr_net(self, m, chan=1):
        assert m is not None
        with argscope([Conv2D, Conv2DTranspose], activation=INLReLU):
            m = tf.keras.layers.UpSampling2D(2, data_format=None)(m)
            l = (LinearWrap(m)
                 .Conv2D('conv0_sr', NF, 7, padding='SAME')
                 .Conv2D('conv1_sr', NF, 3, padding='SAME')
                 .Conv2D('conv2_sr', chan, 7, padding='SAME', activation=tf.tanh, use_bias=True)())
        return l

    @staticmethod
    def build_adain_res_block(x, musigma, name, chan, first=False):
        with tf.variable_scope(name), \
             argscope([Conv2D], kernel_size=3, strides=1):
            musigma = tf.reshape(musigma, [-1, 2, chan])
            mu = musigma[:, 0]
            mu = tf.reshape(mu, [-1, 1, 1, chan])
            sigma = musigma[:, 1]
            sigma = tf.reshape(sigma, [-1, 1, 1, chan])

            input = x
            x = Conv2D('conv0', x, chan, 3, activation=tf.nn.leaky_relu, strides=1)
            x = tf.add(mu * InstanceNorm('in_0', x, use_affine=False), sigma, name='adain_0')

            x = Conv2D('conv1', x, chan, 3, activation=tf.nn.leaky_relu, strides=1)
            x = tf.add(mu * InstanceNorm('in_1', x, use_affine=False), sigma, name='adain_1')

            return x + input

            # x =(LinearWrap(x)
            #         .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
            #         .Conv2D('conv0', chan, 3, padding='VALID')
            #         .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
            #         .Conv2D('conv1', chan, 3, padding='VALID', activation=tf.identity)())
            # return GroupNorm(x) + input

    @staticmethod
    def apply_noise(x, name):
        with tf.variable_scope(name):
            sp = x.shape.as_list()
            noise = tf.random_normal([tf.shape(x)[0], sp[1], sp[2], 1], mean=0, stddev=0.5)
            noise = tf.concat([noise] * sp[3], axis=-1)
            gamma = tf.get_variable('gamma', [sp[3]], trainable=True)
            gamma = tf.reshape(gamma, [1, 1, 1, sp[3]])
        return x + gamma * noise

    @staticmethod
    def build_res_block(x, name, chan):
        with tf.variable_scope(name), \
             argscope([Conv2D], kernel_size=3, strides=1):
            input = x
            x = Conv2D('conv0', x, chan, 3, activation=INLReLU, strides=1)
            x = Conv2D('conv1', x, chan, 3, activation=INLReLU, strides=1)

            return x + input

            # x =(LinearWrap(x)
            #         .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
            #         .Conv2D('conv0', chan, 3, padding='VALID')
            #         .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
            #         .Conv2D('conv1', chan, 3, padding='VALID', activation=tf.identity)())
            # return GroupNorm(x) + input

    @auto_reuse_variable_scope
    def z_reconstructer(self, musigma, dimz, name='z_reconstructer'):
        with tf.variable_scope(name), \
             argscope([tf.layers.dense],
                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
            musigma = tf.layers.flatten(musigma)
            x = tf.layers.dense(musigma, dimz, activation=tf.nn.leaky_relu, name='linear_0')
            x = tf.layers.dense(x, dimz, activation=tf.nn.leaky_relu, name='linear_1')
            x = tf.reshape(x, [-1, 1, 1, dimz])
        return x

    @auto_reuse_variable_scope
    def discrim_enc(self, img):
        with argscope(Conv2D, activation=INLReLU, kernel_size=3, strides=2):
            l1 = Conv2D('conv0', img, NF, 7, strides=1, activation=tf.nn.leaky_relu)
            l2 = Conv2D('conv1', l1, NF * 2)
            l3 = Conv2D('conv2', l2, NF * 4)
            features = Conv2D('conv3', l3, NF * 8)
        return features, [l1, l2, l3, features]

    @auto_reuse_variable_scope
    def discrim_classify(self, img):
        with argscope(Conv2D, activation=INLReLU, kernel_size=3, strides=2):
            l1 = Conv2D('conv3', img, NF * 8, strides=2)
            l2 = tf.reduce_mean(l1, axis=[1, 2])
            l3 = tf.layers.dense(l2, 1, activation=tf.identity, name='imisreal')
            return l3, [l1]

    @auto_reuse_variable_scope
    def discrim_patch_classify(self, img):
        with argscope(Conv2D, activation=INLReLU, kernel_size=3, strides=2):
            l1 = Conv2D('conv3', img, NF * 8, strides=2)
            l2 = Conv2D('conv4', l1, 1, strides=1, activation=tf.identity, use_bias=True)
            return l2, [l1]

    @auto_reuse_variable_scope
    def style_encoder(self, x):
        chan = NF
        with tf.variable_scope('senc'), argscope([tf.layers.conv2d, Conv2D]):
            x = tpad(x, pad=1, mode='reflect')
            x = tf.layers.conv2d(x, chan, kernel_size=3, strides=2, activation=INLReLU, name='conv_0')

            for i in range(3):
                x = tpad(x, pad=1, mode='reflect')
                x = tf.layers.conv2d(x, chan * 2, kernel_size=3, strides=2, activation=INLReLU,
                                     name='conv_%d' % (i + 1))
                chan *= 2

            x = tf.layers.conv2d(x, chan, kernel_size=3, strides=2, activation=INLReLU, name='conv_%d' % (i + 2))
            x = tf.layers.conv2d(x, chan, kernel_size=3, strides=2, activation=INLReLU, name='conv_%d' % (i + 3))

            x = tf.layers.flatten(x)
            mean = tf.layers.dense(x, STYLE_DIM, name='fcmean')
            mean = tf.reshape(mean, [-1, 1, 1, STYLE_DIM])
        return mean

    def get_feature_match_loss(self, feats_real, feats_fake, name):
        losses = []

        for i, (real, fake) in enumerate(zip(feats_real, feats_fake)):
            with tf.variable_scope(name):
                fm_loss_real = tf.get_variable('fm_real_%d' % i,
                                               real.shape[1:],
                                               dtype=tf.float32,
                                               # expected_shape=real.shape[1:],
                                               trainable=False)

                ema_real_op = moving_averages.assign_moving_average(fm_loss_real,
                                                                    tf.reduce_mean(real, 0), 0.99, zero_debias=False,
                                                                    name='EMA_fm_real_%d' % i)

            loss = tf.reduce_mean(tf.squared_difference(
                fm_loss_real,
                tf.reduce_mean(fake, 0)),
                name='mse_feat_' + real.op.name)

            losses.append(loss)

            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_real_op)

        ret = tf.add_n(losses, name='feature_match_loss')
        return ret

    def build_graph(self, img, box, mask, bbx, z, z2):

        with tf.name_scope('preprocess'):
            # img_crop = tf.multiply(img, 1 - box)
            img = (img / 127.5 - 1.0)
            mask = img2mask(img)
            # img_crop = (img_crop / 127.5 - 1.0)
            # bin_mask = mask / 255.
            # mask = (mask / 127.5 - 1.0)
            # box_ = (box - 0.5) * 2
            randomrot = tf.random.uniform(shape=[1], minval=0., maxval=angle * 1.)[0]
            randomrot = tf.identity(randomrot, name='randomrot')
            print(randomrot)

            # zrs = tf.zeros_like(randomrot)
            # ons = tf.ones_like(randomrot)

        def vizN(name, a, b):
            with tf.name_scope(name):
                m = tf.concat(a, axis=2)
                m = tf.image.grayscale_to_rgb(m)
                m = (m + 1.0) * 127.5
                m = tf.clip_by_value(m, 0, 255)

                gm = tf.concat(b, axis=2) * 255
                m = tf.concat([m] + [gm], axis=2)
                # show = tf.concat([m, vid], axis=1)
                show = tf.cast(m, tf.uint8, name='viz')
            tf.summary.image(name, show, max_outputs=50)

        # use the initializers from torch
        with argscope([Conv2D, Conv2DTranspose, tf.layers.conv2d]):
            # Let us encode the images

            with tf.variable_scope('pred_gaussians'):
                mucann, sigmacann = self.get_musigma()
                mucann = tf.identity(mucann, name='mucann')
                sigmacann = tf.identity(sigmacann, name='sigmacann')
            with tf.variable_scope('gen'):
                with tf.variable_scope('foreground_encoder'):
                    z_p, z_f = self.encoder(img, mask)
                    # z_p = tf.identity(z_p, name='z_p')
                    # z_f = tf.identity(z_f, name='z_f')
                with tf.variable_scope('gen_gaussians'):
                    mus, sigmas, thetas = self.transform_mu_sigma(z_p, mucann, sigmacann)
                    zrs = tf.zeros_like(thetas)
                    mus = tf.identity(mus, name='mu3d')
                    sigmas = tf.identity(sigmas, name='sigma3d')
                    thetas = tf.identity(thetas, name='theta3d')

                    # apply rotations
                    mus_org, sigmas_org = apply_rotation(mus, sigmas, zrs, thetas, zrs)
                    mus_org = tf.identity(mus_org, name='mus_org')
                    sigmas_org = tf.identity(sigmas_org, name='sigmas_org')
                    # print(mus_org)
                    # print(sigmas_org)
                    mus_rot, sigmas_rot = apply_rotation(mus, sigmas, zrs, thetas + randomrot, zrs)
                    mus_rot = tf.identity(mus_rot, name='mus_rot')
                    sigmas_rot = tf.identity(sigmas_rot, name='sigmas_rot')

                    # pose_embeddings = get_landmarks(mus, sigmas, zrs, thetas, zrs, 0, 0, -2.)
                    # pose_embeddings_rot = get_landmarks(mus, sigmas, zrs, thetas + randomrot, zrs, 0, 0, -2.)
                    # pose_embeddings_cann = get_landmarks(mucann, sigmacann, zrs, zrs, zrs, 0, 0, -2.)
                with tf.variable_scope('gen_img'):
                    feature_img = self.volume_renderer(mus_org, sigmas_org, z_f, M_S=M_S)
                    gen_img = self.neural_renderer(feature_img, input_dim=M_F)
                    gen_img = gen_img * 2.0 - 1.  # value range -1 to 1
                    gen_mask = img2mask(gen_img)
                    gen_img = tf.identity(gen_img, name='gen_img')
                    # print(gen_img)
                    density_map = self.volume_renderer(mus_org, sigmas_org, z_f,
                                                       M_S=M_S,
                                                       rez=16, include_feature=False)
                    density_map = tf.reduce_sum(density_map, axis=-1)
                    density_map = tf.expand_dims(density_map, axis=-1, name='density_map')
                    # print(density_map)

                    # print()
                    # print(randomrot)
                    # print(mus_rot)
                    # print(sigmas_rot)
                    feature_img_rot = self.volume_renderer(mus_rot, sigmas_rot, z_f, roty=randomrot,
                                                           M_S=M_S)
                    gen_img_rot = self.neural_renderer(feature_img_rot, input_dim=M_F)
                    gen_img_rot = gen_img_rot * 2.0 - 1.  # value range -1 to 1
                    gen_mask_rot = img2mask(gen_img_rot)
                    gen_img_rot = tf.identity(gen_img_rot, name='gen_img_rot')
                    density_map_rot = self.volume_renderer(mus_rot, sigmas_rot, z_f,
                                                           M_S=M_S,
                                                           roty=randomrot, rez=16, include_feature=False)
                    density_map_rot = tf.reduce_sum(density_map_rot, axis=-1)
                    density_map_rot = tf.expand_dims(density_map_rot, axis=-1, name='density_map_rot')

                ##cycle
                with tf.variable_scope('foreground_encoder'):
                    z_p_rot, z_f_rot = self.encoder(gen_img_rot, gen_mask_rot)
                    z_p_rot = tf.identity(z_p_rot, name='z_p_rot')
                    z_f_rot = tf.identity(z_f_rot, name='z_f_rot')
                with tf.variable_scope('gen_gaussians'):
                    mus_cyc, sigmas_cyc, thets_cyc = self.transform_mu_sigma(z_p_rot, mucann, sigmacann)
                    mus_cyc, sigmas_cyc = apply_rotation(mus_cyc, sigmas_cyc, zrs, thets_cyc - randomrot, zrs)
                with tf.variable_scope('gen_img'):
                    feature_img_cyc = self.volume_renderer(mus_cyc, sigmas_cyc, z_f_rot, roty=-1.*randomrot,
                                                           M_S=M_S)
                    gen_img_cyc = self.neural_renderer(feature_img_cyc)
                    gen_img_cyc = gen_img_cyc * 2.0 - 1.
                    gen_img_cyc = tf.identity(gen_img_cyc, name='gen_img_cyc')
                    # print(gen_img_cyc)

            # The final discriminator that takes them both
            discrim_out_mask = []
            discrim_fm_real_mask = []
            discrim_fm_fake_mask = []
            with tf.variable_scope('discrim'):
                with tf.variable_scope('discrim_mask'):
                    def downsample(img):
                        return tf.layers.average_pooling2d(img, 3, 2)

                    D_input_real = img
                    D_input_fake = gen_img
                    D_input_fake_rot = gen_img_rot
                    # D_inputs = [D_input_real, D_input_fake]
                    D_inputs = [D_input_real, D_input_fake, D_input_fake_rot]

                    for s in range(DIS_SCALE):
                        with tf.variable_scope('s%d' % s):
                            if s != 0:
                                D_inputs = [downsample(im) for im in D_inputs]

                            # mask_s, mask_recon_s = D_inputs
                            mask_s, mask_recon_s, mask_recon_s_rot = D_inputs

                            with tf.variable_scope('Ax'):
                                Ax_feats_real, Ax_fm_real = self.discrim_enc(mask_s)
                                Ax_feats_fake, Ax_fm_fake = self.discrim_enc(mask_recon_s)
                                Ax_feats_fake_rot, Ax_fm_fake_rot = self.discrim_enc(mask_recon_s_rot)

                            with tf.variable_scope('Ah'):
                                Ah_dis_real, Ah_fm_real = self.discrim_patch_classify(Ax_feats_real)
                                Ah_dis_fake, Ah_fm_fake = self.discrim_patch_classify(Ax_feats_fake)
                                Ah_dis_fake_rot, Ah_fm_fake_rot = self.discrim_patch_classify(Ax_feats_fake_rot)

                            discrim_out_mask.append((Ah_dis_real, Ah_dis_fake))
                            discrim_out_mask.append((Ah_dis_real, Ah_dis_fake_rot))

                            discrim_fm_real_mask += Ax_fm_real + Ah_fm_real + Ax_fm_real + Ah_fm_real
                            discrim_fm_fake_mask += Ax_fm_fake + Ah_fm_fake + Ax_fm_fake_rot + Ah_fm_fake_rot

                            # discrim_fm_real_mask += Ax_fm_real + Ah_fm_real
                            # discrim_fm_fake_mask += Ax_fm_fake + Ah_fm_fake

            # vizN('A_recon', [mask, gen_mask, gen_mask_rot], [colorize_landmark_maps(pose_embeddings[-1]),
            #                                                  colorize_landmark_maps(pose_embeddings_rot[-1]),
            #                                                  colorize_landmark_maps(pose_embeddings_cann[-1]),
            #                                                  colorize_landmark_maps(pose_embeddings_cyc[-1])])

        def LSGAN_hinge_loss(real, fake):
            d_real = tf.reduce_mean(-tf.minimum(0., tf.subtract(real, 1.)), name='d_real')
            d_fake = tf.reduce_mean(-tf.minimum(0., tf.add(-fake, -1.)), name='d_fake')
            d_loss = tf.multiply(d_real + d_fake, 0.5, name='d_loss')

            g_loss = tf.reduce_mean(-fake, name='g_loss')
            # add_moving_summary(g_loss)
            return g_loss, d_loss

        def IoU_loss(mask, gen_mask):
            mask = (mask + 1) * 0.5
            gen_mask = (gen_mask + 1) * 0.5
            intersection = tf.reduce_sum(tf.multiply(mask, gen_mask), name='intersection')
            union = tf.add(tf.reduce_sum(tf.abs(mask - gen_mask)), intersection, name='union')
            IoU_loss = tf.divide(intersection, union, name='iou_loss')
            return IoU_loss

        with tf.name_scope('losses'):
            with tf.name_scope('mask_losses'):
                with tf.name_scope('GAN_loss'):
                    # gan loss
                    G_loss_img, D_loss_img = zip(*[LSGAN_hinge_loss(real, fake) for real, fake in discrim_out_mask])
                    G_loss_img = tf.add_n(G_loss_img, name='img_lsgan_loss') / len(G_loss_img)
                    G_loss_img = tf.identity(G_loss_img, 'img_Gen_loss')
                    print(G_loss_img)
                    D_loss_img = tf.add_n(D_loss_img, name='img_Disc_loss')
                with tf.name_scope('FM_loss'):
                    FM_loss_mask = self.get_feature_match_loss(discrim_fm_real_mask, discrim_fm_fake_mask,
                                                               'img_fm_loss')
                with tf.name_scope('recon_loss'):
                    img_recon_loss = tf.reduce_mean(tf.abs(img - gen_img), name='img_recon_loss')
                    mask_recon_loss = tf.reduce_mean(tf.abs(mask - gen_mask), name='mask_recon_loss')
                    # mask_recon_loss = tf.subtract(1., IoU_loss(mask, gen_mask), name='mask_recon_loss')
                # with tf.name_scope('m_recon_loss_bis'):
                #     m_b = tf.reduce_sum(pose_embeddings[-1], axis=-1, keepdims=True)
                #     m_r = tf.reduce_sum(pose_embeddings_rot[-1], axis=-1, keepdims=True)
                #     mask_recon_loss_bis = tf.reduce_mean(tf.abs((mask + 1) * 0.5 - m_b), name='mask_recon_loss_bis')
                #     mask_recon_loss_rot_bis = tf.reduce_mean(tf.abs((gen_mask_rot + 1) * 0.5 - m_r),
                #                                              name='mask_recon_loss_rot_bis')
                with tf.name_scope('density_loss'):
                    density_loss = tf.reduce_mean(tf.abs(mask * 0.5 - density_map), name='density_loss')
                    rot_density_loss = tf.reduce_mean(tf.abs(gen_mask_rot * 0.5 - density_map_rot),
                                                      name='rot_density_loss')
                with tf.name_scope('cycle_loss'):
                    kl_cycle_loss = kl_div(mus_org, sigmas_org, mus_cyc, sigmas_cyc)
                    kl_cycle_loss = tf.identity(kl_cycle_loss, name='kl_cycle_loss')
                    recon_cycle_loss = tf.reduce_mean(tf.abs(img - gen_img_cyc), name='recon_cycle_loss')
                    # gm_cycle_loss = tf.reduce_mean(tf.abs(pose_embeddings[-1] - pose_embeddings_cyc[-1]),
                    #                                name='cycle_loss')

        self.g_loss = G_loss_img + LAMBDA * FM_loss_mask + LAMBDA_m * img_recon_loss \
                      + LAMBDA_kl_cyc * kl_cycle_loss + LAMBDA_recon_cyc * recon_cycle_loss \
                      + LAMBDA_dens * density_loss + LAMBDA_dens_rot * rot_density_loss
                      # + LAMBDA_gm * mask_recon_loss_bis + LAMBDA_gm_rot * mask_recon_loss_rot_bis + \

        # self.g_loss = LAMBDA_gm * mask_recon_loss_bis
        self.d_loss = D_loss_img
        self.collect_variables('gen', 'pred_gaussians', 'discrim')
        # self.collect_variables('gen', 'discrim')

        add_moving_summary(G_loss_img, D_loss_img, FM_loss_mask,
                           mask_recon_loss, img_recon_loss,
                           density_loss, rot_density_loss,
                           # mask_recon_loss_bis,
                           # mask_recon_loss_rot_bis,
                           kl_cycle_loss, recon_cycle_loss)

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=LR, trainable=False)
        lrm = ratio * lr
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3), tf.train.AdamOptimizer(lrm, beta1=0.5, epsilon=1e-3)
        # return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def export_compact_a(model_path):
    """Export trained model to use it as a frozen and pruned inference graph in
       mobile applications. """
    pred_config = PredictConfig(
        session_init=SmartInit(model_path),
        model=Model(),
        input_names=['img'],
        output_names=['gen/gen_gaussians/mu3d', 'gen/gen_gaussians/sigma3d',
                      'gen/gen_gaussians/theta3d', 'gen/foreground_encoder/z_f'])
    ModelExporter(pred_config).export_compact(os.path.join(os.path.dirname(model_path), 'frozen_model_a.pb'))


def export_compact_b(model_path):
    """Export trained model to use it as a frozen and pruned inference graph in
       mobile applications. """
    pred_config = PredictConfig(
        session_init=SmartInit(model_path),
        model=Model(),
        input_names=['gen/gen_gaussians/mus_rot', 'gen/gen_gaussians/sigmas_rot',
                     'gen/foreground_encoder/z_f', 'preprocess/randomrot'],
        output_names=['gen/gen_img/gen_img_rot'])
    ModelExporter(pred_config).export_compact(os.path.join(os.path.dirname(model_path), 'frozen_model_b.pb'))


def export_compact(model_path):
    """Export trained model to use it as a frozen and pruned inference graph in
       mobile applications. """
    pred_config = PredictConfig(
        session_init=SmartInit(model_path),
        model=Model(),
        input_names=['img', 'preprocess/randomrot'],
        output_names=['gen/gen_img/gen_img_rot', 'gen/gen_img/density_map_rot'])
    ModelExporter(pred_config).export_compact(os.path.join(os.path.dirname(model_path), 'frozen_model.pb'))



def get_data_synth(isTrain=True):
    def get_images(dir1, image_path, istrain):
        files = sorted(glob.glob(os.path.join(dir1, '*.png')))
        np.random.seed(42)
        np.random.shuffle(files)
        lenfiles = len(files)
        #files = files[:int(lenfiles * 0.9)] if istrain == 'train' else files[int(lenfiles * 0.9):]
        print()
        print('Files length' + str(len(files)))
        df = data_loader(files, image_path, SHAPE, STYLE_DIM, STYLE_DIM_z2, channel=3, shuffle=isTrain)
        return df

    path_type = 'train' if isTrain else 'val'
    path = args.path
    npy_path = os.path.join(args.data, path_type, args.dataset)
    df = get_images(npy_path, path, path_type)
    print()
    print(len(df))
    print()
    df = BatchData(df, BATCH if isTrain else TEST_BATCH)
    df = PrefetchDataZMQ(df, 2 if isTrain else 1)
    return df


class VisualizeTestSet(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['input', 'template', 'mask', 'bbx', 'z1', 'z2'], ['A_recon/viz'])

    def _before_train(self):
        global args
        self.val_ds = datagetter(isTrain=False)
        self.val_ds.reset_state()

    def _trigger(self):
        idx = 0
        for iA, tA, mA, bA, z1, z2 in self.val_ds:
            vizA = self.pred(iA, tA, mA, bA, z1, z2)
            self.trainer.monitors.put_image('testA-{}'.format(idx), vizA[0])
            # self.trainer.monitors.put_image('testB-{}'.format(idx), vizB)
            idx += 1


class TestSetLosses(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['input', 'template', 'mask', 'bbx', 'z1', 'z2'],
            ['losses/mask_losses/kl_cycle_loss/cycle_loss'])

    def _before_train(self):
        global args
        self.losses = []
        self.val_ds = datagetter(isTrain=False)
        self.val_ds.reset_state()

    def _trigger(self):
        cycle_losses = []
        print(len(self.val_ds))
        for iA, tA, mA, bA, z1, z2 in self.val_ds:
            cycle_loss = self.pred(iA, tA, mA, bA, z1, z2)
            cycle_losses.append(cycle_loss)
        self.losses.append([np.mean(cycle_losses)])

        print('Test set losses:')
        print('KL cycle loss is: ' + str(np.mean(cycle_losses)))
        print()

    def after_train(self):
        import csv
        import matplotlib.pyplot as plt
        losses = np.array(self.losses)
        reconst_l = losses[:, 0]
        os.makedirs(os.path.join(logdir, 'Loss'))
        with open(os.path.join(logdir, 'Loss', 'losses.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(reconst_l)

        epochs = range(len(reconst_l))
        plt.plot(epochs, reconst_l, 'g', label='Val KL Cycle Loss')
        plt.title('Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(logdir, 'Loss', 'losses.png'))


class RecordResults(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['img', 'template', 'mask', 'bbx', 'z1', 'z2'],
            ['gen/gen_img/gen_img:0',
             'gen/gen_img/density_map:0',
             'gen/gen_img/gen_img_rot:0',
             'gen/gen_img/density_map_rot:0',
             'gen/gen_img_1/gen_img_cyc:0'])

    def _before_train(self):
        global args
        self.val_ds = datagetter(isTrain=False)
        self.val_ds.reset_state()

    def _trigger(self):
        import matplotlib.pyplot as plt

        print(len(self.val_ds))
        num_epoch = self.epoch_num
        # org_path = os.path.join(logdir, 'results', 'org')
        result_path = os.path.join(logdir, 'results', 'epoch ' + str(num_epoch))
        # rand_rot_path = os.path.join(logdir, 'results', 'epoch ' + str(num_epoch), 'rand_rot')
        # cyc_reconst_path = os.path.join(logdir, 'results', 'epoch ' + str(num_epoch), 'cyc_reconst')

        os.makedirs(result_path)

        num = 1
        for iA, tA, mA, bA, z1, z2 in self.val_ds:
            while num < 2:
                org_img = (np.squeeze(iA))
                gen_img, density_map, gen_img_rot, density_map_rot, gen_img_cyc = self.pred(iA, tA, mA, bA, z1, z2)

                # print('gen mask')
                # print(np.unique(gen_mask))
                gen_img = (np.squeeze((gen_img + 1) * 127.5)).astype(np.uint8)
                gen_img_rot = (np.squeeze((gen_img_rot + 1) * 127.5)).astype(np.uint8)
                gen_img_cyc = (np.squeeze((gen_img_cyc + 1) * 127.5)).astype(np.uint8)
                # print(np.unique(gen_mask))
                density_map = (np.squeeze(density_map * 255.)).astype(np.uint8)
                density_map_rot = (np.squeeze(density_map_rot * 255.)).astype(np.uint8)

                plt.imsave(os.path.join(result_path, 'org.png'), org_img[0, :, :])
                plt.imsave(os.path.join(result_path, 'gen_img.png'), gen_img[0, :, :])
                plt.imsave(os.path.join(result_path, 'gen_img_rot.png'), gen_img_rot[0, :, :])
                plt.imsave(os.path.join(result_path, 'gen_img_cyc.png'), gen_img_cyc[0, :, :])
                plt.imsave(os.path.join(result_path, 'density_map.png'), density_map[0, :, :], cmap='gray')
                plt.imsave(os.path.join(result_path, 'density_map_rot.png'), density_map_rot[0, :, :], cmap='gray')

                num += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', required=True,
        help='name of the class used')
    parser.add_argument(
        '--data', required=True,
        help='directory containing bounding box annotations, should contain train, val folders')
    parser.add_argument(
        '--path',
        default='/Users/harrywang/Documents/Brown/Research/Visual Computing Lab/code/Stamps_cvpr/dataset/CocoPNG/',
        help='the path that contains the raw coco JPEG images')
    parser.add_argument('--gpu', default='0', help='nb gpus to use, to use four gpus specify 0,1,2,3')
    parser.add_argument('--dataloader', help='which dataloader to use',
                        default='CocoLoader_center_scaled_keep_aspect_ratio_adacrop', type=str)
    parser.add_argument('--getdata', help='which datagetter to use',
                        default='get_data', type=str)
    parser.add_argument('--nb_epochs', help='hyperparameter', default=200, type=int)
    parser.add_argument('--batch', help='hyperparameter', default=3, type=int)
    parser.add_argument('--testbatch', help='hyperparameter', default=5, type=int)
    parser.add_argument('--usepencoding', help='hyperparameter', default=True, type=bool)
    parser.add_argument('--featuredim', help='hyperparameter', default=32, type=int)
    # parser.add_argument('--usedensity', help='hyperparameter', default=False, type=bool)
    parser.add_argument('--LAMBDA', help='hyperparameter', default=10.0, type=float)
    parser.add_argument('--LAMBDA_m', help='hyperparameter', default=100.0, type=float)
    parser.add_argument('--LAMBDA_dens', help='hyperparameter', default=100.0, type=float)
    parser.add_argument('--LAMBDA_dens_rot', help='hyperparameter', default=100.0, type=float)
    parser.add_argument('--LAMBDA_kl_cyc', help='hyperparameter', default=5.0, type=float)
    parser.add_argument('--LAMBDA_recon_cyc', help='hyperparameter', default=5.0, type=float)
    parser.add_argument('--LAMBDA_KLm', help='hyperparameter', default=0.5, type=float)
    parser.add_argument('--LAMBDA_KLl', help='hyperparameter', default=1., type=float)
    parser.add_argument('--ratio', help='hyperparameter', default=1., type=float)
    parser.add_argument('--angle', help='hyperparameter', default=90, type=int)
    parser.add_argument('--nb_l', help='hyperparameter', default=5, type=int)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    LAMBDA = args.LAMBDA
    LAMBDA_m = args.LAMBDA_m
    LAMBDA_dens_rot = args.LAMBDA_dens_rot
    LAMBDA_dens = args.LAMBDA_dens
    LAMBDA_kl_cyc = args.LAMBDA_kl_cyc
    LAMBDA_recon_cyc = args.LAMBDA_recon_cyc
    ratio = args.ratio
    angle = args.angle
    BATCH = args.batch
    TEST_BATCH = args.testbatch
    nb_epochs = args.nb_epochs
    use_pencoding = args.usepencoding
    M_F = args.featuredim
    # use_density = args.usedensity
    # print('Use density? ' + str(use_density))
    data_loader = getattr(sys.modules[__name__], args.dataloader)
    nb_landmarks = args.nb_l
    # COLORS = get_n_colors(nb_landmarks, pastel_factor=0.0)
    # TODO: it was nr_tower = max(get_num_gpu(), 1)
    nr_tower = 2
    # nr_tower = max(get_num_gpu(), 1)

    BATCH = BATCH // nr_tower
    mod = sys.modules['__main__']
    basename = os.path.basename(mod.__file__).split('.')[0]
    logdir = os.path.join('train_log', args.dataset, basename,
                          'Ep%d_r%.1f_L%d_Lm%d_Lgm%d_Lgmrot%d_Lklcyc%d_Lreccyc%d_ange%d_nbl%d' % (nb_epochs, ratio, LAMBDA,
                                                                                      LAMBDA_m, LAMBDA_dens,
                                                                                      LAMBDA_dens_rot,
                                                                                      LAMBDA_kl_cyc, LAMBDA_recon_cyc,
                                                                                      angle, nb_landmarks),
                          datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger.set_logger_dir(logdir)
    from shutil import copyfile

    namefile = os.path.basename(os.path.realpath(__file__))
    copyfile(namefile, os.path.join(logdir, namefile))
    datagetter = getattr(sys.modules[__name__], args.getdata)
    df = datagetter()
    df = PrintData(df)
    data = QueueInput(df)

    GANTrainer(data, Model(), num_gpu=nr_tower).train_with_defaults(
        callbacks=[
            CheckNumerics(),
            PeriodicTrigger(ModelSaver(), every_k_epochs=nb_epochs),
            PeriodicTrigger(RecordResults(), every_k_epochs=20),
            ScheduledHyperParamSetter(
                'learning_rate',
                [(int(nb_epochs / 2), LR), (nb_epochs, 0)], interp='linear'),
            # PeriodicTrigger(VisualizeTestSet(), every_k_epochs=50),
        ],
        max_epoch=nb_epochs,
        steps_per_epoch=data.size() // nr_tower,
        # steps_per_epoch=1,
        session_init=SaverRestore(args.load) if args.load else None,
    )

    # export_compact_a(os.path.join(logdir, 'checkpoint'))
    # export_compact_b(os.path.join(logdir, 'checkpoint'))
    export_compact(os.path.join(logdir, 'checkpoint'))
