import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import UpSampling2D
import math

tf.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()

M_S = 25
M_F = 32
batch_size = 5
ray_batch_size = 16


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
    densities = tf.squeeze(densities, [0, 4, 5])
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
    output_list.append(tf.zeros_like(sigmas[:, 0, :]))
    output_list.append(1 - alphas[:, 0, :])
    for i in range(2, M_S):
        output_list.append(tf.multiply(output_list[-1], 1 - alphas[:, i - 1, :]))
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
    focus = np.array((0, 0, 1. + f))
    direction = tf.linalg.l2_normalize(starts - focus, axis=-1)
    steps = direction * ray_len / M_S

    starts = tf.expand_dims(starts, axis=1)
    steps = tf.expand_dims(steps, axis=1)
    points = starts + steps * tf.reshape(np.arange(M_S).astype(np.float64), [1, -1, 1])
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
            output_list.append(tf.sin(2 ** i * t * math.pi))
            output_list.append(tf.cos(2 ** i * t * math.pi))
        return tf.stack(output_list, axis=-1)

    output = tf.map_fn(fn=encoding, elems=tensor)
    # flatten the last two dimensions
    output = tf.reshape(output, list(output.shape[:-2]) + [L * 2 * 3])

    return output


def get_rotation(rotx, roty, rotz):
    """
    Gets a rotation matrix.

    :param rotx: rotation around x-axis
    :param roty: rotation around y-axis
    :param rotz: rotation around z-axis

    :return: a rotation matrix
    """
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
    R = tf.cast(R, tf.float64)

    return R


# R = rotate(0., 90., 0.)
# point = np.random.random([5, 10, 3])
# print(np.matmul(point, R).shape)


# t = np.random.random([2, 4])
# t = np.array([0, 1, math.pi])
# print(positional_encoding(t, L=3))

# print(point_sampling(M_S=10)[0])

def get_musigma(num):
    # mu = np.random.random((1, num, 1, 3)).astype(np.float32) * 2.0 - 1.
    mu = np.array([1., 0., 0.], np.float32)
    mu = np.reshape(mu, [1, 1, 1, 3])
    # mu = np.zeros((1, 10, 1, 3), dtype=np.float32)
    v0 = np.random.random((1, num, 3)) * 2.0 - 1
    v1 = np.random.random((1, num, 3)) * 2.0 - 1

    v0 = tf.math.l2_normalize(v0, axis=-1)
    v1 = tf.linalg.cross(v0, v1)
    v1 = tf.math.l2_normalize(v1, axis=-1)
    v2 = tf.linalg.cross(v1, v0)
    v0 = tf.reshape(v0, [-1, num, 3, 1])
    v1 = tf.reshape(v1, [-1, num, 3, 1])
    v2 = tf.reshape(v2, [-1, num, 3, 1])

    V = tf.concat([v0, v2, v1], axis=-1)
    u = np.random.random((1, num, 3)) * 0.5 + 1e-2
    U = tf.linalg.diag(u)
    sigma = tf.linalg.matmul(tf.linalg.matmul(V, U), V, transpose_b=True)
    sigma = tf.cast(sigma, tf.float32).numpy()
    # sigma = np.array(sigma, np.float32)

    return mu, sigma


def test_volume_rendering(mu, sigma, rez=16, roty=0., M_S=M_S):
    # mu, sigma = get_musigma(num=2)
    # cast rays and sample points
    points, directions = point_sampling(rez=rez, M_S=M_S)

    # get rotation matrix
    # TODO: rotation x y mixed
    R = get_rotation(0., roty, 0.)

    # rotation points and viewing directions
    points = tf.matmul(points, R)
    directions = tf.matmul(directions, R)

    # duplicate points and directions into correct shapes
    points = tf.expand_dims(points, axis=0)
    directions = tf.expand_dims(directions, axis=0)
    points = tf.tile(points, (1, 1, 1, 1))
    directions = tf.tile(directions, (1, 1, points.shape[-2], 1))

    mu = tf.cast(mu, tf.float64)
    sigma = tf.cast(sigma, tf.float64)
    densities = get_density(mu, sigma, points)
    densities2 = tf.reduce_sum(densities, -1)
    densities2 = tf.expand_dims(densities2, axis=-1)
    # feature_maps = numerical_interg(sigmas=densities, fs=np.random.random((256, 64, 20)), M_S=M_S, include_feature=True)
    feature_maps = numerical_interg(sigmas=densities, fs=None, M_S=M_S, include_feature=False)
    feature_maps = tf.reshape(feature_maps, (1, rez, rez, -1))
    feature_maps = tf.transpose(feature_maps, (0, 2, 1, 3))
    # plt.imshow(feature_maps)
    # plt.show()
    feature_maps2 = numerical_interg(sigmas=densities2, fs=None, M_S=M_S, include_feature=False)
    feature_maps2 = tf.reshape(feature_maps2, (1, rez, rez, -1))
    feature_maps2 = tf.transpose(feature_maps2, (0, 2, 1, 3))
    return feature_maps, feature_maps2


def test_rotation(x):
    tf.compat.v1.enable_eager_execution()
    R = get_rotation(0., 90., 0.)
    print(np.matmul(x, R))


def test_feature_map():
    # get volume densities
    mu, sigma = get_musigma(num=1)
    # mu = tf.reshape(np.array([[0., 0., 0.]]), (1, 1, 1, 3))
    # sigma = np.array([[[0.25934, 0.02336, 0.08002],
    #                   [0.02336, 0.15204, -0.03867],
    #                   [0.08002, -0.03867, 0.30347]]])
    # sigma = tf.reshape(sigma, (1, 1, 3, 3))

    # feature_maps1, feature_maps2 = test_volume_rendering(mu, sigma, rez=16, roty=30., M_S=64)
    feature_maps1, feature_maps2 = test_volume_rendering(mu, sigma, rez=64, roty=0., M_S=64)
    # for i in range(4):
    #     feature_maps1 = UpSampling2D(interpolation='bilinear')(feature_maps1)
    #     feature_maps2 = UpSampling2D(interpolation='bilinear')(feature_maps2)

    # process = psutil.Process(os.getpid())
    # print(process.memory_info().rss / 1024. / 1024.)  # in bytes

    feature_mapsa = tf.squeeze(feature_maps1[:, :, :, 0])
    feature_mapsa = np.array(feature_mapsa)
    plt.imshow(feature_mapsa)
    plt.show()
    # feature_mapsb = tf.squeeze(feature_maps1[:, :, :, 1])
    # feature_mapsb = np.array(feature_mapsb)
    # feature_mapsc = tf.squeeze(feature_maps2)
    # feature_mapsc = np.array(feature_mapsc)
    # feature_mapsd = tf.squeeze(tf.reduce_sum(feature_maps1, -1))
    # feature_mapsd = np.array(feature_mapsd)

    # print(np.max(feature_mapsa))
    # print(np.max(feature_mapsb))
    # print(np.sum(feature_mapsa * feature_mapsb))
    # print(np.mean(np.abs(feature_mapsc - feature_mapsd)))
    # print(np.mean(feature_mapsc - feature_mapsd))
    # print(np.max(feature_mapsd))
    # print()

    # plt.subplot(221)
    # plt.imshow(feature_mapsa)
    # plt.subplot(222)
    # plt.imshow(feature_mapsb)
    # plt.subplot(223)
    # plt.imshow(feature_mapsc)
    # plt.subplot(224)
    # plt.imshow(feature_mapsd)
    # plt.show()


# mu, sigma = get_musigma(num=2)
# test_volume_rendering(mu, sigma)

test_feature_map()
