import cv2
import os
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

nb_landmarks = 6


def get_random_color(pastel_factor = 0.5):
  return [(x+pastel_factor)/(1.0+pastel_factor) for x in [np.random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
  return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
  max_distance = None
  best_color = None
  for i in range(0,100):
    color = get_random_color(pastel_factor = pastel_factor)
    if not existing_colors:
      return color
    best_distance = min([color_distance(color,c) for c in existing_colors])
    if not max_distance or best_distance > max_distance:
      max_distance = best_distance
      best_color = color
  return best_color

def get_n_colors(n, pastel_factor=0.9):
  colors = []
  for i in range(n):
    colors.append(generate_new_color(colors,pastel_factor = 0.9))
  return colors

def colorize_landmark_maps(maps):
  """
  Given BxHxWxN maps of landmarks, returns an aggregated landmark map
  in which each landmark is colored randomly. BxHxWxN
  """
  n_maps = maps.shape[-1]
  COLORS = get_n_colors(n_maps, pastel_factor=0.9)
  # get n colors:
  # colors = get_n_colors(n_maps, pastel_factor=0.0)
  hmaps = [np.expand_dims(maps[..., i], axis=3) * np.reshape(COLORS[i], [1, 1, 1, 3])
           for i in range(n_maps)]
  return np.max(hmaps, axis=0)


def apply_rotation(mus, sigma, rotx, roty, rotz, tx, ty, tz, focal=1.):
    assert mus is not None
    assert sigma is not None
    count = 4
    rotXval = rotx
    rotYval = roty
    rotZval = rotz
    rotX = (rotXval) * np.pi / 180
    rotY = (rotYval) * np.pi / 180
    rotZ = (rotZval) * np.pi / 180
    zr = tf.zeros_like(rotY, np.float64)
    ons = tf.ones_like(rotY, np.float64)

    RX = tf.stack([tf.stack([ons, zr, zr], axis=-1), tf.stack([zr, tf.cos(rotX), -tf.sin(rotX)], axis=-1),
                   tf.stack([zr, tf.sin(rotX), tf.cos(rotX)], axis=-1)], axis=-1)
    RY = tf.stack([tf.stack([tf.cos(rotY), zr, tf.sin(rotY)], axis=-1), tf.stack([zr, ons, zr], axis=-1),
                   tf.stack([-tf.sin(rotY), zr, tf.cos(rotY)], axis=-1)], axis=-1)
    RZ = tf.stack([tf.stack([tf.cos(rotZ), -tf.sin(rotZ), zr], axis=-1),
                   tf.stack([tf.sin(rotZ), tf.cos(rotZ), zr], axis=-1),
                   tf.stack([zr, zr, ons], axis=-1)], axis=-1)

    # Composed rotation matrix with (RX,RY,RZ)
    R = tf.matmul(tf.matmul(RX, RY), RZ)
    # R = tf.stack([R] * nb_landmarks, axis=0)[None, :, :, :]

    transvec = tf.constant(np.array([[tx, ty, tz]]), dtype=tf.float64)
    transvec = tf.stack([transvec] * nb_landmarks, axis=1)
    transvec = transvec[:, :, tf.newaxis, :]

    px = tf.zeros([tf.shape(mus)[0], nb_landmarks])
    py = tf.zeros([tf.shape(mus)[0], nb_landmarks])
    fvs = tf.ones_like(px) * focal
    zv = tf.zeros_like(px)
    ov = tf.ones_like(px)
    K = tf.stack([tf.stack([fvs, zv, zv], axis=-1), tf.stack([zv, fvs, zv], axis=-1),
                  tf.stack([px, py, ov], axis=-1)], axis=-1)
    K = tf.cast(K, tf.float64)
    K = tf.identity(K, name='K')

    R = tf.cast(R, tf.float64) * tf.ones_like(sigma)
    sigma = tf.linalg.matmul(tf.linalg.matmul(R, sigma), R, transpose_b=True)
    mus = tf.cast(mus, tf.float64)
    mus = tf.transpose(tf.linalg.matmul(R, tf.transpose(mus, [0, 1, 3, 2])), [0, 1, 3, 2]) + transvec

    return mus, sigma


def get_landmarks(mus, sigma, rotx, roty, rotz, tx, ty, tz, focal=1., nb_landmarks=6):
    assert mus is not None
    assert sigma is not None
    count = 4
    # nb_landmarks = 6
    rotXval = rotx
    rotYval = roty
    rotZval = rotz
    rotX = (rotXval) * np.pi / 180
    rotY = (rotYval) * np.pi / 180
    rotZ = (rotZval) * np.pi / 180
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
    # R = tf.stack([R] * nb_landmarks, axis=0)[None, :, :, :]

    transvec = tf.constant(np.array([[tx, ty, tz]]), dtype=tf.float64)
    transvec = tf.stack([transvec] * nb_landmarks, axis=1)
    transvec = transvec[:, :, tf.newaxis, :]

    px = tf.zeros([tf.shape(mus)[0], nb_landmarks])
    py = tf.zeros([tf.shape(mus)[0], nb_landmarks])
    fvs = tf.ones_like(px) * focal
    zv = tf.zeros_like(px)
    ov = tf.ones_like(px)
    K = tf.stack([tf.stack([fvs, zv, zv], axis=-1), tf.stack([zv, fvs, zv], axis=-1),
                  tf.stack([px, py, ov], axis=-1)], axis=-1)
    K = tf.cast(K, tf.float64)
    K = tf.identity(K, name='K')

    R = tf.cast(R, tf.float64) * tf.ones_like(sigma)
    sigma = tf.linalg.matmul(tf.linalg.matmul(R, sigma), R, transpose_b=True)
    invsigma = tf.linalg.inv(sigma)
    mus = tf.cast(mus, tf.float64)
    mus = tf.transpose(tf.linalg.matmul(R, tf.transpose(mus, [0, 1, 3, 2])), [0, 1, 3, 2]) + transvec

    M0 = tf.matmul(invsigma, tf.matmul(mus, mus, transpose_a=True))
    M0 = tf.matmul(M0, invsigma, transpose_b=True)
    M1 = (tf.matmul(tf.matmul(mus, invsigma), mus, transpose_b=True) - 1)
    M1 = M1 * invsigma

    M = M0 - M1

    Mtmp = tf.constant(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]]), dtype=tf.float64)
    M = -M + 2 * M * Mtmp[tf.newaxis, tf.newaxis, :, :]
    M33 = tf.gather(tf.gather(M, [0, 1], axis=2), [0, 1], axis=3)
    K33 = tf.gather(tf.gather(K, [0, 1], axis=2), [0, 1], axis=3)
    M31 = tf.gather(tf.gather(M, [0, 1], axis=2), [1, 2], axis=3)
    M23 = tf.gather(tf.gather(M, [0, 2], axis=2), [0, 1], axis=3)
    det_m31 = tf.linalg.det(M31)
    det_m23 = tf.linalg.det(M23)
    det_m33 = tf.linalg.det(M33)
    det_m = tf.linalg.det(M)

    mup0 = tf.squeeze(tf.matmul(K33, tf.stack([det_m31, -det_m23], axis=-1)[:, :, :, tf.newaxis]), axis=-1) / (
        det_m33[:, :, tf.newaxis])
    mup1 = tf.stack([K[:, :, 0, 2], K[:, :, 1, 2]], axis=-1)
    mup = mup0 + mup1

    sigma_w = det_m / det_m33
    sigma_w = sigma_w[:, :, None, None]
    invm33 = tf.linalg.inv(M33)
    sigmap = -sigma_w * invm33

    gauss_xy_list = []
    mup = tf.cast(mup, tf.float32)
    sigmap = tf.cast(sigmap, tf.float32)
    mup = tf.identity(mup, name='mu2d')
    sigmap = tf.identity(sigmap, name='sigma2d')
    gm2d = get_gaussian_maps_2d(mup, sigmap, [256, 256], nb_landmarks=nb_landmarks)

    return mup, sigmap, gm2d


def get_gaussian_maps_2d(mu, sigma, shape_hw, mode='rot', nb_landmarks=6):
    """
  Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
  given the gaussian centers: MU [B, NMAPS, 2] tensor.

  STD: is the fixed standard dev.
  """

    with tf.name_scope(None, 'gauss_map', [mu]):
        y = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[0]))

        x = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[1]))
        [x, y] = tf.meshgrid(x, y)
        xy = tf.stack([x, y], axis=-1)
        xy = tf.stack([xy] * nb_landmarks, axis=0)
        xy = xy[tf.newaxis, :, :, :, :]
        mu = mu[:, :, tf.newaxis, tf.newaxis, :]
        invsigma = tf.linalg.inv(sigma)
        invsigma = tf.cast(invsigma, tf.float32)
        pp = tf.tile(invsigma[:, :, tf.newaxis, :, :], [1, 1, shape_hw[1], 1, 1])
        X = xy - mu
        dist = tf.matmul(X, pp)
        dist = tf.reduce_sum((dist * X), axis=-1)

        g_yx = tf.exp(-dist)

        g_yx = tf.transpose(g_yx, perm=[0, 2, 3, 1])

    return g_yx


def resize_mask(mask, size=256):
    def find_bbx(maskj):
        maskj = np.expand_dims(maskj, axis=-1)
        box = np.array([0, 0, 0, 0])

        # Compute Bbx coordinates

        margin = 3
        xs = np.nonzero(np.sum(maskj, axis=0))[0]
        ys = np.nonzero(np.sum(maskj, axis=1))[0]
        box[1] = xs.min() - margin
        box[3] = xs.max() + margin
        box[0] = 0
        box[2] = maskj.shape[0]

        if box[0] < 0: box[0] = 0
        if box[1] < 0: box[1] = 0

        h = box[2] - box[0]
        w = box[3] - box[1]
        if h < w:
            diff = w - h
            half = int(diff / 2)
            box[0] -= half
            if box[0] < 0:
                box[2] -= box[0]
                box[0] = 0
            else:
                box[2] += diff - half

            if box[2] > maskj.shape[0]:
                box[2] = maskj.shape[0]
        else:
            diff = h - w
            half = int(diff / 2)
            box[1] -= half
            if box[1] < 0:
                box[3] -= box[1]
                box[1] = 0
            else:
                box[3] += diff - half
            if box[3] > maskj.shape[1]:
                box[3] = maskj.shape[1]

        if box[3] == box[1]:
            box[3] += 1
        if box[0] == box[2]:
            box[2] += 1

        return box, h - w

    box, crp_dir = find_bbx(mask)
    mask = mask[box[0]:box[2], box[1]:box[3]]
    mask = cv2.resize(mask, (size, size))
    return mask


def z_to_l_infer(model_path, img_path, angle=180):

    frozen_model = model_path + '/frozen_model.pb'

    with tf.gfile.GFile(frozen_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph_model:
        tf.import_graph_def(graph_def, name='')

    input_img = graph_model.get_tensor_by_name('img:0')
    random_rot = graph_model.get_tensor_by_name('preprocess/randomrot:0')
    gen_img_rot = graph_model.get_tensor_by_name('gen/gen_img/gen_img_rot:0')
    gen_densmap_rot = graph_model.get_tensor_by_name('gen/gen_img/density_map_rot:0')

    imgs = os.listdir(img_path)
    for img in imgs:
        try:
            print(img)
            im_f = os.path.join(img_path, img)
            im = cv2.imread(im_f)
            # plt.imshow(im)
            # plt.show()
            # im = cv2.resize(im, (256, 256))
            im = resize_mask(im)
            video_path = os.path.join(model_path, "results", "videos", 'mask')
            dens_path = os.path.join(model_path, "results", "videos", 'density')
            os.makedirs(os.path.join(video_path, img.split('.')[0]))
            os.makedirs(os.path.join(dens_path, img.split('.')[0]))

            mask_video = cv2.VideoWriter(os.path.join(video_path, img.split('.')[0], 'video.mp4'),
                                         cv2.VideoWriter_fourcc(*'MP4V'),
                                         15, (256, 256))
            dens_video = cv2.VideoWriter(os.path.join(dens_path, img.split('.')[0], 'video.mp4'),
                                         cv2.VideoWriter_fourcc(*'MP4V'),
                                         15, (256, 256))

            for x in np.arange(0, angle, 3):
                x = x * 1.
                print(x)

                with tf.Session(
                        graph=graph_model,
                        config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess_m_to_l:
                    gen_img, gen_densmap = sess_m_to_l.run(
                        [gen_img_rot, gen_densmap_rot],
                        feed_dict={input_img: np.expand_dims(im, axis=[0]),
                                   random_rot: x})

                    sess_m_to_l.close()

                gen_img = (np.squeeze((gen_img + 1) * 0.5 * 255)).astype(np.uint8)
                gen_densmap = (np.squeeze(gen_densmap * 255)).astype(np.uint8)
                gen_densmap = cv2.cvtColor(gen_densmap, cv2.COLOR_GRAY2BGR)
                mask_video.write(gen_img)
                dens_video.write(gen_densmap)

            mask_video.release()
            dens_video.release()
            # gm_video.release()
            cv2.destroyAllWindows()
            plt.imsave(os.path.join(video_path, img.split('.')[0], img), im)

            # plt.imsave(os.path.join(out_path, img.split('.')[0],
            #                         'gm_' + img.split('.')[0] + '_' + str(int(x[0, 0])) + '.png'), gm_final)

        except Exception as e:
            print(img + ' wrong!')
            print(e)


def kl(mu1, sigma1, mu2, sigma2):
    # print(tf.log(tf.linalg.det(sigma2) / tf.linalg.det(sigma1)))
    # print(tf.linalg.trace(tf.matmul(tf.linalg.inv(sigma2), sigma1)))
    # print(tf.matmul(tf.matmul((mu1 - mu2), tf.linalg.inv(sigma2)), tf.transpose(mu1 - mu2, (0, 1, 3, 2))))
    return (tf.log(tf.linalg.det(sigma2) / tf.linalg.det(sigma1)) - 3. +
            tf.linalg.trace(tf.matmul(tf.linalg.inv(sigma2), sigma1)) +
            tf.squeeze(tf.matmul(tf.matmul((mu1 - mu2), tf.linalg.inv(sigma2)),
                                 tf.transpose(mu1 - mu2, (0, 1, 3, 2))))) * 0.5


def kl_est(mu1, sigma1, mu2, sigma2):
    kl_est = tf.zeros(mu1.shape[0], np.float64)
    for i in range(mu1.shape[1]):
        mu_a = tf.expand_dims(mu1[:, i, :, :], 1)
        # print(mu_a.shape)
        sigma_a = tf.expand_dims(sigma1[:, i, :, :], 1)
        # print(sigma_a.shape)
        divs = kl(mu_a, sigma_a, mu2, sigma2)
        # print(kl(mu_a, sigma_a, mu1, sigma1))
        # print(tf.reduce_min(divs, 1))
        # print(kl_est)
        kl_est += tf.reduce_min(divs, 1)

    return kl_est


def make_video(model_path, img_path, angle=180, nb_landmarks=6):
    z_to_l_model = model_path + '/frozen_model_z_to_l.pb'
    os.makedirs(os.path.join(model_path, 'video', 'mask'))
    os.makedirs(os.path.join(model_path, 'video', 'gm'))

    with tf.gfile.GFile(z_to_l_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph_z_to_l:
        tf.import_graph_def(graph_def, name='')

    mask_ztl = graph_z_to_l.get_tensor_by_name('mask:0')
    mu3d_ztl = graph_z_to_l.get_tensor_by_name('gen/genlandmarks/mu3d:0')
    sigma3d_ztl = graph_z_to_l.get_tensor_by_name('gen/genlandmarks/sigma3d:0')
    theta3d_ztl = graph_z_to_l.get_tensor_by_name('gen/genlandmarks/theta3d:0')

    l_to_m_model = model_path + '/frozen_model_l_to_m.pb'

    with tf.gfile.GFile(l_to_m_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph_l_to_m:
        tf.import_graph_def(graph_def, name='')

    mu2d_ltm = graph_l_to_m.get_tensor_by_name('gen/genlandmarks/mu2d:0')
    sigma2d_ltm = graph_l_to_m.get_tensor_by_name('gen/genlandmarks/sigma2d:0')
    genm = graph_l_to_m.get_tensor_by_name('gen/genmask/convlast/output:0')

    imgs = os.listdir(img_path)
    for img in imgs:
        try:
            print(img)
            im_f = os.path.join(img_path, img)
            im = cv2.imread(im_f, cv2.IMREAD_GRAYSCALE)
            im = resize_mask(im)

            with tf.Session(
                    graph=graph_z_to_l,
                    config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess_m_to_l:
                mu3d, sig3d, thet3d = sess_m_to_l.run(
                    [mu3d_ztl, sigma3d_ztl, theta3d_ztl],
                    feed_dict={mask_ztl: np.expand_dims(im, axis=[0, -1])})

                sess_m_to_l.close()

            mask_video = cv2.VideoWriter(os.path.join(model_path, 'video', 'mask', img.split('.')[0] + '.mp4'),
                                         cv2.VideoWriter_fourcc(*'MP4V'),
                                         15, (256, 256))
            gm_video = cv2.VideoWriter(os.path.join(model_path, 'video', 'gm', img.split('.')[0] + '.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),
                                       15, (256, 256))
            for x in np.arange(0, angle, 3):
                x = np.array([[x]], dtype=np.float32)
                zrs = tf.zeros_like(x)
                mu2d, sigma2d, gm2d = get_landmarks(mu3d, sig3d, zrs, thet3d[0, 0] + x * 1., zrs, 0., 0., -2.,
                                                    nb_landmarks=nb_landmarks)

                with tf.Session(graph=graph_l_to_m,
                                config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess_l_to_m:
                    m_final = sess_l_to_m.run(genm, feed_dict={mu2d_ltm: mu2d.numpy(),
                                                               sigma2d_ltm: sigma2d.numpy()})

                    m_final = (np.squeeze((m_final + 1) * 0.5 * 255)).astype(np.uint8)
                    m_final = cv2.cvtColor(m_final, cv2.COLOR_GRAY2BGR)
                    gm_final = (np.squeeze(colorize_landmark_maps(gm2d.numpy())) * 255).astype(np.uint8)
                    sess_l_to_m.close()

                mask_video.write(m_final)
                gm_video.write(gm_final)

            mask_video.release()
            gm_video.release()
            cv2.destroyAllWindows()
            print(img + ' successful!')

        except Exception as e:
            print(img + ' wrong!')
            print(e)


def rotation_consistency(model_path, img_path, angle=180):
    import random

    # read graphs
    z_to_l_model = model_path + '/frozen_model_z_to_l.pb'

    with tf.gfile.GFile(z_to_l_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph_z_to_l:
        tf.import_graph_def(graph_def, name='')

    mask_ztl = graph_z_to_l.get_tensor_by_name('mask:0')
    mu3d_ztl = graph_z_to_l.get_tensor_by_name('gen/genlandmarks/mu3d:0')
    sigma3d_ztl = graph_z_to_l.get_tensor_by_name('gen/genlandmarks/sigma3d:0')
    theta3d_ztl = graph_z_to_l.get_tensor_by_name('gen/genlandmarks/theta3d:0')

    l_to_m_model = model_path + '/frozen_model_l_to_m.pb'

    with tf.gfile.GFile(l_to_m_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph_l_to_m:
        tf.import_graph_def(graph_def, name='')

    mu2d_ltm = graph_l_to_m.get_tensor_by_name('gen/genlandmarks/mu2d:0')
    sigma2d_ltm = graph_l_to_m.get_tensor_by_name('gen/genlandmarks/sigma2d:0')
    genm = graph_l_to_m.get_tensor_by_name('gen/genmask/convlast/output:0')

    rot_consists = []
    imgs = os.listdir(img_path)
    for img in imgs:
        try:
            print(img)

            if random.random() < 0.5:
                rot_a = random.uniform(0, angle / 2)
                rot_b = rot_a * 2
                rot_a = np.array([[rot_a]], dtype=np.float32)
                rot_b = np.array([[rot_b]], dtype=np.float32)
            else:
                rot_b = random.uniform(angle / 2, angle)
                rot_a = rot_b / 2
                rot_a = np.array([[rot_a]], dtype=np.float32)
                rot_b = np.array([[rot_b]], dtype=np.float32)

            in_mask = os.path.join(img_path, img)
            in_mask = cv2.imread(in_mask, cv2.IMREAD_GRAYSCALE)
            in_mask = resize_mask(in_mask)

            def z_to_l(mask):
                with tf.Session(
                        graph=graph_z_to_l,
                        config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess_m_to_l:
                    mu3d, sig3d, thet3d = sess_m_to_l.run(
                        [mu3d_ztl, sigma3d_ztl, theta3d_ztl],
                        feed_dict={mask_ztl: np.expand_dims(mask, axis=[0, -1])})

                    sess_m_to_l.close()

                return mu3d, sig3d, thet3d

            def l_to_m(mu2d, sigma2d):
                with tf.Session(graph=graph_l_to_m,
                                config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess_l_to_m:
                    m_final = sess_l_to_m.run(genm, feed_dict={mu2d_ltm: mu2d.numpy(),
                                                               sigma2d_ltm: sigma2d.numpy()})

                    m_final = (np.squeeze((m_final + 1) * 0.5 * 255)).astype(np.uint8)
                    # m_final = cv2.cvtColor(m_final, cv2.COLOR_GRAY2BGR)
                    # gm_final = (np.squeeze(colorize_landmark_maps(gm2d.numpy())) * 255).astype(np.uint8)
                    sess_l_to_m.close()
                return m_final

            zrs = tf.zeros_like(rot_a)
            mu3d_org, sig3d_org, thet3d_org = z_to_l(in_mask)
            mu2d_rot_a1, sigma2d_rot_a1, gm2d_rot_a1 = \
                get_landmarks(mu3d_org, sig3d_org, zrs, thet3d_org[0, 0] + rot_a * 1., zrs, 0., 0., -2.)
            mu2d_rot_b, sigma2d_rot_b, gm2d_rot_b = \
                get_landmarks(mu3d_org, sig3d_org, zrs, thet3d_org[0, 0] + rot_b * 1., zrs, 0., 0., -2.)

            mask_rot_a1 = l_to_m(mu2d_rot_a1, sigma2d_rot_a1)
            mask_rot_b = l_to_m(mu2d_rot_b, sigma2d_rot_b)

            mu3d_a, sig3d_a, thet3d_a = z_to_l(mask_rot_a1)
            mu2d_rot_a2, sigma2d_rot_a2, gm2d_rot_a2 = \
                get_landmarks(mu3d_a, sig3d_a, zrs, thet3d_a[0, 0] + rot_a * 1., zrs, 0., 0., -2.)
            mask_rot_a2 = l_to_m(mu2d_rot_a2, sigma2d_rot_a2)

            rot_consist = np.mean(np.abs(mask_rot_a2 - mask_rot_b)) / 255
            print(rot_consist)
            rot_consists.append(rot_consist)

        except Exception as e:
            print(img + ' wrong!')
            print(e)

    return np.mean(rot_consists)


def kl_score(model_path, img_path, rot=30., angle=180):
    import random

    # read graphs
    z_to_l_model = model_path + '/frozen_model_z_to_l.pb'

    with tf.gfile.GFile(z_to_l_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph_z_to_l:
        tf.import_graph_def(graph_def, name='')

    mask_ztl = graph_z_to_l.get_tensor_by_name('mask:0')
    mu3d_ztl = graph_z_to_l.get_tensor_by_name('gen/genlandmarks/mu3d:0')
    sigma3d_ztl = graph_z_to_l.get_tensor_by_name('gen/genlandmarks/sigma3d:0')
    theta3d_ztl = graph_z_to_l.get_tensor_by_name('gen/genlandmarks/theta3d:0')

    l_to_m_model = model_path + '/frozen_model_l_to_m.pb'

    with tf.gfile.GFile(l_to_m_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph_l_to_m:
        tf.import_graph_def(graph_def, name='')

    mu2d_ltm = graph_l_to_m.get_tensor_by_name('gen/genlandmarks/mu2d:0')
    sigma2d_ltm = graph_l_to_m.get_tensor_by_name('gen/genlandmarks/sigma2d:0')
    genm = graph_l_to_m.get_tensor_by_name('gen/genmask/convlast/output:0')

    imgs = os.listdir(img_path)
    for img in imgs:
        print(img)
        try:
            in_mask = os.path.join(img_path, img)
            in_mask = cv2.imread(in_mask, cv2.IMREAD_GRAYSCALE)
            in_mask = resize_mask(in_mask)

            def z_to_l(mask):
                with tf.Session(
                        graph=graph_z_to_l,
                        config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess_m_to_l:
                    mu3d, sig3d, thet3d = sess_m_to_l.run(
                        [mu3d_ztl, sigma3d_ztl, theta3d_ztl],
                        feed_dict={mask_ztl: np.expand_dims(mask, axis=[0, -1])})

                    sess_m_to_l.close()

                return mu3d, sig3d, thet3d

            def l_to_m(mu2d, sigma2d):
                with tf.Session(graph=graph_l_to_m,
                                config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess_l_to_m:
                    m_final = sess_l_to_m.run(genm, feed_dict={mu2d_ltm: mu2d.numpy(),
                                                               sigma2d_ltm: sigma2d.numpy()})

                    m_final = (np.squeeze((m_final + 1) * 0.5 * 255)).astype(np.uint8)
                    # m_final = cv2.cvtColor(m_final, cv2.COLOR_GRAY2BGR)
                    # gm_final = (np.squeeze(colorize_landmark_maps(gm2d.numpy())) * 255).astype(np.uint8)
                    sess_l_to_m.close()
                return m_final

            zrs = tf.zeros_like(rot, np.float64)
            mu3d_org, sig3d_org, thet3d_org = z_to_l(in_mask)
            mu2d_rot, sigma2d_rot, gm2d_rot = \
                get_landmarks(mu3d_org, sig3d_org, zrs, thet3d_org[0, 0] + rot * 1., zrs, 0., 0., -2.)
            mu3d_org, sig3d_org = apply_rotation(mu3d_org, sig3d_org, zrs, thet3d_org[0, 0], zrs, 0., 0., 0.)

            mask_rot = l_to_m(mu2d_rot, sigma2d_rot)

            mu3d_rot, sig3d_rot, thet3d_rot = z_to_l(mask_rot)
            mu3d_cyc, sig3d_cyc = \
                apply_rotation(mu3d_rot, sig3d_rot, zrs, thet3d_rot[0, 0] - rot * 1., zrs, 0., 0., 0.)

            kl_s = kl_est(mu3d_org, sig3d_org, mu3d_cyc, sig3d_cyc)

            print(kl_s)

        except Exception as e:
            print(img + ' wrong!')
            print(e)


if __name__ == '__main__':
    model_path = 'results/model5_2'
    img_path = 'test/img/synth'
    # out_path = 'results/with_iou/rotation/synth_final'
    z_to_l_infer(model_path, img_path, angle=180)
