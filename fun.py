import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
import matplotlib as mpl


# 导入图片
def load_img(path_to_img):
    max_dim = 512
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    # 图片缩放
    img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
    img = kp_image.img_to_array(img)
    # 增加至4维, batch
    img = np.expand_dims(img, axis=0)
    return img


# 显示函数
def imshow(img, title=None):
    # 4维转3维
    out = np.squeeze(img, axis=0)
    # 标准化
    out = out.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)


# 图像预处理
def load_and_process_img(path_to_img):
    img = load_img(path_to_img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


# 图像后处理
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # 转换为显示图片
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 选择 mode
def choose_model():
    pass


def get_model():
    # 内容层选取 block5_conv2
    content_layers = ['block5_conv2']

    # 风格层 conv1
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    # 导入VGG19模型，权重使用imagenet
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    # 提取风格特征
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    # 提取内容特征
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    # 输出结果特征
    model_outputs = style_outputs + content_outputs
    return models.Model(vgg.input, model_outputs)


# 计算内容损失， 均方误差
def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


# 计算格拉姆矩阵
def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)  # 转置
    return gram / tf.cast(n, tf.float32)


# 计算风格损失
def get_style_loss(base_style, gram_target):
    """Expects two images of dimension h, w, c"""
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    # 均方误差
    return tf.reduce_mean(tf.square(gram_style - gram_target))  # / (4. * (channels ** 2) * (width * height) ** 2)


def get_feature_representations(model, content_path, style_path):
    # 导入图片
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    # 得到风格特征和内容特征
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # 得到风格和内容特征
    style_features = [style_layer[0] for style_layer in style_outputs[:5]]
    content_features = [content_layer[0] for content_layer in content_outputs[5:]]
    return style_features, content_features


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)

    style_output_features = model_outputs[:5]
    content_output_features = model_outputs[5:]

    style_score = 0
    content_score = 0

    # 加权风格损失值
    weight_per_style_layer = 1.0 / float(5)  # 1.0
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    # 加权内容损失值
    weight_per_content_layer = 1.0 / float(1)  # 0.2
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight
    # Get total loss
    loss = style_score + content_score
    return loss, style_score, content_score


# 计算格拉姆矩阵
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss


# 风格迁移
def run_style_transfer(content_path,
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3,
                       style_weight=1e-2):
    model = get_model()
    # layer 不训练
    for layer in model.layers:
        layer.trainable = False

    # 得到特征
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # 第一次内容图作为输出图
    init_image = load_and_process_img(content_path)
    init_image = tfe.Variable(init_image, dtype=tf.float32)

    opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

    # For displaying intermediate images
    iter_count = 1

    best_loss, best_img = float('inf'), None

    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }

    # 每训练10次显示一次结果
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations / (num_rows * num_cols)
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    imgs = []
    for i in range(num_iterations):
        # 计算梯度, loss值
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        # 更新梯度
        opt.apply_gradients([(grads, init_image)])
        # 处理图片
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)

        # 保存loss值最小的
        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

    #     # 每训练100显示一次
    #     if i % display_interval == 0:
    #         start_time = time.time()
    #         # Use the .numpy() method to get the concrete numpy array
    #         plot_img = init_image.numpy()
    #         plot_img = deprocess_img(plot_img)
    #         imgs.append(plot_img)
    #         plt.imshow(Image.fromarray(plot_img))
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.show()
    #         print('Iteration: {}'.format(i))
    #         print('Total loss: {:.4e}, '
    #               'style loss: {:.4e}, '
    #               'content loss: {:.4e}, '
    #               'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
    # print('Total time: {:.4f}s'.format(time.time() - global_start))
    # plt.figure(figsize=(14, 4))
    # for i, img in enumerate(imgs):
    #     plt.subplot(num_rows, num_cols, i + 1)
    #     plt.imshow(img)
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()
    return best_img, best_loss


def show_results(best_img, content_path, style_path, show_large_final=True):
    plt.figure(figsize=(10, 5))
    content = load_img(content_path)
    style = load_img(style_path)

    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')
    plt.xticks([])
    plt.yticks([])

    if show_large_final:
        plt.figure(figsize=(10, 10))
        plt.imshow(best_img)
        plt.xticks([])
        plt.yticks([])
        plt.title('Output Image')
        plt.show()


def run(content_path='static/content/a.jpg', style_path='style/style2.jpg', path='static/out/a.png',
        num_iterations=1000,
        content_weight=1e3,
        style_weight=1e-2):
    # 动态图
    tf.enable_eager_execution()
    best, best_loss = run_style_transfer(content_path, style_path, num_iterations=num_iterations,
                                         content_weight=content_weight, style_weight=style_weight)
    plt.imsave(path, best)
