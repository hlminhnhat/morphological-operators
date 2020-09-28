import numpy as np


def dilate(img, kernel):
    kernel_ = np.rot90(kernel, k=2)
    img_shape = img.shape
    dilated_img = np.zeros(img_shape)

    x_append = np.zeros((img.shape[0], kernel.shape[1] // 2))
    img = np.concatenate((img, x_append), axis=1)
    img = np.concatenate((x_append, img), axis=1)

    y_append = np.zeros((kernel.shape[0] // 2, img.shape[1]))
    img = np.concatenate((img, y_append), axis=0)
    img = np.concatenate((y_append, img), axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel_.shape[0]
            j_ = j + kernel_.shape[1]
            dilated_img[i, j] = min(255, np.max(img[i:i_, j:j_] + kernel_))

    return dilated_img.astype(np.uint8, copy=False)


def erode(img, kernel):
    img_shape = img.shape
    eroded_img = np.zeros(img_shape)

    x_append = np.ones((img.shape[0], kernel.shape[1] // 2)) * 255
    img = np.concatenate((img, x_append), axis=1)
    img = np.concatenate((x_append, img), axis=1)

    y_append = np.ones((kernel.shape[0] // 2, img.shape[1])) * 255
    img = np.concatenate((img, y_append), axis=0)
    img = np.concatenate((y_append, img), axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            eroded_img[i, j] = max(0, np.min(img[i:i_, j:j_] - kernel))

    return eroded_img.astype(np.uint8, copy=False)


def morph_open(img, kernel):
    morph_opened_img = erode(img, kernel)
    morph_opened_img = dilate(morph_opened_img, kernel)

    return morph_opened_img


def morph_close(img, kernel):
    morph_closed_img = dilate(img, kernel)
    morph_closed_img = erode(morph_closed_img, kernel)

    return morph_closed_img


def gradient(img, kernel):
    eroded_img = erode(img, kernel)
    dilated_img = dilate(img, kernel)

    gradient_img = np.maximum(0, dilated_img - eroded_img)
    return gradient_img.astype(np.uint8, copy=False)


def top_hat(img, kernel):
    top_hat_img = morph_open(img, kernel)

    top_hat_img = np.maximum(0, img - top_hat_img)
    return top_hat_img.astype(np.uint8, copy=False)


def black_hat(img, kernel):
    black_hat_img = morph_close(img, kernel)

    black_hat_img = np.maximum(0, black_hat_img - img)
    return black_hat_img.astype(np.uint8, copy=False)


def smoothing(img, kernel):
    smoothing_img = morph_open(img, kernel)
    smoothing_img = morph_close(smoothing_img, kernel)
    return smoothing_img


def segment(img, kernel1, kernel2):
    kernel = np.zeros((3, 3), np.uint8)

    img_ = morph_close(img, kernel1)
    img_ = morph_open(img_, kernel2)
    img_ = gradient(img_, kernel)
    img_ = np.minimum(255, img + img_)
    return img_
