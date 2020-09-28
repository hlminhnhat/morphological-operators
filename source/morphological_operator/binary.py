import numpy as np

print()


def intersect(X, Y):
    res = X * (Y // 255)
    res = res.astype(np.uint8, copy=False)
    return res


def minus(X, Y):
    Y_ = 255 - Y
    return intersect(X, Y_)


def union(X, Y):
    res = ((X > 0) | (Y > 0)) * 255
    res = res.astype(np.uint8, copy=False)
    return res


"""
def erode(img, kernel):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    eroded_img = np.zeros((img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            if kernel_ones_count == (kernel * img[i:i_, j:j_]).sum() / 255:
                eroded_img[i + kernel_center[0], j + kernel_center[1]] = 255

    return eroded_img[:img_shape[0], :img_shape[1]]
"""


def erode(img, kernel):
    kernel_ones_count = kernel.sum()
    img_shape = img.shape
    eroded_img = np.zeros(img_shape)

    x_append = np.zeros((img.shape[0], kernel.shape[1] // 2))
    img = np.concatenate((img, x_append), axis=1)
    img = np.concatenate((x_append, img), axis=1)

    y_append = np.zeros((kernel.shape[0] // 2, img.shape[1]))
    img = np.concatenate((img, y_append), axis=0)
    img = np.concatenate((y_append, img), axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            if kernel_ones_count == (kernel * img[i:i_, j:j_]).sum() / 255:
                eroded_img[i, j] = 255

    return eroded_img


def dilate(img, kernel):
    dilated_img = np.zeros(
        (img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1)
    )
    margin = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ = kernel * 255

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 255:
                i_ = i + kernel.shape[0]
                j_ = j + kernel.shape[1]
                dilated_img[i:i_, j:j_] = np.minimum(
                    255, dilated_img[i:i_, j:j_] + kernel_
                )

    return dilated_img[
        margin[0] : margin[0] + img.shape[0], margin[1] : margin[1] + img.shape[1]
    ]


def morph_open(img, kernel):
    morph_opened_img = erode(img, kernel)
    morph_opened_img = dilate(morph_opened_img, kernel)

    return morph_opened_img


def morph_close(img, kernel):
    morph_closed_img = dilate(img, kernel)
    morph_closed_img = erode(morph_closed_img, kernel)

    return morph_closed_img


'''
def hitmiss(img, kernel):
    kernel_ = np.ones((kernel.shape[0] + 2, kernel.shape[1] + 2))
    kernel_[1 : kernel.shape[0] + 1, 1 : kernel.shape[1] + 1] = 0

    img_ = 255 - img
    img1 = erode(img, kernel)
    img2 = erode(img_, kernel_)
    """
    hitmiss_img=((img1>0)&(img2>0))*255
    hitmiss_img=hitmiss_img.astype(np.uint8,copy=False)
    """
    hitmiss_img = intersect(img1, img2)
    return hitmiss_img
'''


def hitmiss(img, kernel):
    kernel1 = (kernel == 1) * 1
    kernel2 = (kernel == -1) * 1

    img_ = 255 - img
    img1 = erode(img, kernel1)
    img2 = erode(img_, kernel2)

    hitmiss_img = intersect(img1, img2)
    return hitmiss_img


def thinning(img):
    B = [
        [-1, -1, -1, 0, 1, 0, 1, 1, 1],
        [0, -1, -1, 1, 1, -1, 1, 1, 0],
        [1, 0, -1, 1, 1, -1, 1, 0, -1],
        [1, 1, 0, 1, 1, -1, 0, -1, -1],
        [1, 1, 1, 0, 1, 0, -1, -1, -1],
        [0, 1, 1, -1, 1, 1, -1, -1, 0],
        [-1, 0, 1, -1, 1, 1, -1, 0, 1],
        [-1, -1, 0, -1, 1, 1, 0, 1, 1],
    ]

    kernels = [np.array(b).reshape(3, 3) for b in B]

    thinning_img = np.array(img)

    changes = True
    while changes:
        for kernel in kernels:
            img_ = hitmiss(thinning_img, kernel)
            img_ = minus(thinning_img, img_)
            if np.sum(img_ != thinning_img) == 0:
                changes = False
                thinning_img = img_
                break
            else:
                thinning_img = img_

    return thinning_img


def thickening(img):
    B = [
        [-1, -1, -1, 0, 1, 0, 1, 1, 1],
        [0, -1, -1, 1, 1, -1, 1, 1, 0],
        [1, 0, -1, 1, 1, -1, 1, 0, -1],
        [1, 1, 0, 1, 1, -1, 0, -1, -1],
        [1, 1, 1, 0, 1, 0, -1, -1, -1],
        [0, 1, 1, -1, 1, 1, -1, -1, 0],
        [-1, 0, 1, -1, 1, 1, -1, 0, 1],
        [-1, -1, 0, -1, 1, 1, 0, 1, 1],
    ]

    kernels = [np.array(b).reshape(3, 3) * (-1) for b in B]

    thickening_img = np.array(img)

    changes = True
    while changes:
        for kernel in kernels:
            img_ = hitmiss(thickening_img, kernel)
            img_ = union(thickening_img, img_)
            if np.sum(img_ != thickening_img) == 0:
                changes = False
                thickening_img = img_
                break
            else:
                thickening_img = img_

    return thickening_img


def boundary(img, kernel):
    img_ = erode(img, kernel)

    return minus(img, img_)


def convex_hull(img):
    B = [
        [1, 0, 0, 1, -1, 0, 1, 0, 0],
        [1, 1, 1, 0, -1, 0, 0, 0, 0],
        [0, 0, 1, 0, -1, 1, 0, 0, 1],
        [0, 0, 0, 0, -1, 0, 1, 1, 1],
    ]

    kernels = [np.array(b).reshape(3, 3) for b in B]

    convex_hull_img = np.zeros(img.shape)

    for kernel in kernels:
        img_ = np.array(img)
        while True:
            temp = hitmiss(img_, kernel)

            if np.sum(img_ != temp) == 0:
                break
            else:
                img_ = temp

        img_ = union(img_, img)
        convex_hull_img = union(convex_hull_img, img_)
    return convex_hull_img


def hole_filling(img, points):
    kernel = np.zeros((3, 3), np.uint8)
    kernel[1, :] = 1
    kernel[:, 1] = 1

    hole_filling_img = np.zeros(img.shape, np.uint8)
    for p in points:
        hole_filling_img[p[0], p[1]] = 255

    img_ = 255 - img
    while True:
        temp = dilate(hole_filling_img, kernel)
        temp = intersect(temp, img_)
        if np.sum(hole_filling_img != temp) == 0:
            break
        else:
            hole_filling_img = temp

    return hole_filling_img


def skeleton(img, kernel):
    skeleton_img = np.zeros(img.shape, np.uint8)

    eroded_img = img

    while np.sum(eroded_img) > 0:
        opening_img = morph_open(eroded_img, kernel)
        minus_img = minus(eroded_img, opening_img)
        skeleton_img = union(skeleton_img, minus_img)

        eroded_img = erode(eroded_img, kernel)

    return skeleton_img


def extract_component(img):
    kernel = np.ones((3, 3), np.uint8)

    img_ = np.array(img)

    components = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.sum(img) == 0:
                break
            if img[i, j] == 255:
                X = np.zeros(img.shape, np.uint8)
                X[i, j] = 255

                while True:
                    temp = dilate(X, kernel)
                    temp = intersect(temp, img)
                    if np.sum(X != temp) == 0:
                        break
                    else:
                        X = temp
                components.append(X)
                img = minus(img, X)

    return components
