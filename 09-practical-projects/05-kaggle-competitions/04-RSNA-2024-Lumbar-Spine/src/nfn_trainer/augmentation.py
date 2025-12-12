"""
数据增强模块
============

本模块提供了专门针对医学影像（MRI）的数据增强方法。

主要功能：
    1. 图像缩放和居中对齐
    2. 基于关键点的仿射变换
    3. 基于参考形状的旋转和缩放
    4. 透视变换
    5. 随机裁剪

技术要点：
    - 所有变换同步应用于图像和关键点坐标
    - 使用安全策略确保关键点始终在图像范围内
    - 支持形状对齐以标准化不同患者的脊柱位置

应用场景：
    用于训练时的数据增强，提升模型的泛化能力。
    特别适合处理具有关键点标注的医学影像。

参考资料：
    - OpenCV仿射变换: https://docs.opencv.org/master/d4/d61/tutorial_warp_affine.html
    - 医学影像增强: Shape-aware augmentation
"""

import cv2
import pandas as pd
import numpy as np


def do_resize_and_center(image, point, reference_size):
    """
    将图像缩放并居中到指定大小

    该函数执行以下操作：
    1. 计算缩放比例（保持宽高比）
    2. 缩放图像和关键点
    3. 将图像居中放置（两侧填充0）

    参数：
        image: 输入图像，shape=(H, W, C)
        point: 关键点坐标数组，shape=(N, 2)，格式为(x, y)
        reference_size: 目标尺寸（正方形）

    返回：
        resized_image: 缩放后的图像，shape=(reference_size, reference_size, C)
        resized_point: 缩放后的关键点，shape=(N, 2)

    使用示例：
        >>> image = np.random.rand(100, 150, 3)  # 100x150的图像
        >>> points = np.array([[20, 30], [40, 50]])
        >>> resized_img, resized_pts = do_resize_and_center(image, points, 512)
        >>> resized_img.shape
        (512, 512, 3)

    注意：
        - 如果图像已经是目标尺寸，直接返回不做处理
        - 使用零填充（黑色背景）
    """
    point = np.array(point, np.float32)
    H, W = image.shape[:2]

    # 如果已经是目标尺寸，直接返回
    if (W == reference_size) & (H == reference_size):
        return image, point

    # 计算缩放比例（保持宽高比）
    s = reference_size / max(H, W)
    m = cv2.resize(image, dsize=None, fx=s, fy=s)
    h, w = m.shape[:2]

    # 计算填充大小（居中）
    padx0 = (reference_size - w) // 2
    padx1 = reference_size - w - padx0
    pady0 = (reference_size - h) // 2
    pady1 = reference_size - h - pady0

    # 填充图像
    m = np.pad(m, [[pady0, pady1], [padx0, padx1], [0, 0]],
               mode='constant', constant_values=0)

    # 调整关键点坐标
    p = point * s + [[padx0, pady0]]

    return m, p


def get_rotate_scale_by_reference_mat(
    point, image_shape, reference,
    scale_limit=(-0.5, 0.5),
    rotate_limit=(-45, 45),
    shift_limit=(10, 10),
    border=5
):
    """
    生成基于参考形状的旋转缩放变换矩阵

    该方法用于形状对齐（Shape Alignment）：
    1. 首先估计从当前关键点到参考形状的仿射变换
    2. 在参考形状空间中应用随机旋转、缩放和平移
    3. 组合两个变换得到最终变换矩阵

    参数：
        point: 当前关键点，shape=(N, 2)
        image_shape: 图像尺寸 (H, W)
        reference: 参考形状关键点，shape=(N, 2)
        scale_limit: 缩放范围 (min, max)
        rotate_limit: 旋转角度范围（度）
        shift_limit: 平移范围（像素）
        border: 边界安全距离

    返回：
        mat: 2x3仿射变换矩阵

    算法原理：
        M_final = M_augment @ M_align
        其中：
        - M_align: 将当前形状对齐到参考形状
        - M_augment: 在参考空间中的数据增强

    应用场景：
        用于标准化不同患者的脊柱位置，同时保持数据增强的多样性。
    """
    H, W = image_shape
    point = np.array(point, dtype=np.float32)
    reference = np.array(reference, dtype=np.float32)

    # 步骤1: 估计对齐变换矩阵
    mat0, inlier0 = cv2.estimateAffinePartial2D(point, reference)
    point0 = np.concatenate([point, np.ones((len(point), 1))], axis=1) @ mat0.T

    # 步骤2: 在对齐后的空间中进行安全的随机变换
    mat1 = get_safe_rotate_scale_mat(
        point0, image_shape,
        scale_limit=scale_limit,
        rotate_limit=rotate_limit,
        shift_limit=shift_limit,
        border=border,
    )

    # 步骤3: 组合变换矩阵
    mat0 = np.concatenate([mat0, [[0, 0, 1]]])
    mat1 = np.concatenate([mat1, [[0, 0, 1]]])
    mat = mat1 @ mat0
    mat = mat[:2]

    return mat


def get_safe_custom_mat(point, image_shape, affline_limit=(-0.25, 0.25), border=5):
    """
    生成安全的自定义透视变换矩阵

    该函数通过随机扰动图像四个角点来生成透视变换。
    会进行多次试验，确保变换后关键点仍在图像范围内。

    参数：
        point: 关键点坐标，shape=(N, 2)
        image_shape: 图像尺寸 (H, W)
        affline_limit: 角点扰动范围（相对于图像尺寸）
        border: 边界安全距离

    返回：
        mat: 3x3透视变换矩阵（齐次坐标）

    安全策略：
        - 最多尝试20次
        - 检查变换后所有关键点是否在边界内
        - 如果失败，返回单位矩阵（不做变换）

    数学原理：
        透视变换允许更复杂的图像扭曲，模拟不同的成像角度。
        使用齐次坐标系：[x', y', w'] = H @ [x, y, 1]
    """
    H, W = image_shape
    point = np.array(point, dtype=np.float32)
    q = np.array([[x, y, 1] for x, y in point])

    # 定义原始四个角点
    src = np.array([
        [0, 0], [0, H], [W, H], [W, 0]
    ], dtype=np.float32)

    trial_state = 0
    trial = 0
    max_trial = 20

    while trial < max_trial:
        trial += 1
        size = max(H, W)

        # 随机扰动角点
        dsrc = np.random.uniform(*affline_limit, (4, 2)) * size
        dst = src + dsrc
        dst = dst.astype(np.float32)

        # 计算单应性矩阵（透视变换）
        mat, inliner = cv2.findHomography(src, dst)
        p = (q @ mat.T)
        p = p[:, :2] / p[:, [2]]  # 归一化齐次坐标

        # 检查所有点是否在边界内
        xmin, xmax = p[:, 0].min(), p[:, 0].max()
        ymin, ymax = p[:, 1].min(), p[:, 1].max()

        if (xmin > border) & (xmax < W - border) & (ymin > border) & (ymax < H - border):
            trial_state = 1
            break

    # 如果所有尝试都失败，返回单位矩阵
    if trial_state == 0:
        mat = np.array([
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        ], dtype=np.float32).reshape(3, 3)

    return mat


def get_safe_rotate_scale_mat(
    point, image_shape,
    scale_limit=(-0.5, 0.5),
    rotate_limit=(-45, 45),
    shift_limit=(10, 10),
    border=5
):
    """
    生成安全的旋转缩放变换矩阵

    该函数生成随机的旋转和缩放变换，同时确保变换后的关键点
    不会超出图像边界。

    参数：
        point: 关键点坐标，shape=(N, 2)
        image_shape: 图像尺寸 (H, W)
        scale_limit: 缩放因子范围（相对于1.0）
        rotate_limit: 旋转角度范围（度）
        shift_limit: 平移范围或None（自动居中）
        border: 边界安全距离

    返回：
        mat: 2x3仿射变换矩阵

    算法步骤：
        1. 将关键点移到原点（去中心化）
        2. 应用随机旋转和缩放
        3. 计算边界框
        4. 添加安全的随机平移
        5. 检查所有点是否在范围内

    安全机制：
        - 最多尝试20次
        - 如果无法生成安全变换，返回单位矩阵
    """
    H, W = image_shape
    point = np.array(point, dtype=np.float32)
    mean = point.mean(0, keepdims=True)
    mpoint = point - mean

    trial_state = 0
    trial = 0
    max_trial = 20

    while trial < max_trial:
        trial += 1

        # 生成随机旋转和缩放
        scale = np.random.uniform(*scale_limit) + 1
        rotate = np.random.uniform(*rotate_limit)
        cos = np.cos(rotate / 180 * np.pi)
        sin = np.sin(rotate / 180 * np.pi)

        # 构建旋转缩放矩阵
        mat = np.array([
            scale * cos, -scale * sin,
            scale * sin, scale * cos,
        ]).reshape(2, 2)

        # 应用变换
        p = mpoint @ mat.T
        p = p - p.min(axis=0, keepdims=True)
        w, h = p.max(0)

        # 检查是否超出边界
        if (w > W - 1.5 - 2 * border) | (h > H - 1.5 - 2 * border):
            continue

        # 计算平移
        if shift_limit is None:
            shiftx = np.random.uniform(border, W - 1.5 - w - border)
            shifty = np.random.uniform(border, H - 1.5 - h - border)
        else:
            mx = (W - 1.5 - 2 * border - w) / 2 + border
            my = (H - 1.5 - 2 * border - h) / 2 + border
            shiftx = np.random.uniform(*shift_limit) + mx
            shifty = np.random.uniform(*shift_limit) + my

        p = p + [[shiftx, shifty]]
        p = p.astype(np.float32)

        # 估计完整的仿射变换矩阵
        mat, inliner = cv2.estimateAffinePartial2D(point, p)
        trial_state = 1
        break

    # 失败时返回单位矩阵
    if trial_state == 0:
        mat = np.array([
            1, 0, 0,
            0, 1, 0,
        ], dtype=np.float32).reshape(2, 3)

    return mat


def apply_affine(image, point, mat):
    """
    应用仿射变换到图像和关键点

    参数：
        image: 输入图像，shape=(H, W, D)
        point: 关键点，shape=(N, 2)
        mat: 2x3仿射变换矩阵

    返回：
        image_augment: 变换后的图像
        point_augment: 变换后的关键点
    """
    H, W, D = image.shape
    image_augment = cv2.warpAffine(
        image, mat, (W, H),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    point = np.array([[x, y, 1] for x, y in point])
    point_augment = (point @ mat.T).tolist()
    return image_augment, point_augment


def apply_perspective(image, point, mat):
    """
    应用透视变换到图像和关键点

    参数：
        image: 输入图像，shape=(H, W, D)
        point: 关键点，shape=(N, 2)
        mat: 3x3透视变换矩阵

    返回：
        image_augment: 变换后的图像
        point_augment: 变换后的关键点
    """
    H, W, D = image.shape
    image_augment = cv2.warpPerspective(
        image, mat, (W, H),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    point = np.array([[x, y, 1] for x, y in point])
    point_augment = (point @ mat.T)
    point_augment = point_augment[:, :2] / point_augment[:, [2]]
    point_augment = point_augment.tolist()
    return image_augment, point_augment


def do_random_cutout(image, point):
    """
    在图像上方区域进行随机裁剪（Cutout增强）

    该方法在关键点上方区域随机裁剪一个矩形区域，
    模拟部分遮挡，提升模型鲁棒性。

    参数：
        image: 输入图像，shape=(H, W, D)
        point: 关键点，shape=(N, 2)

    返回：
        image: 裁剪后的图像（原地修改）

    策略：
        - 只在关键点上方区域裁剪
        - 避免遮挡关键的解剖结构
        - 宽度随机，高度限制在上方区域
    """
    H, W, D = image.shape
    point = np.array(point, dtype=np.float32)
    xmin, xmax = point[:, 0].min(), point[:, 0].max()
    ymin, ymax = point[:, 1].min(), point[:, 1].max()

    # 随机裁剪尺寸
    w = np.random.randint(10, W)
    h = np.random.randint(1, int(ymin))

    # 随机裁剪位置（在上方区域）
    x = np.random.randint(0, W - w)
    y = np.random.randint(0, int(ymin) - h)

    # 填充为0（黑色）
    image[y:y+h, x:x+w] = 0

    return image
