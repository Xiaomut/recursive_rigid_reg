import torch
import math
import numpy as np
from copy import deepcopy


def cropImageByPoint(array, pt, limit):
    """ 
    以pt为中心裁剪图像, 不按照各个边长一致裁剪, 尽可能取较大的图像块
    """
    p1, p2, p3, p4, p5, p6 = pt[2] - limit[0], pt[2] + limit[1], pt[1] - limit[
        2], pt[1] + limit[3], pt[0] - limit[4], pt[0] + limit[5]
    array = array[p1:p2, p3:p4, p5:p6]
    return array

def cropImageByPointTest(array, pt, limit):
    """ 
    以pt为中心裁剪图像, 不按照各个边长一致裁剪, 尽可能取较大的图像块
    """
    p1, p2, p3, p4, p5, p6 = pt[2] - limit[0], pt[2] + limit[1], pt[1] - limit[
        2], pt[1] + limit[3], pt[0] - limit[4], pt[0] + limit[5]
    # array = array[p1:, p3:, p5:]
    array = array[p1:p2, p3:p4, p5:p6]
    return array


def processOutPt(pt, img_shape, limit):
    """ 限制输出点的范围 """
    assert len(img_shape) == 3, "array must be 3D image"
    ori_D, ori_H, ori_W = img_shape

    limits = [
        pt[2] - limit[0], pt[2] + limit[1], pt[1] - limit[2], pt[1] + limit[3],
        pt[0] - limit[4], pt[0] + limit[5]
    ]
    if max(limits) < ori_D and min(limits) >= 0:
        return pt
    else:
        if limits[0] < 0:
            pt[2] = limit[0] + 1
        if limits[1] > ori_D:
            pt[2] = ori_D - limit[1] - 1
        if limits[2] < 0:
            pt[1] = limit[2] + 1
        if limits[3] > ori_H:
            pt[1] = ori_H - limit[3] - 1
        if limits[4] < 0:
            pt[0] = limit[4] + 1
        if limits[5] > ori_W:
            pt[0] = ori_W - limit[5] - 1
        return pt


def rotationMatrixToEulerAngles(R):
    """由旋转矩阵变成三个方向的旋转角度"""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def decomposeMatrixDegree(matrix):
    """输入整体矩阵, 输出6个角度(平移+旋转)"""
    eus = rotationMatrixToEulerAngles(matrix[:3, :3])
    eus = np.asarray(eus, dtype=np.float)
    params = np.asarray(
        [eus[0], eus[1], eus[2], matrix[0, 3], matrix[1, 3], matrix[2, 3]])
    return params


# def eulerAnglesToRotationMatrix(r):
#     """ 旋转 degree 转换为 旋转 matrix"""
#     rx = torch.FloatTensor([[1, 0, 0], [0, torch.cos(r[0]),
#                                         torch.sin(r[0])],
#                             [0, -torch.sin(r[0]),
#                              torch.cos(r[0])]])
#     ry = torch.FloatTensor([[torch.cos(r[1]), 0, -torch.sin(r[1])], [0, 1, 0],
#                             [torch.sin(r[1]), 0,
#                              torch.cos(r[1])]])
#     rz = torch.FloatTensor([[torch.cos(r[2]),
#                              torch.sin(r[2]), 0],
#                             [-torch.sin(r[2]),
#                              torch.cos(r[2]), 0], [0, 0, 1]])
#     R = torch.FloatTensor(rx @ ry @ rz)
#     return R.to(r.device)

# def composeMatrixFromDegree(degree):
#     """ 由 degree 合成 matrix """
#     if degree.shape != (6, ):
#         degree = degree.reshape(6, )
#     try:
#         degree = torch.from_numpy(degree).type(torch.float)
#     except TypeError:
#         pass
#     R = eulerAnglesToRotationMatrix(degree[:3])
#     T = degree[3:].view(-1, 1)
#     matrix = torch.cat([R, T], dim=1)
#     return matrix.view(1, 3, 4)


def composeMatrixFromDegree(rt):
    rx = torch.cos(rt[0, 0]).repeat(4, 4) * torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        dtype=float) + torch.sin(rt[0, 0]).repeat(4, 4) * torch.tensor(
            [[0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
            dtype=float) + torch.tensor(
                [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
                dtype=float)

    ry = torch.cos(rt[0, 1]).repeat(4, 4) * torch.tensor(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        dtype=float) + torch.sin(rt[0, 1]).repeat(4, 4) * torch.tensor(
            [[0, 0, 1, 0], [0, 0, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 0]],
            dtype=float) + torch.tensor(
                [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
                dtype=float)

    rz = torch.cos(rt[0, 2]).repeat(4, 4) * torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=float) + torch.sin(rt[0, 2]).repeat(4, 4) * torch.tensor(
            [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=float) + torch.tensor(
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype=float)

    # translation x
    d = rt[0, 3:].unsqueeze(1).repeat(1, 4)
    d = d * torch.FloatTensor([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])

    # transform matrix
    R = torch.mm(torch.mm(rx, ry), rz)
    theta = R[0:3, :] + d
    return theta.view(1, 3, 4)


def reviseMtxFromCrop(imgA_shape, imgB_shape, mtx_res):
    """ 由部分还原到原图像的映射关系 """
    rate1 = imgA_shape[0] / imgB_shape[0]
    rate2 = imgA_shape[1] / imgB_shape[1]
    rate3 = imgA_shape[2] / imgB_shape[2]

    mtx = deepcopy(mtx_res)

    if mtx.shape == (6, ) or mtx.shape == (1, 6):
        if isinstance(mtx, np.ndarray):
            mtx.reshape((6, ))
        else:
            mtx.view((6, ))
        mtx[3] = mtx[3] / rate3
        mtx[4] = mtx[4] / rate2
        mtx[5] = mtx[5] / rate1
    elif mtx.shape == (3, 4) or mtx.shape == (1, 3, 4):
        if isinstance(mtx, np.ndarray):
            mtx.reshape(1, 3, 4)
        else:
            mtx.view((1, 3, 4))
        mtx[:, 0, 3] = mtx[:, 0, 3] / rate3
        mtx[:, 1, 3] = mtx[:, 1, 3] / rate2
        mtx[:, 2, 3] = mtx[:, 2, 3] / rate1
    else:
        raise IndexError(f"The mtx's shape {mtx.shape} is not matched!")
    return mtx


def invMatrix(m, change=True, degree=True):
    """对矩阵求逆"""
    m_add = np.vstack([m, [[0, 0, 0, 1]]])
    m_add_inv = np.linalg.inv(m_add)
    if degree:
        res = decomposeMatrixDegree(m_add_inv, change=change)
        return res
    return m_add_inv[:3, :]