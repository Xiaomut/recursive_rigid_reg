import os
import shutil
from glob import glob



def copyLeft2Train():
    """ 训练集原数据 """
    src_dir = "Y:/nii2"
    dst_root_dir = "Y:/traindata2"

    src_imgs = sorted(glob(src_dir + "/*"))

    i, file_num = 0, 1
    while i < len(src_imgs):
        dst_dir = os.path.join(dst_root_dir, "img{}".format(file_num))
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        # else:
        #     raise OSError(f"The dir {dst_dir} exsit!")
        shutil.copy(src_imgs[i], os.path.join(dst_dir, "imgA.nii.gz"))
        print(f"{src_imgs[i][4:]} done!")
        shutil.copy(src_imgs[i + 2], os.path.join(dst_dir, "imgB.nii.gz"))
        print(f"{src_imgs[i+2][4:]} done!")
        i += 4
        file_num += 1
        print(f"----- {dst_dir} -----")


def copyOnlyLeft2OneDir():
    """ 
    把需要分割的图像单独放到一个文件夹用于放到服务器
    这一步做完需要将其进行直方图匹配, 才能做分割
    """
    src_dir = "Y:/nii2"
    dst_dir = "Y:/trainSeg"

    src_imgs = sorted(glob(src_dir + "/*"))

    i = 0
    while i < len(src_imgs):
        shutil.copy(src_imgs[i],
                    os.path.join(dst_dir, os.path.basename(src_imgs[i])))
        print(f"{src_imgs[i][4:]} done!")
        shutil.copy(src_imgs[i + 2],
                    os.path.join(dst_dir, os.path.basename(src_imgs[i + 2])))
        print(f"{src_imgs[i+2][4:]} done!")
        i += 4


if __name__ == "__main__":
    copyOnlyLeft2OneDir()