import os
from multiprocessing.dummy import Pool
from glob import glob

import sys

sys.path.append("../")
sys.path.append("./")
from utils import base_util, image_util, histmatching

root_dir = "Y:/trainSeg"
# images = glob(f"{root_dir}/*")
images = os.listdir(root_dir)
dst_dir = "Y:/trainSeg_his"
# images_done = glob(f"{dst_dir}/*")
images_done = os.listdir(dst_dir)

for img in images_done:
    images.remove(img)

images = [os.path.join(root_dir, i) for i in images]
print(len(images))


def main():
    template, infos = base_util.readNiiImage("Y:/CBCT_Images/base.nii", True)
    pool = Pool(3)
    for i in range(len(images)):
        pt_data = base_util.readNiiImage(images[i])
        pool.apply_async(histmatching.histForSave,
                         (template, pt_data, images[i].replace(
                             "trainSeg", "trainSeg_his"), infos))
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
    # pass