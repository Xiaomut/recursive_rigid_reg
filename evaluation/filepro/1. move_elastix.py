import os
import shutil


def cp2segment():
    base_dir = "Y:/testdata"
    out_dir = "Y:/evals"

    os.makedirs(os.path.join(out_dir, "fixed"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "elastix"), exist_ok=True)

    for i in range(1, 45):
        prefix = "00" if i < 10 else "0"

        # cur_fixed_path = os.path.join(base_dir, f"img{i}", "imgA.nii.gz")
        # dst_fixed_path = os.path.join(out_dir, "fixed", f"{prefix}{i}")

        cur_elastix_path = os.path.join(base_dir, f"img{i}", "elastix.nii.gz")
        dst_elastix_path = os.path.join(out_dir, "elastix",
                                        f"{prefix}{i}.nii.gz")

        shutil.copy(cur_elastix_path, dst_elastix_path)


def cpFromSeg():
    base_dir = "Y:/evals/results"
    out_dir = "Y:/testdata"

    for i in range(1, 45):
        prefix = "00" if i < 10 else "0"
        cur_path = os.path.join(base_dir, f"{prefix}{i}.nii.gz")
        dst_path = os.path.join(out_dir, f"img{i}/gt_elstix.nii.gz")

        shutil.copy(cur_path, dst_path)
        print(f"---- {prefix}{i}.nii.gz done! ----")


if __name__ == "__main__":
    cpFromSeg()