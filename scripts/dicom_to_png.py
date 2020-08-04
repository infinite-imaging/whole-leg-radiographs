from skimage.io import imsave
import SimpleITK as sitk
import os
from tqdm import tqdm
import shutil

in_path = ""
out_path = ""

dirs_to_process = [in_path]
files = []

while dirs_to_process:
    curr_dir = dirs_to_process.pop()
    subitems = [os.path.join(curr_dir, x) for x in os.listdir(curr_dir)]

    for item in subitems:
        if "DICOMDIR" in item:
            continue
        if os.path.isfile(item):
            files.append(item)
        else:
            dirs_to_process.append(item)

invalid_files = []
for file in tqdm(files):
    try:
        img = sitk.GetArrayFromImage(sitk.ReadImage(file))
    except RuntimeError:
        invalid_files.append(file)
        continue

    save_path = file.replace(in_path, out_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = img[0][..., None]

    img = img.astype("float32")
    img = img - img.min()
    img = img / img.max()

    try:
        imsave(save_path + ".png", img)
    except ValueError:
        shutil.rmtree(os.path.dirname(save_path))

print(invalid_files)
