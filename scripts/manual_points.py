import napari
import numpy as np
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_fill_holes, generate_binary_structure
import os
import numpy as np

if __name__ == "__main__":

    INPUT_PATH = ""

    image = sitk.ReadImage(INPUT_PATH)
    img = sitk.GetArrayFromImage(image)
    with napari.gui_qt():
        viewer = napari.view_image(img)

    masks = {}
    for layer in viewer.layers:
        if layer.name == "Image":
            continue

        if isinstance(layer, napari.layers.Shapes):

            polygons = layer.data
            curr_mask = np.zeros_like(img)
            contour_pts = []
            for nodes in polygons:
                nodes = nodes[:-1]

            for pt in contour_pts:
                curr_mask[int(pt[0]), int(pt[1]), int(pt[2])] = 1
        elif isinstance(layer, napari.layers.Labels):
            curr_mask = layer.data
        else:
            raise RuntimeError

        filled_slices = []
        for _slice in curr_mask:
            filled_slices.append(
                binary_fill_holes(
                    _slice,
                    structure=generate_binary_structure(_slice.ndim, _slice.ndim),
                )
            )
        curr_mask = np.array(filled_slices)
        masks[layer.name.lower()] = curr_mask

    for key, mask in masks.items():
        mask = mask.astype(np.uint8)
        mask = sitk.GetImageFromArray(mask)

        sitk.WriteImage(mask, os.path.join(OUTPUT_PATH, "%s.dcm" % key))
