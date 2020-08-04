from pyorthanc import Orthanc
import os
from tqdm import tqdm
import zipfile
import SimpleITK as sitk
import pydicom
from io import BytesIO
import numpy as np
from functools import partial
from multiprocessing import Pool


def retrieve_image(
    server: Orthanc,
    series_id: str,
    name: list = None,
    slice_idx=None,
    norm_range=True,
    convert_to_uint=None,
):

    if convert_to_uint is None:
        convert_to_uint = not norm_range
    zfile = zipfile.ZipFile(BytesIO(server.get_series_archives(series_id)))

    # return all elements if no name is given
    if name is None:
        name = zfile.namelist()

    try:
        name.remove("DICOMDIR")
    except ValueError:
        pass

    imgs = []

    for _name in name:
        with zfile.open(_name) as f:
            img = pydicom.read_file(f).pixel_array

        if norm_range:
            img = img - img.min()
            img = img / img.max()

        if convert_to_uint:
            img = ((img + 32768) / (32767 * 2 + 1)) * 255
            img = img.astype(np.uint8)

        if slice_idx is not None:
            img = img[slice_idx]
        imgs.append(img)

    if len(imgs) == 1:
        return imgs[0]
    else:
        return imgs


def process_patient(args, base_out_dir):
    try:
        puid, pid = args
        series_information = orthanc.get_patient_series(puid)
        patient_path = os.path.join(base_out_dir, pid)

        sorted_series = []
        for series in series_information:
            series_id = series["ID"]
            series_number = series["MainDicomTags"]["SeriesNumber"]
            sorted_series.append((series_id, series_number))

        sorted_series = sorted(sorted_series, key=lambda x: int(x[1]))
        img = None
        for series_id, series_number in sorted_series:

            if series_number in ("01", "02"):
                assert img is not None
                # retrieve masks from orthanc
                _masks = retrieve_image(orthanc, series_id, slice_idx=slice(0, 2))
                masks = {k: _masks[idx] for k, idx in instance_frame_mapping.items()}

                series_path = os.path.join(
                    patient_path, series_number_mapping[series_number]
                )

                os.makedirs(series_path)

                # write masks and image to disk
                for k, v in masks.items():
                    sitk.WriteImage(
                        sitk.GetImageFromArray((v * 255).astype(np.uint8)),
                        os.path.join(series_path, "mask_%s.png" % k),
                    )
                sitk.WriteImage(
                    sitk.GetImageFromArray(img), os.path.join(series_path, "image.png")
                )

            elif series_number == "00":
                img = retrieve_image(orthanc, series_id, norm_range=False)
            else:
                raise ValueError

        tqdm.write("Finished Patient %s" % pid)
    except Exception as e:
        return str(e)


if __name__ == "__main__":

    import sys

    base_out_dir = "/work/scratch/schock/Temp/FromOrthanc"
    valid_ids = ["0000000001"]

    series_number_mapping = {"00": "img", "01": "left", "02": "right"}
    instance_frame_mapping = {"femur": 0, "tibia": 1}

    orthanc = Orthanc("http://orthanc:8043")
    orthanc.setup_credentials(sys.argv[1], sys.argv[2])  # If needed

    patient_ids = []
    tqdm.write("Retrieve Patient IDs")
    for study_identifier in orthanc.get_studies():
        study_information = orthanc.get_study_information(study_identifier)
        study_id = study_information["MainDicomTags"]["StudyID"]
        patient_id = study_information["PatientMainDicomTags"]["PatientID"]
        if study_id in valid_ids:
            # Add Unique Patient ID and Readable (but possibly ambiguous)
            # Patient ID
            patient_ids.append((study_information["ParentPatient"], patient_id))

    tqdm.write("Start Processing Patients")
    func = partial(process_patient, base_out_dir=base_out_dir)

    with Pool() as p:
        ret_vals = p.map(func, patient_ids)

    for val in ret_vals:
        if val is not None:
            print(val)
