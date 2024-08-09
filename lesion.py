import numpy as np
import nibabel as nib
from longitudinal_reg.longitudinal_utils import run_command

def inverse_nifti_values(image):
    """
    Replace zeros by ones and non-zero values by 1.

    Parameters
    ----------
    image : str or nibabel.nifti1.Nifti1Image
        path to the nifti file to be inverted or
        the image already loaded through nibabel

    Returns
    -------
    output : Nibabel Nifti1Image
    """
    img = nifti_image_input(image)
    data = img.get_fdata()
    zeros = np.where(data)
    out_data = np.ones(data.shape)
    out_data[zeros] = 0

    return nib.nifti1.Nifti1Image(out_data, img.affine)

def inverse_lesion(lesion_path):
    """Replace non-zeroes with zeroes and zeroes with ones.

    Check if the image contains more zeros than non-zeros, if so,
    replaces non-zeros by zeros and zeros by ones.

    Parameters
    ----------
    lesion_path : str
        path to the nifti file to be checked and inverted if needed

    Returns
    -------
    lesion_out : str
        path to the output file, if the lesion does not require to be inverted
        it returns the unchanged lesion_path input
    """
    import ntpath
    import os
    import shutil

    import nibabel as nib

    import CPAC.utils.nifti_utils as nu

    lesion_out = lesion_path

    if nu.more_zeros_than_ones(image=lesion_path):
        lesion_out = os.path.join(os.getcwd(), ntpath.basename(lesion_path))
        shutil.copyfile(lesion_path, lesion_out)
        nii = nu.inverse_nifti_values(image=lesion_path)
        nib.save(nii, lesion_out)
        return lesion_out
    return lesion_out

def create_lesion_preproc(lesion_mask):
    """Process lesions masks.

    Lesion mask file is deobliqued and reoriented in the same way as the T1 in
    the anat_preproc function.

    Returns
    -------
    output : dict
        Dictionary containing paths to the deobliqued and reoriented lesion masks.

    Workflow Inputs::
        inputspec.lesion : string
            User input lesion mask, in any of the 8 orientations

    Workflow Outputs::

        outputspec.refit : string
            Path to deobliqued lesion mask

        outputspec.reorient : string
            Path to RPI oriented lesion mask

    Order of commands:
    - Deobliqing the scans. ::
        3drefit -deoblique lesion.nii.gz

    - Re-orienting the Image into Right-to-Left Posterior-to-Anterior
    Inferior-to-Superior  (RPI) orientation ::
        3dresample -orient RPI
                   -prefix lesion_RPI.nii.gz
                   -inset lesion.nii.gz

    Examples
    --------
    >>> output = create_lesion_preproc()
    >>> print(output['refit'])
    >>> print(output['reorient'])
    """
    
    # Step 1: Invert the lesion if necessary
    lesion_out = inverse_lesion(lesion_mask)

    # Step 2: Deoblique the lesion mask
    deoblique_output = lesion_out.replace(".nii.gz", "_deoblique.nii.gz")
    deoblique_command = f"3drefit -deoblique {lesion_out}"
    run_command(deoblique_command)

    # Step 3: Reorient the lesion mask to RPI orientation
    reorient_output = lesion_out.replace(".nii.gz", "_RPI.nii.gz")
    reorient_command = f"3dresample -orient RPI -prefix {reorient_output} -inset {deoblique_output}"
    run_command(reorient_command)

    # Outputs
    outputs = {
        "refit": deoblique_output,
        "reorient": reorient_output
    }

    return outputs

def nifti_image_input(
    image: str | nib.nifti1.Nifti1Image,
) -> nib.nifti1.Nifti1Image:
    """Test if an input is a path or a nifti.image and the image loaded through nibabel.

    Parameters
    ----------
    image : str or nibabel.nifti1.Nifti1Image
        path to the nifti file or the image already loaded through nibabel

    Returns
    -------
    img : nibabel.nifti1.Nifti1Image
        load and return the nifti image if image is a path, otherwise simply
        return image
    """
    if isinstance(image, nib.nifti1.Nifti1Image):
        img = image
    elif isinstance(image, str):
        if not os.path.exists(image):
            msg = f"{image} does not exist."
            raise FileNotFoundError(msg)
        img = nib.load(image)
    else:
        msg = "Image can be either a string or a nifti1.Nifti1Image"
        raise TypeError(msg)
    return img
