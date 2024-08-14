import os
import warnings
import six
import numpy as np
import nibabel as nib
import subprocess
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool

def read_ants_mat(ants_mat_file):
    """Read a matrix, returning (translation) and (other transformations) matrices."""
    if not os.path.exists(ants_mat_file):
        raise ValueError(str(ants_mat_file) + " does not exist.")

    with open(ants_mat_file) as f:
        for line in f:
            tmp = line.split(":")
            if tmp[0] == "Parameters":
                oth_transform = np.reshape(
                    np.fromstring(tmp[1], float, sep=" "), (-1, 3)
                )
            if tmp[0] == "FixedParameters":
                translation = np.fromstring(tmp[1], float, sep=" ")
    return translation, oth_transform


def read_mat(input_mat):
    """Read a matrix, returning (translation) and (other transformations) matrices."""
    if isinstance(input_mat, np.ndarray):
        mat = input_mat
    elif isinstance(input_mat, str):
        if os.path.exists(input_mat):
            mat = np.loadtxt(input_mat)
        else:
            raise IOError(
                "ERROR norm_transformation: " + input_mat + " file does not exist"
            )
    else:
        raise TypeError(
            "ERROR norm_transformation: input_mat should be"
            + " either a str or a numpy.ndarray matrix"
        )

    if mat.shape != (4, 4):
        msg = "ERROR norm_transformation: the matrix should be 4x4"
        raise ValueError(msg)

    # Translation vector
    translation = mat[0:3, 3]
    # 3x3 matrice of rotation, scaling and skewing
    oth_transform = mat[0:3, 0:3]

    return translation, oth_transform

def norm_transformations(translation, oth_transform):
    """Calculate the sum of squares of norm translation and Frobenius norm."""
    tr_norm = np.linalg.norm(translation)
    affine_norm = np.linalg.norm(oth_transform - np.identity(3), "fro")
    return pow(tr_norm, 2) + pow(affine_norm, 2)


def norm_transformation(input_mat):
    """Calculate the sum of squares of norm translation and Frobenius norm.

    Calculate the squared norm of the translation + squared Frobenium norm
    of the difference between other affine transformations and the identity
    from an fsl FLIRT transformation matrix.

    Parameters
    ----------
    input_mat : str or ~numpy.ndarray
        Either the path to text file matrix or a matrix already imported.

    Returns
    -------
    ~numpy.float64
        squared norm of the translation + squared Frobenius norm of the
        difference between other affine transformations and the identity
    """
    if isinstance(input_mat, np.ndarray):
        mat = input_mat
    elif isinstance(input_mat, str):
        if os.path.exists(input_mat):
            mat = np.loadtxt(input_mat)
        else:
            msg = f"ERROR norm_transformation: {input_mat} file does not exist"
            raise IOError(msg)
    else:
        msg = (
            "ERROR norm_transformation: input_mat should be either a str"
            " (file_path) or a numpy.ndarray matrix"
        )
        raise TypeError(msg)

    if mat.shape != (4, 4):
        msg = "ERROR norm_transformation: the matrix should be 4x4"
        raise ValueError(msg)

    # Translation vector
    translation = mat[0:3, 3]
    # 3x3 matrice of rotation, scaling and skewing
    oth_affine_transform = mat[0:3, 0:3]
    tr_norm = np.linalg.norm(translation)
    affine_norm = np.linalg.norm(oth_affine_transform - np.identity(3), "fro")
    return pow(tr_norm, 2) + pow(affine_norm, 2)

def template_convergence(
    mat_file, mat_type="matrix", convergence_threshold=np.finfo(np.float64).eps
):
    """Check that the deistance between matrices is smaller than the threshold.

    Calculate the distance between transformation matrix with a matrix of no transformation.

    Parameters
    ----------
    mat_file : str
        path to an fsl flirt matrix
    mat_type : str
        'matrix'(default), 'ITK'
        The type of matrix used to represent the transformations
    convergence_threshold : float
        (numpy.finfo(np.float64).eps (default)) threshold for the convergence
        The threshold is how different from no transformation is the
        transformation matrix.

    Returns
    -------
    bool
    """
    if mat_type == "matrix":
        translation, oth_transform = read_mat(mat_file)
    elif mat_type == "ITK":
        translation, oth_transform = read_ants_mat(mat_file)
    else:
        msg = f"template_convergence: matrix type {mat_type} does not exist"
        raise ValueError(msg)
    distance = norm_transformations(translation, oth_transform)
    print("distance = %s", abs(distance))

    return abs(distance) <= convergence_threshold


def create_temporary_template(
    input_brain_list,
    input_skull_list,
    output_brain_path,
    output_skull_path,
    avg_method="median",
):
    """Average all the 3D images of the list into one 3D image.

    Warnings
    --------
    The function assumes that all the images have the same header,
    the output image will have the same header as the first image of the list.

    Parameters
    ----------
    input_brain_list : list of str
        list of brain image paths
    input_skull_list : list of str
        list of skull image paths
    output_brain_path : ~nibabel.Nifti1Image
        temporary longitudinal brain template
    output_skull_path : ~nibabel.Nifti1Image
        temporary longitudinal skull template
    avg_method : str
        function names from numpy library such as 'median', 'mean', 'std' ...

    Returns
    -------
    output_brain_path : ~nibabel.Nifti1Image
        temporary longitudinal brain template
    output_skull_path : ~nibabel.Nifti1Image
        temporary longitudinal skull template
    """
    if not input_brain_list or not input_skull_list:
        msg = "ERROR create_temporary_template: image list is empty"
        raise ValueError(msg)

    if len(input_brain_list) == 1 and len(input_skull_list) == 1:
        return input_brain_list[0], input_skull_list[0]

    # ALIGN CENTERS
    avg_brain_data = getattr(np, avg_method)(
        np.asarray([nifti_image_input(img).get_fdata() for img in input_brain_list]), 0
    )

    avg_skull_data = getattr(np, avg_method)(
        np.asarray([nifti_image_input(img).get_fdata() for img in input_skull_list]), 0
    )

    nii_brain = nib.Nifti1Image(
        avg_brain_data, nifti_image_input(input_brain_list[0]).affine
    )
    nii_skull = nib.Nifti1Image(
        avg_skull_data, nifti_image_input(input_skull_list[0]).affine
    )

    nib.save(nii_brain, output_brain_path)
    nib.save(nii_skull, output_skull_path)

    return output_brain_path, output_skull_path

def template_creation_flirt(
    input_brain_list,
    input_skull_list,
    init_reg=None,
    avg_method="median",
    dof=12,
    interp="trilinear",
    cost="corratio",
    mat_type="matrix",
    convergence_threshold=-1,
    thread_pool=2,
    unique_id_list=None,
):
    """Create a temporary template from a list of images.

    Parameters
    ----------
    input_brain_list : list of str
        list of brain images paths
    input_skull_list : list of str
        list of skull images paths
    init_reg : list of Node
        (default None so no initial registration performed)
        the output of the function register_img_list with another reference
        Reuter et al. 2012 (NeuroImage) section "Improved template estimation"
        doi:10.1016/j.neuroimage.2012.02.084 uses a ramdomly
        selected image from the input dataset
    avg_method : str
        function names from numpy library such as 'median', 'mean', 'std' ...
    dof : integer (int of long)
        number of transform degrees of freedom (FLIRT) (12 by default)
    interp : str
        ('trilinear' (default) or 'nearestneighbour' or 'sinc' or 'spline')
        final interpolation method used in reslicing
    cost : str
        ('mutualinfo' or 'corratio' (default) or 'normcorr' or 'normmi' or
         'leastsq' or 'labeldiff' or 'bbr')
        cost function
    mat_type : str
        'matrix'(default), 'ITK'
        The type of matrix used to represent the transformations
    convergence_threshold : float
        (numpy.finfo(np.float64).eps (default)) threshold for the convergence
        The threshold is how different from no transformation is the
        transformation matrix.
    thread_pool : int or multiprocessing.dummy.Pool
        (default 2) number of threads. You can also provide a Pool so the
        node will be added to it to be run.
    unique_id_list : list of str
        list of unique IDs in data config

    Returns
    -------
    template : str
        path to the final template

    """
    # DEBUG to skip the longitudinal template generation which takes a lot of time.
    # return 'CECI_EST_UN_TEST'

    if not input_brain_list or not input_skull_list:
        msg = "ERROR create_temporary_template: image list is empty"
        raise ValueError(msg)

    warp_list = []

    # check if image basename_list are the same
    basename_list = [
        str(os.path.basename(img).split(".")[0]) for img in input_brain_list
    ]
    counter = Counter(basename_list)
    duplicated_basename_list = [i for i, j in counter.items() if j > 1]

    if (
        not duplicated_basename_list
    ):  # if duplicated_basename_list is empty, no duplicated basenames
        warp_list_filenames = [
            os.path.join(
                os.getcwd(),
                str(os.path.basename(img).split(".")[0]) + "_anat_to_template.mat",
            )
            for img in input_brain_list
        ]
    elif len(unique_id_list) == len(input_brain_list):
        warp_list_filenames = [
            os.path.join(
                os.getcwd(),
                str(os.path.basename(img).split(".")[0])
                + "_"
                + unique_id_list[i]
                + "_anat_to_template.mat",
            )
            for i, img in enumerate(input_brain_list)
        ]

    if isinstance(thread_pool, int):
        pool = ThreadPool(thread_pool)
    else:
        pool = thread_pool

    if convergence_threshold == -1:
        convergence_threshold = np.finfo(np.float64).eps

    if len(input_brain_list) == 1 or len(input_skull_list) == 1:
        print(
            "input_brain_list or input_skull_list contains only 1 image, "
            "no need to calculate template"
        )
        warp_list.append(np.identity(4, dtype=float))  # return an identity matrix
        return (
            input_brain_list[0],
            input_skull_list[0],
            input_brain_list,
            input_skull_list,
            warp_list,
        )

    #TODO: fix up init_reg


    # Chris: I added this part because it is mentioned in the paper but I actually never used it
    # You could run a first register_img_list() with a selected image as starting point and
    # give the output to this function
    if init_reg is not None:
        if isinstance(init_reg, list):
            output_brain_list = [node.inputs.out_file for node in init_reg]
            mat_list = [node.inputs.out_matrix_file for node in init_reg]
            warp_list = mat_list
            # test if every transformation matrix has reached the convergence
            convergence_list = [
                template_convergence(mat, mat_type, convergence_threshold)
                for mat in mat_list
            ]
            converged = all(convergence_list)
        else:
            msg = "init_reg must be a list of FLIRT nipype nodes files"
            raise ValueError(msg)
    else:
        output_brain_list = input_brain_list
        output_skull_list = input_skull_list
        converged = False

    temporary_brain_template = os.path.join(
        os.getcwd(), "temporary_brain_template.nii.gz"
    )
    temporary_skull_template = os.path.join(
        os.getcwd(), "temporary_skull_template.nii.gz"
    )

    """ First is calculated an average image of the dataset to be the temporary template
    and the loop stops when this temporary template is close enough (with a transformation
    distance smaller than the threshold) to all the images of the precedent iteration.
    """
    while not converged:
        temporary_brain_template, temporary_skull_template = create_temporary_template(
            input_brain_list=output_brain_list,
            input_skull_list=output_skull_list,
            output_brain_path=temporary_brain_template,
            output_skull_path=temporary_skull_template,
            avg_method=avg_method,
        )

        reg_list_dict = register_img_list(
            input_brain_list=output_brain_list,
            ref_img=temporary_brain_template,
            dof=dof,
            interp=interp,
            cost=cost,
            unique_id_list=unique_id_list,
        )

        mat_list = [item["output_matrix"] for item in reg_list_dict]

        # TODO clean code, refactor variables
        if len(warp_list) == 0:
            warp_list = mat_list

        for index, mat in enumerate(mat_list):
            cmd = (
                "flirt -in %s -ref %s -applyxfm -init %s -dof %s -interp %s -cost %s -out %s"
                % (
                    output_skull_list[index],
                    temporary_skull_template,
                    mat,
                    dof,
                    interp,
                    cost,
                    os.path.join(
                        os.getcwd(), os.path.basename(output_skull_list[index])
                    ),
                )
            )
            os.system(cmd)

            output_skull_list[index] = os.path.join(
                os.getcwd(), os.path.basename(output_skull_list[index])
            )

            # why inverse?
            cmd = "convert_xfm -omat %s -inverse %s" % (
                warp_list_filenames[index],
                warp_list[index],
            )
            os.system(cmd)

            warp_list[index] = warp_list_filenames[index]

        output_brain_list = [item["output_image"] for item in reg_list_dict]

        # test if every transformation matrix has reached the convergence
        convergence_list = [
            template_convergence(mat, mat_type, convergence_threshold)
            for mat in mat_list
        ]
        converged = all(convergence_list)
        print(converged)

    if isinstance(thread_pool, int):
        pool.close()
        pool.join()

    brain_template = temporary_brain_template
    skull_template = temporary_skull_template

    # register T1 to longitudinal template space
    reg_list_dict = register_img_list(
        input_brain_list,
        ref_img=temporary_brain_template,
        dof=dof,
        interp=interp,
        cost=cost,
        unique_id_list=unique_id_list,
    )

    warp_list = [item["output_matrix"] for item in reg_list_dict]

    return (
        brain_template,
        skull_template,
        output_brain_list,
        output_skull_list,
        warp_list,
    )

def subject_specific_template(input_brain_list,
            input_skull_list,
            init_reg,
            avg_method,
            dof,
            interp,
            cost,
            mat_type,
            convergence_threshold,
            thread_pool,
            unique_id_list, method="flirt"
):
    """Create a subject-specific template from a list of images.

    Parameters
    ----------
    workflow_name : str

    method : str

    Returns
    -------
    result : dict
        A dictionary containing the results of the template creation.
    """
    if method == "flirt":
        # Run the template creation function
        result = template_creation_flirt(
            input_brain_list,
            input_skull_list,
            init_reg, #TODO - what is this
            avg_method,
            dof,
            interp,
            cost,
            mat_type,
            convergence_threshold,
            thread_pool,
            unique_id_list
        )

        return result
    else:
        raise ValueError(f"{method} method has not yet been implemented")


def register_img_list(
    input_brain_list,
    ref_img,
    dof=12,
    interp="trilinear",
    cost="corratio",
    thread_pool=2,
    unique_id_list=None,
):
    """
    Register a list of images to the reference image.

    Parameters
    ----------
    input_brain_list : list of str
        list of brain image paths
    ref_img : str
        path to the reference image to which the images will be registered
    dof : integer (int of long)
        number of transform degrees of freedom (FLIRT) (12 by default)
    interp : str
        ('trilinear' (default) or 'nearestneighbour' or 'sinc' or 'spline')
        final interpolation method used in reslicing
    cost : str
        ('mutualinfo' or 'corratio' (default) or 'normcorr' or 'normmi' or
         'leastsq' or 'labeldiff' or 'bbr')
        cost function
    thread_pool : int or multiprocessing.dummy.Pool
        (default 2) number of threads. You can also provide a Pool so the
        node will be added to it to be run.
    duplicated_basename : boolean
        whether there exists duplicated basename which may happen in non-BIDS dataset
    unique_id_list : list
        a list of unique IDs in data

    Returns
    -------
    node_list : list of Node
        each Node 'node' has been run and
        node.inputs.out_file contains the path to the registered image
        node.inputs.out_matrix_file contains the path to the transformation
        matrix
    """
    if not input_brain_list:
        msg = "ERROR register_img_list: image list is empty"
        raise ValueError(msg)

    basename_list = [
        str(os.path.basename(img).split(".")[0]) for img in input_brain_list
    ]
    counter = Counter(basename_list)
    duplicated_basename_list = [i for i, j in counter.items() if j > 1]

    if not duplicated_basename_list:
        output_img_list = [
            os.path.join(os.getcwd(), os.path.basename(img)) for img in input_brain_list
        ]

        output_mat_list = [
            os.path.join(os.getcwd(), str(os.path.basename(img).split(".")[0]) + ".mat")
            for img in input_brain_list
        ]
    else:
        output_img_list = [
            os.path.join(
                os.getcwd(),
                str(os.path.basename(img).split(".")[0])
                + "_"
                + unique_id_list[i]
                + ".nii.gz",
            )
            for i, img in enumerate(input_brain_list)
        ]

        output_mat_list = [
            os.path.join(
                os.getcwd(),
                str(os.path.basename(img).split(".")[0])
                + "_"
                + unique_id_list[i]
                + ".mat",
            )
            for i, img in enumerate(input_brain_list)
        ]

    def run_flirt(in_img, ref_img, output_img, output_mat, cost, dof, interp):
        flirt_cmd = [
            "flirt",
            "-in", in_img,
            "-ref", ref_img,
            "-out", output_img,
            "-omat", output_mat,
            "-cost", cost,
            "-dof", str(dof),
            "-interp", interp
        ]
        subprocess.run(flirt_cmd, check=True)
        outputs = {
            "output_image": output_img,
            "output_matrix": output_mat
        }
        return outputs

    if isinstance(thread_pool, int):
        pool = ThreadPool(thread_pool)
    else:
        pool = thread_pool

    tasks = [
        {
            "in_img": img,
            "ref_img": ref_img,
            "output_img": out_img,
            "output_mat": out_mat,
            "cost": cost,
            "dof": dof,
            "interp": interp
        }
        for img, out_img, out_mat in zip(input_brain_list, output_img_list, output_mat_list)
    ]

    out_list = pool.map(lambda task: run_flirt(**task), tasks)

    if isinstance(thread_pool, int):
        pool.close()
        pool.join()

    return out_list

def run_command(cmd):
    """Run a shell command and handle errors."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with error: {result.stderr}")
    return result.stdout

def apply_transform(name, reg_tool, time_series=False, num_cpus=1, num_ants_cores=1):
    """
    Function to set up the command for applying transformations.
    """
    def apply_xfm(input_image, reference, transform, interpolation):
        output_image = f"{name}_warped.nii.gz"
        if reg_tool == "ants":
            command = (
                f"antsApplyTransforms -d 3 -i {input_image} -r {reference} "
                f"-t {transform} -o {output_image} --interpolation {interpolation}"
            )
        elif reg_tool == "fsl":
            command = (
                f"applywarp --ref={reference} --in={input_image} --out={output_image} "
                f"--warp={transform} --interp={interpolation}"
            )
        run_command(command)
        return output_image

    return apply_xfm

def warp_longitudinal_T1w_to_template(cfg, pipe_num, input_image, reference, ):
    """
    Warp longitudinal T1w image to a template using either ANTs or FSL based on
    the registration tool used.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing pipeline setup information.
    strat_pool : object
        Strategy pool containing data and methods for retrieving relevant data.
    pipe_num : int
        Pipeline number for naming purposes.

    Returns
    -------
    output : dict
        Dictionary containing paths to the warped T1w image in template space.
    """

    # Determine the registration tool based on provenance
    xfm_prov = strat_pool.get_cpac_provenance(
        "from-longitudinal_to-template_mode-image_xfm"
    )
    reg_tool = check_prov_for_regtool(xfm_prov)

    num_cpus = cfg["pipeline_setup"]["system_config"]["max_cores_per_participant"]
    num_ants_cores = cfg["pipeline_setup"]["system_config"]["num_ants_threads"]

    # Get interpolation setting based on the registration tool
    if reg_tool == "ants":
        interpolation = cfg["registration_workflows"][
            "anatomical_registration"
        ]["registration"]["ANTs"]["interpolation"]
    elif reg_tool == "fsl":
        interpolation = cfg["registration_workflows"][
            "anatomical_registration"
        ]["registration"]["FSL-FNIRT"]["interpolation"]

    # Get the necessary inputs from the strat_pool
    input_image, _ = strat_pool.get_data("space-longitudinal_desc-brain_T1w")
    reference, _ = strat_pool.get_data("T1w_brain_template")
    transform, _ = strat_pool.get_data("from-longitudinal_to-template_mode-image_xfm")

    # Apply the transformation using the appropriate tool and settings
    apply_xfm_func = apply_transform(
        f"warp_longitudinal_to_T1template_{pipe_num}",
        reg_tool,
        time_series=False,
        num_cpus=num_cpus,
        num_ants_cores=num_ants_cores,
    )

    output_image = apply_xfm_func(input_image, reference, transform, interpolation)

    # Return the output path
    outputs = {"space-template_desc-brain_T1w": output_image}

    return outputs

def warp_longitudinal_seg_to_T1w(cfg, pipe_num, images, reference, transform):
    """
    Warp longitudinal segmentation masks and probability maps to T1w space using
    either ANTs or FSL based on the registration tool used.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing pipeline setup information.
    strat_pool : object
        Strategy pool containing data and methods for retrieving relevant data.
    pipe_num : int
        Pipeline number for naming purposes.

    Returns
    -------
    output : dict
        Dictionary containing paths to the warped segmentation masks and probability maps in T1w space.
    """

    # Determine the registration tool based on provenance
    xfm_prov = strat_pool.get_cpac_provenance(
        "from-longitudinal_to-T1w_mode-image_desc-linear_xfm"
    )
    reg_tool = check_prov_for_regtool(xfm_prov)

    num_cpus = cfg["pipeline_setup"]["system_config"]["max_cores_per_participant"]
    num_ants_cores = cfg["pipeline_setup"]["system_config"]["num_ants_threads"]

    # Get interpolation setting based on the registration tool
    if reg_tool == "ants":
        interpolation = cfg["registration_workflows"][
            "anatomical_registration"
        ]["registration"]["ANTs"]["interpolation"]
    elif reg_tool == "fsl":
        interpolation = cfg["registration_workflows"][
            "anatomical_registration"
        ]["registration"]["FSL-FNIRT"]["interpolation"]

    # Labels to process
    labels = [
        "CSF_mask",
        "CSF_desc-preproc_mask",
        "CSF_probseg",
        "GM_mask",
        "GM_desc-preproc_mask",
        "GM_probseg",
        "WM_mask",
        "WM_desc-preproc_mask",
        "WM_probseg",
    ]

    outputs = {}

    for label in labels:
        # Get the necessary inputs from the strat_pool
        input_image = images[f"space-longitudinal_label-{label}"]
    
        # Apply the transformation using the appropriate tool and settings
        apply_xfm_func = apply_transform(
            f"warp_longitudinal_seg_to_T1w_{label}_{pipe_num}",
            reg_tool,
            time_series=False,
            num_cpus=num_cpus,
            num_ants_cores=num_ants_cores,
        )

        output_image = apply_xfm_func(input_image, reference, transform, interpolation)

        # Store the output path
        outputs[f"label-{label}"] = output_image

    return outputs

def fs_generate_template():
    template = ""
    transforms = []
    return template, transforms 

def nifti_image_input(image):
    """
    Test if an input is a path or a nifti.image and the image loaded through
    nibabel
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
    elif isinstance(image, six.string_types):
        if not os.path.exists(image):
            raise ValueError(str(image) + " does not exist.")
        else:
            img = nib.load(image)
    else:
        raise TypeError("Image can be either a string or a nifti1.Nifti1Image")
    return img