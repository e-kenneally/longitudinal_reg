import subprocess
import os
from longitudinal_reg.longitudinal_utils import run_command
from longitudinal_reg.lesion import create_lesion_preproc

def register_ANTs_anat_to_template(cfg, input_brain, input_head, input_mask, reference_brain, reference_head,
    reference_mask, lesion_mask):
    """
    Register T1w to template with ANTs.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    strat_pool : object
        An object with methods to get and check data.
    pipe_num : int
        Pipeline number.
    opt : optional
        Optional parameter.

    Returns
    -------
    outputs : dict
        A dictionary with the output file paths.
    """
    params = cfg['registration_workflows']['anatomical_registration']['registration']['ANTs']['T1_registration']
    interpolation = cfg['registration_workflows']['anatomical_registration']['registration']['ANTs']['interpolation']

    outputs = ANTs_registration_connector(
        cfg, params, input_brain, reference_brain, input_head, reference_head, input_mask, reference_mask, 
        lesion_mask, interpolation, orig="T1w"
    )

    # Set output paths
    output_prefix = f"ANTS_T1_to_template_{pipe_num}"
    output_paths = {
        "from-longitudinal_to-template_mode-image_desc-linear_xfm": f"{output_prefix}_linear_xfm.mat",
        "from-template_to-longitudinal_mode-image_desc-linear_xfm": f"{output_prefix}_inverse_linear_xfm.mat",
        "from-longitudinal_to-template_mode-image_desc-nonlinear_xfm": f"{output_prefix}_nonlinear_xfm.nii.gz",
        "from-template_to-longitudinal_mode-image_desc-nonlinear_xfm": f"{output_prefix}_inverse_nonlinear_xfm.nii.gz",
        "from-longitudinal_to-template_mode-image_xfm": f"{output_prefix}_composite_xfm.nii.gz",
        "from-template_to-longitudinal_mode-image_xfm": f"{output_prefix}_inverse_composite_xfm.nii.gz"
    }

    return outputs

def mask_longitudinal_T1w_brain(brain_template, pipe_num):
    """
    Apply a binary mask to the longitudinal T1w brain image using FSL's fslmaths command.

    Parameters
    ----------
    strat_pool : object
        An object with a method get_data that returns the node and output path.
    pipe_num : int
        Pipeline number.

    Returns
    -------
    outputs : dict
        A dictionary with the output file paths.
    """
    # Get input file from strat_pool
    input_file = brain_template
    output_file = f"longitudinal_T1w_brain_mask_{pipe_num}.nii.gz"

    # Run the fslmaths command
    fslmaths_cmd = [
        "fslmaths", input_file, "-bin", output_file
    ]

    run_command(fslmaths_cmd)

    # Output dictionary
    outputs = {"space-longitudinal_desc-brain_mask": output_file}

    return outputs


def ANTs_registration_connector(
    cfg, params, input_brain, reference_brain, input_head, reference_head, input_mask, reference_mask, 
    lesion_mask, interpolation, orig="T1w", symmetric=False, template="T1w",
    
):
    """Transform raw data to template with ANTs."""

    if params is None:
        err_msg = (
            "\n\n[!] C-PAC says: \nYou have selected ANTs as your"
            " anatomical registration method.\nHowever, no ANTs parameters were"
            " specified.\nPlease specify ANTs parameters properly and try again."
        )
        raise Exception(err_msg)

    sym = "sym" if symmetric else ""
    symm = "_symmetric" if symmetric else ""
    tmpl = "EPI" if template == "EPI" else ""


    if cfg.registration_workflows["anatomical_registration"]["registration"]["ANTs"]["use_lesion_mask"]:
        fixed_image_mask = create_lesion_preproc(lesion_mask)
    else:
        fixed_image_mask = None

    transforms, normalized_output_brain = create_wf_calculate_ants_warp(
        num_threads=cfg.pipeline_setup["system_config"]["num_ants_threads"],
        reg_ants_skull=cfg["registration_workflows"]["anatomical_registration"][
        "reg_with_skull"], 
        moving_brain=input_brain, 
        reference_brain=reference_brain, 
        moving_skull=input_head, 
        reference_skull=reference_head, 
        reference_mask=reference_mask,
        moving_mask=input_mask, 
        fixed_image_mask=fixed_image_mask,
        ants_para=params, 
        interp=interpolation,
    )

    # combine the linear xfm's into one - makes it easier downstream
    
    output_image = (
        f"from-{orig}_to-{sym}{tmpl}template_mode-image_desc-linear_xfm.nii.gz"
    )

    # write_composite_linear_xfm.inputs.input_image_type = 0

    linear_transforms = [transforms[1], transforms[2]]

    checked_linear_transforms, _ = check_transforms(linear_transforms)
    
    # Collect inverse transforms
    inverse_transforms = transforms[-3:]
    checked_inverse_transforms, _ = check_transforms(inverse_transforms[::-1])

    # Collect all transforms
    all_transforms = transforms[:4]  # Includes Initial, Rigid, Affine, Warp
    checked_all_transforms, _ = check_transforms(all_transforms)

    # Generate inv transform flags
    inverse_transform_flags = generate_inverse_transform_flags(checked_inverse_transforms)

    # Gather outputs
    output_image_linear = f"from-{orig}_to-{sym}{tmpl}template_mode-image_desc-linear_xfm.nii.gz"
    output_image_nonlinear = f"from-{orig}_to-{sym}{tmpl}template_mode-image_desc-nonlinear_xfm.nii.gz"
    output_image_invlinear = f"from-{sym}{tmpl}template_to-{orig}_mode-image_desc-linear_xfm.nii.gz"
    output_image_invnonlinear = f"from-{sym}{tmpl}template_to-{orig}_mode-image_desc-nonlinear_xfm.nii.gz"

    apply_transforms(input_brain, reference_brain, output_image_linear, checked_linear_transforms, interpolation)
    apply_transforms(input_brain, reference_brain, output_image_nonlinear, checked_all_transforms, interpolation)
    apply_transforms(reference_brain, input_brain, output_image_invlinear, checked_inverse_transforms, interpolation, invert=True)
    apply_transforms(reference_brain, input_brain, output_image_invnonlinear, checked_inverse_transforms, interpolation, invert=True)

    outputspec = {
        f"space-{sym}template_desc-preproc_{orig}": normalized_output_brain,
        f"from-{orig}_to-{sym}{tmpl}template_mode-image_xfm": output_image_nonlinear,
        f"from-{sym}{tmpl}template_to-{orig}_mode-image_xfm": output_image_invnonlinear,
        f"from-{orig}_to-{sym}{tmpl}template_mode-image_desc-linear_xfm": output_image_linear,
        f"from-{sym}{tmpl}template_to-{orig}_mode-image_desc-linear_xfm": output_image_invlinear,
        f"from-{orig}_to-{sym}{tmpl}template_mode-image_desc-nonlinear_xfm": transforms[3],  # warp_field
        f"from-{sym}{tmpl}template_to-{orig}_mode-image_desc-nonlinear_xfm": transforms[4],  # inverse_warp_field
    }

    return outputspec


def apply_transforms(input_image, reference_image, output_image, transforms, interpolation, invert=False):
    transform_cmd = [
        'antsApplyTransforms',  # ANTs command
        '-d', '3',  # Dimension (3 for 3D images)
        '-i', input_image,  # Input image
        '-r', reference_image,  # Reference image
        '-o', output_image,  # Output image
        '-t'
    ] + transforms  # List of transforms

    if invert:
        transform_cmd += ['--invert-transform-flags'] + ['1'] * len(transforms)

    transform_cmd += ['-n', interpolation]
    run_command(transform_cmd)

def separate_warps_list(warp_list, selection):
    """Select the warp from the warp list."""
    selected_warp = None
    for warp in warp_list:
        if selection == "Warp":
            if "3Warp" in warp or "2Warp" in warp or "1Warp" in warp:
                selected_warp = warp
        elif selection in warp:
            selected_warp = warp
    return selected_warp

def create_wf_calculate_ants_warp(
        moving_brain,
        reference_brain,
        moving_skull,
        input_reference_skull,
        reference_mask,
        moving_mask,
        fixed_image_mask,
        ants_para,
        interp
):
    """Calculate the nonlinear ANTS registration transform.

    This workflow employs the antsRegistration tool:

    http://stnava.github.io/ANTs/


    Parameters
    ----------
    name : string, optional
        Name of the workflow.

    Returns
    -------
    calc_ants_warp_wf : nipype.pipeline.engine.Workflow

    Notes
    -----
    Some of the inputs listed below are lists or lists of lists. This is
    because antsRegistration can perform multiple stages of calculations
    depending on how the user configures their registration.

    For example, if one wants to employ a different metric (with different
    parameters) at each stage, the lists would be configured like this:

    warp_wf.inputs.inputspec.transforms = ['Rigid','Affine','SyN']
    warp_wf.inputs.inputspec.transform_parameters = [[0.1],[0.1],[0.1,3,0]]

    ..where each element in the first list is a metric to be used at each
    stage, 'Rigid' being for stage 1, 'Affine' for stage 2, etc. The lists
    within the list for transform_parameters would then correspond to each
    stage's metric, with [0.1] applying to 'Rigid' and 'Affine' (stages 1 and
    2), and [0.1,3,0] applying to 'SyN' of stage 3.

    In some cases, when a parameter is not needed for a stage, 'None' must be
    entered in its place if there are other parameters for other stages.


    Workflow Inputs::

        inputspec.moving_brain : string (nifti file)
            File of brain to be normalized (registered)
        inputspec.reference_brain : string (nifti file)
            Target brain file to normalize to
        inputspec.dimension : integer
            Dimension of the image (default: 3)
        inputspec.use_histogram_matching : boolean
            Histogram match the images before registration
        inputspec.winsorize_lower_quantile : float
            Winsorize data based on quantiles (lower range)
        inputspec.winsorize_higher_quantile : float
            Winsorize data based on quantiles (higher range)
        inputspec.metric : list of strings
            Image metric(s) to be used at each stage
        inputspec.metric_weight : list of floats
            Modulate the per-stage weighting of the corresponding metric
        inputspec.radius_or_number_of_bins : list of integers
            Number of bins in each stage for the MI and Mattes metric, the
            radius for other metrics
        inputspec.sampling_strategy : list of strings
            Sampling strategy (or strategies) to use for the metrics
            {None, Regular, or Random}
        inputspec.sampling_percentage : list of floats
            Defines the sampling strategy
            {float value, or None}
        inputspec.number_of_iterations : list of lists of integers
            Determines the convergence
        inputspec.convergence_threshold : list of floats
            Threshold compared to the slope of the line fitted in convergence
        inputspec.convergence_window_size : list of integers
            Window size of convergence calculations
        inputspec.transforms : list of strings
            Selection of transform options. See antsRegistration documentation
            for a full list of options and their descriptions
        inputspec.transform_parameters : list of lists of floats
            Fine-tuning for the different transform options
        inputspec.shrink_factors : list of lists of integers
            Specify the shrink factor for the virtual domain (typically the
            fixed image) at each level
        inputspec.smoothing_sigmas : list of lists of floats
            Specify the sigma of gaussian smoothing at each level
        inputspec.fixed_image_mask: (an existing file name)
            Mask used to limit metric sampling region of the fixed imagein all
            stages
        inputspec.interp : string
            Type of interpolation to use
            ('Linear' or 'BSpline' or 'LanczosWindowedSinc')

    Workflow Outputs::

        outputspec.warp_field : string (nifti file)
            Output warp field of registration
        outputspec.inverse_warp_field : string (nifti file)
            Inverse of the warp field of the registration
        outputspec.ants_affine_xfm : string (.mat file)
            The affine matrix of the registration
        outputspec.ants_inverse_affine_xfm : string (.mat file)
            The affine matrix of the reverse registration
        outputspec.composite_transform : string (nifti file)
            The combined transform including the warp field and rigid & affine
            linear warps
        outputspec.normalized_output_brain : string (nifti file)
            Template-registered version of input brain

    Registration Procedure:

    1. Calculates a nonlinear anatomical-to-template registration.

    .. exec::
        from CPAC.registration import create_wf_calculate_ants_warp
        wf = create_wf_calculate_ants_warp()
        wf.write_graph(
            graph2use='orig',
            dotfilename='./images/generated/calculate_ants_warp.dot'
        )

    Workflow Graph:
    .. image::
        :width: 500

    Detailed Workflow Graph:

    .. image::
        :width: 500
    """

    # use ANTS to warp the masked anatomical image to a template image
    """
    calculate_ants_warp = pe.Node(interface=ants.Registration(),
            name='calculate_ants_warp')

    calculate_ants_warp.inputs.output_warped_image = True
    calculate_ants_warp.inputs.initial_moving_transform_com = 0
    """

    warp_list, warped_image = hardcoded_reg(
        moving_brain,
        moving_skull,
        input_reference_skull,
        ants_para,
        moving_mask,
        reference_mask,
        fixed_image_mask,
        interp,
        reg_with_skull,
        num_threads,
        mem_gb=2.8,
        mem_x=(2e-7, "moving_brain", "xyz"),
        throttle=True)
    
    #TODO: threads
    
    ants_initial_xfm = separate_warps_list(warp_list, "Initial")
    ants_rigid_xfm = separate_warps_list(warp_list, "Rigid")
    ants_affine_xfm = separate_warps_list(warp_list, "Affine")
    warp_field = separate_warps_list(warp_list, "Warp")
    inverse_warp_field = separate_warps_list(warp_list, "Inverse")

    
    return (ants_initial_xfm,
        ants_rigid_xfm,
        ants_affine_xfm,
        warp_field,
        inverse_warp_field,
        composite_transform,
        wait,
        normalized_output_brain)

def hardcoded_reg(
    moving_brain,
    moving_skull,
    reference_skull,
    ants_para,
    moving_mask=None,
    reference_mask=None,
    fixed_image_mask=None,
    interp=None,
    reg_with_skull=0,
):
    """Run ANTs registration."""
    # TODO: expand transforms to cover all in ANTs para

    regcmd = ["antsRegistration"]
    for para_index in range(len(ants_para)):
        for para_type in ants_para[para_index]:
            if para_type == "dimensionality":
                if ants_para[para_index][para_type] not in [2, 3, 4]:
                    err_msg = (
                        "Dimensionality specified in ANTs parameters:"
                        f" {ants_para[para_index][para_type]}, is not supported."
                        " Change to 2, 3, or 4 and try again"
                    )
                    raise ValueError(err_msg)
                regcmd.append("--dimensionality")
                regcmd.append(str(ants_para[para_index][para_type]))

            elif para_type == "verbose":
                if ants_para[para_index][para_type] not in [0, 1]:
                    err_msg = (
                        "Verbose output option in ANTs parameters:"
                        f" {ants_para[para_index][para_type]}, is not supported."
                        " Change to 0 or 1 and try again"
                    )
                    raise ValueError(err_msg)
                regcmd.append("--verbose")
                regcmd.append(str(ants_para[para_index][para_type]))

            elif para_type == "float":
                if ants_para[para_index][para_type] not in [0, 1]:
                    err_msg = (
                        "Float option in ANTs parameters:"
                        f" {ants_para[para_index][para_type]}, is not supported."
                        " Change to 0 or 1 and try again"
                    )
                    raise ValueError(err_msg)
                regcmd.append("--float")
                regcmd.append(str(ants_para[para_index][para_type]))

            elif para_type == "collapse-output-transforms":
                if ants_para[para_index][para_type] not in [0, 1]:
                    err_msg = (
                        "collapse-output-transforms specified in ANTs parameters:"
                        f" {ants_para[para_index][para_type]}, is not supported."
                        " Change to 0 or 1 and try again"
                    )
                    raise ValueError(err_msg)
                regcmd.append("--collapse-output-transforms")
                regcmd.append(str(ants_para[para_index][para_type]))

            elif para_type == "winsorize-image-intensities":
                if (
                    ants_para[para_index][para_type]["lowerQuantile"] is None
                    or ants_para[para_index][para_type]["upperQuantile"] is None
                ):
                    err_msg = (
                        "Please specifiy lowerQuantile and upperQuantile of ANTs"
                        " parameters --winsorize-image-intensities in pipeline config."
                    )
                    raise Exception(err_msg)
                regcmd.append("--winsorize-image-intensities")
                _quantile = ants_para[para_index][para_type]
                regcmd.append(
                    f"[{_quantile['lowerQuantile']},{_quantile['upperQuantile']}]"
                )

            elif para_type == "initial-moving-transform":
                if ants_para[para_index][para_type]["initializationFeature"] is None:
                    err_msg = (
                        "Please specifiy initializationFeature of ANTs parameters in"
                        " pipeline config."
                    )
                    raise Exception(err_msg)
                regcmd.append("--initial-moving-transform")
                initialization_feature = ants_para[para_index][para_type][
                    "initializationFeature"
                ]
                if reg_with_skull == 1:
                    regcmd.append(
                        f"[{reference_skull},{moving_skull},{initialization_feature}]"
                    )
                else:
                    regcmd.append(
                        f"[{reference_brain},{moving_brain},{initialization_feature}]"
                    )

            elif para_type == "transforms":
                for trans_index in range(len(ants_para[para_index][para_type])):
                    for trans_type in ants_para[para_index][para_type][trans_index]:
                        regcmd.append("--transform")
                        if trans_type in ("Rigid", "Affine"):
                            if (
                                ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["gradientStep"]
                                is None
                            ):
                                err_msg = (
                                    f"Please specifiy {trans_type} Gradient Step of"
                                    " ANTs parameters in pipeline config."
                                )
                                raise Exception(err_msg)
                            gradient_step = ants_para[para_index][para_type][
                                trans_index
                            ][trans_type]["gradientStep"]
                            regcmd.append(f"{trans_type}[{gradient_step}]")

                        if trans_type == "SyN":
                            if (
                                ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["gradientStep"]
                                is None
                            ):
                                err_msg = f"Please specifiy {trans_type} Gradient Step of ANTs parameters in pipeline config."
                                raise Exception(err_msg)
                            SyN_para = []
                            SyN_para.append(
                                str(
                                    ants_para[para_index][para_type][trans_index][
                                        trans_type
                                    ]["gradientStep"]
                                )
                            )
                            if (
                                ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["updateFieldVarianceInVoxelSpace"]
                                is not None
                            ):
                                SyN_para.append(
                                    str(
                                        ants_para[para_index][para_type][trans_index][
                                            trans_type
                                        ]["updateFieldVarianceInVoxelSpace"]
                                    )
                                )
                            if (
                                ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["totalFieldVarianceInVoxelSpace"]
                                is not None
                            ):
                                SyN_para.append(
                                    str(
                                        ants_para[para_index][para_type][trans_index][
                                            trans_type
                                        ]["totalFieldVarianceInVoxelSpace"]
                                    )
                                )
                            SyN_para = ",".join([str(elem) for elem in SyN_para])
                            regcmd.append(f"{trans_type}[{SyN_para}]")

                        if (
                            ants_para[para_index][para_type][trans_index][trans_type][
                                "metric"
                            ]["type"]
                            == "MI"
                        ):
                            if (
                                ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["metric"]["metricWeight"]
                                is None
                                or ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["metric"]["numberOfBins"]
                                is None
                            ):
                                err_msg = (
                                    "Please specifiy metricWeight and numberOfBins for"
                                    " metric MI of ANTs parameters in pipeline config."
                                )
                                raise Exception(err_msg)
                            MI_para = []
                            _metric = ants_para[para_index][para_type][trans_index][
                                trans_type
                            ]["metric"]
                            MI_para.append(
                                f"{_metric['metricWeight']},{_metric['numberOfBins']}"
                            )
                            if "samplingStrategy" in ants_para[para_index][para_type][
                                trans_index
                            ][trans_type]["metric"] and ants_para[para_index][
                                para_type
                            ][trans_index][trans_type]["metric"][
                                "samplingStrategy"
                            ] in ["None", "Regular", "Random"]:
                                MI_para.append(
                                    str(
                                        ants_para[para_index][para_type][trans_index][
                                            trans_type
                                        ]["metric"]["samplingStrategy"]
                                    )
                                )
                            if (
                                "samplingPercentage"
                                in ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["metric"]
                                and ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["metric"]["samplingPercentage"]
                                is not None
                            ):
                                MI_para.append(
                                    str(
                                        ants_para[para_index][para_type][trans_index][
                                            trans_type
                                        ]["metric"]["samplingPercentage"]
                                    )
                                )
                            MI_para = ",".join([str(elem) for elem in MI_para])
                            regcmd.append("--metric")
                            if reg_with_skull == 1:
                                regcmd.append(
                                    f"MI[{reference_skull},{moving_skull},{MI_para}]"
                                )
                            else:
                                regcmd.append(
                                    f"MI[{reference_brain},{moving_brain},{MI_para}]"
                                )

                        if (
                            ants_para[para_index][para_type][trans_index][trans_type][
                                "metric"
                            ]["type"]
                            == "CC"
                        ):
                            if (
                                ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["metric"]["metricWeight"]
                                is None
                                or ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["metric"]["radius"]
                                is None
                            ):
                                err_msg = (
                                    "Please specifiy metricWeight and radius for metric"
                                    " CC of ANTs parameters in pipeline config."
                                )
                                raise Exception(err_msg)
                            CC_para = []
                            _metric = ants_para[para_index][para_type][trans_index][
                                trans_type
                            ]["metric"]
                            CC_para.append(
                                f"{_metric['metricWeight']},{_metric['radius']}"
                            )
                            if "samplingStrategy" in ants_para[para_index][para_type][
                                trans_index
                            ][trans_type]["metric"] and ants_para[para_index][
                                para_type
                            ][trans_index][trans_type]["metric"][
                                "samplingStrategy"
                            ] in ["None", "Regular", "Random"]:
                                CC_para.append(
                                    str(
                                        ants_para[para_index][para_type][trans_index][
                                            trans_type
                                        ]["metric"]["samplingStrategy"]
                                    )
                                )
                            if (
                                "samplingPercentage"
                                in ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["metric"]
                                and ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["metric"]["samplingPercentage"]
                                is not None
                            ):
                                CC_para.append(
                                    str(
                                        ants_para[para_index][para_type][trans_index][
                                            trans_type
                                        ]["metric"]["samplingPercentage"]
                                    )
                                )
                            CC_para = ",".join([str(elem) for elem in CC_para])
                            regcmd.append("--metric")
                            regcmd.append(
                                f"CC[{reference_skull},{moving_skull},{CC_para}]"
                            )

                        if (
                            "convergence"
                            in ants_para[para_index][para_type][trans_index][trans_type]
                        ):
                            convergence_para = []
                            if (
                                ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["convergence"]["iteration"]
                                is None
                            ):
                                err_msg = (
                                    "Please specifiy convergence iteration of ANTs"
                                    " parameters in pipeline config."
                                )
                                raise Exception(err_msg)
                            convergence_para.append(
                                str(
                                    ants_para[para_index][para_type][trans_index][
                                        trans_type
                                    ]["convergence"]["iteration"]
                                )
                            )
                            if (
                                "convergenceThreshold"
                                in ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["convergence"]
                                and ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["convergence"]["convergenceThreshold"]
                                is not None
                            ):
                                convergence_para.append(
                                    str(
                                        ants_para[para_index][para_type][trans_index][
                                            trans_type
                                        ]["convergence"]["convergenceThreshold"]
                                    )
                                )
                            if (
                                "convergenceWindowSize"
                                in ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["convergence"]
                                and ants_para[para_index][para_type][trans_index][
                                    trans_type
                                ]["convergence"]["convergenceWindowSize"]
                                is not None
                            ):
                                convergence_para.append(
                                    str(
                                        ants_para[para_index][para_type][trans_index][
                                            trans_type
                                        ]["convergence"]["convergenceWindowSize"]
                                    )
                                )
                            convergence_para = ",".join(
                                [str(elem) for elem in convergence_para]
                            )
                            regcmd.append("--convergence")
                            regcmd.append(f"[{convergence_para}]")

                        if (
                            "smoothing-sigmas"
                            in ants_para[para_index][para_type][trans_index][trans_type]
                            and ants_para[para_index][para_type][trans_index][
                                trans_type
                            ]["smoothing-sigmas"]
                            is not None
                        ):
                            regcmd.append("--smoothing-sigmas")
                            regcmd.append(
                                str(
                                    ants_para[para_index][para_type][trans_index][
                                        trans_type
                                    ]["smoothing-sigmas"]
                                )
                            )

                        if (
                            "shrink-factors"
                            in ants_para[para_index][para_type][trans_index][trans_type]
                            and ants_para[para_index][para_type][trans_index][
                                trans_type
                            ]["shrink-factors"]
                            is not None
                        ):
                            regcmd.append("--shrink-factors")
                            regcmd.append(
                                str(
                                    ants_para[para_index][para_type][trans_index][
                                        trans_type
                                    ]["shrink-factors"]
                                )
                            )

                        if (
                            "use-histogram-matching"
                            in ants_para[para_index][para_type][trans_index][trans_type]
                        ):
                            if ants_para[para_index][para_type][trans_index][
                                trans_type
                            ]["use-histogram-matching"]:
                                regcmd.append("--use-histogram-matching")
                                regcmd.append("1")

                        if (
                            "winsorize-image-intensities"
                            in ants_para[para_index][para_type][trans_index][trans_type]
                            and ants_para[para_index][para_type][trans_index][
                                trans_type
                            ]["winsorize-image-intensities"]["lowerQuantile"]
                            is not None
                            and ants_para[para_index][para_type][trans_index][
                                trans_type
                            ]["winsorize-image-intensities"]["upperQuantile"]
                            is not None
                        ):
                            regcmd.append("--winsorize-image-intensities")
                            _quantile = ants_para[para_index][para_type][trans_index][
                                trans_type
                            ]["winsorize-image-intensities"]
                            regcmd.append(
                                f"[{_quantile['lowerQuantile']},{_quantile['upperQuantile']}]"
                            )

                        if (
                            "masks"
                            in ants_para[para_index][para_type][trans_index][trans_type]
                            and ants_para[para_index][para_type][trans_index][
                                trans_type
                            ]["masks"]
                            is not None
                        ):
                            if ants_para[para_index][para_type][trans_index][
                                trans_type
                            ]["masks"]:
                                regcmd.append("--masks")
                                regcmd.append(f"[{reference_mask},{moving_mask}]")
                            else:
                                regcmd.append("--masks")
                                regcmd.append("[NULL,NULL]")

            elif para_type == "masks":
                # lesion preproc has
                if fixed_image_mask is not None:
                    regcmd.append("--masks")
                    regcmd.append(str(fixed_image_mask))
                else:
                    if (
                        not ants_para[para_index][para_type]["fixed_image_mask"]
                        and ants_para[para_index][para_type]["moving_image_mask"]
                    ):
                        err_msg = (
                            "Masks option in ANTs parameters:"
                            f" {ants_para[para_index][para_type]} is not supported."
                            " Please set `fixed_image_mask` as True. Or set both"
                            " `fixed_image_mask` and `moving_image_mask` as False"
                        )
                        raise NotImplementedError(err_msg)
                    if (
                        ants_para[para_index][para_type]["fixed_image_mask"]
                        and ants_para[para_index][para_type]["moving_image_mask"]
                    ):
                        regcmd.append("--masks")
                        regcmd.append(
                            "[" + str(reference_mask) + "," + str(moving_mask) + "]"
                        )
                    elif (
                        ants_para[para_index][para_type]["fixed_image_mask"]
                        and ants_para[para_index][para_type]["moving_image_mask"]
                    ):
                        regcmd.append("--masks")
                        regcmd.append("[" + str(reference_mask) + "]")
                    else:
                        continue

    if interp is not None:
        regcmd.append("--interpolation")
        regcmd.append(f"{interp}")

    regcmd.append("--output")
    regcmd.append("[transform,transform_Warped.nii.gz]")

    # write out the actual command-line entry for testing/validation later
    command_file = os.path.join(os.getcwd(), "command.txt")
    with open(command_file, "wt") as f:
        f.write(" ".join(regcmd))

    try:
        subprocess.check_output(regcmd)
    except Exception as e:
        msg = (
            "[!] ANTS registration did not complete successfully."
            f"\n\nError details:\n{e}\n{e.output}\n"
        )
        raise RuntimeError(msg)

    warp_list = []
    warped_image = None

    files = [f for f in os.listdir(".") if os.path.isfile(f)]

    for f in files:
        if ("transform" in f) and ("Warped" not in f):
            warp_list.append(os.getcwd() + "/" + f)
        if "Warped" in f:
            warped_image = os.getcwd() + "/" + f

    if not warped_image:
        msg = (
            "\n\n[!] No registration output file found. ANTS registration may not have"
            " completed successfully.\n\n"
        )
        raise RuntimeError(msg)

    return warp_list, warped_image

def check_transforms(transform_list):
    """Check if the transform list is empty."""
    transform_number = list(filter(None, transform_list))
    return [(transform_number[index]) for index in range(len(transform_number))], len(
        transform_number
    )

def generate_inverse_transform_flags(transform_list):
    """List whether each transform has an inverse."""
    inverse_transform_flags = []
    for transform in transform_list:
        # check `blip_warp_inverse` file name and rename it
        if "WARPINV" in transform:
            inverse_transform_flags.append(False)
        if "updated_affine" in transform:
            inverse_transform_flags.append(True)
        if "Initial" in transform:
            inverse_transform_flags.append(True)
        if "Rigid" in transform:
            inverse_transform_flags.append(True)
        if "Affine" in transform:
            inverse_transform_flags.append(True)
        if "InverseWarp" in transform:
            inverse_transform_flags.append(False)
    return inverse_transform_flags

def register_FSL_anat_to_template(cfg, strat_pool, pipe_num, space_longitudinal_desc_reorient_T1w, space_longitudinal_desc_preproc_T1w, opt=None):
    """Register T1w to template with FSL."""
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), f"register_{opt}_anat_to_template_{pipe_num}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get interpolation and FNIRT config from config
    interpolation = cfg['registration_workflows']['anatomical_registration']['registration']['FSL-FNIRT']['interpolation']
    fnirt_config = cfg['registration_workflows']['anatomical_registration']['registration']['FSL-FNIRT']['fnirt_config']
    
    # Get data from strat_pool
    input_brain = strat_pool.get_data(['desc-brain_T1w', 'space-longitudinal_desc-preproc_T1w'])
    input_head = strat_pool.get_data(['desc-preproc_T1w', 'space-longitudinal_desc-reorient_T1w'])
    reference_mask = strat_pool.get_data('template-ref-mask')
    
    if cfg['registration_workflows']['anatomical_registration']['registration']['FSL-FNIRT']['ref_resolution'] == cfg['registration_workflows']['anatomical_registration']['resolution_for_anat']:
        reference_brain = strat_pool.get_data('T1w-brain-template')
        reference_head = strat_pool.get_data('T1w-template')
    else:
        reference_brain = strat_pool.get_data('FNIRT-T1w-brain-template')
        reference_head = strat_pool.get_data('FNIRT-T1w-template')
    
    # Call the FSL registration function
    outputs = create_fsl_fnirt_nonlinear_reg_nhp(
        input_brain=input_brain,
        input_skull=input_head,
        reference_brain=reference_brain,
        reference_skull=reference_head,
        interp=interpolation,
        ref_mask=reference_mask,
        linear_aff="linear_aff.mat",  # Example affine file
        fnirt_config=fnirt_config
    )
    
    if "space-longitudinal" in input_brain:
        replace_outputs = {}
        for key in outputs.keys():
            if "from-T1w" in key:
                new_key = key.replace("from-T1w", "from-longitudinal")
                replace_outputs[key] = new_key
            if "to-T1w" in key:
                new_key = key.replace("to-T1w", "to-longitudinal")
                replace_outputs[key] = new_key
        for key in replace_outputs:
            tmp = outputs[key]
            outputs[replace_outputs[key]] = tmp
            del outputs[key]
    
    return outputs


def FSL_registration_connector(input_brain, reference_brain, input_head, reference_head, input_mask, reference_mask, transform, interpolation, fnirt_config, wf_name, cfg, orig="T1w", opt=None, symmetric=False, template="T1w"):
    """Transform raw data to template with FSL."""
    
    sym = "sym" if symmetric else ""
    symm = "_symmetric" if symmetric else ""
    tmpl = "EPI" if template == "EPI" else ""

    outputs = {}

    if opt in ("FSL", "FSL-linear"):
        # Linear registration with FLIRT
        #linear_xfm = os.path.join(os.getcwd(), f"{wf_name}_linear.mat")
        #output_brain = os.path.join(os.getcwd(), f"{wf_name}_brain.nii.gz")

        flirt_outputs = create_fsl_flirt_linear_reg(input_brain, reference_brain, interpolation, f"anat_mni_flirt_register{symm}")
        
        reference_brain_out = reference_brain
        lin_xfm_out = flirt_outputs["linear_xfm"]
        invlin_xfm_out = flirt_outputs["invlinear_xfm"]
        
        write_lin_composite_xfm = os.path.join(os.getcwd(), f"fsl_lin-warp_to_nii{symm}.nii.gz")
        write_invlin_composite_xfm = os.path.join(os.getcwd(), f"fsl_invlin-warp_to_nii{symm}.nii.gz")
        
        # ConvertWarp commands
        run_command([
            'convertwarp', '--ref', reference_brain_out, '--premat', lin_xfm_out,
            '--out', write_lin_composite_xfm
        ])
        
        run_command([
            'convertwarp', '--ref', reference_brain_out, '--premat', invlin_xfm_out,
            '--out', write_invlin_composite_xfm
        ])
        
        outputs.update({
            f"space-{sym}template_desc-preproc_{orig}": flirt_outputs["output_brain"],
            f"from-{orig}_to-{sym}{tmpl}template_mode-image_desc-linear_xfm": write_lin_composite_xfm,
            f"from-{sym}{tmpl}template_to-{orig}_mode-image_desc-linear_xfm": write_invlin_composite_xfm,
            f"from-{orig}_to-{sym}{tmpl}template_mode-image_xfm": write_lin_composite_xfm
        })

    if (cfg.registration_workflows["anatomical_registration"]["registration"]["FSL-FNIRT"]["ref_resolution"]
            == cfg.registration_workflows["anatomical_registration"]["resolution_for_anat"]):
            fnirt_reg_anat_mni = create_fsl_fnirt_nonlinear_reg(f"anat_mni_fnirt_register{symm}")
    else:
        fnirt_reg_anat_mni = create_fsl_fnirt_nonlinear_reg_nhp(f"anat_mni_fnirt_register{symm}")
    
    # FNIRT commands
    fnirt_out_brain = os.path.join(os.getcwd(), f"{fnirt_reg_anat_mni}_output_brain.nii.gz")
    fnirt_out_warp = os.path.join(os.getcwd(), f"{fnirt_reg_anat_mni}_warpcoef.nii.gz")
    fnirt_out_image = os.path.join(os.getcwd(), f"{fnirt_reg_anat_mni}_warped.nii.gz")
    
    run_command([
        'fnirt', '--in', input_brain, '--aff', lin_xfm_out, '--cout', fnirt_out_warp,
        '--config', fnirt_config, '--ref', reference_brain, '--refmask', reference_mask,
        '--iout', fnirt_out_image
    ])
    
    added_outputs = {
        f"space-{sym}template_desc-preproc_{orig}": fnirt_out_brain,
        f"from-{orig}_to-{sym}{tmpl}template_mode-image_xfm": fnirt_out_warp
    }
    
    if (cfg.registration_workflows["anatomical_registration"]["registration"]["FSL-FNIRT"]["ref_resolution"]
        != cfg.registration_workflows["anatomical_registration"]["resolution_for_anat"]):
        added_outputs.update({
            f"space-{sym}template_desc-head_{orig}": fnirt_out_brain,  # assuming a head output is also generated
            f"space-{sym}template_desc-{orig}_mask": reference_mask,  # similarly assuming a mask output
            f"space-{sym}template_desc-T1wT2w_biasfield": reference_mask,  # and a biasfield
            f"from-{orig}_to-{sym}{tmpl}template_mode-image_warp": fnirt_out_warp
        })
    
    outputs.update(added_outputs)

    return outputs

def create_fsl_flirt_linear_reg(input_brain, reference_brain, interp, ref_mask=None, name="fsl_flirt_linear_reg"):
    """Create a FLIRT registration process."""
    
    output_brain = os.path.join(os.getcwd(), f"{name}_output_brain.nii.gz")
    linear_xfm = os.path.join(os.getcwd(), f"{name}_linear.mat")
    invlinear_xfm = os.path.join(os.getcwd(), f"{name}_invlinear.mat")
    
    flirt(input_brain, reference_brain, interp, linear_xfm, output_brain)
    
    #make into command fsl.convertxfm
    convertxfm(linear_xfm, invlinear_xfm)
    
    outputs = {
        "output_brain": output_brain,
        "linear_xfm": linear_xfm,
        "invlinear_xfm": invlinear_xfm
    }
    
    return outputs

def flirt(input_brain, reference_brain, interp, output_matrix, output_image):
    """Perform FSL FLIRT registration."""
    cmd = [
        'flirt', '-in', input_brain, '-ref', reference_brain, '-out', output_image,
        '-omat', output_matrix, '-interp', interp
    ]
    run_command(cmd)

def fnirt(input_skull, reference_skull, affine_file, ref_mask, fnirt_config, fieldcoeff_file, jacobian_file):
    """Perform FSL FNIRT non-linear registration."""
    cmd = [
        'fnirt', '--in', input_skull, '--ref', reference_skull, '--aff', affine_file,
        '--refmask', ref_mask, '--config', fnirt_config, '--fout', fieldcoeff_file,
        '--jout', jacobian_file
    ]
    run_command(cmd)

def applywarp(input_brain, reference_brain, fieldcoeff_file, output_brain, interp):
    """Apply a warp using FSL ApplyWarp."""
    cmd = [
        'applywarp', '--in', input_brain, '--ref', reference_brain, '--warp', fieldcoeff_file,
        '--out', output_brain, '--interp', interp
    ]
    run_command(cmd)

def create_fsl_fnirt_nonlinear_reg(input_brain, input_skull, reference_brain, reference_skull, interp, ref_mask, linear_aff, fnirt_config, name="fsl_fnirt_nonlinear_reg"):
    
    """Perform non-linear registration of an input to a reference using FSL FNIRT."""
    fieldcoeff_file = os.path.join(os.getcwd(), f"{name}_fieldcoeff.nii.gz")
    jacobian_file = os.path.join(os.getcwd(), f"{name}_jacobian.nii.gz")
    output_brain = os.path.join(os.getcwd(), f"{name}_output_brain.nii.gz")

    fnirt(input_skull, reference_skull, linear_aff, ref_mask, fnirt_config, fieldcoeff_file, jacobian_file)
    applywarp(input_brain, reference_brain, fieldcoeff_file, output_brain, interp)

    outputs = {
        "output_brain": output_brain,
        "nonlinear_xfm": fieldcoeff_file
    }
    
    return outputs

def create_fsl_fnirt_nonlinear_reg_nhp(input_brain, input_skull, reference_brain, reference_skull, interp, ref_mask, linear_aff, fnirt_config, name="fsl_fnirt_nonlinear_reg_nhp"):
    
    """Perform non-linear registration of an input to a reference using FSL FNIRT."""
    fieldcoeff_file = os.path.join(os.getcwd(), f"{name}_fieldcoeff.nii.gz")
    jacobian_file = os.path.join(os.getcwd(), f"{name}_jacobian.nii.gz")
    field_file = os.path.join(os.getcwd(), f"{name}_field.nii.gz")
    output_brain = os.path.join(os.getcwd(), f"{name}_output_brain.nii.gz")
    output_head = os.path.join(os.getcwd(), f"{name}_output_head.nii.gz")
    output_mask = os.path.join(os.getcwd(), f"{name}_output_mask.nii.gz")
    output_biasfield = os.path.join(os.getcwd(), f"{name}_output_biasfield.nii.gz")

    fnirt(input_skull, reference_skull, linear_aff, ref_mask, fnirt_config, fieldcoeff_file, jacobian_file, field_file)

    applywarp(input_brain, reference_skull, field_file, output_brain, interp="nn", relwarp=True)
    applywarp(input_skull, reference_skull, field_file, output_head, interp="spline", relwarp=True)
    applywarp(input_brain, reference_skull, field_file, output_mask, interp="nn", relwarp=True)
    applywarp(input_brain, reference_skull, field_file, output_biasfield, interp="spline", relwarp=True)

    outputs = {
        "output_brain": output_brain,
        "output_head": output_head,
        "output_mask": output_mask,
        "output_biasfield": output_biasfield,
        "nonlinear_xfm": fieldcoeff_file,
        "nonlinear_warp": field_file
    }

    return outputs

def overwrite_transform_anat_to_template(cfg, strat_pool, pipe_num, opt=None):
    """Overwrite ANTs transforms with FSL transforms."""
    
    xfm_prov = strat_pool.get_cpac_provenance("from-T1w_to-template_mode-image_xfm")

    reg_tool = check_prov_for_regtool(xfm_prov)

    if opt.lower() == "fsl" and reg_tool.lower() == "ants":
        # Apply head-to-head transforms on brain using ABCD-style registration
        # Convert ANTs warps to FSL warps to be consistent with the functional registration

        input_brain = strat_pool.get_data(["desc-restore_T1w", "desc-preproc_T1w"])
        reference_image = strat_pool.get_data("T1w-template")
        transforms = strat_pool.get_data("from-T1w_to-template_mode-image_xfm")
        input_image = strat_pool.get_data("desc-preproc_T1w")

        # Create output directory
        output_dir = os.path.join(os.getcwd(), f"overwrite_transform_anat_to_template_{pipe_num}")
        os.makedirs(output_dir, exist_ok=True)
        
        # ANTs apply transforms T1 to template
        ants_apply_warp_t1_to_template = os.path.join(output_dir, "ANTs_CombinedWarp.nii.gz")
        cmd1 = [
            "antsApplyTransforms", "-d", "3", "-i", input_brain, "-r", reference_image,
            "-t", transforms, "-o", f"[{ants_apply_warp_t1_to_template},1]"
        ]
        run_command(cmd1)
        
        # ANTs apply transforms template to T1
        ants_apply_warp_template_to_t1 = os.path.join(output_dir, "ANTs_CombinedInvWarp.nii.gz")
        cmd2 = [
            "antsApplyTransforms", "-d", "3", "-i", input_image, "-r", reference_image,
            "-t", transforms, "-o", f"[{ants_apply_warp_template_to_t1},1]"
        ]
        run_command(cmd2)

        # Split combined warp and inverse warp using c4d
        
        combined_warp_outputs = run_c4d(ants_apply_warp_t1_to_template, "e")
        combined_inv_warp_outputs = run_c4d(ants_apply_warp_template_to_t1, "einv")

        # Change sign of second component using fslmaths
        def change_sign(input_file, output_file):
            cmd4 = ["fslmaths", input_file, "-mul", "-1", output_file]
            run_command(cmd4)
        
        change_sign(combined_warp_outputs[1], f"{output_dir}/e-2.nii.gz")
        change_sign(combined_inv_warp_outputs[1], f"{output_dir}/e-2inv.nii.gz")

        # Merge transformations using fslmerge
        def fslmerge(inputs, output):
            cmd5 = ["fslmerge", "-t", output] + inputs
            run_command(cmd5)

        merged_xfms = os.path.join(output_dir, "merged_xfms.nii.gz")
        merged_inv_xfms = os.path.join(output_dir, "merged_inv_xfms.nii.gz")

        fslmerge([combined_warp_outputs[0], f"{output_dir}/e-2.nii.gz", combined_warp_outputs[2]], merged_xfms)
        fslmerge([combined_inv_warp_outputs[0], f"{output_dir}/e-2inv.nii.gz", combined_inv_warp_outputs[2]], merged_inv_xfms)

        # Apply FSL warp T1 to template
        output_t1w_image_restore = os.path.join(output_dir, "OutputT1wImageRestore.nii.gz")
        cmd6 = [
            "applywarp", "--rel", "--interp=spline", "-i", input_brain, "-r", reference_image,
            "-w", merged_xfms, "-o", output_t1w_image_restore
        ]
        run_command(cmd6)

        # Apply FSL warp T1 brain to template
        output_t1w_image_restore_brain = os.path.join(output_dir, "OutputT1wImageRestoreBrain.nii.gz")
        cmd7 = [
            "applywarp", "--rel", "--interp=nn", "-i", input_image, "-r", reference_image,
            "-w", merged_xfms, "-o", output_t1w_image_restore_brain
        ]
        run_command(cmd7)

        # Apply FSL warp T1 brain mask to template
        brain_mask = strat_pool.get_data("space-T1w_desc-brain_mask")
        output_t1w_brain_mask_to_template = os.path.join(output_dir, "OutputT1wBrainMaskToTemplate.nii.gz")
        cmd8 = [
            "applywarp", "--rel", "--interp=nn", "-i", brain_mask, "-r", reference_image,
            "-w", merged_xfms, "-o", output_t1w_brain_mask_to_template
        ]
        run_command(cmd8)

        # Apply mask
        apply_mask_output = os.path.join(output_dir, "ApplyMaskOutput.nii.gz")
        cmd9 = [
            "fslmaths", output_t1w_image_restore, "-mas", output_t1w_image_restore_brain, apply_mask_output
        ]
        run_command(cmd9)

        outputs = {
            "space-template_desc-preproc_T1w": apply_mask_output,
            "space-template_desc-head_T1w": output_t1w_image_restore,
            "space-template_desc-T1w_mask": output_t1w_brain_mask_to_template,
            "from-T1w_to-template_mode-image_xfm": merged_xfms,
            "from-template_to-T1w_mode-image_xfm": merged_inv_xfms,
        }

        return outputs

def run_c4d(input_name, output_name):
    """Run c4d to split a 4D image into 3D images."""
    import os

    output1 = os.path.join(os.getcwd(), output_name + "1.nii.gz")
    output2 = os.path.join(os.getcwd(), output_name + "2.nii.gz")
    output3 = os.path.join(os.getcwd(), output_name + "3.nii.gz")

    cmd = f"c4d -mcs {input_name} -oo {output1} {output2} {output3}"
    os.system(cmd)

    return output1, output2, output3

def register_symmetric_ANTs_anat_to_template(cfg, strat_pool, pipe_num, opt=None):
    
    """Register T1 to symmetric template with ANTs."""
    params = cfg["registration_workflows"]["anatomical_registration"]["registration"]["ANTs"]["T1_registration"]
    interpolation = cfg["registration_workflows"]["anatomical_registration"]["registration"]["ANTs"]["interpolation"]

    # File paths
    input_brain = strat_pool.get_data(["desc-preproc_T1w", "space-longitudinal_desc-preproc_T1w"])
    reference_brain = strat_pool.get_data("T1w-brain-template-symmetric")
    input_head = strat_pool.get_data(["desc-head_T1w", "desc-preproc_T1w", "space-longitudinal_desc-reorient_T1w"])
    reference_head = strat_pool.get_data("T1w-template-symmetric")
    input_mask = strat_pool.get_data(["space-T1w_desc-brain_mask", "space-longitudinal_desc-brain_mask"])
    reference_mask = strat_pool.get_data("dilated-symmetric-brain-mask")
    lesion_mask = strat_pool.get_data("label-lesion_mask") if strat_pool.check_rpool("label-lesion_mask") else None
    
    # Output paths
    output_dir = f"ANTS_T1_to_template_symmetric_{pipe_num}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build the ANTs registration command
    command = (
        f"antsRegistration "
        f"-d 3 "
        f"-r [ {reference_brain} , {input_brain} , 1 ] "
        f"-m MI[ {reference_brain} , {input_brain} , 1 , 32 ] "
        f"-t SyN[ 0.1, 3, 0 ] "
        f"-c [ 100x70x50x20, 1e-8, 10 ] "
        f"-s 4x2x1x0 "
        f"-f 8x4x2x1 "
        f"-u 1 "
        f"-z 1 "
        f"-o [ {output_dir}/transform_, {output_dir}/output.nii.gz ] "
        f"-x {input_mask} "
        f"-v"
    )

    # Run the ANTs registration
    run_command(command)
    
    # Handle output renaming if necessary
    outputs = {
        "space-symtemplate_desc-preproc_T1w": f"{output_dir}/output.nii.gz",
        "from-T1w_to-symtemplate_mode-image_desc-linear_xfm": f"{output_dir}/transform_0GenericAffine.mat",
        "from-symtemplate_to-T1w_mode-image_desc-linear_xfm": f"{output_dir}/transform_0GenericAffine.mat",
        "from-T1w_to-symtemplate_mode-image_desc-nonlinear_xfm": f"{output_dir}/transform_1Warp.nii.gz",
        "from-symtemplate_to-T1w_mode-image_desc-nonlinear_xfm": f"{output_dir}/transform_1Warp.nii.gz",
        "from-T1w_to-symtemplate_mode-image_xfm": f"{output_dir}/transform_1Warp.nii.gz",
        "from-symtemplate_to-T1w_mode-image_xfm": f"{output_dir}/transform_1Warp.nii.gz",
        "from-longitudinal_to-symtemplate_mode-image_desc-linear_xfm": f"{output_dir}/transform_0GenericAffine.mat",
        "from-symtemplate_to-longitudinal_mode-image_desc-linear_xfm": f"{output_dir}/transform_0GenericAffine.mat",
        "from-longitudinal_to-symtemplate_mode-image_desc-nonlinear_xfm": f"{output_dir}/transform_1Warp.nii.gz",
        "from-symtemplate_to-longitudinal_mode-image_desc-nonlinear_xfm": f"{output_dir}/transform_1Warp.nii.gz",
        "from-longitudinal_to-symtemplate_mode-image_xfm": f"{output_dir}/transform_1Warp.nii.gz",
        "from-symtemplate_to-longitudinal_mode-image_xfm": f"{output_dir}/transform_1Warp.nii.gz",
    }

    return outputs

def register_symmetric_FSL_anat_to_template(cfg, strat_pool, pipe_num, opt=None):
    """Register T1w to symmetric template with FSL."""
    # Get FSL parameters
    interpolation = cfg["registration_workflows"]["anatomical_registration"]["registration"]["FSL-FNIRT"]["interpolation"]
    fnirt_config = cfg["registration_workflows"]["anatomical_registration"]["registration"]["FSL-FNIRT"]["fnirt_config"]

    # File paths
    input_brain = strat_pool.get_data(["desc-brain_T1w", "space-longitudinal_desc-preproc_T1w"])
    reference_brain = strat_pool.get_data("T1w-brain-template-symmetric")
    input_head = strat_pool.get_data(["desc-preproc_T1w", "space-longitudinal_desc-reorient_T1w"])
    reference_head = strat_pool.get_data("T1w-template-symmetric")
    reference_mask = strat_pool.get_data("dilated-symmetric-brain-mask")

    # Output directory and files
    output_dir = f"register_{opt}_anat_to_template_symmetric_{pipe_num}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Build the FSL registration command
    command = (
        f"fnirt --in={input_brain} "
        f"--ref={reference_brain} "
        f"--iout={output_dir}/output.nii.gz "
        f"--fout={output_dir}/transform.nii.gz "
        f"--aff={output_dir}/affine.mat "
        f"--config={fnirt_config} "
        f"--warpres=10,10,10 "
        f"--refmask={reference_mask} "
        f"--cout={output_dir}/coef.nii.gz "
        f"--reffwhm=8 "
        f"--refwarp={output_dir}/refwarp.nii.gz "
        f"--refhead={reference_head} "
        f"--interp={interpolation}"
    )

    # Run the FSL registration
    run_command(command)
    
    # Handle output renaming if necessary
    outputs = {
        "space-symtemplate_desc-preproc_T1w": f"{output_dir}/output.nii.gz",
        "from-T1w_to-symtemplate_mode-image_desc-linear_xfm": f"{output_dir}/affine.mat",
        "from-symtemplate_to-T1w_mode-image_desc-linear_xfm": f"{output_dir}/affine.mat",
        "from-T1w_to-symtemplate_mode-image_xfm": f"{output_dir}/transform.nii.gz",
        "from-longitudinal_to-symtemplate_mode-image_desc-linear_xfm": f"{output_dir}/affine.mat",
        "from-symtemplate_to-longitudinal_mode-image_desc-linear_xfm": f"{output_dir}/affine.mat",
        "from-longitudinal_to-symtemplate_mode-image_xfm": f"{output_dir}/transform.nii.gz",
    }

    return outputs
