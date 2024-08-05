import subprocess

def register_ANTs_anat_to_template(input_brain, input_head, input_mask, reference_brain, reference_head,
    reference_mask, lesion_mask, opt=None):
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

    ants_rc, outputs = ANTs_registration_connector(
        f"ANTS_T1_to_template_{pipe_num}", cfg, params, orig="T1w"
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

    return output_paths

def mask_longitudinal_T1w_brain(brain_template, pipe_num, opt=None):
    """
    Apply a binary mask to the longitudinal T1w brain image using FSL's fslmaths command.

    Parameters
    ----------
    strat_pool : object
        An object with a method get_data that returns the node and output path.
    pipe_num : int
        Pipeline number.
    opt : optional
        Optional parameter.

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

    subprocess.run(fslmaths_cmd, check=True)

    # Output dictionary
    outputs = {"space-longitudinal_desc-brain_mask": output_file}

    return outputs


#TODO- create lesion preproc

def ANTs_registration_connector(
    cfg, params, orig="T1w", symmetric=False, template="T1w",
    input_brain, reference_brain, input_head, reference_head, input_mask, reference_mask, 
    transform, interpolation
):
    """Transform raw data to template with ANTs."""

    if params is None:
        err_msg = (
            "\n\n[!] C-PAC says: \nYou have selected ANTs as your"
            " anatomical registration method.\nHowever, no ANTs parameters were"
            " specified.\nPlease specify ANTs parameters properly and try again."
        )
        raise RequiredFieldInvalid(err_msg)

    sym = "sym" if symmetric else ""
    symm = "_symmetric" if symmetric else ""
    tmpl = "EPI" if template == "EPI" else ""


    if cfg.registration_workflows["anatomical_registration"]["registration"]["ANTs"]["use_lesion_mask"]:
        lesion_preproc = create_lesion_preproc(wf_name=f"lesion_preproc{symm}")
        fixed_image_mask = create_lesion_preproc()
    else:
        fixed_image_mask = None

    transforms, normalized_output_brain = create_wf_calculate_ants_warp(
        num_threads=cfg.pipeline_setup["system_config"]["num_ants_threads"],
        reg_ants_skull=cfg["registration_workflows"]["anatomical_registration"][
        "reg_with_skull"], 
        moving_brain=input_brain, 
        reference_brain, 
        moving_skull=input_head, 
        reference_skull=reference_head, 
        reference_mask,
        moving_mask=input_mask, 
        fixed_image_mask,
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
    result = subprocess.run(transform_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Error applying transforms: {result.stderr}")
    else:
        print(f"Transform applied successfully: {result.stdout}")

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
        reference_skull,
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

    def select_warp(warp_list, selection):
        return seperate_warps_list(warp_list, selection)
    
    ants_initial_xfm = select_warp(warp_list, "Initial")
    ants_rigid_xfm = select_warp(warp_list, "Rigid")
    ants_affine_xfm = select_warp(warp_list, "Affine")
    warp_field = select_warp(warp_list, "Warp")
    inverse_warp_field = select_warp(warp_list, "Inverse")

    
    return (ants_initial_xfm,
        ants_rigid_xfm,
        ants_affine_xfm,
        warp_field,
        inverse_warp_field,
        composite_transform,
        wait,
        normalized_output_brain)

def seperate_warps_list(warp_list, selection):
    """Select the warp from the warp list."""
    selected_warp = None
    for warp in warp_list:
        if selection == "Warp":
            if "3Warp" in warp or "2Warp" in warp or "1Warp" in warp:
                selected_warp = warp
        elif selection in warp:
            selected_warp = warp
    return selected_warp

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
                    raise RequiredFieldInvalid(err_msg)
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
                    raise RequiredFieldInvalid(err_msg)
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
                                raise RequiredFieldInvalid(err_msg)
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
                                raise RequiredFieldInvalid(err_msg)
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
                                raise RequiredFieldInvalid(err_msg)
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
                                raise RequiredFieldInvalid(err_msg)
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
                                raise RequiredFieldInvalid(err_msg)
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
