import subprocess
import os

from longitudinal_reg.longitudinal_utils import run_command

# Main function to perform FSL-FAST tissue segmentation
def tissue_seg_fsl_fast(wf, cfg, strat_pool, pipe_num, opt=None):
    # Configuration parameters
    img_type = 1
    segments = True
    probability_maps = True
    out_basename = "segment"

    # Get data from strat_pool
    space_longitudinal_desc_preproc_T1w = \
        strat_pool.get_data(["space-longitudinal_desc-preproc_T1w"], report_fetched=True)
    space_longitudinal_desc_brain_mask = \
        strat_pool.get_data("space-longitudinal_desc-brain_mask")
    from_template_to_longitudinal_mode_image_desc_linear_xfm = \
        strat_pool.get_data(["from-template_to-longitudinal_mode-image_desc-linear_xfm"])
    
    # Perform FSL FAST segmentation
    segment_command = f"fast -t {img_type} -g {int(segments)} -p {int(probability_maps)} -o {out_basename} {desc_brain_T1w}"
    run_command(segment_command)

    use_custom_threshold = cfg["segmentation"]["tissue_segmentation"]["FSL-FAST"]["thresholding"]["use"] == "Custom"
    use_priors = cfg["segmentation"]["tissue_segmentation"]["FSL-FAST"]["use_priors"]["run"]

    # IT looks like the 'check-if-file-empty' nodes never run - they 
    # are never given inputs. 
    
    if use_priors:
        xfm = from_template_to_longitudinal_mode_image_desc_linear_xfm
        xfm_prov = strat_pool.get_cpac_provenance(xfm)
        reg_tool = check_prov_for_regtool(xfm_prov)
    else:
        xfm_prov = None
        reg_tool = None
        xfm = None

    csf_threshold = cfg["segmentation"]["tissue_segmentation"]["FSL-FAST"]["thresholding"]["Custom"]["CSF_threshold_value"]
    gm_threshold = cfg["segmentation"]["tissue_segmentation"]["FSL-FAST"]["thresholding"]["Custom"]["GM_threshold_value"]
    wm_threshold = cfg["segmentation"]["tissue_segmentation"]["FSL-FAST"]["thresholding"]["Custom"]["WM_threshold_value"]

    # Process CSF
    process_csf = process_segment_map(f"CSF_{pipe_num}", use_priors, use_custom_threshold, reg_tool, csf_threshold)

    # Generate CSF PVE mask
    pve_csf_command = f"fslmaths {out_basename}_pve_0 -thr 0.5 -uthr 1.5 -bin {out_basename}_pve_0_bin"
    run_command(pve_csf_command)

    # Process GM
    process_gm = process_segment_map(f"GM_{pipe_num}", use_priors, use_custom_threshold, reg_tool, gm_threshold)

    # Generate GM PVE mask
    pve_gm_command = f"fslmaths {out_basename}_pve_1 -thr 1.5 -uthr 2.5 -bin {out_basename}_pve_1_bin"
    run_command(pve_gm_command)

    # Process WM
    process_wm = process_segment_map(f"WM_{pipe_num}", use_priors, use_custom_threshold, reg_tool, wm_threshold)

    # Generate WM PVE mask
    pve_wm_command = f"fslmaths {out_basename}_pve_2 -thr 2.5 -uthr 3.5 -bin {out_basename}_pve_2_bin"
    run_command(pve_wm_command)

    # Outputs
    outputs = {
        "label-CSF_probseg": f"{out_basename}_pve_0",
        "label-GM_probseg": f"{out_basename}_pve_1",
        "label-WM_probseg": f"{out_basename}_pve_2",
        "label-CSF_mask": f"{out_basename}_seg_0",
        "label-GM_mask": f"{out_basename}_seg_1",
        "label-WM_mask": f"{out_basename}_seg_2",
        "label-CSF_desc-preproc_mask": process_csf,
        "label-GM_desc-preproc_mask": process_gm,
        "label-WM_desc-preproc_mask": process_wm,
        "label-CSF_pveseg": f"{out_basename}_pve_0_bin",
        "label-GM_pveseg": f"{out_basename}_pve_1_bin",
        "label-WM_pveseg": f"{out_basename}_pve_2_bin"
    }

    return wf, outputs

def process_segment_map(wf_name, use_priors, use_custom_threshold, reg_tool):
    """
    Create a sub workflow used inside segmentation workflow to process probability maps obtained in segmentation.

    Parameters
    ----------
    wf_name : string
        Workflow Name
    use_priors : boolean
        Whether or not to use template-space tissue priors to further refine the resulting segmentation tissue masks.
    use_custom_threshold : boolean
        Whether or not to use a custom threshold.
    reg_tool : string
        The registration tool to be used.

    Returns
    -------
    dict
        Dictionary of output file paths for process_segment_map workflow
    """

    def form_threshold_string(threshold):
        return f"-thr {threshold} "

    output_files = {}

    if use_priors:
        apply_xfm = f"apply_transform --input_image {inputNode['tissue_prior']} --reference {inputNode['brain']} --transform {inputNode['template_to_T1_xfm']} --interpolation NearestNeighbor"
        run_command(apply_xfm)

        overlap_command = f"fslmaths {inputNode['tissue_class_file' if not use_custom_threshold else 'probability_tissue_map']} -mas {apply_xfm_output} overlap_output"
        run_command(overlap_command)

        input_file = "overlap_output"
    else:
        input_file = inputNode['tissue_class_file' if not use_custom_threshold else 'probability_tissue_map']

    if use_custom_threshold:
        threshold_command = f"fslmaths {input_file} {form_threshold_string(inputNode['threshold'])} threshold_output"
        run_command(threshold_command)

        binarize_command = "fslmaths threshold_output -bin binarize_output"
        run_command(binarize_command)

        output_files['segment_mask'] = "binarize_output"
    else:
        output_files['segment_mask'] = input_file

    return output_files


def pick_wm_prob_0(probability_maps):
    """Returns the csf probability map from the list of segmented
    probability maps.

    Parameters
    ----------
    probability_maps : list (string)
        List of Probability Maps

    Returns
    -------
    file : string
        Path to segment_prob_0.nii.gz is returned
    """
    if isinstance(probability_maps, list):
        if len(probability_maps) == 1:
            probability_maps = probability_maps[0]
        for filename in probability_maps:
            if filename.endswith("prob_0.nii.gz"):
                return filename
    return None


def pick_wm_prob_1(probability_maps):
    """Returns the gray matter probability map from the list of segmented probability maps.

    Parameters
    ----------
    probability_maps : list (string)
        List of Probability Maps

    Returns
    -------
    file : string
        Path to segment_prob_1.nii.gz is returned
    """
    if isinstance(probability_maps, list):
        if len(probability_maps) == 1:
            probability_maps = probability_maps[0]
        for filename in probability_maps:
            if filename.endswith("prob_1.nii.gz"):
                return filename
    return None


def pick_wm_prob_2(probability_maps):
    """Returns the white matter probability map from the list of segmented probability maps.

    Parameters
    ----------
    probability_maps : list (string)
        List of Probability Maps

    Returns
    -------
    file : string
        Path to segment_prob_2.nii.gz is returned
    """
    if isinstance(probability_maps, list):
        if len(probability_maps) == 1:
            probability_maps = probability_maps[0]
        for filename in probability_maps:
            if filename.endswith("prob_2.nii.gz"):
                return filename
    return None


def pick_wm_class_0(tissue_class_files):
    """Returns the csf tissue class file from the list of segmented tissue class files.

    Parameters
    ----------
    tissue_class_files : list (string)
        List of tissue class files

    Returns
    -------
    file : string
        Path to segment_seg_0.nii.gz is returned
    """
    if isinstance(tissue_class_files, list):
        if len(tissue_class_files) == 1:
            tissue_class_files = tissue_class_files[0]
        for filename in tissue_class_files:
            if filename.endswith("seg_0.nii.gz"):
                return filename
    return None


def pick_wm_class_1(tissue_class_files):
    """Returns the gray matter tissue class file from the list of segmented tissue class files.

    Parameters
    ----------
    tissue_class_files : list (string)
        List of tissue class files

    Returns
    -------
    file : string
        Path to segment_seg_1.nii.gz is returned
    """
    if isinstance(tissue_class_files, list):
        if len(tissue_class_files) == 1:
            tissue_class_files = tissue_class_files[0]
        for filename in tissue_class_files:
            if filename.endswith("seg_1.nii.gz"):
                return filename
    return None


def pick_wm_class_2(tissue_class_files):
    """Returns the white matter tissue class file from the list of segmented tissue class files.

    Parameters
    ----------
    tissue_class_files : list (string)
        List of tissue class files

    Returns
    -------
    file : string
        Path to segment_seg_2.nii.gz is returned
    """
    if isinstance(tissue_class_files, list):
        if len(tissue_class_files) == 1:
            tissue_class_files = tissue_class_files[0]
        for filename in tissue_class_files:
            if filename.endswith("seg_2.nii.gz"):
                return filename
    return None
