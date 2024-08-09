
# Separated and de-nipype'd C-PAC longitudinal workflow
# Anatomical preproc runs before this in C-PAC

import subprocess
import os

from longitudinal_reg.longitudinal_utils import (
    subject_specific_template,
    warp_longitudinal_T1w_to_template,
    warp_longitudinal_seg_to_T1w
)

from longitudinal_reg.register import (
    mask_longitudinal_T1w_brain,
    register_ANTS_anat_to_template,
    register_fsl_anat_to_template,
    overwrite_transform_anat_to_template,
    register_symmetric_ANTs_anat_to_template,
    register_symmetric_FSL_anat_to_template
)

from longitudinal_reg.segment import (
    tissue_seg_fsl_fast
)


def run_workflow(config, out_dir):

    # cd into output dir
    # maybe cd into subdir for each session???

    cpac_dirs = []
    sub_list = []
    cpac_dir = out_dir
    
    cpac_dirs.append(os.path.join(cpac_dir, "anat"))
    sessions = []

    # loop over the different anat preproc strategies
    strats_brain_dct = {}
    strats_head_dct = {}
    for cpac_dir in cpac_dirs:
        if os.path.isdir(cpac_dir):
            for filename in os.listdir(cpac_dir):
                if "T1w.nii" in filename:
                    for tag in filename.split("_"):
                        if "desc-" in tag and "preproc" in tag:
                            if tag not in strats_brain_dct:
                                strats_brain_dct[tag] = []
                            strats_brain_dct[tag].append(
                                os.path.join(cpac_dir, filename)
                            )
                            if tag not in strats_head_dct:
                                strats_head_dct[tag] = []
                            head_file = filename.replace(tag, "desc-head")
                            strats_head_dct[tag].append(
                                os.path.join(cpac_dir, head_file)
                            )

    for strat in strats_brain_dct.keys():

        # This will generate the longitudinal template 
        brain_template, skull_template, output_brain_list, \
            output_skull_list, warp_list = subject_specific_template( 
                input_brain_list=strats_brain_dct[strat], 
                input_skull_list=strats_head_dct[strat],
                avg_method=config.longitudinal_template_generation["average_method"],
                dof=config.longitudinal_template_generation["dof"],
                interp=config.longitudinal_template_generation["interp"],
                cost=config.longitudinal_template_generation["cost"],
                convergence_threshold=config.longitudinal_template_generation[
                    "convergence_threshold"],
                thread_pool=config.longitudinal_template_generation["thread_pool"],
                unique_id_list=list(sessions))

        longitudinal_brain_mask = mask_longitudinal_T1w_brain(brain_template)
        
        #TODO - how to deal w templates?? just paths for now
        reference_head = config.registration_workflows.anatomical_registration["T1w_template"]
        reference_mask = config.registration_workflows.anatomical_registration["T1w_brain_template_mask"] if \
            config.registration_workflows.anatomical_registration["T1w_brain_template_mask"] else None
        reference_brain = config.registration_workflows.anatomical_registration["T1w_brain_template"] if \
            config.registration_workflows.anatomical_registration["T1w_brain_template"] else None
        template_ref_mask = config.registration_workflows.anatomical_registration.registration.FSL_FNIRT["ref_mask"]
        T1w_template_symmetric = config.voxel_mirrored_homotopic_connectivity["symmetric_registration_T1w_template_symmetric"]

        #TODO: symmetric templates!!!


        #rpool.get_data("T1w-brain-template-mask") if strat_pool.check_rpool("T1w-brain-template-mask") else None
        # lesion_mask = 
        #strat_pool.get_data("label-lesion_mask") if strat_pool.check_rpool("label-lesion_mask") else None

        # registration - once for each session
        for session in sub_list:
            # cd into session folder
            if config.registration_workflows.anatomical_registration["run"] and \
                config.registration_workflows.anatomical_registration["using"] == "ANTS":
                if config.voxel_mirrored_homotopic_connectivity["run"]:
                    reg_outputs = register_symmetric_ANTs_anat_to_template()
                else:
                    reg_outputs = register_ANTS_anat_to_template(config, input_brain=brain_template, input_head=skull_template,
                            input_mask=longitudinal_brain_mask, reference_brain=reference_brain, 
                            reference_head=reference_head, reference_mask=reference_mask, 
                            lesion_mask=None)
            # shouldn't fsl take the longitudinal mask? what
            elif config.registration_workflows.anatomical_registration["run"] and \
                (config.registration_workflows.anatomical_registration["using"] == "FSL" or 
                config.registration_workflows.anatomical_registration["using"] == "FSL-linear"):
                if config.voxel_mirrored_homotopic_connectivity["run"]:
                    reg_outputs = register_symmetric_FSL_anat_to_template()
                else:
                    reg_outputs = register_fsl_anat_to_template(config, skull_template, reference_head, reference_brain, template_ref_mask)
            if config.registration_workflows.anatomical_registration.overwrite_transform["run"]:
                overwrite_transform_anat_to_template() 
            
            # tissue segmentation
            if config.segmentation["run"] and config.segmentation.tissue_segmentation["using"] == "FSL-FAST":
                tissue_seg_fsl_fast()


        # begin single-session stuff again
        for session in sub_list:
            warp_longitudinal_T1w_to_template()
            warp_longitudinal_seg_to_T1w()


def main():
    config = "/path/to/config"
    out_dir = "/cpac/preproc/outputs"
    
    # for subject in out_dir:
    run_workflow(config, out_dir)

if __name__ == "__main__":
    main()
