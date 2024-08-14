
# Separated and de-nipype'd C-PAC longitudinal workflow
# Anatomical preproc runs before this in C-PAC

import os

from longitudinal_reg.longitudinal_utils import (
    subject_specific_template,
    warp_longitudinal_T1w_to_template,
    warp_longitudinal_seg_to_T1w,
    fs_generate_template
)

from longitudinal_reg.register import (
    mask_longitudinal_T1w_brain,
    register_ANTs_anat_to_template,
    register_FSL_anat_to_template,
    overwrite_transform_anat_to_template,
    register_symmetric_ANTs_anat_to_template,
    register_symmetric_FSL_anat_to_template
)

from longitudinal_reg.segment import (
    tissue_seg_fsl_fast
)


def run_workflow(config, cpac_dir, out_dir, subject_id):


    cpac_dirs = []
    sessions = []
    cpac_dirs.append(os.path.join(cpac_dir, "anat"))

    for root, dirs, files in os.walk(cpac_dir):
        # The root is the current folder, dirs are the subfolders in the current folder
        for dir_name in dirs:
            sessions.append(dir_name)
        break
    print(sessions)
    # To change template generation algorithm
    use_fs = False

    # loop over the different anat preproc strategies
    strats_brain_dct = {}
    strats_head_dct = {}
    for dir in os.listdir(cpac_dir):
        if os.path.isdir(os.path.join(cpac_dir, dir)):
            anat_dir = os.path.join(cpac_dir, dir, "anat")
            if os.path.isdir(anat_dir):
                for filename in os.listdir(anat_dir):
                    if "T1w.nii" in filename:
                        for tag in filename.split("_"):
                            if "desc-" in tag and "preproc" in tag:
                                if tag not in strats_brain_dct:
                                    strats_brain_dct[tag] = []
                                strats_brain_dct[tag].append(
                                    os.path.join(dir, filename)
                                )
                                if tag not in strats_head_dct:
                                    strats_head_dct[tag] = []
                                head_file = filename.replace(tag, "desc-head")
                                strats_head_dct[tag].append(
                                    os.path.join(dir, head_file)
                                )
    print(strats_head_dct)
    print(strats_brain_dct)
    for strat in strats_brain_dct.keys():

        # This will generate the longitudinal template 

        if use_fs:
            warp_list, brain_template = fs_generate_template()
        else:
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

        # TODO: does this happen in fs command? I don't think so
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
        # lesion_mask = strat_pool.get_data("label-lesion_mask") if strat_pool.check_rpool("label-lesion_mask") else None

        # registration - once for each session
        for session in sessions:
            # cd into session folder
            # pipe num
            if config.registration_workflows.anatomical_registration["run"] and \
                config.registration_workflows.anatomical_registration["using"] == "ANTS":
                if config.voxel_mirrored_homotopic_connectivity["run"]:
                    reg_outputs = register_symmetric_ANTs_anat_to_template()
                else:
                    reg_outputs = register_ANTs_anat_to_template(config, input_brain=brain_template, input_head=skull_template,
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
                    reg_outputs = register_FSL_anat_to_template(config, skull_template, reference_head, reference_brain, template_ref_mask)
            if config.registration_workflows.anatomical_registration.overwrite_transform["run"]:
                overwrite_transform_anat_to_template() 
            
            # tissue segmentation
            if config.segmentation["run"] and config.segmentation.tissue_segmentation["using"] == "FSL-FAST":
                labels = tissue_seg_fsl_fast(config, pipe_num=session)

        transform = reg_outputs["from-longitudinal_to-template_mode-image_xfm"]
    
        seg_transform = reg_outputs["from-longitudinal_to-T1w_mode-image_desc-linear_xfm"]
        # begin single-session stuff again
        for session in sessions:
            brain_template = warp_longitudinal_T1w_to_template(config, pipe_num=session, input_image=brain_template, 
                                                                                reference=reference_brain, transform=transform)
            label_outputs = warp_longitudinal_seg_to_T1w(config, pipe_num=session, images=labels, reference=reference_brain, transform=seg_transform)


def main():
    config = "/home/c-pac_user/CMI/longitudinal/config.yml"
    cpac_dir = "/home/c-pac_user/CMI/data/pipeline_cpac-default-pipeline"
    out_dir = "/home/c-pac_user/CMI/longitudinal"
    
    print(out_dir)
    print(os.path.isdir(cpac_dir))
    for root, dirs, files in os.walk(cpac_dir):
        for dir_name in dirs:
            print('dir name', dir_name)
            full_path = os.path.join(root, dir_name)
            output_subfolder = os.path.join(out_dir, dir_name)
            os.makedirs(output_subfolder, exist_ok=True)
            
            # Change the working directory to the output folder
            # THis is necessary for the os.cwd calls in cpac functions
            os.chdir(output_subfolder)
            run_workflow(config, full_path, output_subfolder, dir_name)
            os.chdir(out_dir)
        break

if __name__ == "__main__":
    main()
