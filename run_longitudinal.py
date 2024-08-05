
# Separated and de-nipype'd C-PAC longitudinal workflow
# Anatomical preproc runs before this in C-PAC

import subprocess
import os


def run_workflow(config):

    cpac_dir = os.path.join(
            out_dir, f"pipeline_{orig_pipe_name}", subject_id, unique_id
        )
        cpac_dirs.append(os.path.join(cpac_dir, "anat"))

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
        brain_template, skull_template, output_brain_list, output_skull_list,
            warp_list = subject_specific_template(
                input_brain_list=strats_brain_dct[strat], 
                input_skull_list=strats_head_dct[strat],
                avg_method=config.longitudinal_template_generation["average_method"],
                dof=config.longitudinal_template_generation["dof"],
                interp=config.longitudinal_template_generation["interp"],
                cost=config.longitudinal_template_generation["cost"],
                convergence_threshold=config.longitudinal_template_generation[
                    "convergence_threshold"],
                thread_pool=config.longitudinal_template_generation["thread_pool"],
                unique_id_list=list(session_wfs.keys()))

        longitudinal_brain_mask = mask_longitudinal_T1w_brain(brain_template)
        
        #TODO - how to deal w templates?? idk
        reference_head = rpool.get_data("T1w-template")
        reference_mask = rpool.get_data("T1w-brain-template-mask") if strat_pool.check_rpool("T1w-brain-template-mask") else None
        lesion_mask = strat_pool.get_data("label-lesion_mask") if strat_pool.check_rpool("label-lesion_mask") else None

        # register ants
        if config.registration_workflows.anatomical_registration["run"] and 
            config.registration_workflows.anatomical_registration["using"] == "ANTS":
            register_ants_anat_to_template(input_brain=brain_template, input_head=skull_template, 
                        input_mask=longitudinal_brain_mask, reference_brain, 
                        reference_head, reference_mask, lesion_mask, opt=None)
        # TODO
        if config.registration_workflows.anatomical_registration["run"] and 
            (config.registration_workflows.anatomical_registration["using"] == "FSL" or 
            config.registration_workflows.anatomical_registration["using"] == "FSL-linear"):
            register_fsl_anat_to_template()
        
        # To implement:
        # overwrite_transform_anat_to_template, 
        # register_symmetric_ANTs_anat_to_template
        # register_symmetric_FSL_anat_to_template
        # tissue_seg_fsl_fast

''' 
de-nipype work ends here
Rest is WIP

'''
        # now, just write out a copy of the above to each session
        config.pipeline_setup["pipeline_name"] = orig_pipe_name
        for session in sub_list:
            unique_id = session["unique_id"]

            try:
                creds_path = session["creds_path"]
                if creds_path and "none" not in creds_path.lower():
                    if os.path.exists(creds_path):
                        input_creds_path = os.path.abspath(creds_path)
                    else:
                        err_msg = (
                            'Credentials path: "%s" for subject "%s" '
                            'session "%s" was not found. Check this path '
                            "and try again." % (creds_path, subject_id, unique_id)
                        )
                        raise Exception(err_msg)
                else:
                    input_creds_path = None
            except KeyError:
                input_creds_path = None

            wf = initialize_nipype_wf(config, sub_list[0])

            wf, rpool = initiate_rpool(wf, config, session)

            config.pipeline_setup["pipeline_name"] = f"longitudinal_{orig_pipe_name}"
            
            data_paths = {}
            data_paths["derivatives_dir"] = 
            rpool = ingress_output_dir(
                wf, config, rpool, long_id, creds_path=input_creds_path
            )

            select_node_name = f"select_{unique_id}"
            select_sess = pe.Node(
                Function(
                    input_names=["session", "output_brains", "warps"],
                    output_names=["brain_path", "warp_path"],
                    function=select_session,
                ),
                name=select_node_name,
            )
            select_sess.inputs.session = unique_id

            wf.connect(template_node, "output_brain_list", select_sess, "output_brains")
            wf.connect(template_node, "warp_list", select_sess, "warps")

            rpool.set_data(
                "space-longitudinal_desc-preproc_T1w",
                select_sess,
                "brain_path",
                {},
                "",
                select_node_name,
            )

            rpool.set_data(
                "from-T1w_to-longitudinal_mode-image_desc-linear_xfm",
                select_sess,
                "warp_path",
                {},
                "",
                select_node_name,
            )

            config.pipeline_setup["pipeline_name"] = orig_pipe_name
            excl = ["space-template_desc-brain_T1w", "space-T1w_desc-brain_mask"]

            rpool.gather_pipes(wf, config, add_excl=excl)
            
            wf.run()

    # begin single-session stuff again
    for session in sub_list:
        unique_id = session["unique_id"]

        try:
            creds_path = session["creds_path"]
            if creds_path and "none" not in creds_path.lower():
                if os.path.exists(creds_path):
                    input_creds_path = os.path.abspath(creds_path)
                else:
                    err_msg = (
                        'Credentials path: "%s" for subject "%s" '
                        'session "%s" was not found. Check this path '
                        "and try again." % (creds_path, subject_id, unique_id)
                    )
                    raise Exception(err_msg)
            else:
                input_creds_path = None
        except KeyError:
            input_creds_path = None

        wf = initialize_nipype_wf(config, sub_list[0])

        wf, rpool = initiate_rpool(wf, config, session)

        pipeline_blocks = [
            warp_longitudinal_T1w_to_template,
            warp_longitudinal_seg_to_T1w,
        ]

        wf = connect_pipeline(wf, config, rpool, pipeline_blocks)

        rpool.gather_pipes(wf, config)

        # this is going to run multiple times!
        # once for every strategy!
        wf.run()


def main():
    config = "/path/to/config"
    run_workflow(config)

if __name__ == "__main__":
    main()
