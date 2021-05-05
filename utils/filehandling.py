
import os
import shutil
from datetime import datetime

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        shutil.rmtree(dir_name, ignore_errors=True)
        os.makedirs(dir_name)


def get_str_experiments(flags):
    dateTimeObj = datetime.now()
    dateStr = dateTimeObj.strftime("%Y_%m_%d_%H_%M_%S_%f")
    str_experiments = flags.dataset + '_' + dateStr;
    return str_experiments


def create_dir_structure_testing(exp):
    flags = exp.flags;
    for k, label_str in enumerate(exp.labels):
        dir_gen_eval_label = os.path.join(flags.dir_gen_eval, label_str)
        create_dir(dir_gen_eval_label)
        dir_inference_label = os.path.join(flags.dir_inference, label_str)
        create_dir(dir_inference_label)


def create_dir_structure(flags, train=True):
    if train:
        str_experiments = get_str_experiments(flags)
        flags.dir_experiment_run = os.path.join(flags.dir_experiment, str_experiments)
        flags.str_experiment = str_experiments;
    else:
        flags.dir_experiment_run = flags.dir_experiment;

    print(flags.dir_experiment_run)
    if train:
        create_dir(flags.dir_experiment_run)

    flags.dir_checkpoints = os.path.join(flags.dir_experiment_run, 'checkpoints')
    if train:
        create_dir(flags.dir_checkpoints)

    flags.dir_logs = os.path.join(flags.dir_experiment_run, 'logs')
    if train:
        create_dir(flags.dir_logs)
    print(flags.dir_logs)

    flags.dir_logs_clf = os.path.join(flags.dir_experiment_run, 'logs_clf')
    if train:
        create_dir(flags.dir_logs_clf)

    flags.dir_gen_eval = os.path.join(flags.dir_experiment_run, 'generation_evaluation')
    if train:
        create_dir(flags.dir_gen_eval)

    flags.dir_inference = os.path.join(flags.dir_experiment_run, 'inference')
    if train:
        create_dir(flags.dir_inference)

    if flags.dir_fid is None:
        flags.dir_fid = flags.dir_experiment_run;
    elif not train:
        flags.dir_fid = os.path.join(flags.dir_experiment_run, 'fid_eval');
        if not os.path.exists(flags.dir_fid):
            os.makedirs(flags.dir_fid);
    flags.dir_gen_eval_fid = os.path.join(flags.dir_fid, 'fid');
    create_dir(flags.dir_gen_eval_fid)

    flags.dir_plots = os.path.join(flags.dir_experiment_run, 'plots')
    if train:
        create_dir(flags.dir_plots)
    flags.dir_swapping = os.path.join(flags.dir_plots, 'swapping')
    if train:
        create_dir(flags.dir_swapping)

    flags.dir_random_samples = os.path.join(flags.dir_plots, 'random_samples')
    if train:
        create_dir(flags.dir_random_samples)

    flags.dir_cond_gen = os.path.join(flags.dir_plots, 'cond_gen')
    if train:
        create_dir(flags.dir_cond_gen)
    return flags;
