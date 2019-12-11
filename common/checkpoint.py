from __future__ import print_function

import glob
import os
import shutil
import sys

import torch


def save_models(sample_dir, deblend_net, Inp_net, Dis_net):
    model_dir = os.path.join(sample_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    deblend_pth = os.path.join(model_dir, 'deblend.pth')
    Inp_pth = os.path.join(model_dir, 'Inp.pth')
    Dis_pth = os.path.join(model_dir, 'Dis.pth')
    torch.save(deblend_net.state_dict(), deblend_pth)
    torch.save(Inp_net.state_dict(), Inp_pth)
    torch.save(Dis_net.state_dict(), Dis_pth)


def save_models_stack(sample_dir, deblend_net, refine_net, critic_net):
    model_dir = os.path.join(sample_dir, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    deblend_pth = os.path.join(model_dir, 'deblend.pth')
    refine_pth = os.path.join(model_dir, 'refine.pth')
    critic_pth = os.path.join(model_dir, 'critic.pth')
    torch.save(deblend_net.state_dict(), deblend_pth)
    torch.save(refine_net.state_dict(), refine_pth)
    torch.save(critic_net.state_dict(), critic_pth)


def save_sources(sample_dir):
    source_dir = os.path.join(sample_dir, 'sources')
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)

    main_src = sys.modules['__main__'].__file__
    main_dir = os.path.dirname(os.path.abspath(main_src))
    py_files = glob.glob(os.path.join(main_dir, '*.py'))
    for py in py_files:
        shutil.copy(py, source_dir)


def write_summary(summary_dir, sample_dir, epoch, image_samples, image_names):
    image_ext = '.png'

    summary_file = os.path.join(summary_dir, '{}.html'.format(epoch))
    with open(summary_file, 'w') as f:
        print('<!DOCTYPE html>', file=f)
        print('<title>Epoch {}</title>'.format(epoch), file=f)

        print('<table>', file=f)

        print('<thead>', file=f)
        print('<tr>', file=f)
        for name in image_names:
            print('<th scope="col">{}'.format(name), file=f)

        print('<tbody>', file=f)
        num_samples = len(image_samples)
        relative_sample_dir = os.path.join('..', 'samples', str(epoch))
        relative_image_dir = os.path.join(relative_sample_dir, 'images')
        for i in range(num_samples):
            print('<tr>', file=f)
            prefix = os.path.join(relative_image_dir, '{}_'.format(i))
            for name in image_names:
                image_path = prefix + name + image_ext
                print('<td><img src="{}">'.format(image_path), file=f)

        print('</table>', file=f)

    save_images(sample_dir, image_samples, image_names, image_ext)


def save_images(sample_dir, image_samples, image_names, image_ext):
    image_dir = os.path.join(sample_dir, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    num_samples = len(image_samples)
    for i in range(num_samples):
        prefix = os.path.join(image_dir, '{}_'.format(i))
        for image, name in zip(image_samples[i], image_names):
            image.save(prefix + name + image_ext)


def save_checkpoint(chkpt_root, epoch, image_samples, image_names, deblend_net,Inp_net, Dis_net):
    summary_dir = os.path.join(chkpt_root, 'summaries')
    sample_dir = os.path.join(chkpt_root, 'samples', str(epoch))

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    save_models(sample_dir, deblend_net, Inp_net, Dis_net)
    save_sources(sample_dir)
    write_summary(summary_dir, sample_dir, epoch, image_samples, image_names)

def save_validate_images(val_root, image_samples, image_names, epoch=0):
    summary_dir = os.path.join(val_root, 'htmls')
    sample_dir = os.path.join(val_root, 'samples', str(epoch))

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    save_sources(sample_dir)
    write_summary(summary_dir, sample_dir, epoch, image_samples, image_names)


def save_checkpoint_stack(chkpt_root, epoch, image_samples, image_names, deblend_net, refine_net, critic_net):
    summary_dir = os.path.join(chkpt_root, 'summaries')
    sample_dir = os.path.join(chkpt_root, 'samples', str(epoch))

    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    save_models_stack(sample_dir, deblend_net, refine_net, critic_net)
    save_sources(sample_dir)
    write_summary(summary_dir, sample_dir, epoch, image_samples, image_names)
