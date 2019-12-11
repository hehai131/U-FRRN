from __future__ import print_function

from PIL import Image
import torch
import torch.autograd as autograd
import torchvision.transforms as transforms
import math

to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()


def load_image(image, scale):
    if isinstance(image, str):
        image = Image.open(image)
        image = image.convert('RGB')

        ori_w, ori_h = image.size
        if scale != -1 and max(ori_w, ori_h) > scale:
            ratio = max(ori_w, ori_h) / scale
            ori_w, ori_h = ori_w / ratio, ori_h / ratio
        new_w = int(32 * math.ceil(ori_w / 32))
        new_h = int(32 * math.ceil(ori_h / 32))
        image = image.resize((new_w, new_h), Image.ANTIALIAS)

    return image, to_tensor(image)


def do_deblend(deblend_net, deblend_pth, tensor, decompose):
    state_dict = torch.load(deblend_pth, map_location=lambda storage, loc: storage)
    deblend_net.load_state_dict(state_dict)
    deblend_net.eval()

    tensor = tensor.unsqueeze(0)
    tensor = autograd.Variable(tensor, volatile=True)

    if decompose:
        output_tensor1, output_tensor2 = deblend_net(tensor)
    else:
        output_tensor1 = deblend_net(tensor)
        output_tensor2 = tensor - output_tensor1

    output_tensor1 = output_tensor1.cpu().data.squeeze(0)
    output_tensor2 = output_tensor2.cpu().data.squeeze(0)
    output_tensor1 = output_tensor1.clamp(0, 1)
    output_tensor2 = output_tensor2.clamp(0, 1)

    output_image1 = to_image(output_tensor1)
    output_image2 = to_image(output_tensor2)
    return output_image1, output_image2


def do_stack(deblend_net, deblend_pth, refine_net, refine_pth, tensor, decompose):
    state_dict = torch.load(deblend_pth, map_location=lambda storage, loc: storage)
    deblend_net.load_state_dict(state_dict)
    deblend_net.eval()
    state_dict = torch.load(refine_pth, map_location=lambda storage, loc: storage)
    refine_net.load_state_dict(state_dict)
    refine_net.eval()

    tensor = tensor.unsqueeze(0)
    tensor = autograd.Variable(tensor, volatile=True)
    tensor = tensor.cuda()
    if decompose:
        output_tensor1, output_tensor2 = deblend_net(tensor)
        output_tensor3 = refine_net(output_tensor1, output_tensor2)
    else:
        output_tensor1 = deblend_net(tensor)
        output_tensor2 = tensor - output_tensor1
        output_tensor3 = refine_net(output_tensor1, output_tensor2)

    output_tensor1 = output_tensor1.cpu().data.squeeze(0)
    output_tensor2 = output_tensor2.cpu().data.squeeze(0)
    output_tensor3 = output_tensor3.cpu().data.squeeze(0)
    output_tensor1 = output_tensor1.clamp(0, 1)
    output_tensor2 = output_tensor2.clamp(0, 1)
    output_tensor3 = output_tensor3.clamp(0, 1)

    output_image1 = to_image(output_tensor1)
    output_image2 = to_image(output_tensor2)
    output_image3 = to_image(output_tensor3)
    return output_image1, output_image2, output_image3


def handle_original(deblend_net, deblend_pth, image, scale, decompose):
    image, tensor = load_image(image, scale)
    output_image1, output_image2 = do_deblend(deblend_net, deblend_pth, tensor, decompose)
    return image, output_image1, output_image2


def handle_stack_original(deblend_net, deblend_pth, refine_net, refine_pth, image, scale, decompose):
    image, tensor = load_image(image, scale)
    output_image1, output_image2, output_image3 = do_stack(deblend_net, deblend_pth, refine_net, refine_pth, tensor,
                                                           decompose)
    return image, output_image1, output_image2, output_image3


def handle_multi_scale(deblend_net, deblend_pth, image, scale, decompose):
    image, tensor = load_image(image, -1)

    output_image1, output_image2 = do_deblend(deblend_net, deblend_pth, tensor, decompose)
    return image, output_image1, output_image2


def handle_synthetic(deblend_net, deblend_pth, image1, image2, ratio, scale, decompose):
    image1, tensor1 = load_image(image1, scale)
    image2, tensor2 = load_image(image2, scale)

    tensor = tensor1 * ratio + tensor2 * (1 - ratio)
    image = to_image(tensor)

    output_image1, output_image2 = do_deblend(deblend_net, deblend_pth, tensor, decompose)
    return image1, image2, image, output_image1, output_image2


def decompose_original(deblend_net, deblend_pth, image, scale=128):
    return handle_original(deblend_net, deblend_pth, image, scale, True)


def stack_original(deblend_net, deblend_pth, refine_net, refine_pth, image, scale=128):
    return handle_stack_original(deblend_net, deblend_pth, refine_net, refine_pth, image, scale, True)


def decompose_synthetic(deblend_net, deblend_pth, image1, image2, ratio=0.5, scale=128):
    return handle_synthetic(deblend_net, deblend_pth, image1, image2, ratio, scale, True)


def suppress_original(deblend_net, deblend_pth, image, scale=128):
    return handle_original(deblend_net, deblend_pth, image, scale, False)


def suppress_synthetic(deblend_net, deblend_pth, image1, image2, ratio=0.5, scale=128):
    return handle_synthetic(deblend_net, deblend_pth, image1, image2, ratio, scale, False)
