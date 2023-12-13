"""
here are some funcitons used in cuda + pytorch for image processing
"""
import cv2, torch 

def kernel_circular(width, normalize):
    ker = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width,width))).unsqueeze_(0).unsqueeze_(0)
    if normalize:
        return ker/torch.sum(ker)
    else:
        return ker

def morph_erode_dilate(im_tensor, kernel, mode):
    """
    mode = 0 = erosion; mode = 1 = dilation; im_tensor is 1s and 0s
    convolve image (stack) with a kernel. 
    dilate  = keep partial/full overlap between image and kernel.   means masked sum > 0
    erosion = keep only full overlap.                               means masked sum  = kernel sum.
    erode: subtract full overlap area from result, pixels with full overlap will be 0. rest will go below 0.
    erode:add 1 to bring full overlap pixels to value of 1. partial overlap will be below 1 and will be clamped to 0.
    dilate: just clamp
    """
    padding = (kernel.shape[-1]//2,)*2
    torch_result0   = torch.nn.functional.conv2d(im_tensor, kernel, padding = padding, groups = 1)
    if mode == 0:
        full_area = torch.sum(kernel)
        torch_result0.add_(-full_area + 1)
    return torch_result0.clamp_(0, 1)