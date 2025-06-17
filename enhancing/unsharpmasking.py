from model.model_utils.img_process_utils import USMSharp


def UnsharpMasking(gt):
    usm_sharpener = USMSharp().cuda() 
    gt_usm = usm_sharpener(gt, weight=0.5)
    
    return gt_usm