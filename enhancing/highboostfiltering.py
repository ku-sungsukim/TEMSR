from model.model_utils.img_process_utils import USMSharp


def HighBoostFiltering(gt):
    usm_sharpener = USMSharp().cuda() 
    gt_hbf = usm_sharpener(gt, weight=2)
    
    return gt_hbf