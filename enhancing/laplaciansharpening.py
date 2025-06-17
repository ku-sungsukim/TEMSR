from model.model_utils.img_process_utils import LaplacianSharp


def LaplacianSharpening(gt):
    laplacian_sharpener = LaplacianSharp().cuda() 
    gt_lps = laplacian_sharpener(gt, weight=1)
    
    return gt_lps