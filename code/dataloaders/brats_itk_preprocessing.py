import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
import nibabel as nib
import SimpleITK as sitk
import glob

def brain_bbox_itk(data, gt):
    bbox=[]
    labelimfilter=sitk.LabelShapeStatisticsImageFilter()
    # Cast images to compatible pixel types
    mask = sitk.Cast(gt, sitk.sitkUInt8)
    labelimfilter.Execute(mask)
    for i in range(1,labelimfilter.GetNumberOfLabels()+1):
        box=labelimfilter.GetBoundingBox(i)
        bbox.append(box)
        
    start = [max(0, bbox[0][i]) for i in range(3)]
    end = [min(data.GetSize()[i], bbox[0][i + 3]) for i in range(3)]
    
    roi_filter = sitk.RegionOfInterestImageFilter()
    size = [end[i] - start[i] for i in range(3)]
    roi_filter.SetIndex(start)
    roi_filter.SetSize(size)
    
    cropped_image = roi_filter.Execute(data)
    cropped_mask = roi_filter.Execute(gt)
    
    return cropped_image, cropped_mask


def itensity_normalize_one_volume_itk(image):
    """
    Normalize the intensity of an image based on non-zero region statistics.
    """
    stats = sitk.StatisticsImageFilter()
    stats.Execute(image)
    mean = stats.GetMean()
    std_dev = stats.GetSigma()
    
    image = sitk.ShiftScale(image, shift=-mean, scale=1 / std_dev)
    return image




def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean)/std
    out_random = np.random.normal(0, 1, size=volume.shape)
#     out[volume == 0] = out_random[volume == 0]
    out = out.astype(np.float32)
    return out


class MedicalImageProcessor:
    def __init__(self, image, percent=0.999):
        self.image = image
        self.percent = percent

    def clip_intensity(self):
        """
        Clip the intensity values based on a specified cumulative percentage.
        """
        histogram = sitk.Histogram(self.image, 256)
        cumulative = np.cumsum(histogram)
        threshold = np.searchsorted(cumulative, self.percent * cumulative[-1])
        return sitk.Clamp(self.image, lowerBound=self.image.GetPixelIDTypeMin(), upperBound=threshold)

    def normalize(self):
        """
        Normalize image to [0, 1] range.
        """
        stats = sitk.StatisticsImageFilter()
        stats.Execute(self.image)
        min_val = stats.GetMinimum()
        max_val = stats.GetMaximum()
        return sitk.RescaleIntensity(self.image, outputMinimum=0, outputMaximum=1)


all_flair = glob.glob("flair/*_flair.nii.gz")
for p in all_flair:
    dataimg=sitk.ReadImage(p)
    # data = sitk.GetArrayFromImage(dataimg)
    labimg=sitk.ReadImage(p.replace("flair", "seg"))


    # lab = sitk.GetArrayFromImage(labimg)
    # img, lab = brain_bbox(data, lab)
    croppedimg,croppedlab=brain_bbox_itk(dataimg,labimg)
    img = MedicalImageProcessor(croppedimg, percent=0.999).clip_intensity
    # img = itensity_normalize_one_volume(img)
    img=itensity_normalize_one_volume_itk(croppedimg)
    
    # lab[lab > 0] = 1
    data=sitk.GetArrayFromImage(img)
    lab=sitk.GetArrayFromImage(croppedlab)
    lab[lab > 0] = 1
    
        
    print(f"Image shape: {data.shape}")
    print(f"Label shape: {lab.shape}")
    
    uid = p.split("/")[-1]
    sitk.WriteImage(sitk.GetImageFromArray(
        data), "/home/shihan/data/course/SSL4MIS/preprocessing_results/brats19/data/flair{}".format(uid))
    sitk.WriteImage(sitk.GetImageFromArray(
        lab), "/home/shihan/data/course/SSL4MIS/preprocessing_results/brats19/data/label/{}".format(uid))
