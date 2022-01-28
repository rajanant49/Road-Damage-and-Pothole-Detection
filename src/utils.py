import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import DEVICE
from albumentations.pytorch import ToTensorV2
import numpy as np
from config import NUM_CLASSES
from mean_average_precision import MetricBuilder

# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

# define the training tranforms
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

# define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })


def show_tranformed_image(train_loader):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
    """
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            for box in boxes:
                cv2.rectangle(sample,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            (0, 0, 255), 2)
            plt.imshow(sample)
            plt.title('Transformed Image')
            plt.show()


def get_batch_mAP(outputs,targets):
    # print list of available metrics
#     print(MetricBuilder.get_metrics_list())
    # create metric_fn
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=NUM_CLASSES)
    # add some samples to evaluation
    
    for b in range(len(outputs)):
        preds = []
        for i in range(len(outputs[b]['boxes'])):
            bboxes = outputs[b]['boxes'][i].tolist()
            label = outputs[b]['labels'][i].tolist()
            score = outputs[b]['scores'][i].tolist()
            pred = bboxes + [label,score]
            preds.append(pred)
        preds = np.array(preds)
        
        gts = []
        for i in range(len(targets[b]['boxes'])):
            bboxes = targets[b]['boxes'][i].tolist()
            label = targets[b]['labels'][i].tolist()
            crowd = targets[b]['iscrowd'][i].tolist()
            gt = bboxes + [label,0,crowd]
            gts.append(gt)
        gts = np.array(gts)
        
        metric_fn.add(preds, gts)
    # compute PASCAL VOC metric
#     print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")
    voc_pascal_map = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']
    # compute PASCAL VOC metric at the all points
#     print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")
    voc_pascal_map_allpts = metric_fn.value(iou_thresholds=0.5)['mAP']

    # compute metric COCO metric
#     print(f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")
    coco_map = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']
    return voc_pascal_map,voc_pascal_map_allpts,coco_map