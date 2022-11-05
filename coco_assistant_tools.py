from coco_assistant import COCO_Assistant

img_dir = r'C:\Users\Admin\Desktop\sp_ppe_data_coco_assistant\images'
ann_dir = r'C:\Users\Admin\Desktop\sp_ppe_data_coco_assistant\annotations'

cas = COCO_Assistant(img_dir, ann_dir)
cas.visualise()