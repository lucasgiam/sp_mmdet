# Ensure that the following packages are installed before running this script:
# sklearn
# funcy
# shutil

import os
import json
import funcy
import shutil
from sklearn.model_selection import train_test_split

# User configs:
seed = 2022
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
data_root_dir = "./data/sp_ppe"
ann_file_path = "./data/sp_ppe/all_annotations.json"
image_path = "./data/sp_ppe/raw_img"


def save_coco(file, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'images': images, 'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def main():
    with open(os.path.join(ann_file_path), 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        # x: train, y: test z: val
        xz, y = train_test_split(images, test_size=test_ratio, random_state=seed)

        ratio_remaining = 1 - test_ratio
        val_ratio_adjusted = val_ratio / ratio_remaining

        x, z = train_test_split(xz, test_size=val_ratio_adjusted, random_state=seed)
        
        train_path = os.path.join(data_root_dir, 'train')
        train_image_path = os.path.join(train_path, 'images')
        val_path = os.path.join(data_root_dir, 'val')
        val_image_path = os.path.join(val_path, 'images')
        test_path = os.path.join(data_root_dir, 'test')
        test_image_path = os.path.join(test_path, 'images')

        for pp in [train_path, val_path, test_path]:
            if not os.path.exists(pp):
                os.mkdir(pp)
            if not os.path.exists(os.path.join(pp, 'images')):
                os.mkdir(os.path.join(pp, 'images'))
        
        for train_image in x:
            fname = train_image['file_name']
            shutil.copyfile(os.path.join(image_path, fname), os.path.join(train_image_path, fname))
        print(f'Completed {len(x)} train images')
        for test_image in y:
            fname = test_image['file_name']
            shutil.copyfile(os.path.join(image_path, fname), os.path.join(test_image_path, fname))
        print(f'Completed {len(y)} test images')
        for val_image in z:
            fname = val_image['file_name']
            shutil.copyfile(os.path.join(image_path, fname), os.path.join(val_image_path, fname))
        print(f'Completed {len(z)} val images')

        save_coco(os.path.join(train_path, "train.json"), x, filter_annotations(annotations, x), categories)
        save_coco(os.path.join(test_path, "test.json"), y, filter_annotations(annotations, y), categories)
        save_coco(os.path.join(val_path, "val.json"), z, filter_annotations(annotations, z), categories)

        print("Saved {} entries in {}, {} in {}, and {} in {}".format(len(x), train_path, len(y), test_path, len(z), val_path))

if __name__ == "__main__":
    main()