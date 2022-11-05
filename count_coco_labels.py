import os
import json
from collections import Counter


ann_file = "./data/sp_ppe/train/train.json"


def cat_count(annotations=None):
    cats = []
    for i in annotations['annotations']:
        j = i['category_id']
        for cat in annotations['categories']:
            if j == cat['id']:
                cats.append(cat['name'])
    count_dict = dict(Counter(cats))
    num_classes = len(count_dict)
    print("num_classes:", num_classes)
    print("count_instances:", count_dict)

def main():
    with open(os.path.join(ann_file), 'rt', encoding='UTF-8') as annotations:
        ann = json.load(annotations)
    cat_count(ann)

if __name__ == "__main__":
    main()