from typing import Any, Dict
import numpy as np

from peekingduck.pipeline.nodes.node import AbstractNode

from mmdet.apis import inference_detector, init_detector


class Node(AbstractNode):

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.config_file: str          # config file
        self.checkpoint_file: str      # checkpoint file
        self.device: str               # device used for inference (default = 'cuda:0')
        self.score_thre: float         # conf score threshold
        self.model = self.load_model()

    def load_model(self):
        model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        return model
    
    def post_process(self, result, height, width):     
        bboxes = []
        class_ids = []
        scores = []
        masks = []
        for id, (pred_bbox, pred_mask) in enumerate(zip(result[0], result[1])):
            if len(pred_bbox) > 0:
                for v_bbox, v_mask in zip(pred_bbox, pred_mask):
                    xyxy = v_bbox[0:4]
                    score = v_bbox[4]
                    mask = v_mask * 1
                    if score >= self.score_thre:
                        class_ids.append(id)
                        bboxes.append(xyxy)
                        scores.append(score)
                        masks.append(mask)
        for bbox in bboxes:
            bbox[[0,2]] = bbox[[0,2]] / width
            bbox[[1,3]] = bbox[[1,3]] / height
        return bboxes, class_ids, scores, masks
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        img = inputs["img"]
        height, width = img.shape[:2]
        result = inference_detector(self.model, img)
        bboxes, class_ids, scores, masks = self.post_process(result, height, width)
        bboxes = np.array(bboxes, dtype="float32")
        class_ids = np.array(class_ids, dtype="str")
        scores = np.array(scores, dtype="float32")
        masks = np.array(masks, dtype="uint8")
        outputs = {"bboxes": bboxes, "bbox_labels": class_ids, "bbox_scores": scores, "masks": masks}  ## class_ids supposed to change to class_labels defined by dictionary mapping
        return outputs
