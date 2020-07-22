import torch.nn as nn

from models.backbone import ResNet
from models.necks import FPN, BFP
from models.head import AnchorHead, KpHead
from models.py_utils import parse_losses, parse_det_offset, parse_det_bbox
import pdb


class SingleStageDetector(nn.Module):
    def __init__(self, cfg):
        super(SingleStageDetector, self).__init__()

        self.backbone = ResNet(**cfg.backbone)

        self.fpn = FPN(**cfg.fpn) if 'fpn' in cfg else None

        self.bfp = BFP(**cfg.bfp) if 'bfp' in cfg else None

        self.bbox_head = AnchorHead(**cfg.anchor_head) if 'anchor_head' in cfg else None

        self.csp_head = KpHead(**cfg.kp_head) if 'kp_head' in cfg else None

        self.test = True if cfg.test_cfg.test else False
        self.cfg = cfg

    def forward(self, img, **kwargs):
        x = self.backbone(img)

        if self.fpn:
            outs_fpn = self.fpn(x)
        if self.bfp:
            outs_bfp = self.bfp(outs_fpn)
        if self.bbox_head:
            outs_bbox = self.bbox_head(outs_bfp)
        if self.csp_head and self.fpn:
            outs_csp = self.csp_head(outs_fpn)
        else:
            outs_csp = self.csp_head(x)

        if self.bbox_head and self.csp_head:
            if self.test:
                detections = self.simple_test([outs_bbox, outs_csp], **kwargs)
                return detections
            else:
                return outs_bbox, outs_csp
        elif self.bbox_head:
            if self.test:
                detections = self.simple_test(outs_bbox, **kwargs)
                return detections
            else:
                return outs_bbox
        elif self.csp_head:
            if self.test:
                detections = self.simple_test(outs_csp, **kwargs)
                return detections
            else:
                return outs_csp
        else:
            raise TypeError(f'There is not a head defined')

    def loss(self,
             preds,
             targets,
             img_metas):

        if self.bbox_head and self.csp_head:
            preds_anchor, preds_csp = preds[0], preds[1]
        elif self.bbox_head:
            preds_anchor = preds
        else:
            preds_csp = preds

        seman_map, scale_map, offset_map = targets[0], targets[1], targets[2]

        if self.bbox_head:
            losses_anchor = self.bbox_head.loss(preds_anchor, img_metas)
            loss_anchor, loss_anchor_vars = parse_losses(losses_anchor)

        if self.csp_head:
            losses_csp = self.csp_head.loss(preds_csp, seman_map=seman_map, scale_map=scale_map, offset_map=offset_map)
            loss_csp, loss_csp_var = parse_losses(losses_csp)

        if self.bbox_head and self.csp_head:
            return loss_anchor, loss_csp
        elif self.bbox_head:
            return loss_anchor
        else:
            return loss_csp

    def simple_test(self, outs, **kwargs):

        if self.csp_head and self.bbox_head:
            outs_bbox, outs_csp = outs[0], outs[1]
        elif self.csp_head:
            outs_csp = outs
        else:
            outs_bbox = outs

        detections = []

        nms_algorithm = {
            "nms": 0,
            "linear_soft_nms": 1,
            "exp_soft_nms": 2
        }[self.cfg.test_cfg.nms_algorithm]

        if self.csp_head:
            outs = []
            scores_csp = self.cfg.test_cfg.scores_csp
            for out_csp in outs_csp:
                out_csp = out_csp.data.cpu().numpy()
                outs.append(out_csp)
            dets_csp = parse_det_offset(outs, self.cfg, nms_algorithm, score=scores_csp, down=4)
            detections.append(dets_csp)

        if self.bbox_head:
            scores_bbox = self.cfg.test_cfg.scores_bbox
            bbox_list = self.bbox_head.get_bboxes(outs_bbox, img_metas=kwargs, cfg=self.cfg.test_cfg)
            dets_bbox = parse_det_bbox(bbox_list, self.cfg, nms_algorithm, score_thr=scores_bbox)
            detections.append(dets_bbox)

        return detections
