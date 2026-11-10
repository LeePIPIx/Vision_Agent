from .base_tool import BaseTool
from pydantic import BaseModel, Field
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
from uuid import uuid4
import numpy as np
import os
from utils.logger import get_logger

logger = get_logger(__name__)


class SegmentationTool(BaseTool):
    name = "segmentation_tool"
    description = "使用YOLO对图片进行视觉分析。支持目标检测、实例分割、旋转框检测(obb)、姿态估计(pose)、图像分类(classify)。"

    class Parameters(BaseModel):
        image_path: str = Field(..., description="要分析的图片路径")
        task: str = Field(
            default="detect",
            description="任务类型：detect(目标检测), segment(实例分割), obb(旋转框检测), pose(姿态估计), classify(图像分类)"
        )

    def _load_model(self, task):
        """按任务加载对应的 YOLO 模型权重"""
        model_map = {
            "detect":   "yolo26n.pt",
            "segment":  "yolo26n-seg.pt",
            "obb":      "yolo26n-obb.pt",
            "pose":     "yolo26n-pose.pt",
            "classify": "yolo26n-cls.pt",
        }
        weight_file = model_map.get(task, "yolo26n.pt")
        model_path = f"tools/model_weight/{weight_file}"
        logger.info(f"[SegmentationTool] 加载模型: {model_path}")
        return YOLO(model_path)

    def _load_image(self, image_path: str) -> np.ndarray:
        """用 PIL 加载图片并转为 numpy array，解决 OpenCV 不支持 .jfif 等格式的问题"""
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)

    def run(self, image_path: str, task: str = "detect"):
        logger.info("[SegmentationTool] task=%s", task)

        model = self._load_model(task)
        # 用 PIL 加载图片，避免 OpenCV 不支持 .jfif 等格式
        image = self._load_image(image_path)
        results = model(image)
        r = results[0]

        # 创建输出目录，保存可视化结果
        time = uuid4().hex
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(project_root, "outputs", f"session-{time}")
        from medical_app.context import current_task
        ctx = current_task.get()
        if ctx:
            save_dir = str(ctx.output / time)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "result.jpg")
        r.save(filename=save_path)
        logger.info("[SegmentationTool] 可视化结果已保存")

        # === 按 task 提取不同的统计信息 ===
        if task == "detect":
            return self._summarize_detect(r, save_path)
        elif task == "segment":
            return self._summarize_segment(r, save_path)
        elif task == "obb":
            return self._summarize_obb(r, save_path)
        elif task == "pose":
            return self._summarize_pose(r, save_path)
        elif task == "classify":
            return self._summarize_classify(r, save_path)
        else:
            return {"task": task, "error": f"不支持的任务类型: {task}", "save_path": save_path}

    # ========== 各任务统计方法 ==========

    def _summarize_detect(self, r, save_path):
        """目标检测：统计每类物体的数量和平均置信度"""
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return {"task": "detect", "objects": [], "total": 0, "message": "未检测到任何物体", "save_path": save_path}

        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        names = r.names

        unique_cls, counts = np.unique(cls_ids, return_counts=True)
        objects = []
        for cls_id, count in zip(unique_cls, counts):
            cls_mask = cls_ids == cls_id
            avg_conf = round(float(confs[cls_mask].mean()), 4)
            objects.append({
                "name": names[int(cls_id)],
                "count": int(count),
                "avg_confidence": avg_conf,
            })

        return {
            "task": "detect",
            "total_objects": int(len(cls_ids)),
            "objects": objects,
            "save_path": save_path,
        }

    def _summarize_segment(self, r, save_path):
        """实例分割：统计每个物体的类别和 mask 像素占比"""
        masks = r.masks
        boxes = r.boxes
        if masks is None or boxes is None or len(boxes) == 0:
            return {"task": "segment", "objects": [], "total": 0, "message": "未检测到任何物体", "save_path": save_path}

        cls_ids = boxes.cls.cpu().numpy().astype(int)
        mask_data = masks.data.cpu().numpy()  # shape: (N, H, W)
        names = r.names

        objects = []
        for i, cls_id in enumerate(cls_ids):
            area_ratio = round(float(mask_data[i].mean()), 4)
            objects.append({
                "name": names[int(cls_id)],
                "mask_area_ratio": area_ratio,
            })

        return {
            "task": "segment",
            "total_objects": len(objects),
            "objects": objects,
            "save_path": save_path,
        }

    def _summarize_obb(self, r, save_path):
        """旋转框检测：统计每类物体的数量"""
        obb = r.obb
        if obb is None or len(obb) == 0:
            return {"task": "obb", "objects": [], "total": 0, "message": "未检测到任何物体", "save_path": save_path}

        cls_ids = obb.cls.cpu().numpy().astype(int)
        confs = obb.conf.cpu().numpy()
        names = r.names

        unique_cls, counts = np.unique(cls_ids, return_counts=True)
        objects = []
        for cls_id, count in zip(unique_cls, counts):
            cls_mask = cls_ids == cls_id
            avg_conf = round(float(confs[cls_mask].mean()), 4)
            objects.append({
                "name": names[int(cls_id)],
                "count": int(count),
                "avg_confidence": avg_conf,
            })

        return {
            "task": "obb",
            "total_objects": int(len(cls_ids)),
            "objects": objects,
            "save_path": save_path,
        }

    def _summarize_pose(self, r, save_path):
        """姿态估计：统计每人的关键点数量和平均置信度"""
        keypoints = r.keypoints
        if keypoints is None or len(keypoints) == 0:
            return {"task": "pose", "persons_detected": 0, "details": [], "message": "未检测到任何人", "save_path": save_path}

        kp_xy = keypoints.xy.cpu().numpy()     # (N, num_keypoints, 2)
        kp_conf = keypoints.conf.cpu().numpy()  # (N, num_keypoints)

        persons = []
        for i, (kp, kc) in enumerate(zip(kp_xy, kp_conf)):
            visible_count = int((kp.sum(axis=1) > 0).sum())
            avg_conf = round(float(kc.mean()), 4)
            persons.append({
                "person": i + 1,
                "visible_keypoints": visible_count,
                "avg_confidence": avg_conf,
            })

        return {
            "task": "pose",
            "persons_detected": len(persons),
            "details": persons,
            "save_path": save_path,
        }

    def _summarize_classify(self, r, save_path):
        """图像分类：返回 top5 分类结果"""
        probs = r.probs
        if probs is None:
            return {"task": "classify", "top5": [], "message": "分类失败", "save_path": save_path}

        top5_idx = probs.top5
        top5_conf = probs.top5conf
        names = r.names

        top5 = []
        for idx, conf in zip(top5_idx, top5_conf):
            top5.append({
                "name": names[int(idx)],
                "confidence": round(float(conf), 4),
            })

        return {
            "task": "classify",
            "top1": top5[0] if top5 else None,
            "top5": top5,
            "save_path": save_path,
        }