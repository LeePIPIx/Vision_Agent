"""NIfTI geometry, measured label statistics and CT overlays (no LLM measurements)."""
import hashlib
import io
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

MAX_VOXELS = 128_000_000


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_volume(path):
    image = nib.load(str(path))
    if len(image.shape) != 3 or min(image.shape) < 2 or np.prod(image.shape) > MAX_VOXELS:
        raise ValueError("仅接受三维 NIfTI；体素数不得超过 1.28 亿")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
        raise ValueError("影像空间矩阵无效")
    return image


def metadata(path):
    image = load_volume(path)
    return {"shape": list(image.shape), "spacing": list(map(float, image.header.get_zooms())),
            "spatial_unit": image.header.get_xyzt_units()[0], "orientation": list(nib.aff2axcodes(image.affine)),
            "affine": image.affine.tolist(), "dtype": str(image.get_data_dtype()), "sha256": digest(path)}


def label_statistics(image_path, mask_path, labels=None):
    from scipy import ndimage
    image, mask = load_volume(image_path), load_volume(mask_path)
    if image.shape != mask.shape or not np.allclose(image.affine, mask.affine, atol=1e-4, rtol=1e-5):
        raise ValueError("mask 与 CT 的尺寸或空间坐标不一致，必须先完成配准/重采样")
    unit = image.header.get_xyzt_units()[0]
    factors = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    if unit not in factors:
        raise ValueError("影像未声明空间单位，不能计算真实体积；请核对并修正 NIfTI header")
    if mask.header.get_xyzt_units()[0] != unit:
        raise ValueError("mask 与 CT 的空间单位不一致")
    data = mask.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all() or np.any(data < 0) or not np.all(data == np.rint(data)):
        raise ValueError("mask 必须是非负整数标签")
    values = np.unique(data)
    if len(values) > 128:
        raise ValueError("mask 标签超过 127 个，请确认上传的是分割标签而非原始 CT")
    affine = image.affine.copy()
    affine[:3, :] *= factors[unit]
    voxel_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    area_mm2 = float(np.linalg.norm(np.cross(affine[:3, 0], affine[:3, 1])))
    rows = []
    for label in values[values > 0].astype(int):
        selected = data == label
        coords = np.argwhere(selected)
        counts = selected.sum(axis=(0, 1))
        _, components = ndimage.label(selected, structure=ndimage.generate_binary_structure(3, 1))
        rows.append({"label": int(label), "name": (labels or {}).get(str(label), f"标签 {label}"),
                     "voxel_count": int(len(coords)), "volume_ml": round(len(coords) * voxel_mm3 / 1000, 5),
                     "connected_regions_6": int(components),
                     "centroid_ras_mm": nib.affines.apply_affine(affine, coords.mean(axis=0)).round(3).tolist(),
                     "bbox_voxel_min": coords.min(axis=0).tolist(), "bbox_voxel_max": coords.max(axis=0).tolist(),
                     "largest_slice_index": int(counts.argmax()),
                     "largest_slice_area_mm2": round(float(counts.max()) * area_mm2, 3)})
    return {"image_sha256": digest(image_path), "mask_sha256": digest(mask_path),
            "voxel_volume_mm3": voxel_mm3, "labels": rows,
            "method": "label-count-v1; volume_ml = voxel_count * abs(det(affine_mm[:3,:3])) / 1000",
            "warnings": ["连通区域数不等同于病灶数；空 mask 不代表没有疾病"]}


def slice_png(image_path, mask_path=None, index=None, window=400, level=50):
    original = load_volume(image_path)
    image = nib.as_closest_canonical(original)
    index = image.shape[2] // 2 if index is None else max(0, min(int(index), image.shape[2] - 1))
    plane = np.asarray(image.dataobj[:, :, index], dtype=np.float32)
    plane = np.nan_to_num(plane)
    grey = (np.clip((plane - (level - window / 2)) / window, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([grey] * 3, axis=-1)
    if mask_path:
        original_mask = load_volume(mask_path)
        if original.shape != original_mask.shape or not np.allclose(original.affine, original_mask.affine):
            raise ValueError("mask 与 CT 不在相同空间")
        mask = nib.as_closest_canonical(original_mask)
        selected = np.asarray(mask.dataobj[:, :, index]) > 0
        rgb[selected] = (rgb[selected] * 0.5 + np.array([255, 70, 50]) * 0.5).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(np.rot90(rgb)).save(buffer, format="PNG")
    return buffer.getvalue(), image.shape[2]


def report_markdown(case_id, stats, model):
    lines = ["# CT 分割定量报告（待人工复核）", f"病例编号：{case_id}",
             f"模型：{model['name']}；配置：{model['configuration']}",
             "| 目标 | 标签 | 体素数 | 体积（mL） | 六邻域连通区域数 |", "|---|---:|---:|---:|---:|"]
    for row in stats["labels"]:
        lines.append(f"| {row['name']} | {row['label']} | {row['voxel_count']} | {row['volume_ml']} | {row['connected_regions_6']} |")
    if not stats["labels"]:
        lines.append("未得到非零分割标签，未完成目标测量。")
    lines += ["", "连通区域数不等同于病灶数；本报告记录模型输出的定量结果，需核对覆盖范围和分割边界。",
              f"计算方法：{stats['method']}", f"输入 SHA256：{stats['image_sha256']}",
              f"Mask SHA256：{stats['mask_sha256']}", f"权重来源：{model.get('source', '人工上传')}"]
    return "\n\n".join(lines)
