"""
测试脚本：D405 相机 + SAM3 分割 + DINOv3 关键点提议
- 从 RealSense D405 拍摄 RGB + 深度图（1280x720）
- 用 SAM3 文本 prompt 分割物体
- 用 DINOv3 提取特征并聚类提议关键点
- 可视化结果
"""

import sys
sys.path.insert(0, "/home/ypf/sam3-main")

import numpy as np
import torch
import cv2
import pyrealsense2 as rs
from transformers import AutoModel
from torch.nn.functional import interpolate
from kmeans_pytorch import kmeans
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import os
import time
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ============================================
# 配置
# ============================================
CAMERA_SERIAL = "230422271972"
CAMERA_RESOLUTION = (640, 480)  # 提高分辨率，SAM3 分割效果更好
MODEL_PATH = "/home/ypf/.cache/modelscope/hub/models/facebook/dinov3-vitb16-pretrain-lvd1689m"
SAM3_CHECKPOINT = "/home/ypf/sam3-main/checkpoint/sam3.pt"
SAVE_DIR = "/home/ypf/ReKep/test_output"
PATCH_SIZE = 16
NUM_PREFIX_TOKENS = 5
NUM_CANDIDATES_PER_MASK = 5
MIN_DIST_BT_KEYPOINTS = 0.06
MAX_MASK_RATIO = 0.5
DEVICE = "cuda"

# 场景中的物体 prompt 列表（根据实际场景修改）
OBJECT_PROMPTS = ["bowl", "pen", "cup", "rectangular object", "lego brick", "building block"]

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================
# 1. D405 相机
# ============================================
def capture_rgbd(serial, resolution=(1280, 720)):
    """从 D405 拍摄一帧 RGB + 对齐的深度图 + 相机内参"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, resolution[0], resolution[1], rs.format.z16, 30)

    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)

    print("等待相机自动曝光稳定...")
    for _ in range(30):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    rgb = np.asanyarray(color_frame.get_data())
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth_meters = depth * depth_scale

    pipeline.stop()

    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

    print(f"RGB shape: {rgb.shape}, Depth shape: {depth_meters.shape}")
    print(f"内参: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    valid_depth = depth_meters[depth_meters > 0]
    if len(valid_depth) > 0:
        print(f"深度范围: {valid_depth.min():.3f}m - {valid_depth.max():.3f}m")

    return rgb, depth_meters, intrinsic_matrix


def depth_to_pointcloud(depth, intrinsic_matrix):
    """深度图转 3D 点云"""
    H, W = depth.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=-1)


# ============================================
# 2. SAM3 分割
# ============================================
def sam3_segment(rgb, prompts, checkpoint=SAM3_CHECKPOINT, confidence=0.3):
    """
    用 SAM3 对场景中的多个物体进行分割
    
    Args:
        rgb: RGB 图像 (H, W, 3)
        prompts: 物体 prompt 列表，如 ["bowl", "pen", "cup"]
        checkpoint: SAM3 模型路径
        confidence: 检测置信度阈值
    
    Returns:
        seg_mask: (H, W) 整数数组，0=背景，1,2,...=各物体
        prompt_labels: 每个 mask ID 对应的 prompt 名称
    """
    print(f"加载 SAM3 模型...")
    model = build_sam3_image_model(checkpoint_path=checkpoint)
    processor = Sam3Processor(model, confidence_threshold=confidence)
    
    H, W = rgb.shape[:2]
    pil_img = Image.fromarray(rgb)
    
    # 对每个 prompt 分割
    seg_mask = np.zeros((H, W), dtype=np.int32)
    prompt_labels = {0: "background"}
    current_id = 1
    
    for prompt in prompts:
        print(f"  分割: '{prompt}' ...", end=" ")
        state = processor.set_image(pil_img)
        out = processor.set_text_prompt(state=state, prompt=prompt)
        
        if out["masks"] is None:
            print("未检测到")
            continue
        
        masks = out["masks"].cpu().numpy()  # (N, 1, H, W) or (N, H, W)
        scores = out["scores"].cpu().numpy()
        print(f"检测到 {len(masks)} 个实例")
        
        for i in range(len(masks)):
            mask = masks[i]
            if len(mask.shape) == 3:
                mask = mask[0]
            # resize 到原始分辨率（SAM3 内部可能 resize 过）
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.float32), (W, H),
                                  interpolation=cv2.INTER_NEAREST)
            
            binary = mask > 0.5
            # 不覆盖已有的分割区域（先来的优先级高）
            new_region = binary & (seg_mask == 0)
            if np.sum(new_region) < 100:
                continue
            
            seg_mask[new_region] = current_id
            prompt_labels[current_id] = f"{prompt}_{i}"
            print(f"    实例 {current_id}: '{prompt}' #{i}  "
                  f"({np.sum(new_region)} px, score={scores[i]:.3f})")
            current_id += 1
    
    torch.cuda.empty_cache()
    
    total_objects = current_id - 1
    print(f"SAM3 分割完成: {total_objects} 个物体")
    return seg_mask, prompt_labels


# ============================================
# 3. DINOv3 特征提取 + 关键点提议
# ============================================
class DINOv3KeypointProposer:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device)
        print("加载 DINOv3 模型...")
        self.model = AutoModel.from_pretrained(model_path).eval().to(self.device)
        self.patch_size = PATCH_SIZE
        self.num_prefix_tokens = NUM_PREFIX_TOKENS
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(self.device)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).to(self.device)
        self.mean_shift = MeanShift(bandwidth=MIN_DIST_BT_KEYPOINTS, bin_seeding=True, n_jobs=32)
        print("DINOv3 模型加载完成")

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def get_features(self, rgb):
        """提取 DINOv3 patch 特征"""
        H, W, _ = rgb.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        new_H = patch_h * self.patch_size
        new_W = patch_w * self.patch_size

        transformed = cv2.resize(rgb, (new_W, new_H))
        transformed = transformed.astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(transformed).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_tensor = (img_tensor - self.img_mean) / self.img_std

        outputs = self.model(pixel_values=img_tensor)
        patch_tokens = outputs.last_hidden_state[:, self.num_prefix_tokens:, :]
        patch_tokens = patch_tokens.reshape(1, patch_h, patch_w, -1)

        feature_grid = interpolate(
            patch_tokens.permute(0, 3, 1, 2),
            size=(H, W),
            mode='bilinear'
        ).permute(0, 2, 3, 1).squeeze(0)

        features_flat = feature_grid.reshape(-1, feature_grid.shape[-1])

        print(f"DINOv3 特征提取完成: patch 网格 {patch_h}x{patch_w}, 特征维度 {features_flat.shape[-1]}")
        return features_flat

    def propose_keypoints(self, rgb, points, seg_mask):
        """完整的关键点提议流程"""
        H, W, _ = rgb.shape
        features_flat = self.get_features(rgb)

        masks = [seg_mask == uid for uid in np.unique(seg_mask)]

        candidate_keypoints = []
        candidate_pixels = []
        candidate_rigid_group_ids = []

        for rigid_group_id, binary_mask in enumerate(masks):
            if np.mean(binary_mask) > MAX_MASK_RATIO:
                continue
            if np.sum(binary_mask) < 100:
                continue

            obj_features = features_flat[binary_mask.reshape(-1)]
            feature_pixels = np.argwhere(binary_mask)
            feature_points = points[binary_mask]

            valid = feature_points[:, 2] > 0
            if np.sum(valid) < 50:
                continue
            obj_features = obj_features[valid]
            feature_pixels = feature_pixels[valid]
            feature_points = feature_points[valid]

            obj_features = obj_features.double()
            (u, s, v) = torch.pca_lowrank(obj_features, center=False)
            features_pca = torch.mm(obj_features, v[:, :3])
            feat_min = features_pca.min(0)[0]
            feat_max = features_pca.max(0)[0]
            feat_range = feat_max - feat_min
            feat_range[feat_range == 0] = 1
            features_pca = (features_pca - feat_min) / feat_range

            feature_points_torch = torch.tensor(feature_points, dtype=features_pca.dtype, device=features_pca.device)
            pts_min = feature_points_torch.min(0)[0]
            pts_max = feature_points_torch.max(0)[0]
            pts_range = pts_max - pts_min
            pts_range[pts_range == 0] = 1
            feature_points_torch = (feature_points_torch - pts_min) / pts_range

            X = torch.cat([features_pca, feature_points_torch], dim=-1)

            num_clusters = min(NUM_CANDIDATES_PER_MASK, len(X))
            if num_clusters < 2:
                continue

            cluster_ids, cluster_centers = kmeans(
                X=X, num_clusters=num_clusters, distance='euclidean', device=self.device
            )
            cluster_centers = cluster_centers.to(self.device)

            for cid in range(num_clusters):
                center = cluster_centers[cid][:3]
                member_idx = cluster_ids == cid
                if member_idx.sum() == 0:
                    continue
                member_points = feature_points[member_idx.cpu().numpy()]
                member_pixels = feature_pixels[member_idx.cpu().numpy()]
                member_features = features_pca[member_idx]
                dist = torch.norm(member_features - center, dim=-1)
                closest = torch.argmin(dist)
                candidate_keypoints.append(member_points[closest])
                candidate_pixels.append(member_pixels[closest])
                candidate_rigid_group_ids.append(rigid_group_id)

        if len(candidate_keypoints) == 0:
            print("未找到任何关键点候选！")
            return np.array([]), np.array([]), rgb.copy()

        candidate_keypoints = np.array(candidate_keypoints)
        candidate_pixels = np.array(candidate_pixels)

        if len(candidate_keypoints) >= 2:
            try:
                self.mean_shift.fit(candidate_keypoints)
                cluster_centers = self.mean_shift.cluster_centers_
                merged_indices = []
                for center in cluster_centers:
                    dist = np.linalg.norm(candidate_keypoints - center, axis=-1)
                    merged_indices.append(np.argmin(dist))
                candidate_keypoints = candidate_keypoints[merged_indices]
                candidate_pixels = candidate_pixels[merged_indices]
            except Exception as e:
                print(f"MeanShift 合并失败，跳过: {e}")

        sort_idx = np.lexsort((candidate_pixels[:, 0], candidate_pixels[:, 1]))
        candidate_keypoints = candidate_keypoints[sort_idx]
        candidate_pixels = candidate_pixels[sort_idx]

        projected = draw_keypoints(rgb, candidate_pixels)

        print(f"提议了 {len(candidate_keypoints)} 个关键点")
        return candidate_keypoints, candidate_pixels, projected


def draw_keypoints(rgb, pixels):
    """在图像上标注关键点"""
    img = rgb.copy()
    for i, pixel in enumerate(pixels):
        text = str(i)
        text_len = len(text)
        bw, bh = 30 + 10 * (text_len - 1), 30
        r, c = int(pixel[0]), int(pixel[1])
        cv2.rectangle(img, (c - bw // 2, r - bh // 2), (c + bw // 2, r + bh // 2), (255, 255, 255), -1)
        cv2.rectangle(img, (c - bw // 2, r - bh // 2), (c + bw // 2, r + bh // 2), (0, 0, 0), 2)
        cv2.putText(img, text, (c - 7 * text_len, r + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return img


def visualize_sam3_masks(rgb, seg_mask, prompt_labels):
    """可视化 SAM3 分割结果"""
    H, W = seg_mask.shape
    overlay = rgb.copy().astype(np.float32)
    
    colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255],
        [128, 255, 0], [255, 128, 0], [0, 128, 255],
    ]
    
    for uid in np.unique(seg_mask):
        if uid == 0:
            continue
        color = colors[(uid - 1) % len(colors)]
        mask = seg_mask == uid
        overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5
        
        # 标注物体名称
        ys, xs = np.where(mask)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))
        label = prompt_labels.get(uid, f"obj_{uid}")
        cv2.putText(overlay.astype(np.uint8), label, (cx - 30, cy),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return overlay.astype(np.uint8)


# ============================================
# 4. 主流程
# ============================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str,
                        default=",".join(OBJECT_PROMPTS),
                        help="逗号分隔的物体 prompt 列表")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="SAM3 检测置信度阈值")
    args = parser.parse_args()
    
    prompts = [p.strip() for p in args.prompts.split(",")]
    print(f"物体 prompt: {prompts}")

    # --- 拍照 ---
    print("=" * 50)
    print("步骤 1：从 D405 相机拍摄 RGB + 深度 (1280x720)")
    print("=" * 50)
    rgb, depth, intrinsics = capture_rgbd(CAMERA_SERIAL, resolution=CAMERA_RESOLUTION)

    cv2.imwrite(os.path.join(SAVE_DIR, "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    depth_vis = (depth / max(depth.max(), 1e-6) * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(SAVE_DIR, "depth_colormap.png"), depth_vis)

    # --- 点云 ---
    print("\n" + "=" * 50)
    print("步骤 2：深度图转 3D 点云")
    print("=" * 50)
    points = depth_to_pointcloud(depth, intrinsics)
    print(f"点云 shape: {points.shape}")

    # --- SAM3 分割 ---
    print("\n" + "=" * 50)
    print("步骤 3：SAM3 实例分割")
    print("=" * 50)
    seg_mask, prompt_labels = sam3_segment(
        rgb, prompts, confidence=args.confidence
    )
    
    seg_vis = visualize_sam3_masks(rgb, seg_mask, prompt_labels)
    cv2.imwrite(os.path.join(SAVE_DIR, "segmentation.png"), cv2.cvtColor(seg_vis, cv2.COLOR_RGB2BGR))

    # --- DINOv3 关键点提议 ---
    print("\n" + "=" * 50)
    print("步骤 4：DINOv3 关键点提议")
    print("=" * 50)
    proposer = DINOv3KeypointProposer(MODEL_PATH, device=DEVICE)

    t0 = time.time()
    keypoints_3d, keypoints_2d, projected_img = proposer.propose_keypoints(rgb, points, seg_mask)
    t1 = time.time()
    print(f"关键点提议耗时: {t1 - t0:.2f}s")

    if len(keypoints_3d) > 0:
        print("\n关键点 3D 坐标 (相机坐标系):")
        for i, kp in enumerate(keypoints_3d):
            print(f"  关键点 {i}: x={kp[0]:.4f}, y={kp[1]:.4f}, z={kp[2]:.4f} (m)")

    cv2.imwrite(os.path.join(SAVE_DIR, "keypoints.png"), cv2.cvtColor(projected_img, cv2.COLOR_RGB2BGR))

    # --- 展示结果 ---
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB (1280x720)")
    axes[1].imshow(depth, cmap='jet')
    axes[1].set_title("Depth")
    axes[2].imshow(seg_vis)
    axes[2].set_title(f"SAM3 Segmentation ({len(prompt_labels)-1} objects)")
    axes[3].imshow(projected_img)
    axes[3].set_title(f"Keypoints ({len(keypoints_3d)})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "overview.png"), dpi=150)
    print(f"总览图已保存到 {SAVE_DIR}/overview.png")
    plt.show()


if __name__ == "__main__":
    main()