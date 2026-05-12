import os
import cv2
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import spconv.pytorch as spconv
from torch_scatter import scatter_mean

import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.models import Base3DDetector
from .mask_matrix_nms import mask_matrix_nms
from val import get_total_vram_usage
import sys
sys.path.append("/root/wish/oneformer3d_main/segment-anything")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


class ScanNetOneFormer3DMixin:
    """Class contains common methods for ScanNet and ScanNet200."""

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        inst_res = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.predict_by_feat_semantic(out, superpoints)
        pan_res = self.predict_by_feat_panoptic(out, superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]
      
        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]
    
    def predict_by_feat_instance(self, out, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred_sigmoid = mask_pred_sigmoid[:, superpoints]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

    def predict_by_feat_semantic(self, out, superpoints, classes=None):
        """Predict semantic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `sem_preds` of shape (n_queries, n_semantic_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            classes (List[int] or None): semantic (stuff) class ids.
        
        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, n_semantic_classe + 1),
        """
        if classes is None:
            classes = list(range(out['sem_preds'][0].shape[1] - 1))
        return out['sem_preds'][0][:, classes].argmax(dim=1)[superpoints]

    def predict_by_feat_panoptic(self, out, superpoints):
        """Predict panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        sem_map = self.predict_by_feat_semantic(
            out, superpoints, self.test_cfg.stuff_classes)
        mask_pred, labels, scores  = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.pan_score_thr)
        if mask_pred.shape[0] == 0:
            return sem_map, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        n_stuff_classes = len(self.test_cfg.stuff_classes)
        inst_idxs = torch.arange(
            n_stuff_classes, 
            mask_pred.shape[0] + n_stuff_classes, 
            device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs] + n_stuff_classes

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map[things_inst_mask != 0] = 0
        inst_map = sem_map.clone()
        inst_map += things_inst_mask
        sem_map += things_sem_mask
        return sem_map, inst_map
    
    def _select_queries(self, x, gt_instances):
        queries = []
        for i in range(len(x)):
            # Apply self-attention to get importance scores
            self_attn_output, _ = self.self_attn(x[i].unsqueeze(1), x[i].unsqueeze(1), x[i].unsqueeze(1))
            self_attn_output = self.self_attn_norm(self_attn_output.squeeze(1) + x[i])
            self_attn_output = self.self_attn_linear(self_attn_output)
            
            # Calculate importance scores from self-attention output
            importance_scores = F.softmax(self_attn_output, dim=0)
            importance_scores = importance_scores.mean(dim=1, keepdim=True)
            
            if self.query_thr < 1:
                # Dynamically adjust the number of queries based on importance scores
                n = int(self.query_selection_threshold * len(x[i]))
                # Sample points based on importance scores
                ids = torch.multinomial(importance_scores.squeeze(), n, replacement=False)
                queries.append(x[i][ids])
                gt_instances[i].query_masks = gt_instances[i].sp_masks[:, ids]
            else:
                queries.append(x[i])
                gt_instances[i].query_masks = gt_instances[i].sp_masks
        
        return queries, gt_instances


@MODELS.register_module()
class ScanNetOneFormer3D(ScanNetOneFormer3DMixin, Base3DDetector):
    r"""OneFormer3D for ScanNet dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        query_thr (float): We select >= query_thr * n_queries queries
            for training and all n_queries for testing.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 query_thr,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)

        self.scannet_colormap = np.array([
                    [0.5, 0.5, 0.5],  # 背景色（灰色）
                    [174/255, 199/255, 232/255], [152/255, 223/255, 138/255],
                    [31/255, 119/255, 180/255], [255/255, 187/255, 120/255],
                    [188/255, 189/255, 34/255], [140/255, 86/255, 75/255],
                    [255/255, 152/255, 150/255], [214/255, 39/255, 40/255],
                    [197/255, 176/255, 213/255], [148/255, 103/255, 189/255],
                    [196/255, 156/255, 148/255], [23/255, 190/255, 207/255],
                    [247/255, 182/255, 210/255], [219/255, 219/255, 141/255],
                    [255/255, 127/255, 14/255], [158/255, 218/255, 229/255],
                    [44/255, 160/255, 44/255], [112/255, 128/255, 144/255],
                    [227/255, 119/255, 194/255]
                ])
        self.query_thr = query_thr
        self.hidden_dim = in_channels
        
        # Self-Attention Mechanism
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=32, num_heads=8)
        
        # Linear layer for processing
        self.self_attn_linear = torch.nn.Linear(32, 32)
        
        # Layer normalization
        self.self_attn_norm = torch.nn.LayerNorm(32)
        
        # Query selection threshold adjustment
        self.query_selection_threshold = torch.nn.Parameter(torch.tensor(query_thr))
        
        self.sample_img_num = 6
        self.position_embedding = torch.nn.Sequential(
            nn.Linear(12, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256)
        )
        #创建sam模型
        sam_checkpoint = "/root/wish/oneformer3d_main/sam_vit_l_0b3195.pth"
        model_type = "vit_l"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def _select_queries(self, x, gt_instances):
        queries = []
        for i in range(len(x)):
            # Apply self-attention to get importance scores
            self_attn_output, _ = self.self_attn(x[i].unsqueeze(1), x[i].unsqueeze(1), x[i].unsqueeze(1))
            self_attn_output = self.self_attn_norm(self_attn_output.squeeze(1) + x[i])
            self_attn_output = self.self_attn_linear(self_attn_output)
            
            # Calculate importance scores from self-attention output
            importance_scores = F.softmax(self_attn_output, dim=0)
            importance_scores = importance_scores.mean(dim=1, keepdim=True)
            # print(importance_scores)
            # Adjust importance scores based on class labels
            # class_weights = self.calculate_class_weights(gt_instances[i].labels_3d)
            # importance_scores = importance_scores * class_weights.unsqueeze(1).t()
            
            if self.query_thr < 1:
                # Dynamically adjust the number of queries based on importance scores
                n = int(self.query_selection_threshold * len(x[i]))
                # Sample points based on importance scores
                ids = torch.multinomial(importance_scores.squeeze(), n, replacement=False)
                queries.append(x[i][ids])
                gt_instances[i].query_masks = gt_instances[i].sp_masks[:, ids]
            else:
                queries.append(x[i])
                gt_instances[i].query_masks = gt_instances[i].sp_masks
        
        return queries, gt_instances
    
    def calculate_class_weights(self, labels):
        """Calculate class weights based on label distribution."""
        unique_labels, counts = torch.unique(labels, return_counts=True)
        class_weights = 1.0 / counts.float()
        class_weights = class_weights / class_weights.sum()
        return class_weights
    
    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x, superpoints, inverse_mapping, batch_offsets):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).
            superpoints (Tensor): of shape (n_points,).
            inverse_mapping (Tesnor): of shape (n_points,).
            batch_offsets (List[int]): of len batch_size + 1.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = scatter_mean(x.features[inverse_mapping], superpoints, dim=0)
        out = []
        for i in range(len(batch_offsets) - 1):
            out.append(x[batch_offsets[i]: batch_offsets[i + 1]])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])
        
        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass
    
    def preprocess_img(self, img_path):

        image = cv2.imread(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack((img, m*0.35)))

    def load_pose_to_tensor(self, pose_path):
        with open(pose_path, 'r') as f:
            lines = f.readlines()[:3]  
            values = []
            for line in lines:
                values.extend([float(x) for x in line.strip().split()])
        return torch.tensor(values).reshape(1, 12).to('cuda')

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` and `elastic_coords` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
                (`gt_pts_seg`, `gt_instances_3d`, `eval_ann_info` and `gt_instances`)
        Returns:
            dict: A dictionary of loss components.
        """
        # self.visualize_instance(batch_inputs_dict, batch_data_samples)
        batch_offsets = [0]
        superpoint_bias = 0
        sp_gt_instances = []
        sp_pts_masks = []
        total_feat = []
        feat_tokens = []
        image_tokens = []
        pose_tokens = []
        for i in range(len(batch_data_samples)):
            original_path = batch_data_samples[i].lidar_path
            img_path = original_path.replace('/points/', '/scans/').replace('.bin', '/color')
            pose_path = original_path.replace('/points/', '/scans/').replace('.bin', '/pose')
            all_pose_names = sorted(os.listdir(pose_path), key=lambda x: int(x.split(".")[0]))
            valid_pose_names = []
            valid_img_paths = []
            valid_pose_paths = []
            for fname in all_pose_names:
                full_pose_path = os.path.join(pose_path, fname)
                pose_tensor = self.load_pose_to_tensor(full_pose_path).squeeze(0)
                if not torch.isinf(pose_tensor).any():
                    valid_pose_names.append(fname)
                    valid_pose_paths.append(full_pose_path)
                    img_name = fname.replace(".txt", ".jpg") 
                    full_img_path = os.path.join(img_path, img_name)
                    valid_img_paths.append(full_img_path)
            if len(valid_pose_names) < self.sample_img_num:
                raise ValueError(f"Insufficient valid frames: only {len(valid_pose_names)} frames available, but {self.sample_img_num} frames required.")
            step = len(valid_pose_names) // self.sample_img_num
            selected_pose_paths = valid_pose_paths[::step][:self.sample_img_num]
            selected_img_paths = valid_img_paths[::step][:self.sample_img_num]
            pose_token = torch.stack([self.load_pose_to_tensor(p).squeeze(0) for p in selected_pose_paths], dim=0)
            img_list = [self.preprocess_img(p) for p in selected_img_paths]
            pose_token = self.position_embedding(pose_token)
            processed_feats = []
            image_tokens_list = []
            feat_tokens_list = []
            for img in img_list:
                feat, image_token = self.mask_generator.generate(img, True)
                processed_feat = torch.mean(feat, dim=0).reshape(1, -1)
                processed_feats.append(processed_feat)
                feat_tokens_list.append(feat)
                image_tokens_list.append(image_token)
            # feat_list = [torch.mean(mask_generator.generate(_, True), dim=0).reshape(1,-1) for _ in img_list]
            feat_list = torch.cat(processed_feats,dim=0)
            image_tokens_list = torch.cat(image_tokens_list, dim=0)
            feat_tokens_list = torch.cat(feat_tokens_list, dim=0)
            total_feat.append(feat_list)
            feat_tokens.append(feat_tokens_list)
            image_tokens.append(image_tokens_list)
            pose_tokens.append(pose_token)
            del feat_list
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)

            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))

        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)

        x = self.extract_feat(
            x, sp_pts_masks, inverse_mapping, batch_offsets)
        queries, sp_gt_instances = self._select_queries(x, sp_gt_instances)
        x, distill_loss = self.decoder(x, queries, total_feat, image_tokens, feat_tokens, pose_tokens)
        loss = self.criterion(x, sp_gt_instances, weights=None)
        loss['distill_loss'] = distill_loss
        # saved_files = self.save_training_data(
        #             batch_inputs_dict, 
        #             batch_data_samples, 
        #             x
        #         )
        # print(f"Saved training data to: {saved_files}")
        return loss
    
    # def save_training_data(self, batch_inputs_dict, batch_data_samples, x):  
    #     """
    #     保存训练数据到文件
    #     :param output_dir: 输出目录（自动按时间创建子目录）
    #     :return: 保存的文件路径列表
    #     """
    #     # 创建带时间戳的输出目录
    #     from datetime import datetime
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     save_dir = os.path.join("/root/wish/oneformer3d_main", timestamp)
    #     os.makedirs(save_dir, exist_ok=True)

    #     saved_files = []

    #     # 1. 保存batch_inputs_dict（点云数据）
    #     points_data = {
    #         'points': [tensor.cpu().numpy() for tensor in batch_inputs_dict['points']],
    #         'elastic_coords': batch_inputs_dict.get('elastic_coords', None)
    #     }
    #     points_path = os.path.join(save_dir, "batch_inputs.pkl")
    #     with open(points_path, 'wb') as f:
    #         pickle.dump(points_data, f)
    #     saved_files.append(points_path)

    #     return saved_files

    # def save_predictions(self, batch_data_samples, save_dir="predictions_clean"):
    #     """
    #     将 batch_data_samples 中的预测结果安全转换并保存为 .npz 文件
        
    #     Args:
    #         batch_data_samples (list): 模型输出的预测结果集合
    #         save_dir (str): 保存目录
    #     """
    #     os.makedirs(save_dir, exist_ok=True)

    #     def safe_convert(data):
    #         """增强版类型转换（处理非均匀数据）"""
    #         if isinstance(data, torch.Tensor):
    #             return data.detach().cpu().numpy()
    #         elif isinstance(data, np.ndarray):
    #             return data
    #         elif isinstance(data, (list, tuple)):
    #             if len(data) == 0:
    #                 return np.array([])
                
    #             elem_types = {type(x) for x in data}
                
    #             if all(isinstance(x, (int, float)) for x in data):
    #                 return np.array(data)
    #             elif all(isinstance(x, (list, tuple, np.ndarray)) for x in data):
    #                 try:
    #                     return np.array(data)
    #                 except ValueError:
    #                     arr = np.empty(len(data), dtype=object)
    #                     arr[:] = data
    #                     return arr
    #             else:
    #                 raise ValueError(f"Mixed types in data: {elem_types}")
    #         else:
    #             raise TypeError(f"Cannot convert type {type(data)}")

    #     saved_files = []  # 用来返回所有保存的文件路径

    #     for idx, sample in enumerate(batch_data_samples):
    #         pred = sample.pred_pts_seg

    #         if not hasattr(pred, 'pts_instance_mask'):
    #             raise AttributeError("pred object missing pts_instance_mask")

    #         try:
    #             masks = safe_convert(pred.pts_instance_mask)
    #         except ValueError:
    #             print(f"Warning: Irregular mask shape in sample {idx}, using object array")
    #             masks = np.array(pred.pts_instance_mask, dtype=object)

    #         # 获取 lidar_path（可能在 sample 或 sample.metainfo 中）
    #         lidar_path = getattr(sample, "lidar_path", None)
    #         if lidar_path is None and hasattr(sample, "metainfo"):
    #             lidar_path = sample.metainfo.get("lidar_path", "unknown")

    #         data_dict = {
    #             "labels": safe_convert(pred.instance_labels),
    #             "scores": safe_convert(pred.instance_scores),
    #             "masks": masks,
    #             "meta": {
    #                 "sample_id": idx,
    #                 "mask_type": str(type(masks[0])) if len(masks) > 0 else "empty",
    #                 "lidar_path": lidar_path
    #             }
    #         }

    #         save_path = os.path.join(save_dir, f"pred_{idx}.npz")
    #         np.savez_compressed(save_path, **data_dict)
    #         saved_files.append(save_path)

    #     return saved_files

    def save_training_data(self, batch_inputs_dict, scene_dir):
        """
        保存 batch_inputs_dict 到指定场景目录
        """
        os.makedirs(scene_dir, exist_ok=True)
        points_data = {
            'points': [tensor.cpu().numpy() for tensor in batch_inputs_dict['points']],
            'elastic_coords': batch_inputs_dict.get('elastic_coords', None)
        }

        points_path = os.path.join(scene_dir, "batch_inputs.pkl")
        with open(points_path, 'wb') as f:
            pickle.dump(points_data, f)
        return points_path

    def save_predictions(self, batch_inputs_dict, batch_data_samples):
        """
        根据 batch_data_samples 中的 lidar_path 自动创建目录，
        并保存对应的输入点云和预测结果。
        """
        base_dir = "/root/wish/oneformer3d_main/prediction"
        os.makedirs(base_dir, exist_ok=True)

        def safe_convert(data):
            """增强版类型转换（处理非均匀数据）"""
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            elif isinstance(data, (list, tuple)):
                if len(data) == 0:
                    return np.array([])
                elem_types = {type(x) for x in data}
                if all(isinstance(x, (int, float)) for x in data):
                    return np.array(data)
                elif all(isinstance(x, (list, tuple, np.ndarray)) for x in data):
                    try:
                        return np.array(data)
                    except ValueError:
                        arr = np.empty(len(data), dtype=object)
                        arr[:] = data
                        return arr
                else:
                    raise ValueError(f"Mixed types in data: {elem_types}")
            else:
                raise TypeError(f"Cannot convert type {type(data)}")

        saved_files = []

        for idx, sample in enumerate(batch_data_samples):
            # 1️⃣ 获取场景名
            lidar_path = getattr(sample, "lidar_path", None)
            if lidar_path is None and hasattr(sample, "metainfo"):
                lidar_path = sample.metainfo.get("lidar_path", "unknown")
            if lidar_path is None:
                scene_name = f"sample_{idx}"
            else:
                scene_name = os.path.splitext(os.path.basename(lidar_path))[0]

            scene_dir = os.path.join(base_dir, scene_name)
            os.makedirs(scene_dir, exist_ok=True)

            # 2️⃣ 保存输入点云
            input_path = self.save_training_data(batch_inputs_dict, scene_dir)

            # 3️⃣ 保存预测结果
            pred = sample.pred_pts_seg
            if not hasattr(pred, 'pts_instance_mask'):
                raise AttributeError("pred object missing pts_instance_mask")

            try:
                masks = safe_convert(pred.pts_instance_mask)
            except ValueError:
                print(f"Warning: Irregular mask shape in sample {idx}, using object array")
                masks = np.array(pred.pts_instance_mask, dtype=object)

            data_dict = {
                "labels": safe_convert(pred.instance_labels),
                "scores": safe_convert(pred.instance_scores),
                "masks": masks,
                "meta": {
                    "sample_id": idx,
                    "lidar_path": lidar_path
                }
            }

            pred_path = os.path.join(scene_dir, "pred.npz")
            np.savez_compressed(pred_path, **data_dict)

            saved_files.append({
                "scene": scene_name,
                "input": input_path,
                "pred": pred_path
            })

        return saved_files

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        # self.visualize_instance(batch_inputs_dict, batch_data_samples)
        batch_offsets = [0]
        superpoint_bias = 0
        sp_gt_instances = []
        sp_pts_masks = []
        total_feat = []
        feat_tokens = []
        image_tokens = []
        pose_tokens = []
        for i in range(len(batch_data_samples)):
            original_path = batch_data_samples[i].lidar_path
            img_path = original_path.replace('/points/', '/scans/').replace('.bin', '/color')
            pose_path = original_path.replace('/points/', '/scans/').replace('.bin', '/pose')
            all_pose_names = sorted(os.listdir(pose_path), key=lambda x: int(x.split(".")[0]))
            valid_pose_names = []
            valid_img_paths = []
            valid_pose_paths = []
            for fname in all_pose_names:
                full_pose_path = os.path.join(pose_path, fname)
                pose_tensor = self.load_pose_to_tensor(full_pose_path).squeeze(0)
                if not torch.isinf(pose_tensor).any():
                    valid_pose_names.append(fname)
                    valid_pose_paths.append(full_pose_path)
                    img_name = fname.replace(".txt", ".jpg") 
                    full_img_path = os.path.join(img_path, img_name)
                    valid_img_paths.append(full_img_path)
            if len(valid_pose_names) < self.sample_img_num:
                raise ValueError(f"Insufficient valid frames: only {len(valid_pose_names)} frames available, but {self.sample_img_num} frames required.")
            step = len(valid_pose_names) // self.sample_img_num
            selected_pose_paths = valid_pose_paths[::step][:self.sample_img_num]
            selected_img_paths = valid_img_paths[::step][:self.sample_img_num]
            pose_token = torch.stack([self.load_pose_to_tensor(p).squeeze(0) for p in selected_pose_paths], dim=0)
            img_list = [self.preprocess_img(p) for p in selected_img_paths]
            pose_token = self.position_embedding(pose_token)
            processed_feats = []
            image_tokens_list = []
            feat_tokens_list = []
            for img in img_list:
                feat, image_token = self.mask_generator.generate(img, True)
                processed_feat = torch.mean(feat, dim=0).reshape(1, -1)
                processed_feats.append(processed_feat)
                feat_tokens_list.append(feat)
                image_tokens_list.append(image_token)
            # feat_list = [torch.mean(mask_generator.generate(_, True), dim=0).reshape(1,-1) for _ in img_list]
            feat_list = torch.cat(processed_feats,dim=0)
            image_tokens_list = torch.cat(image_tokens_list, dim=0)
            feat_tokens_list = torch.cat(feat_tokens_list, dim=0)
            total_feat.append(feat_list)
            feat_tokens.append(feat_tokens_list)
            image_tokens.append(image_tokens_list)
            pose_tokens.append(pose_token)
            del feat_list
            gt_pts_seg = batch_data_samples[i].gt_pts_seg
            gt_pts_seg.sp_pts_mask += superpoint_bias
            superpoint_bias = gt_pts_seg.sp_pts_mask.max().item() + 1
            batch_offsets.append(superpoint_bias)
            sp_pts_masks.append(gt_pts_seg.sp_pts_mask)

        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])

        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        sp_pts_masks = torch.hstack(sp_pts_masks)
        x = self.extract_feat(
            x, sp_pts_masks, inverse_mapping, batch_offsets)
        x, _ = self.decoder(x, x, total_feat, image_tokens, feat_tokens, pose_tokens)
        # self.save_training_data(batch_inputs_dict, batch_data_samples, x)
        results_list = self.predict_by_feat(x, sp_pts_masks)
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]

        file = self.save_predictions(batch_inputs_dict, batch_data_samples)
        print("保存的文件列表:", file)
        return batch_data_samples
 
@MODELS.register_module()
class ScanNet200OneFormer3D(ScanNetOneFormer3DMixin, Base3DDetector):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 backbone=None,
                 neck=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.sample_img_num = 6
        self.position_embedding = torch.nn.Sequential(
            nn.Linear(12, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256)
        )
        self.query_thr = query_thr
        
        # Self-Attention Mechanism
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=96, num_heads=8)
        
        # Linear layer for processing
        self.self_attn_linear = torch.nn.Linear(96, 96)
        
        # Layer normalization
        self.self_attn_norm = torch.nn.LayerNorm(96)
        
        # Query selection threshold adjustment
        self.query_selection_threshold = torch.nn.Parameter(torch.tensor(query_thr))
        #创建sam模型
        sam_checkpoint = "/root/wish/oneformer3d_main/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.sample_img_num = 6

    def extract_feat(self, batch_inputs_dict, batch_data_samples):
        """Extract features from sparse tensor.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.

        Returns:
            Tuple:
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_channels).
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_classes + 1).
        """
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][:, :3])
            features.append(batch_inputs_dict['points'][i][:, 3:])
        
        coordinates, features = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse())
        if self.with_neck:
            x = self.neck(x)
        x = x.slice(field).features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        x = scatter_mean(x, torch.cat(sp_pts_masks), dim=0)  # todo: do we need dim?

        # apply cls_layer
        features = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end])
        return features

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def preprocess_img(self, img_path):

        image = cv2.imread(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image

    def calculate_class_weights(self, labels):
        """Calculate class weights based on label distribution."""
        unique_labels, counts = torch.unique(labels, return_counts=True)
        class_weights = 1.0 / counts.float()
        class_weights = class_weights / class_weights.sum()
        return class_weights
    
    def load_pose_to_tensor(self, pose_path):
        with open(pose_path, 'r') as f:
            lines = f.readlines()[:3]  
            values = []
            for line in lines:
                values.extend([float(x) for x in line.strip().split()])
        return torch.tensor(values).reshape(1, 12).to('cuda')

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        total_feat = []
        feat_tokens = []
        image_tokens = []
        pose_tokens = []
        for i in range(len(batch_data_samples)):
            original_path = batch_data_samples[i].lidar_path
            img_path = original_path.replace('/ScanNet200/', '/ScanNet V2/').replace('/points/', '/scans/').replace('.bin', '/color')
            pose_path = original_path.replace('/ScanNet200/', '/ScanNet V2/').replace('/points/', '/scans/').replace('.bin', '/pose')
            all_pose_names = sorted(os.listdir(pose_path), key=lambda x: int(x.split(".")[0]))
            valid_pose_names = []
            valid_img_paths = []
            valid_pose_paths = []
            for fname in all_pose_names:
                full_pose_path = os.path.join(pose_path, fname)
                pose_tensor = self.load_pose_to_tensor(full_pose_path).squeeze(0)
                if not torch.isinf(pose_tensor).any():
                    valid_pose_names.append(fname)
                    valid_pose_paths.append(full_pose_path)
                    img_name = fname.replace(".txt", ".jpg") 
                    full_img_path = os.path.join(img_path, img_name)
                    valid_img_paths.append(full_img_path)
            if len(valid_pose_names) < self.sample_img_num:
                raise ValueError(f"Insufficient valid frames: only {len(valid_pose_names)} frames available, but {self.sample_img_num} frames required.")
            step = len(valid_pose_names) // self.sample_img_num
            selected_pose_paths = valid_pose_paths[::step][:self.sample_img_num]
            selected_img_paths = valid_img_paths[::step][:self.sample_img_num]
            pose_token = torch.stack([self.load_pose_to_tensor(p).squeeze(0) for p in selected_pose_paths], dim=0)
            img_list = [self.preprocess_img(p) for p in selected_img_paths]

            pose_token = self.position_embedding(pose_token)
            processed_feats = []
            image_tokens_list = []
            feat_tokens_list = []
            for img in img_list:
                feat, image_token = self.mask_generator.generate(img, True)
                processed_feat = torch.mean(feat, dim=0).reshape(1, -1)
                processed_feats.append(processed_feat)
                feat_tokens_list.append(feat)
                image_tokens_list.append(image_token)
            # feat_list = [torch.mean(mask_generator.generate(_, True), dim=0).reshape(1,-1) for _ in img_list]
            feat_list = torch.cat(processed_feats,dim=0)
            image_tokens_list = torch.cat(image_tokens_list, dim=0)
            feat_tokens_list = torch.cat(feat_tokens_list, dim=0)
            total_feat.append(feat_list)
            feat_tokens.append(feat_tokens_list)
            image_tokens.append(image_tokens_list)
            pose_tokens.append(pose_token)
            del feat_list
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        gt_instances = [s.gt_instances_3d for s in batch_data_samples]
        queries, gt_instances = self._select_queries(x, gt_instances)
        x, distill_loss = self.decoder(x, queries, total_feat, image_tokens, feat_tokens, pose_tokens)
        loss = self.criterion(x, gt_instances, weights=None)
        loss['distill_loss'] = distill_loss
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        total_feat = []
        feat_tokens = []
        image_tokens = []
        pose_tokens = []
        for i in range(len(batch_data_samples)):
            original_path = batch_data_samples[i].lidar_path
            img_path = original_path.replace('/ScanNet200/', '/ScanNet V2/').replace('/points/', '/scans/').replace('.bin', '/color')
            pose_path = original_path.replace('/ScanNet200/', '/ScanNet V2/').replace('/points/', '/scans/').replace('.bin', '/pose')
            all_pose_names = sorted(os.listdir(pose_path), key=lambda x: int(x.split(".")[0]))
            valid_pose_names = []
            valid_img_paths = []
            valid_pose_paths = []
            for fname in all_pose_names:
                full_pose_path = os.path.join(pose_path, fname)
                pose_tensor = self.load_pose_to_tensor(full_pose_path).squeeze(0)
                if not torch.isinf(pose_tensor).any():
                    valid_pose_names.append(fname)
                    valid_pose_paths.append(full_pose_path)
                    img_name = fname.replace(".txt", ".jpg") 
                    full_img_path = os.path.join(img_path, img_name)
                    valid_img_paths.append(full_img_path)
            if len(valid_pose_names) < self.sample_img_num:
                raise ValueError(f"Insufficient valid frames: only {len(valid_pose_names)} frames available, but {self.sample_img_num} frames required.")
            step = len(valid_pose_names) // self.sample_img_num
            selected_pose_paths = valid_pose_paths[::step][:self.sample_img_num]
            selected_img_paths = valid_img_paths[::step][:self.sample_img_num]
            pose_token = torch.stack([self.load_pose_to_tensor(p).squeeze(0) for p in selected_pose_paths], dim=0)
            img_list = [self.preprocess_img(p) for p in selected_img_paths]

            pose_token = self.position_embedding(pose_token)
            processed_feats = []
            image_tokens_list = []
            feat_tokens_list = []
            for img in img_list:
                feat, image_token = self.mask_generator.generate(img, True)
                processed_feat = torch.mean(feat, dim=0).reshape(1, -1)
                processed_feats.append(processed_feat)
                feat_tokens_list.append(feat)
                image_tokens_list.append(image_token)
            # feat_list = [torch.mean(mask_generator.generate(_, True), dim=0).reshape(1,-1) for _ in img_list]
            feat_list = torch.cat(processed_feats,dim=0)
            image_tokens_list = torch.cat(image_tokens_list, dim=0)
            feat_tokens_list = torch.cat(feat_tokens_list, dim=0)
            total_feat.append(feat_list)
            feat_tokens.append(feat_tokens_list)
            image_tokens.append(image_tokens_list)
            pose_tokens.append(pose_token)
            del feat_list
        assert len(batch_data_samples) == 1
        x = self.extract_feat(batch_inputs_dict, batch_data_samples)
        x, _ = self.decoder(x, x, total_feat, image_tokens, feat_tokens, pose_tokens)
        pred_pts_seg = self.predict_by_feat(
            x, batch_data_samples[0].gt_pts_seg.sp_pts_mask)
        batch_data_samples[0].pred_pts_seg = pred_pts_seg[0]
        return batch_data_samples


@MODELS.register_module()
class S3DISOneFormer3D(Base3DDetector):
    r"""OneFormer3D for S3DIS dataset.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): NUmber of output channels.
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes,
                 min_spatial_shape,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)
        self.sample_img_num = 6
        self.query_thr = 0.5
        # Self-Attention Mechanism
        self.self_attn = torch.nn.MultiheadAttention(embed_dim=64, num_heads=8)
        # Linear layer for processing
        self.self_attn_linear = torch.nn.Linear(64, 64)
        
        # Layer normalization
        self.self_attn_norm = torch.nn.LayerNorm(64)
        
        # Query selection threshold adjustment
        self.query_selection_threshold = torch.nn.Parameter(torch.tensor(0.5))
        sam_checkpoint = "/root/wish/oneformer3d_main/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.position_embedding = torch.nn.Sequential(
            nn.Linear(12, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256)
        )

    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))
        
    def preprocess_img(self, img_path):

        image = cv2.imread(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _select_queries(self, x, gt_instances):
        queries = []
        for i in range(len(x)):
            # Apply self-attention to get importance scores
            self_attn_output, _ = self.self_attn(x[i].unsqueeze(1), x[i].unsqueeze(1), x[i].unsqueeze(1))
            self_attn_output = self.self_attn_norm(self_attn_output.squeeze(1) + x[i])
            self_attn_output = self.self_attn_linear(self_attn_output)
            
            # Calculate importance scores from self-attention output
            importance_scores = F.softmax(self_attn_output, dim=0)
            importance_scores = importance_scores.mean(dim=1, keepdim=True)
            
            if self.query_thr < 1:
                # Dynamically adjust the number of queries based on importance scores
                n = int(self.query_selection_threshold * len(x[i]))
                # Sample points based on importance scores
                ids = torch.multinomial(importance_scores.squeeze(), n, replacement=False)
                queries.append(x[i][ids])
                gt_instances[i].query_masks = gt_instances[i].sp_masks[:, ids]
            else:
                queries.append(x[i])
                gt_instances[i].query_masks = gt_instances[i].sp_masks
        return queries, gt_instances

    def calculate_class_weights(self, labels):
        """Calculate class weights based on label distribution."""
        unique_labels, counts = torch.unique(labels, return_counts=True)
        class_weights = 1.0 / counts.float()
        class_weights = class_weights / class_weights.sum()
        return class_weights
    
    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])

        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def load_pose_to_tensor(self, pose_path):
        with open(pose_path, 'r') as f:
            lines = f.readlines()[:3]  
            values = []
            for line in lines:
                values.extend([float(x) for x in line.strip().split()])
        return torch.tensor(values).reshape(1, 12).to('cuda')

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        total_feat = []
        feat_tokens = []
        image_tokens = []
        pose_tokens = []
        for i in range(len(batch_data_samples)):
            original_path = batch_data_samples[i].lidar_path
            filename = os.path.splitext(os.path.basename(original_path))[0]  # "Area_1_office_5"
            area, room = filename.split('_', maxsplit=2)[:2], filename.split('_', maxsplit=2)[2]
            area = '_'.join([area[0], area[1]])  # "Area_1"
            room = room  # "office_5"

            # 2. 拼接目标路径
            target_path = os.path.join(
                "/mnt/d/s3dis/Stanford3dDataset_v1.2_Aligned_Version",
                area,
                room
            )
            img_path = os.path.join(target_path, "rgb")
            pose_path = os.path.join(target_path, "pose")
            all_pose_names = sorted(os.listdir(pose_path), key=lambda x: int(x.split(".")[0]))
            valid_pose_names = []
            valid_img_paths = []
            valid_pose_paths = []
            for fname in all_pose_names:
                full_pose_path = os.path.join(pose_path, fname)
                pose_tensor = self.load_pose_to_tensor(full_pose_path).squeeze(0)
                if not torch.isinf(pose_tensor).any():
                    valid_pose_names.append(fname)
                    valid_pose_paths.append(full_pose_path)
                    img_name = fname.replace(".txt", ".jpg") 
                    full_img_path = os.path.join(img_path, img_name)
                    valid_img_paths.append(full_img_path)
            if len(valid_pose_names) < self.sample_img_num:
                raise ValueError(f"Insufficient valid frames: only {len(valid_pose_names)} frames available, but {self.sample_img_num} frames required.")
            step = len(valid_pose_names) // self.sample_img_num
            selected_pose_paths = valid_pose_paths[::step][:self.sample_img_num]
            selected_img_paths = valid_img_paths[::step][:self.sample_img_num]
            pose_token = torch.stack([self.load_pose_to_tensor(p).squeeze(0) for p in selected_pose_paths], dim=0)
            img_list = [self.preprocess_img(p) for p in selected_img_paths]

            pose_token = self.position_embedding(pose_token)
            processed_feats = []
            image_tokens_list = []
            feat_tokens_list = []
            for img in img_list:
                feat, image_token = self.mask_generator.generate(img, True)
                processed_feat = torch.mean(feat, dim=0).reshape(1, -1)
                processed_feats.append(processed_feat)
                feat_tokens_list.append(feat)
                image_tokens_list.append(image_token)
            feat_list = torch.cat(processed_feats,dim=0)
            image_tokens_list = torch.cat(image_tokens_list, dim=0)
            feat_tokens_list = torch.cat(feat_tokens_list, dim=0)
            total_feat.append(feat_list)
            feat_tokens.append(feat_tokens_list)
            image_tokens.append(image_tokens_list)
            pose_tokens.append(pose_token)
            del feat_list
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))
        x = self.extract_feat(x)
        x, distill_loss = self.decoder(x, None, total_feat, image_tokens, feat_tokens, pose_tokens)
        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[coordinates[:, 0][ \
                                                        inverse_mapping] == i]
            voxel_superpoints = torch.unique(voxel_superpoints,
                                             return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            sem_mask = batch_data_samples[i].gt_pts_seg.pts_semantic_mask
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_sem_masks = \
                                self.get_gt_semantic_masks(sem_mask,
                                                            voxel_superpoints,
                                                            self.num_classes)
            batch_data_samples[i].gt_instances_3d.sp_inst_masks = \
                                self.get_gt_inst_masks(inst_mask,
                                                       voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)
        loss = self.criterion(x, sp_gt_instances)
        loss['distill_loss'] = distill_loss
        return loss

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        total_feat = []
        feat_tokens = []
        image_tokens = []
        pose_tokens = []
        for i in range(len(batch_data_samples)):
            original_path = batch_data_samples[i].lidar_path
            filename = os.path.splitext(os.path.basename(original_path))[0]  # "Area_1_office_5"
            area, room = filename.split('_', maxsplit=2)[:2], filename.split('_', maxsplit=2)[2]
            area = '_'.join([area[0], area[1]])  # "Area_1"
            room = room  # "office_5"
            # 2. 拼接目标路径
            target_path = os.path.join(
                "/mnt/d/s3dis/Stanford3dDataset_v1.2_Aligned_Version",
                area,
                room
            )
            img_path = os.path.join(target_path, "rgb")
            pose_path = os.path.join(target_path, "pose")
            all_pose_names = sorted(os.listdir(pose_path), key=lambda x: int(x.split(".")[0]))
            valid_pose_names = []
            valid_img_paths = []
            valid_pose_paths = []
            for fname in all_pose_names:
                full_pose_path = os.path.join(pose_path, fname)
                pose_tensor = self.load_pose_to_tensor(full_pose_path).squeeze(0)
                if not torch.isinf(pose_tensor).any():
                    valid_pose_names.append(fname)
                    valid_pose_paths.append(full_pose_path)
                    img_name = fname.replace(".txt", ".jpg") 
                    full_img_path = os.path.join(img_path, img_name)
                    valid_img_paths.append(full_img_path)
            if len(valid_pose_names) < self.sample_img_num:
                raise ValueError(f"Insufficient valid frames: only {len(valid_pose_names)} frames available, but {self.sample_img_num} frames required.")
            step = len(valid_pose_names) // self.sample_img_num
            selected_pose_paths = valid_pose_paths[::step][:self.sample_img_num]
            selected_img_paths = valid_img_paths[::step][:self.sample_img_num]
            pose_token = torch.stack([self.load_pose_to_tensor(p).squeeze(0) for p in selected_pose_paths], dim=0)
            img_list = [self.preprocess_img(p) for p in selected_img_paths]

            pose_token = self.position_embedding(pose_token)
            processed_feats = []
            image_tokens_list = []
            feat_tokens_list = []
            for img in img_list:
                feat, image_token = self.mask_generator.generate(img, True)
                processed_feat = torch.mean(feat, dim=0).reshape(1, -1)
                processed_feats.append(processed_feat)
                feat_tokens_list.append(feat)
                image_tokens_list.append(image_token)
            # feat_list = [torch.mean(mask_generator.generate(_, True), dim=0).reshape(1,-1) for _ in img_list]
            feat_list = torch.cat(processed_feats,dim=0)
            image_tokens_list = torch.cat(image_tokens_list, dim=0)
            feat_tokens_list = torch.cat(feat_tokens_list, dim=0)
            total_feat.append(feat_list)
            feat_tokens.append(feat_tokens_list)
            image_tokens.append(image_tokens_list)
            pose_tokens.append(pose_token)
            del feat_list
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        x,_ = self.decoder(x, None, total_feat, image_tokens, feat_tokens, pose_tokens)

        results_list = self.predict_by_feat(x, inverse_mapping)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints):
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).

        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        pred_labels = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        pred_scores = out['scores'][0]

        inst_res = self.pred_inst(pred_masks[:-self.test_cfg.num_sem_cls, :],
                                  pred_scores[:-self.test_cfg.num_sem_cls, :],
                                  pred_labels[:-self.test_cfg.num_sem_cls, :],
                                  superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.pred_sem(pred_masks[-self.test_cfg.num_sem_cls:, :],
                                superpoints)
        pan_res = self.pred_pan(pred_masks, pred_scores, pred_labels,
                                superpoints)

        pts_semantic_mask = [sem_res.cpu().numpy(), pan_res[0].cpu().numpy()]
        pts_instance_mask = [inst_res[0].cpu().bool().numpy(),
                             pan_res[1].cpu().numpy()]

        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=inst_res[1].cpu().numpy(),
                instance_scores=inst_res[2].cpu().numpy())]

    def pred_inst(self, pred_masks, pred_scores, pred_labels,
                  superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        scores = F.softmax(pred_labels, dim=-1)[:, :-1]
        scores *= pred_scores

        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                self.decoder.num_queries - self.test_cfg.num_sem_cls,
                1).flatten(0, 1)
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores
   
    def pred_sem(self, pred_masks, superpoints):
        """Predict semantic masks for a single scene.

        Args:
            pred_masks (Tensor): of shape (n_points, n_semantic_classes).
            superpoints (Tensor): of shape (n_raw_points,).        

        Returns:
            Tensor: semantic preds of shape
                (n_raw_points, 1).
        """
        mask_pred = pred_masks.sigmoid()
        mask_pred = mask_pred[:, superpoints]
        seg_map = mask_pred.argmax(0)
        return seg_map

    def pred_pan(self, pred_masks, pred_scores, pred_labels,
                 superpoints):
        """Predict panoptic masks for a single scene.
        
        Args:
            pred_masks (Tensor): of shape (n_queries, n_points).
            pred_scores (Tensor): of shape (n_queris, 1).
            pred_labels (Tensor): of shape (n_queries, n_instance_classes + 1).
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        stuff_cls = pred_masks.new_tensor(self.test_cfg.stuff_cls).long()
        sem_map = self.pred_sem(
            pred_masks[-self.test_cfg.num_sem_cls + stuff_cls, :], superpoints)
        sem_map_src_mapping = stuff_cls[sem_map]

        n_cls = self.test_cfg.num_sem_cls
        thr = self.test_cfg.pan_score_thr
        mask_pred, labels, scores = self.pred_inst(
            pred_masks[:-n_cls, :], pred_scores[:-n_cls, :],
            pred_labels[:-n_cls, :], superpoints, thr)
        
        thing_idxs = torch.zeros_like(labels)
        for thing_cls in self.test_cfg.thing_cls:
            thing_idxs = thing_idxs.logical_or(labels == thing_cls)
        
        mask_pred = mask_pred[thing_idxs]
        scores = scores[thing_idxs]
        labels = labels[thing_idxs]

        if mask_pred.shape[0] == 0:
            return sem_map_src_mapping, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        inst_idxs = torch.arange(
            0, mask_pred.shape[0], device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(axis=0)
        things_sem_mask = labels[idxs]

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_inst_mask = torch.unique(
            things_inst_mask, return_inverse=True)[1]
        things_inst_mask[things_inst_mask != 0] += len(stuff_cls) - 1
        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map_src_mapping[things_inst_mask != 0] = 0
        sem_map[things_inst_mask != 0] = 0
        sem_map += things_inst_mask
        sem_map_src_mapping += things_sem_mask
        return sem_map_src_mapping, sem_map

    @staticmethod    
    def get_gt_semantic_masks(mask_src, sp_pts_mask, num_classes):    
        """Create ground truth semantic masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
            num_classes (Int): number of classes.
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_classes).
        """

        mask = torch.nn.functional.one_hot(
            mask_src, num_classes=num_classes + 1)

        mask = mask.T
        sp_masks = scatter_mean(mask.float(), sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5
        sp_masks[-1, sp_masks.sum(axis=0) == 0] = True
        assert sp_masks.sum(axis=0).max().item() == 1

        return sp_masks

    @staticmethod
    def get_gt_inst_masks(mask_src, sp_pts_mask):
        """Create ground truth instance masks.
        
        Args:
            mask_src (Tensor): of shape (n_raw_points, 1).
            sp_pts_mask (Tensor): of shape (n_raw_points, 1).
        
        Returns:
            sp_masks (Tensor): semantic mask of shape (n_points, num_inst_obj).
        """
        mask = mask_src.clone()
        if torch.sum(mask == -1) != 0:
            mask[mask == -1] = torch.max(mask) + 1
            mask = torch.nn.functional.one_hot(mask)[:, :-1]
        else:
            mask = torch.nn.functional.one_hot(mask)

        mask = mask.T
        sp_masks = scatter_mean(mask, sp_pts_mask, dim=-1)
        sp_masks = sp_masks > 0.5

        return sp_masks


@MODELS.register_module()
class InstanceOnlyOneFormer3D(Base3DDetector):
    r"""InstanceOnlyOneFormer3D for training on different datasets jointly.

    Args:
        in_channels (int): Number of input channels.
        num_channels (int): Number of output channels.
        voxel_size (float): Voxel size.
        num_classes_1dataset (int): Number of classes in the first dataset.
        num_classes_2dataset (int): Number of classes in the second dataset.
        prefix_1dataset (string): Prefix for the first dataset.
        prefix_2dataset (string): Prefix for the second dataset.
        min_spatial_shape (int): Minimal shape for spconv tensor.
        backbone (ConfigDict): Config dict of the backbone.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 num_channels,
                 voxel_size,
                 num_classes_1dataset,
                 num_classes_2dataset,
                 prefix_1dataset,
                 prefix_2dataset,
                 min_spatial_shape,
                 backbone=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(InstanceOnlyOneFormer3D, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.num_classes_1dataset = num_classes_1dataset 
        self.num_classes_2dataset = num_classes_2dataset
        
        self.prefix_1dataset = prefix_1dataset 
        self.prefix_2dataset = prefix_2dataset
        
        self.unet = MODELS.build(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.min_spatial_shape = min_spatial_shape
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)
    
    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1'))
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True))

    def extract_feat(self, x):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        out = []
        for i in x.indices[:, 0].unique():
            out.append(x.features[x.indices[:, 0] == i])
        return out

    def collate(self, points, elastic_points=None):
        """Collate batch of points to sparse tensor.

        Args:
            points (List[Tensor]): Batch of points.
            quantization_mode (SparseTensorQuantizationMode): Minkowski
                quantization mode. We use random sample for training
                and unweighted average for inference.

        Returns:
            TensorField: Containing features and coordinates of a
                sparse tensor.
        """
        if elastic_points is None:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - p[:, :3].min(0)[0]) / self.voxel_size,
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for p in points])
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((el_p - el_p.min(0)[0]),
                  torch.hstack((p[:, 3:], p[:, :3] - p[:, :3].mean(0))))
                 for el_p, p in zip(elastic_points, points)])
        
        spatial_shape = torch.clip(
            coordinates.max(0)[0][1:] + 1, self.min_spatial_shape)
        field = ME.TensorField(features=features, coordinates=coordinates)
        tensor = field.sparse()
        coordinates = tensor.coordinates
        features = tensor.features
        inverse_mapping = field.inverse_mapping(tensor.coordinate_map_key)

        return coordinates, features, inverse_mapping, spatial_shape

    def _forward(*args, **kwargs):
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'],
            batch_inputs_dict.get('elastic_coords', None))
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        scene_names = []
        for i in range(len(batch_data_samples)):
           scene_names.append(batch_data_samples[i].lidar_path)
        x = self.decoder(x, scene_names)

        sp_gt_instances = []
        for i in range(len(batch_data_samples)):
            voxel_superpoints = inverse_mapping[
                coordinates[:, 0][inverse_mapping] == i]
            voxel_superpoints = torch.unique(
                voxel_superpoints, return_inverse=True)[1]
            inst_mask = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            assert voxel_superpoints.shape == inst_mask.shape

            batch_data_samples[i].gt_instances_3d.sp_masks = \
                S3DISOneFormer3D.get_gt_inst_masks(inst_mask, voxel_superpoints)
            sp_gt_instances.append(batch_data_samples[i].gt_instances_3d)

        loss = self.criterion(x, sp_gt_instances)
        return loss
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instance_3d` and `gt_sem_seg_3d`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        
        coordinates, features, inverse_mapping, spatial_shape = self.collate(
            batch_inputs_dict['points'])
        x = spconv.SparseConvTensor(
            features, coordinates, spatial_shape, len(batch_data_samples))

        x = self.extract_feat(x)

        scene_names = []
        for i in range(len(batch_data_samples)):
            scene_names.append(batch_data_samples[i].lidar_path)
        x = self.decoder(x, scene_names)

        results_list = self.predict_by_feat(x, inverse_mapping, scene_names)

        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_pts_seg = results_list[i]
        return batch_data_samples

    def predict_by_feat(self, out, superpoints, scene_names):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            scene_names (List[string]): of len 1, which contain scene name.

        Returns:
            List[PointData]: of len 1 with `pts_instance_mask`, 
                `instance_labels`, `instance_scores`.
        """
        pred_labels = out['cls_preds']
        pred_masks = out['masks']
        pred_scores = out['scores']
        scene_name = scene_names[0]

        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        scores *= pred_scores[0]

        if self.prefix_1dataset in scene_name:
            labels = torch.arange(
                self.num_classes_1dataset,
                device=scores.device).unsqueeze(0).repeat(
                    self.decoder.num_queries_1dataset,  
                    1).flatten(0, 1)
        elif self.prefix_2dataset in scene_name:
            labels = torch.arange(
                self.num_classes_2dataset,
                device=scores.device).unsqueeze(0).repeat(
                    self.decoder.num_queries_2dataset,
                    1).flatten(0, 1)          
        else:
            raise RuntimeError(f'Invalid scene name "{scene_name}".')
        
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]

        if self.prefix_1dataset in scene_name:
            topk_idx = torch.div(topk_idx, self.num_classes_1dataset, 
                                 rounding_mode='floor')
        elif self.prefix_2dataset in scene_name:
            topk_idx = torch.div(topk_idx, self.num_classes_2dataset,
                                 rounding_mode='floor')        
        else:
            raise RuntimeError(f'Invalid scene name "{scene_name}".')
        
        mask_pred = pred_masks[0]
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        if self.test_cfg.get('obj_normalization', None):
            mask_pred_thr = mask_pred_sigmoid > \
                self.test_cfg.obj_normalization_thr
            mask_scores = (mask_pred_sigmoid * mask_pred_thr).sum(1) / \
                (mask_pred_thr.sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr
        mask_pred = mask_pred[:, superpoints]
        # score_thr
        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return [
            PointData(
                pts_instance_mask=mask_pred,
                instance_labels=labels,
                instance_scores=scores)
        ]



