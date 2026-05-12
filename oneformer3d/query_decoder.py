import torch
import torch.nn as nn


import math

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
import sys
# sys.path.append("/root/wish/Point-NN-main")
# from models.transformer import *
import torch.nn.functional as F
import yaml
# with open("config.yaml","r") as file:
#     config = yaml.safe_load(file)
# KAN1_en = config['KAN1_en']
# KAN2_en = config['KAN2_en']
# Cross_en = config['Cross_en']
class CrossAttentionLayer(BaseModule):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # todo: why BaseModule doesn't call it without us?
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(sources)):
            k = v = sources[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(queries[i], k, v, attn_mask=attn_mask) #k, v均为自身
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z, _ = self.attn(y, y, y)
            z = self.dropout(z) + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN(BaseModule):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()

        # self.KAN_en = KAN1_en

        
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))   
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out

@MODELS.register_module()
class QueryDecoder(BaseModule):
    """Query decoder.

    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
                 num_classes, in_channels, d_model, num_heads, hidden_dim,
                 dropout, activation_fn, iter_pred, attn_mask, fix_attention,
                 objectness_flag, vit_embed_dim=768, vit_num_heads=12,
                 vit_mlp_ratio=4., vit_qkv_bias=True, vit_drop_rate=0.,
                 vit_attn_drop_rate=0., vit_norm_layer=nn.LayerNorm,
                 vit_act_layer=nn.GELU, vit_depth=12, vit_drop_path_rate=0., class_balanced_attention=True, 
                 class_weights=None, **kwargs):
        super().__init__()
        self.objectness_flag = objectness_flag
        self.input_proj = nn.Sequential(
             nn.Linear(in_channels, d_model // 2), nn.LayerNorm(d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.num_queries = num_instance_queries + num_semantic_queries
        if num_instance_queries + num_semantic_queries > 0:
            self.query = nn.Embedding(num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
             nn.Linear(in_channels, d_model // 2), nn.LayerNorm(d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        self.class_balanced_attention = class_balanced_attention
        if class_balanced_attention:
            # 初始化权重，如果没有提供则使用均匀权重
            if class_weights is None:
                self.register_buffer('class_weights', torch.ones(num_classes + 1))
            else:
                self.register_buffer('class_weights', torch.tensor(class_weights))
                
            # 类别感知注意力模块
            self.class_attention = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, num_classes + 1),
            )
            
            # 用于特征重调权重的模块
            self.feature_reweight = nn.Sequential(
                nn.Linear(d_model + num_classes + 1, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU()
            )
        # if self.Cross_en:
        #     self.Cross_layer = nn.ModuleList([])

        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)

        self.out_cls = nn.Sequential(        ######
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes + 1))            
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
             nn.Linear(in_channels, d_model // 2), nn.LayerNorm(d_model // 2), nn.ReLU(),
             nn.Linear(d_model // 2, d_model), nn.LayerNorm(d_model), nn.ReLU(),
             nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
    
    def _apply_class_balanced_attention(self, queries, cls_logits):
        """应用类别平衡注意力机制
        
        Args:
            queries (List[Tensor]): 查询特征，每个形状为 (n_queries_i, d_model)
            cls_logits (List[Tensor]): 分类logits，每个形状为 (n_queries_i, n_instance_classes + 1)
            
        Returns:
            List[Tensor]: 调整后的查询特征，每个形状为 (n_queries_i, d_model)
        """
        balanced_queries = []
        for i in range(len(queries)):
            # 计算每个类别的注意力权重
            class_attn = self.class_attention(queries[i])  # [n_queries, n_classes+1]
            
            # 将 logits 转换为概率
            cls_probs = F.softmax(cls_logits[i], dim=-1)  # [n_queries, n_classes+1]
            
            # 应用类别权重（反比于样本数量）
            weighted_probs = cls_probs * self.class_weights.unsqueeze(0)  # [n_queries, n_classes+1]
            
            # 计算归一化的注意力权重
            attn_weights = F.softmax(class_attn * weighted_probs, dim=-1)  # [n_queries, n_classes+1]
            
            # 将注意力权重与查询特征连接，并应用特征重调权重
            query_concat = torch.cat([queries[i], attn_weights], dim=1)  # [n_queries, d_model+n_classes+1]
            reweighted_query = self.feature_reweight(query_concat)  # [n_queries, d_model]
            
            # 残差连接
            balanced_query = queries[i] + reweighted_query  # [n_queries, d_model]
            balanced_queries.append(balanced_query)
            
        return balanced_queries

    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)
        
        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                result_query.append(self.query_proj(queries[i]))
            result_queries.append(torch.cat(result_query))
            # result_query.append(self.query_proj(queries[i]))
        return result_queries

    def _forward_head(self, queries, mask_feats, cnt=0, flag=False):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            if flag and cnt == 2:
                norm_query_3d = norm_query[12:, :]
                mask_feats_3d = mask_feats[i][12:, :]
            else:
                norm_query_3d = norm_query
                mask_feats_3d = mask_feats[i]
            cls_preds.append(self.out_cls(norm_query_3d))
            # if last_flag:
            #     sem_preds.append(self.out_sem(norm_query_3d))
            pred_score = self.out_score(norm_query_3d) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query_3d, mask_feats_3d)
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return cls_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        emo_losses = []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            vit_queries = []
            for x in queries:
                x = self.blocks[2 * i](x.unsequeen(0)).sequeen(0)
                x = self.blocks[2 * i + 1](x.unsequeen(0), 2 * i + 1).sequeen(0)
                vit_queries.append(self.blocks[i](x))
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            queries = [self.fuse_blocks[i](torch.cat([s1.unsequeen(0), s2.unsequeen(0)])).sequeen(0) for s1, s2 in zip(vit_queries, queries)]
            
        cls_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores), emo_losses

    def token_distill_loss(self, pc_feats, img_tokens):
        pc_feats = F.normalize(pc_feats, dim=-1)       # [N, D]
        img_tokens = F.normalize(img_tokens, dim=-1)   # [T, D]
        sim = pc_feats @ img_tokens.T  # cosine similarity
        nearest_idx = sim.argmax(dim=1)  # [N]
        matched_tokens = img_tokens[nearest_idx]  # [N, D]
        cos_sim = (pc_feats * matched_tokens).sum(dim=1)  # [N]
        loss = 1 - cos_sim.mean()
        return loss

    def forward_iter_pred(self, x, queries, img_feat, image_tokens, feat_tokens, pose_token):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        for i in range(len(img_feat)):
            img_feat[i] = torch.cat([pose_token[i], img_feat[i]], dim=0)
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        distill_loss = 0
        for i in range(len(queries)):
            distill_loss = self.token_distill_loss(queries[i], image_tokens[i]) + distill_loss
        del image_tokens
        del feat_tokens
        cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        queries = self._apply_class_balanced_attention(queries, cls_pred)
        cls_pred, pred_score, pred_mask, _ = self._forward_head(
                queries, mask_feats)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)

        for i in range(len(self.cross_attn_layers)):
            if (i == 2):
                for j in range(len(inst_feats)):
                    inst_feats[j] = torch.cat([img_feat[j], inst_feats[j]], dim=0)
                    queries[j] = torch.cat([img_feat[j], queries[j]], dim=0)
                    mask_feats[j] = torch.cat([img_feat[j], mask_feats[j]], dim=0)
                    attn_mask[j] = torch.einsum('nd,md->nm', queries[j], mask_feats[j])
                    attn_mask[j] = (attn_mask[j].sigmoid() < 0.5).bool()
                    attn_mask[j][torch.where(
                        attn_mask[j].sum(-1) == attn_mask[j].shape[-1])] = False
                    attn_mask[j] = attn_mask[j].detach()
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            queries = self._apply_class_balanced_attention(queries, cls_pred)
            cls_pred, pred_score, pred_mask, _ = self._forward_head(
                queries, mask_feats, cnt=i, flag=True)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)
        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            for cls_pred, scores, masks in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs), distill_loss

    def forward(self, x, queries=None, img_feat=None, image_tokens=None, feat_tokens=None, pose_tokens=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries, img_feat, image_tokens, feat_tokens, pose_tokens)
        else:
            return self.forward_simple(x, queries)


@MODELS.register_module()
class ScanNetQueryDecoder(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears,class_balanced_attention=True, class_weights=None,
                 **kwargs):
        super().__init__(
            num_classes=num_instance_classes, d_model=d_model, **kwargs)
        assert num_semantic_linears in [1, 2]
        if num_semantic_linears == 2:
            self.out_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_semantic_classes + 1))
        else:
            self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)
        self.class_balanced_attention = class_balanced_attention
        if class_balanced_attention:
            # 初始化权重，如果没有提供则使用均匀权重
            if class_weights is None:
                self.register_buffer('class_weights', torch.ones(num_instance_classes + 1))
            else:
                self.register_buffer('class_weights', torch.tensor(class_weights))
                
            # 类别感知注意力模块
            self.class_attention = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.LayerNorm(d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, num_instance_classes + 1),
            )
            
            # 用于特征重调权重的模块
            self.feature_reweight = nn.Sequential(
                nn.Linear(d_model + num_instance_classes + 1, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU()
            )

    def _apply_class_balanced_attention(self, queries, cls_logits):
        """应用类别平衡注意力机制
        
        Args:
            queries (List[Tensor]): 查询特征，每个形状为 (n_queries_i, d_model)
            cls_logits (List[Tensor]): 分类logits，每个形状为 (n_queries_i, n_instance_classes + 1)
            
        Returns:
            List[Tensor]: 调整后的查询特征，每个形状为 (n_queries_i, d_model)
        """
        balanced_queries = []
        for i in range(len(queries)):
            # 计算每个类别的注意力权重
            class_attn = self.class_attention(queries[i])  # [n_queries, n_classes+1]
            
            # 将 logits 转换为概率
            cls_probs = F.softmax(cls_logits[i], dim=-1)  # [n_queries, n_classes+1]
            
            # 应用类别权重（反比于样本数量）
            weighted_probs = cls_probs * self.class_weights.unsqueeze(0)  # [n_queries, n_classes+1]
            
            # 计算归一化的注意力权重
            attn_weights = F.softmax(class_attn * weighted_probs, dim=-1)  # [n_queries, n_classes+1]
            
            # 将注意力权重与查询特征连接，并应用特征重调权重
            query_concat = torch.cat([queries[i], attn_weights], dim=1)  # [n_queries, d_model+n_classes+1]
            reweighted_query = self.feature_reweight(query_concat)  # [n_queries, d_model]
            
            # 残差连接
            balanced_query = queries[i] + reweighted_query  # [n_queries, d_model]
            balanced_queries.append(balanced_query)
            
        return balanced_queries
    
    def _forward_head(self, queries, mask_feats, last_flag, cnt=0, flag=False):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks = \
            [], [], [], [], []
        
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            if flag and cnt == 5:
                norm_query_3d = norm_query[12:, :]
                mask_feats_3d = mask_feats[i][12:, :]
            else:
                norm_query_3d = norm_query
                mask_feats_3d = mask_feats[i]

            cls_preds.append(self.out_cls(norm_query_3d)) # 一个全连接层
            if last_flag:
                sem_preds.append(self.out_sem(norm_query_3d))
            pred_score = self.out_score(norm_query_3d) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query_3d, mask_feats_3d)
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            # print("tfq",queries.shape)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, sem_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats, last_flag=True)
        queries = self._apply_class_balanced_attention(queries, cls_preds)
        cls_preds, sem_preds, pred_scores, pred_masks, _ = self._forward_head(
                queries, mask_feats, last_flag=True)
        exit()
        return dict(
            cls_preds=cls_preds,
            sem_preds=sem_preds,
            masks=pred_masks,
            scores=pred_scores)
    def token_distill_loss(self, pc_feats, img_tokens):
        pc_feats = F.normalize(pc_feats, dim=-1)       # [N, D]
        img_tokens = F.normalize(img_tokens, dim=-1)   # [T, D]
        sim = pc_feats @ img_tokens.T  # cosine similarity
        nearest_idx = sim.argmax(dim=1)  # [N]
        matched_tokens = img_tokens[nearest_idx]  # [N, D]
        cos_sim = (pc_feats * matched_tokens).sum(dim=1)  # [N]
        loss = 1 - cos_sim.mean()
        return loss

    def forward_iter_pred(self, x, queries, img_feat=None, image_tokens=None, feat_tokens=None, pose_token=None):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,                     
                and aux_outputs.S
        """
        for i in range(len(img_feat)):
            img_feat[i] = torch.cat([pose_token[i], img_feat[i]], dim=0)
            # img_feat[i] = pose_token[i] + img_feat[i]
            # img_feat[i] = (img_feat[i] - img_feat[i].mean()) / (img_feat[i].std() + 1e-8)
        cls_preds, sem_preds, pred_scores, pred_masks = [], [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        distill_loss = 0
        for i in range(len(queries)):
            distill_loss = self.token_distill_loss(queries[i], image_tokens[i]) + distill_loss
        del image_tokens
        del feat_tokens
        cls_pred, sem_pred, pred_score, pred_mask, _ = \
                self._forward_head(queries, mask_feats, last_flag=False)     
        queries = self._apply_class_balanced_attention(queries, cls_pred)
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask = \
                self._forward_head(queries, mask_feats, last_flag=False)
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            if (i == 5):
                for j in range(len(inst_feats)):
                    inst_feats[j] = torch.cat([img_feat[j], inst_feats[j]], dim=0)
                    queries[j] = torch.cat([img_feat[j], queries[j]], dim=0)
                    mask_feats[j] = torch.cat([img_feat[j], mask_feats[j]], dim=0)
                    attn_mask[j] = torch.einsum('nd,md->nm', queries[j], mask_feats[j])
                    attn_mask[j] = (attn_mask[j].sigmoid() < 0.5).bool()
                    attn_mask[j][torch.where(
                        attn_mask[j].sum(-1) == attn_mask[j].shape[-1])] = False
                    attn_mask[j] = attn_mask[j].detach()
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)  #这里是自注意力：即k, q, v均为同一个值


            queries = self.ffn_layers[i](queries)  #一个全连接层
            last_flag = i == len(self.cross_attn_layers) - 1

            cls_pred, sem_pred, pred_score, pred_mask, _ = \
                self._forward_head(queries, mask_feats, last_flag)
            
            queries = self._apply_class_balanced_attention(queries, cls_pred)
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask = \
                self._forward_head(queries, mask_feats, last_flag, cnt=i, flag=True)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            dict(
                cls_preds=cls_pred,
                sem_preds=sem_pred,
                masks=masks,
                scores=scores)
            for cls_pred, sem_pred, scores, masks in zip(
                cls_preds[:-1], sem_preds[:-1],
                pred_scores[:-1], pred_masks[:-1])]

        return dict(
            cls_preds=cls_preds[-1],
            sem_preds=sem_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs), distill_loss


@MODELS.register_module()
class OneDataQueryDecoder(BaseModule):
    """Query decoder. The same as above, but for 2 datasets.

    Args:
        num_layers (int): Number of transformer layers.
        num_queries_1dataset (int): Number of queries for the first dataset.
        num_queries_2dataset (int): Number of queries for the second dataset.
        num_classes_1dataset (int): Number of classes in the first dataset.
        num_classes_2dataset (int): Number of classes in the second dataset.
        prefix_1dataset (string): Prefix for the first dataset.
        prefix_2dataset (string): Prefix for the second dataset.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, 
                 num_layers, 
                 num_queries_1dataset, 
                 num_queries_2dataset,
                 num_classes_1dataset, 
                 num_classes_2dataset,
                 prefix_1dataset,
                 prefix_2dataset,
                 in_channels, 
                 d_model, 
                 num_heads, 
                 hidden_dim,
                 dropout, 
                 activation_fn, 
                 iter_pred, 
                 attn_mask, 
                 fix_attention, 
                 **kwargs):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())

        self.num_queries_1dataset = num_queries_1dataset
        self.num_queries_2dataset = num_queries_2dataset

        self.queries_1dataset = nn.Embedding(num_queries_1dataset, d_model)
        self.queries_2dataset = nn.Embedding(num_queries_2dataset, d_model)
        
        self.prefix_1dataset = prefix_1dataset 
        self.prefix_2dataset = prefix_2dataset

        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls_1dataset = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes_1dataset + 1))
        self.out_cls_2dataset = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes_2dataset + 1))
        self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.num_classes_1dataset = num_classes_1dataset 
        self.num_classes_2dataset = num_classes_2dataset

    def _get_queries(self, batch_size, scene_names):
        """Get query tensor.

        Args:
            batch_size (int, optional): batch size.
            scene_names (List[string]): list of len batch size, which 
                contains scene names.
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        
        result_queries = []
        for i in range(batch_size):
            if self.prefix_1dataset in scene_names[i]:
                result_queries.append(self.queries_1dataset.weight)
            elif self.prefix_2dataset in scene_names[i]:
                result_queries.append(self.queries_2dataset.weight)
            else:
                raise RuntimeError(f'Invalid scene name "{scene_names[i]}".')

        return result_queries

    def _forward_head(self, queries, mask_feats, scene_names):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            scene_names (List[string]): list of len batch size, which 
                contains scene names.

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor]: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            
            if self.prefix_1dataset in scene_names[i]:
                cls_preds.append(self.out_cls_1dataset(norm_query))
            elif self.prefix_2dataset in scene_names[i]:
                cls_preds.append(self.out_cls_2dataset(norm_query))
            else:
                raise RuntimeError(f'Invalid scene name "{scene_names[i]}".')
            

            pred_scores.append(self.out_score(norm_query))
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        return  cls_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, scene_names):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            scene_names (List[string]): list of len batch size, which 
                contains scene names.
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(len(x), scene_names)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats, scene_names)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, scene_names):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            scene_names (List[string]): list of len batch size, which 
                contains scene names.
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(len(x), scene_names)
        cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats, scene_names)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            print(queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats, scene_names)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            for cls_pred, scores, masks in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, scene_names):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            scene_names (List[string]): list of len batch size, which 
                contains scene names.
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, scene_names)
        else:
            return self.forward_simple(x, scene_names)


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        x = x.contiguous()
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
