import torch
import torch.nn.functional as F
import math
import torch.nn as nn

from lib.ada_net import mask_xattn_one_text


def is_sqr(n):
    a = int(math.sqrt(n))
    return a * a == n


class TokenSparse(nn.Module):
    def __init__(self, embed_dim=512, sparse_ratio=0.6):
        super().__init__()

        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio

    def forward(self, tokens, attention_x=None, attention_y=None, score=None):
        """
        两种用法：
        1. 原始 SEPS 风格：
           forward(tokens, attention_x=..., attention_y=...)
        2. 外部直接给最终分数：
           forward(tokens, score=...)
        """
        B_v, L_v, C = tokens.size()

        if score is None:
            if attention_x is None or attention_y is None:
                raise ValueError('Either score or (attention_x, attention_y) must be provided.')
            score = attention_x + attention_y

        num_keep_token = math.ceil(L_v * self.sparse_ratio)

        # select the top-k index, (B_v, L_v)
        score_sort, score_index = torch.sort(score, dim=1, descending=True)

        # (B_v, K)
        keep_policy = score_index[:, :num_keep_token]

        # (B_v, L_v)
        score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1)

        # (B_v, K, C)
        select_tokens = torch.gather(tokens, dim=1, index=keep_policy.unsqueeze(-1).expand(-1, -1, C))

        # fusion token
        non_keep_policy = score_index[:, num_keep_token:]
        non_tokens = torch.gather(tokens, dim=1, index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C))

        non_keep_score = score_sort[:, num_keep_token:]
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)

        # get fusion token (B_v, 1, C)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True)

        return select_tokens, extra_token, score_mask


# dim_ratio affect GPU memory
class TokenAggregation(nn.Module):
    def __init__(self, dim=512, keeped_patches=64, dim_ratio=0.2):
        super().__init__()

        hidden_dim = int(dim * dim_ratio)

        self.weight = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, keeped_patches)
        )

        self.scale = nn.Parameter(torch.ones(1, 1, 1))

    def forward(self, x, keep_policy=None):
        # (B, N, C) -> (B, N, N_s)
        weight = self.weight(x)

        # (B, N, N_s) -> (B, N_s, N)
        weight = weight.transpose(2, 1) * self.scale

        if keep_policy is not None:
            keep_policy = keep_policy.unsqueeze(1)
            weight = weight - (1 - keep_policy) * 1e10

        weight = F.softmax(weight, dim=2)
        x = torch.bmm(weight, x)
        return x


class DenseScoreDeltaPredictor(nn.Module):
    """
    v2.1 保守版：
    只为 dense branch 预测一个小的 delta，用于微调原始 heuristic score，
    不直接替代原始分数。
    """

    def __init__(self, embed_dim=512, hidden_dim=128):
        super().__init__()
        in_dim = embed_dim + 3
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, patch_tokens, img_salience, primary_rel, aux_rel):
        """
        patch_tokens: (B_v, L_v, C)
        img_salience / primary_rel / aux_rel: (B_v, L_v)
        return: delta_logit (B_v, L_v)
        """
        score_feat = torch.stack([img_salience, primary_rel, aux_rel], dim=-1)
        fused_feat = torch.cat([patch_tokens, score_feat], dim=-1)
        delta_logit = self.mlp(fused_feat).squeeze(-1)
        return delta_logit


## sparse + aggregation
class CrossSparseAggrNet_v2(nn.Module):
    def __init__(self, opt=None):
        super().__init__()

        self.opt = opt

        self.hidden_dim = opt.embed_size
        self.num_patches = opt.num_patches
        self.sparse_ratio = opt.sparse_ratio
        self.aggr_ratio = opt.aggr_ratio
        self.attention_weight = opt.attention_weight
        self.ratio_weight = opt.ratio_weight

        # v2.1 新增参数，使用 getattr 保证 arguments.py 暂未添加时也能直接运行
        self.score_pred_hidden = getattr(opt, 'score_pred_hidden', 128)
        self.score_delta_scale = getattr(opt, 'score_delta_scale', 0.1)

        # the number of aggregated patches
        self.keeped_patches = int(self.num_patches * self.aggr_ratio * self.sparse_ratio)

        # sparse network for cap and long_cap
        self.sparse_net_cap = TokenSparse(
            embed_dim=self.hidden_dim,
            sparse_ratio=self.sparse_ratio,
        )
        self.sparse_net_long = TokenSparse(
            embed_dim=self.hidden_dim,
            sparse_ratio=self.sparse_ratio,
        )

        # aggregation network
        self.aggr_net = TokenAggregation(
            dim=self.hidden_dim,
            keeped_patches=self.keeped_patches,
        )

        # 只对 dense branch 做一个小幅分数修正
        self.dense_delta_predictor = DenseScoreDeltaPredictor(
            embed_dim=self.hidden_dim,
            hidden_dim=self.score_pred_hidden,
        )

        # 为兼容你之前 train.py 中读取 latest_dense_gate 的逻辑，这里保留该属性
        self.latest_dense_gate = None

        # 便于后续加日志观察
        self.latest_dense_delta_mean = None
        self.latest_dense_score_mean = None
        self.latest_sparse_score_mean = None

    def reset_latest_stats(self):
        self.latest_dense_gate = None
        self.latest_dense_delta_mean = None
        self.latest_dense_score_mean = None
        self.latest_sparse_score_mean = None

    def get_text_global(self, text_embs, text_lens):
        global_embs = []
        for i in range(len(text_lens)):
            n_word = text_lens[i]
            emb = text_embs[i, :n_word, :].mean(dim=0)
            global_embs.append(emb)
        global_embs = torch.stack(global_embs, dim=0)
        global_embs = F.normalize(global_embs, dim=-1)
        return global_embs

    def build_dense_score_v21(self, patch_tokens, img_salience, dense_rel, sparse_rel):
        """
        v2.1 保守版核心：
        dense_score = heuristic_dense_score + small_delta
        其中 small_delta = scale * tanh(mlp(...))
        """
        heuristic_dense_score = img_salience + dense_rel

        delta_logit = self.dense_delta_predictor(
            patch_tokens=patch_tokens,
            img_salience=img_salience,
            primary_rel=dense_rel,
            aux_rel=sparse_rel,
        )
        small_delta = self.score_delta_scale * torch.tanh(delta_logit)
        final_dense_score = heuristic_dense_score + small_delta

        return final_dense_score, small_delta

    def forward(self, img_embs, cap_embs, cap_lens, long_cap_embs=None, long_cap_lens=None):
        B_v, L_v, C = img_embs.shape
        self.reset_latest_stats()

        # feature normalization
        img_embs_norm = F.normalize(img_embs, dim=-1)
        cap_embs_norm = F.normalize(cap_embs, dim=-1)
        long_cap_embs_norm = F.normalize(long_cap_embs, dim=-1)

        self.has_cls_token = False if is_sqr(img_embs.shape[1]) else True

        # whether it exists [cls] token
        if self.has_cls_token:
            img_cls_emb = img_embs[:, 0:1, :]
            img_spatial_embs = img_embs[:, 1:, :]
            img_spatial_embs_norm = img_embs_norm[:, 1:, :]
        else:
            img_cls_emb = None
            img_spatial_embs = img_embs
            img_spatial_embs_norm = img_embs_norm

        # compute self-attention
        with torch.no_grad():
            # (B_v, L_v, C) ->  (B_v, 1, C)
            img_spatial_glo_norm = F.normalize(img_spatial_embs.mean(dim=1, keepdim=True), dim=-1)
            # (B_v, L_v, C) -> (B_v, L_v)
            img_spatial_self_attention = (img_spatial_glo_norm * img_spatial_embs_norm).sum(dim=-1)

        improve_sims = []
        long_sims = []
        score_mask_all = []
        score_mask_long_all = []

        sparse_text_globals = self.get_text_global(cap_embs, cap_lens)
        dense_text_globals = self.get_text_global(long_cap_embs, long_cap_lens)

        dense_delta_means = []
        dense_score_means = []
        sparse_score_means = []

        for i in range(len(cap_lens)):
            n_word = cap_lens[i]
            cap_i = cap_embs[i, :n_word, :]
            cap_i_expand = cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)

            n_long_word = long_cap_lens[i]
            long_cap_i = long_cap_embs[i, :n_long_word, :]
            long_cap_i_expand = long_cap_embs_norm[i, :n_long_word, :].unsqueeze(0).repeat(B_v, 1, 1)

            # 相关性分数保持和原始风格接近，先用 no_grad 保守处理
            with torch.no_grad():
                cap_i_glo = F.normalize(cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)
                attn_cap = (cap_i_glo * img_spatial_embs_norm).sum(dim=-1)

                long_cap_i_glo = F.normalize(long_cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)
                long_attn_cap = (long_cap_i_glo * img_spatial_embs_norm).sum(dim=-1)

            # sparse branch：完全保持原始 heuristic score，不做 learnable 替换
            select_tokens_cap, extra_token_cap, score_mask_cap = self.sparse_net_cap(
                tokens=img_spatial_embs,
                attention_x=img_spatial_self_attention,
                attention_y=attn_cap,
            )

            aggr_tokens = self.aggr_net(select_tokens_cap)
            keep_spatial_tokens = torch.cat([aggr_tokens, extra_token_cap], dim=1)

            if self.has_cls_token:
                select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
            else:
                select_tokens = keep_spatial_tokens

            select_tokens = F.normalize(select_tokens, dim=-1)
            sim_one_text = mask_xattn_one_text(
                img_embs=select_tokens,
                cap_i_expand=cap_i_expand,
            )

            improve_sims.append(sim_one_text)
            score_mask_all.append(score_mask_cap)

            # dense branch：只做小幅修正，不推翻原始 heuristic score
            dense_score, small_delta = self.build_dense_score_v21(
                patch_tokens=img_spatial_embs,
                img_salience=img_spatial_self_attention,
                dense_rel=long_attn_cap,
                sparse_rel=attn_cap,
            )

            select_tokens_long, extra_token_long, score_mask_long = self.sparse_net_long(
                tokens=img_spatial_embs,
                score=dense_score,
            )

            aggr_tokens_long = self.aggr_net(select_tokens_long)
            keep_spatial_tokens = torch.cat([aggr_tokens_long, extra_token_long], dim=1)

            if self.has_cls_token:
                select_tokens_long = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
            else:
                select_tokens_long = keep_spatial_tokens

            select_tokens_long = F.normalize(select_tokens_long, dim=-1)
            sim_one_text = mask_xattn_one_text(
                img_embs=select_tokens_long,
                cap_i_expand=long_cap_i_expand,
            )

            long_sims.append(sim_one_text)
            score_mask_long_all.append(score_mask_long)

            dense_delta_means.append(small_delta.mean())
            dense_score_means.append(dense_score.mean())
            sparse_score_means.append((img_spatial_self_attention + attn_cap).mean())

        # (B_v, B_t)
        improve_sims = torch.cat(improve_sims, dim=1) + torch.cat(long_sims, dim=1)
        score_mask_all = torch.stack(score_mask_all, dim=0) + torch.stack(score_mask_long_all, dim=0)

        if len(dense_delta_means) > 0:
            self.latest_dense_delta_mean = torch.stack(dense_delta_means).mean().detach()
            self.latest_dense_score_mean = torch.stack(dense_score_means).mean().detach()
            self.latest_sparse_score_mean = torch.stack(sparse_score_means).mean().detach()

        if self.training:
            return improve_sims, score_mask_all
        else:
            return improve_sims


if __name__ == '__main__':
    pass
