import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


class RWConv1d(nn.Conv1d):
    """View-aware Conv1d used by the VISTA-Seg shared representation module."""

    def forward(self, input_x):
        x, views = input_x
        weights = self.weight[:, views, :]
        return F.conv1d(x, weights, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ConvBlock(nn.Module):
    """Conv1d + BN + ReLU block used by the VISTA-Seg meta network."""

    def __init__(self, in_channels, out_channels, use_maxpool=False, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2) if use_maxpool else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class RWConvBlock(nn.Module):
    """First MetaNet block whose input channels are selected by view ids."""

    def __init__(self, in_channels, out_channels, use_maxpool=False, kernel_size=3, padding=1):
        super(RWConvBlock, self).__init__()
        self.conv = RWConv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2) if use_maxpool else nn.Identity()

    def forward(self, x, views):
        return self.pool(self.relu(self.bn(self.conv([x, views]))))


class SpatialMetaNet(nn.Module):
    """VISTA-Seg meta network applied token-wise to 3D feature maps.

    The network receives normalized view embeddings shaped [B, V, D].
    Here each spatial location of the encoder bottleneck x4 is treated as one embedding sample,
    so [B, M, C, H, W, D] becomes [B*H*W*D, V, C] inside each sample.
    """

    def __init__(self, channels=128, num_modalities=4, meta_channels=None, kernel_size=3):
        super(SpatialMetaNet, self).__init__()
        if meta_channels is None:
            meta_channels = [num_modalities, 32]
        meta_channels = list(meta_channels)
        meta_channels[0] = num_modalities
        padding = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4}[kernel_size]
        self.conv_first = RWConvBlock(
            meta_channels[0], meta_channels[1], use_maxpool=False,
            kernel_size=kernel_size, padding=padding)
        hidden = []
        for i in range(1, len(meta_channels) - 1):
            hidden.append(ConvBlock(
                meta_channels[i], meta_channels[i + 1], use_maxpool=False,
                kernel_size=kernel_size, padding=padding))
        self.hidden = nn.Sequential(*hidden)
        self.conv_last = ConvBlock(
            meta_channels[-1], 1, use_maxpool=False,
            kernel_size=kernel_size, padding=padding)

    def forward(self, x, views):
        # x: [N_tokens, len(views), C]
        x = self.conv_first(x, views)
        x = self.hidden(x)
        return self.conv_last(x).squeeze(1)


class ViewFeatureReconstructor(nn.Module):
    """Small view-conditioned x4 feature reconstructor used by the meta inner-loop."""

    def __init__(self, channels=128, num_modalities=4, hidden_channels=None):
        super(ViewFeatureReconstructor, self).__init__()
        hidden_channels = channels if hidden_channels is None else hidden_channels
        self.view_embed = nn.Embedding(num_modalities, channels)
        self.conv1 = nn.Conv3d(channels * 2, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(hidden_channels, channels, kernel_size=1)

    @staticmethod
    def _param(params, name, default):
        return default if params is None else params[name]

    def forward(self, z_shared, view_ids, params=None):
        embed_weight = self._param(params, "view_embed.weight", self.view_embed.weight)
        view_context = F.embedding(view_ids, embed_weight)
        view_context = view_context[:, :, None, None, None].expand_as(z_shared)
        x = torch.cat((z_shared, view_context), dim=1)
        x = F.conv3d(
            x,
            self._param(params, "conv1.weight", self.conv1.weight),
            self._param(params, "conv1.bias", self.conv1.bias),
        )
        x = F.gelu(x)
        x = F.conv3d(
            x,
            self._param(params, "conv2.weight", self.conv2.weight),
            self._param(params, "conv2.bias", self.conv2.bias),
            padding=1,
        )
        x = F.gelu(x)
        return F.conv3d(
            x,
            self._param(params, "conv3.weight", self.conv3.weight),
            self._param(params, "conv3.bias", self.conv3.bias),
        )


class VISTASegSharedAdapter(nn.Module):
    def __init__(self, channels=128, num_modalities=4, meta_channels=None, meta_kernels=3):
        super(VISTASegSharedAdapter, self).__init__()
        self.num_modalities = num_modalities
        self.meta_net = SpatialMetaNet(
            channels=channels,
            num_modalities=num_modalities,
            meta_channels=meta_channels,
            kernel_size=meta_kernels,
        )
        self.output_proj = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
        )
        self.common_target_head = nn.Conv3d(channels, channels, kernel_size=1)
        self.feature_reconstructor = ViewFeatureReconstructor(
            channels=channels, num_modalities=num_modalities)

    @staticmethod
    def _submodule_state(module, params, prefix):
        state = OrderedDict((name, buf) for name, buf in module.named_buffers())
        if params is None:
            return state
        prefix = prefix + "."
        for name, value in params.items():
            if name.startswith(prefix):
                state[name[len(prefix):]] = value
        return state

    def _call_with_params(self, module, params, prefix, *args):
        if params is None:
            return module(*args)
        return functional_call(
            module,
            self._submodule_state(module, params, prefix),
            args,
        )

    @staticmethod
    def _submodule_params(params, prefix):
        if params is None:
            return None
        prefix = prefix + "."
        return OrderedDict(
            (name[len(prefix):], value)
            for name, value in params.items()
            if name.startswith(prefix)
        )

    def sample_sample_split(self, batch_size, support_ratio=0.5, device=None):
        support = torch.zeros(batch_size, dtype=torch.bool, device=device)
        query = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if batch_size <= 1:
            support[:] = True
            return support, query
        support_count = max(1, int(math.ceil(float(batch_size) * support_ratio)))
        support_count = min(support_count, batch_size - 1)
        perm = torch.randperm(batch_size, device=device)
        support[perm[:support_count]] = True
        query[perm[support_count:]] = True
        return support, query

    def sample_support_mask(self, mask, support_ratio=1.0):
        support = torch.zeros_like(mask, dtype=torch.bool)
        query = torch.zeros_like(mask, dtype=torch.bool)
        for b in range(mask.shape[0]):
            available = torch.nonzero(mask[b] > 0, as_tuple=False).flatten()
            if available.numel() == 0:
                continue
            if support_ratio >= 1.0 or available.numel() == 1:
                support[b, available] = True
                continue
            support_count = max(1, int(math.ceil(float(available.numel()) * support_ratio)))
            support_count = min(support_count, int(available.numel()) - 1)
            perm = available[torch.randperm(available.numel(), device=available.device)]
            support_idx = perm[:support_count]
            query_idx = perm[support_count:]
            support[b, support_idx] = True
            query[b, query_idx] = True
        return support, query

    def aggregate(self, features, mask, params=None):
        mask = mask.bool()
        bsz, _, channels, height, width, depth = features.shape
        outputs = []
        weights = features.new_zeros(bsz, self.num_modalities)
        for b in range(bsz):
            views = torch.nonzero(mask[b] > 0, as_tuple=False).flatten()
            if views.numel() == 0:
                outputs.append(features.new_zeros(channels, height, width, depth))
                continue
            tokens = features[b, views].permute(2, 3, 4, 0, 1).reshape(-1, views.numel(), channels)
            tokens = F.normalize(tokens, dim=2, eps=1.0e-6)
            meta_tokens = self._call_with_params(
                self.meta_net, params, "meta_net", tokens, views.tolist())
            meta_feature = meta_tokens.reshape(height, width, depth, channels).permute(3, 0, 1, 2)
            outputs.append(meta_feature)
            weights[b, views] = 1.0 / float(views.numel())
        common = torch.stack(outputs, dim=0)
        common = self._call_with_params(self.output_proj, params, "output_proj", common)
        common_target = self._call_with_params(
            self.common_target_head, params, "common_target_head", common)
        return common, weights, common_target

    def _feature_reconstruction_tensors(self, z_shared, features, mask):
        mask = mask.bool()
        z_items = []
        view_items = []
        target_items = []
        for b in range(mask.shape[0]):
            views = torch.nonzero(mask[b] > 0, as_tuple=False).flatten()
            for view in views:
                z_items.append(z_shared[b:b + 1])
                view_items.append(view)
                target_items.append(features[b:b + 1, view].detach())
        if len(z_items) == 0:
            return None, None, None
        z_batch = torch.cat(z_items, dim=0)
        view_ids = torch.stack(view_items, dim=0).long()
        targets = torch.cat(target_items, dim=0)
        return z_batch, view_ids, targets

    def feature_reconstruction_loss(self, z_shared, features, mask, params=None):
        z_batch, view_ids, targets = self._feature_reconstruction_tensors(
            z_shared, features, mask)
        if z_batch is None:
            return z_shared.new_tensor(0.0)
        recon_params = params
        if params is not None and any(
                name.startswith("feature_reconstructor.") for name in params):
            recon_params = self._submodule_params(params, "feature_reconstructor")
        recon = self.feature_reconstructor(
            z_batch,
            view_ids,
            params=recon_params,
        )
        return F.mse_loss(recon, targets)

    def shared_alignment_loss(self, z_student, z_teacher):
        z_student = F.normalize(z_student.flatten(1), dim=1, eps=1.0e-6)
        z_teacher = F.normalize(z_teacher.flatten(1), dim=1, eps=1.0e-6)
        return (1.0 - (z_student * z_teacher).sum(dim=1)).mean()

    def initial_reconstructor_params(self):
        return OrderedDict(
            (name, parameter) for name, parameter in self.feature_reconstructor.named_parameters())

    def initial_vista_seg_params(self):
        return OrderedDict((name, parameter) for name, parameter in self.named_parameters())

    def adapt_reconstructor(self, z_support, support_features, support_mask,
                            inner_steps=1, inner_lr=1.0e-3, first_order=True):
        params = self.initial_reconstructor_params()
        inner_loss = z_support.new_tensor(0.0)
        z_adapt = z_support.detach() if first_order else z_support
        for _ in range(max(0, inner_steps)):
            inner_loss = self.feature_reconstruction_loss(
                z_adapt, support_features, support_mask, params=params)
            grads = torch.autograd.grad(
                inner_loss,
                tuple(params.values()),
                create_graph=not first_order,
                retain_graph=not first_order,
                allow_unused=True,
            )
            updated = OrderedDict()
            for (name, value), grad in zip(params.items(), grads):
                updated[name] = value if grad is None else value - inner_lr * grad
            params = updated
        if inner_steps <= 0:
            inner_loss = self.feature_reconstruction_loss(
                z_adapt, support_features, support_mask, params=params)
        return params, inner_loss

    def adapt_vista_seg(self, support_features, support_mask,
                         inner_steps=1, inner_lr=1.0e-3, first_order=True):
        """Inner loop that adapts the VISTA-Seg adapter on support samples.

        The original cleaned model only adapted the view-conditioned reconstructor.
        This full-meta variant updates fast parameters for the adapter itself, so
        query samples and the segmentation decoder use support-adapted z_shared.
        """
        params = self.initial_vista_seg_params()
        inner_loss = support_features.new_tensor(0.0)
        for _ in range(max(0, inner_steps)):
            z_support, _, _ = self.aggregate(
                support_features, support_mask, params=params)
            inner_loss = self.feature_reconstruction_loss(
                z_support, support_features, support_mask, params=params)
            grads = torch.autograd.grad(
                inner_loss,
                tuple(params.values()),
                create_graph=not first_order,
                retain_graph=not first_order,
                allow_unused=True,
            )
            updated = OrderedDict()
            for (name, value), grad in zip(params.items(), grads):
                updated[name] = value if grad is None else value - inner_lr * grad
            params = updated
        if inner_steps <= 0:
            z_support, _, _ = self.aggregate(
                support_features, support_mask, params=params)
            inner_loss = self.feature_reconstruction_loss(
                z_support, support_features, support_mask, params=params)
        return params, inner_loss

    def meta_forward(self, features, mask, support_ratio=0.5, inner_steps=1,
                     inner_lr=1.0e-3, first_order=True):
        """Full support-query VISTA-Seg path.

        Support samples first adapt the VISTA-Seg adapter. The adapted adapter
        then produces z_shared for query samples and for the decoder path.
        """
        mask = mask.bool()
        support_samples, query_samples = self.sample_sample_split(
            features.shape[0], support_ratio=support_ratio, device=features.device)

        support_features = features[support_samples]
        support_mask = mask[support_samples]
        fast_params, support_inner_loss = self.adapt_vista_seg(
            support_features,
            support_mask,
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            first_order=first_order,
        )

        z_initial, _, _ = self.aggregate(features, mask)
        z_initial = z_initial.detach()
        z_shared, weights, common_target = self.aggregate(
            features, mask, params=fast_params)
        full_mask = torch.ones_like(mask, dtype=torch.bool)
        z_teacher, _, _ = self.aggregate(features, full_mask)
        z_teacher = z_teacher.detach()
        meta_align_loss = self.shared_alignment_loss(z_shared, z_teacher)

        query_feature_loss = z_shared.new_tensor(0.0)
        if query_samples.any():
            query_feature_loss = self.feature_reconstruction_loss(
                z_shared[query_samples],
                features[query_samples],
                mask[query_samples],
                params=fast_params,
            )
            meta_feature_loss = query_feature_loss
        else:
            meta_feature_loss = self.feature_reconstruction_loss(
                z_shared[support_samples],
                support_features,
                support_mask,
                params=fast_params,
            )

        return {
            "z_shared": z_shared,
            "z_pre_adapt": z_initial,
            "z_teacher": z_teacher,
            "support_mask": mask,
            "query_mask": mask.new_zeros(mask.shape),
            "common_weights": weights,
            "common_target_prediction": common_target,
            "support_sample_mask": support_samples,
            "query_sample_mask": query_samples,
            "support_feature_loss": support_inner_loss.detach(),
            "query_feature_loss": query_feature_loss.detach(),
            "meta_feature_loss": meta_feature_loss,
            "meta_align_loss": meta_align_loss,
        }

    def forward(self, features, mask, support_ratio=1.0):
        """Return z_S, z_A teacher, support/query masks, and slot weights."""
        support_mask, query_mask = self.sample_support_mask(mask.bool(), support_ratio=support_ratio)
        z_support, weights, common_target = self.aggregate(features, support_mask)
        with torch.no_grad():
            z_all, _, _ = self.aggregate(features, mask.bool())
        return {
            "z_shared": z_support,
            "z_teacher": z_all.detach(),
            "support_mask": support_mask,
            "query_mask": query_mask,
            "common_weights": weights,
            "common_target_prediction": common_target,
        }
