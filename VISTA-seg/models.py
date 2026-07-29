import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from layers import general_conv3d
from layers import prm_generator_laststage, prm_generator, region_aware_modal_fusion
from layers import *
from metaviewer_shared_adapter import MetaViewerSharedAdapter

basic_dims = 16
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.e1_c1 = general_conv3d(1, basic_dims, pad_type='reflect')
        self.e1_c2 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d(basic_dims, basic_dims, pad_type='reflect')

        self.e2_c1 = general_conv3d(basic_dims, basic_dims*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e2_c3 = general_conv3d(basic_dims*2, basic_dims*2, pad_type='reflect')

        self.e3_c1 = general_conv3d(basic_dims*2, basic_dims*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e3_c3 = general_conv3d(basic_dims*4, basic_dims*4, pad_type='reflect')

        self.e4_c1 = general_conv3d(basic_dims*4, basic_dims*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e4_c3 = general_conv3d(basic_dims*8, basic_dims*8, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        return x1, x2, x3, x4

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4, activation = 'softmax'):
        super(Decoder_sep, self).__init__()

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        if activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('activation function not supported')

    def forward(self, x1, x2, x3, x4):
        de_x4 = self.d3_c1(self.d3(x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        pred = self.activation(logits)

        return pred

class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_fuse, self).__init__()

        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)

        self.RFM4 = region_aware_modal_fusion(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM3 = region_aware_modal_fusion(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM2 = region_aware_modal_fusion(in_channel=basic_dims*2, num_cls=num_cls)
        self.RFM1 = region_aware_modal_fusion(in_channel=basic_dims*1, num_cls=num_cls)

        self.prm_generator4 = prm_generator_laststage(in_channel=basic_dims*8, num_cls=num_cls)
        self.prm_generator3 = prm_generator(in_channel=basic_dims*4, num_cls=num_cls)
        self.prm_generator2 = prm_generator(in_channel=basic_dims*2, num_cls=num_cls)
        self.prm_generator1 = prm_generator(in_channel=basic_dims*1, num_cls=num_cls)


    def forward(self, x1, x2, x3, x4, mask, bottleneck_override=None, bottleneck_mask=None):
        active_mask = mask if bottleneck_mask is None else bottleneck_mask
        prm_pred4 = self.prm_generator4(x4, mask)
        # x4 is [B, M, 128, H/8, W/8, D/8]; MetaViewer mode swaps only this fused shared tensor.
        de_x4 = self.RFM4(x4, prm_pred4.detach(), mask) if bottleneck_override is None else bottleneck_override
        fusion_x4 = de_x4
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_generator3(de_x4, x3, active_mask)
        de_x3 = self.RFM3(x3, prm_pred3.detach(), active_mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_generator2(de_x3, x2, active_mask)
        de_x2 = self.RFM2(x2, prm_pred2.detach(), active_mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_generator1(de_x2, x1, active_mask)
        de_x1 = self.RFM1(x1, prm_pred1.detach(), active_mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)

        return pred, (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), self.up8(prm_pred4)), fusion_x4

class MetaViewerSeg(nn.Module):
    """Full-meta MetaViewer BraTS segmentation model.

    During training, support samples adapt the MetaViewer adapter and the decoder
    receives the support-adapted bottleneck representation.
    """

    def __init__(self, num_cls=4, fusion_type='RFM', shared_rep_method='metaviewer',
                 meta_support_ratio=1.0, meta_inner_steps=1, meta_inner_lr=1.0e-3,
                 meta_first_order=True):
        super().__init__()
        if shared_rep_method != 'metaviewer':
            raise ValueError('Only shared_rep_method="metaviewer" is supported in this cleaned model.')
        if fusion_type != 'RFM':
            raise ValueError('Only fusion_type="RFM" is supported by MetaViewerSeg.')

        self.shared_rep_method = 'metaviewer'
        self.meta_support_ratio = meta_support_ratio
        self.meta_inner_steps = meta_inner_steps
        self.meta_inner_lr = meta_inner_lr
        self.meta_first_order = meta_first_order
        self.use_checkpoint = False

        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()
        self.metaviewer_adapter = MetaViewerSharedAdapter(channels=basic_dims*8, num_modalities=4)

        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)
        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def _run_with_checkpoint(self, module, *inputs):
        if self.use_checkpoint and self.is_training:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def _run_metaviewer_adapter(self, features, mask):
        if self.is_training:
            return self.metaviewer_adapter.meta_forward(
                features,
                mask.bool(),
                support_ratio=self.meta_support_ratio,
                inner_steps=self.meta_inner_steps,
                inner_lr=self.meta_inner_lr,
                first_order=self.meta_first_order,
            )
        z_support, weights, common_target = self.metaviewer_adapter.aggregate(features, mask.bool())
        return {
            "z_shared": z_support,
            "z_teacher": z_support.detach(),
            "support_mask": mask.bool(),
            "query_mask": torch.zeros_like(mask, dtype=torch.bool),
            "common_weights": weights,
            "common_target_prediction": common_target,
        }

    def forward(self, x, mask):
        flair_x1, flair_x2, flair_x3, flair_x4 = self._run_with_checkpoint(self.flair_encoder, x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4 = self._run_with_checkpoint(self.t1ce_encoder, x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4 = self._run_with_checkpoint(self.t1_encoder, x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4 = self._run_with_checkpoint(self.t2_encoder, x[:, 3:4, :, :, :])

        x1 = torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1)
        x2 = torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1)
        x3 = torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1)
        x4 = torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1)

        meta_outputs = self._run_metaviewer_adapter(x4, mask)
        fuse_pred, prm_preds, _ = self._run_with_checkpoint(
            self.decoder_fuse, x1, x2, x3, x4, mask, meta_outputs["z_shared"], mask.bool())

        if self.is_training:
            flair_pred = self._run_with_checkpoint(self.decoder_sep, flair_x1, flair_x2, flair_x3, flair_x4)
            t1ce_pred = self._run_with_checkpoint(self.decoder_sep, t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4)
            t1_pred = self._run_with_checkpoint(self.decoder_sep, t1_x1, t1_x2, t1_x3, t1_x4)
            t2_pred = self._run_with_checkpoint(self.decoder_sep, t2_x1, t2_x2, t2_x3, t2_x4)
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), prm_preds, meta_outputs
        return fuse_pred

class Decoder_fuse_wmh(nn.Module):
    def __init__(self, num_cls=4, num_modal=2, activation='softmax'):
        super(Decoder_fuse_wmh, self).__init__()

        self.d3_c1 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')

        self.d2_c1 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')

        self.d1_c1 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        if activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('activation function not supported')

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)

        self.RFM4 = region_aware_modal_fusion_wmh(in_channel=basic_dims*8, num_cls=num_cls, num_modal=num_modal)
        self.RFM3 = region_aware_modal_fusion_wmh(in_channel=basic_dims*4, num_cls=num_cls, num_modal=num_modal)
        self.RFM2 = region_aware_modal_fusion_wmh(in_channel=basic_dims*2, num_cls=num_cls, num_modal=num_modal)
        self.RFM1 = region_aware_modal_fusion_wmh(in_channel=basic_dims*1, num_cls=num_cls, num_modal=num_modal)

        self.prm_generator4 = prm_generator_laststage(in_channel=basic_dims*8, num_cls=num_cls, num_modal=num_modal)
        self.prm_generator3 = prm_generator(in_channel=basic_dims*4, num_cls=num_cls, num_modal=num_modal)
        self.prm_generator2 = prm_generator(in_channel=basic_dims*2, num_cls=num_cls, num_modal=num_modal)
        self.prm_generator1 = prm_generator(in_channel=basic_dims*1, num_cls=num_cls, num_modal=num_modal)


    def forward(self, x1, x2, x3, x4, mask, bottleneck_override=None, bottleneck_mask=None):
        active_mask = mask if bottleneck_mask is None else bottleneck_mask
        prm_pred4 = self.prm_generator4(x4, mask)
        # x4 is [B, M, 128, H/8, W/8, D/8]; MetaViewer mode swaps only this fused shared tensor.
        de_x4 = self.RFM4(x4, prm_pred4.detach(), mask) if bottleneck_override is None else bottleneck_override
        fusion_x4 = de_x4
        de_x4 = self.d3_c1(self.up2(de_x4))

        prm_pred3 = self.prm_generator3(de_x4, x3, active_mask)
        de_x3 = self.RFM3(x3, prm_pred3.detach(), active_mask)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))

        prm_pred2 = self.prm_generator2(de_x3, x2, active_mask)
        de_x2 = self.RFM2(x2, prm_pred2.detach(), active_mask)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))

        prm_pred1 = self.prm_generator1(de_x2, x1, active_mask)
        de_x1 = self.RFM1(x1, prm_pred1.detach(), active_mask)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        logits = self.seg_layer(de_x1)
        pred = self.activation(logits)

        return pred, (prm_pred1, self.up2(prm_pred2), self.up4(prm_pred3), self.up8(prm_pred4)), fusion_x4

class MetaViewerSegWMH(nn.Module):
    def __init__(self, num_cls=1, fusion_type='RFM', activation='sigmoid', shared_rep_method='metaviewer',
                 meta_support_ratio=1.0, meta_inner_steps=1, meta_inner_lr=1.0e-3,
                 meta_first_order=True):
        super().__init__()
        if shared_rep_method != 'metaviewer':
            raise ValueError('Only shared_rep_method="metaviewer" is supported in this cleaned model.')
        if fusion_type != 'RFM':
            raise ValueError('Only fusion_type="RFM" is supported by MetaViewerSegWMH.')

        self.shared_rep_method = 'metaviewer'
        self.meta_support_ratio = meta_support_ratio
        self.meta_inner_steps = meta_inner_steps
        self.meta_inner_lr = meta_inner_lr
        self.meta_first_order = meta_first_order
        self.use_checkpoint = False

        self.flair_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.metaviewer_adapter = MetaViewerSharedAdapter(channels=basic_dims*8, num_modalities=2)

        self.decoder_fuse = Decoder_fuse_wmh(num_cls=num_cls, activation=activation)
        self.decoder_sep = Decoder_sep(num_cls=num_cls, activation=activation)
        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def _run_with_checkpoint(self, module, *inputs):
        if self.use_checkpoint and self.is_training:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def _run_metaviewer_adapter(self, features, mask):
        if self.is_training:
            return self.metaviewer_adapter.meta_forward(
                features,
                mask.bool(),
                support_ratio=self.meta_support_ratio,
                inner_steps=self.meta_inner_steps,
                inner_lr=self.meta_inner_lr,
                first_order=self.meta_first_order,
            )
        z_support, weights, common_target = self.metaviewer_adapter.aggregate(features, mask.bool())
        return {
            "z_shared": z_support,
            "z_teacher": z_support.detach(),
            "support_mask": mask.bool(),
            "query_mask": torch.zeros_like(mask, dtype=torch.bool),
            "common_weights": weights,
            "common_target_prediction": common_target,
        }

    def forward(self, x, mask):
        flair_x1, flair_x2, flair_x3, flair_x4 = self._run_with_checkpoint(self.flair_encoder, x[:, 0:1, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4 = self._run_with_checkpoint(self.t1_encoder, x[:, 1:2, :, :, :])

        x1 = torch.stack((flair_x1, t1_x1), dim=1)
        x2 = torch.stack((flair_x2, t1_x2), dim=1)
        x3 = torch.stack((flair_x3, t1_x3), dim=1)
        x4 = torch.stack((flair_x4, t1_x4), dim=1)

        meta_outputs = self._run_metaviewer_adapter(x4, mask)
        fuse_pred, prm_preds, _ = self._run_with_checkpoint(
            self.decoder_fuse, x1, x2, x3, x4, mask, meta_outputs["z_shared"], mask.bool())

        if self.is_training:
            flair_pred = self._run_with_checkpoint(self.decoder_sep, flair_x1, flair_x2, flair_x3, flair_x4)
            t1_pred = self._run_with_checkpoint(self.decoder_sep, t1_x1, t1_x2, t1_x3, t1_x4)
            return fuse_pred, (flair_pred, t1_pred), prm_preds, meta_outputs
        return fuse_pred


if __name__ == '__main__':
    model = MetaViewerSeg()
    model.is_training = True
    x = torch.randn(3, 4, 112, 112, 112)
    mask = torch.ones(3, 4, dtype=torch.bool)
    fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), prm_preds, meta_outputs = model(x, mask)
    print(fuse_pred.shape, flair_pred.shape, t1ce_pred.shape, t1_pred.shape, t2_pred.shape)

    # model = MetaViewerSegWMH()
    # model.is_training = True
    # input = torch.randn(3, 2, 128, 128, 128)
    # mask = [[True, False], [True, True], [True, False]]
    # mask = torch.tensor(mask)
    # fuse_pred, (flair_pred, t1_pred), prm_preds, meta_outputs = model(input, mask)
    # print(fuse_pred.shape, flair_pred.shape, t1_pred.shape)
