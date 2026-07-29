import argparse

import torch
import torch.nn.functional as F

import models
from utils import criterions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_size", default=32, type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = models.MetaViewerSeg(
        num_cls=4,
        shared_rep_method="metaviewer",
        meta_support_ratio=0.5,
    ).to(device)
    model.is_training = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    masks = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 1, 0], [0, 0, 1, 0]],
        dtype=torch.bool,
        device=device,
    )
    x = torch.randn(3, 4, args.crop_size, args.crop_size, args.crop_size, device=device)
    y = torch.randint(0, 4, (3, args.crop_size, args.crop_size, args.crop_size), device=device)
    target = F.one_hot(y, num_classes=4).permute(0, 4, 1, 2, 3).float()

    fuse_pred, _, _, meta_outputs = model(x, masks)
    seg_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=4)
    seg_loss = seg_loss + criterions.dice_loss(fuse_pred, target, num_cls=4)
    z_shared = F.normalize(meta_outputs["z_shared"].flatten(1), dim=1, eps=1.0e-6)
    z_teacher = F.normalize(meta_outputs["z_teacher"].flatten(1), dim=1, eps=1.0e-6)
    align_loss = (1.0 - (z_shared * z_teacher).sum(dim=1)).mean()
    feature_loss = meta_outputs["meta_feature_loss"]
    assert "z_pre_adapt" in meta_outputs
    assert "support_sample_mask" in meta_outputs
    assert "query_sample_mask" in meta_outputs
    loss = seg_loss + 0.01 * align_loss + 0.2 * feature_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(
        "dummy ok",
        "fuse_pred", tuple(fuse_pred.shape),
        "support_samples", meta_outputs["support_sample_mask"].int().tolist(),
        "query_samples", meta_outputs["query_sample_mask"].int().tolist(),
        "loss", float(loss.detach()),
    )


if __name__ == "__main__":
    main()
