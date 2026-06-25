#!/usr/bin/env python3
"""V3.1 joint training: BiomedCLIP prompts + contrastive loss + channel fusion."""

from __future__ import annotations
import argparse, sys, gc, itertools, os
import numpy as np, pandas as pd, psutil, torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from cdf_vlm.biomedclip import load_local_biomedclip
from cdf_vlm.config import save_json
from cdf_vlm.ddsm import DDSMPairedViewDataset, DDSMSingleViewDataset
from cdf_vlm.io import ensure_dir, seed_everything
from cdf_vlm.lora import inject_decoupled_lora
from cdf_vlm.metrics import binary_metrics
from cdf_vlm.multiview_v31 import (
    DDSMGeometryPromptModelV31, DDSMSingleViewPromptModelV31,
)


def _single_forward(model, batch, device):
    return model(image=batch["image"].to(device), mask=batch["mask"].to(device),
                 description=list(batch["description"]))


def _pair_forward(model, batch, device):
    return model(
        cc_image=batch["cc_image"].to(device), cc_mask=batch["cc_mask"].to(device),
        cc_description=list(batch["cc_description"]),
        mlo_image=batch["mlo_image"].to(device), mlo_mask=batch["mlo_mask"].to(device),
        mlo_description=list(batch["mlo_description"]),
        side_ids=list(batch["side_id"]),
    )


def _label_smoothing(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def main():
    p = argparse.ArgumentParser(description="V3.1 joint training")
    p.add_argument("--single-csv", required=True)
    p.add_argument("--pair-csv", required=True)
    p.add_argument("--val-pair-csv"); p.add_argument("--test-pair-csv")
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--biomedclip-dir", default="external/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-single", type=float, default=1.0)
    p.add_argument("--lambda-pair", type=float, default=1.0)
    p.add_argument("--lambda-proj", type=float, default=0.1)
    p.add_argument("--lambda-consistency", type=float, default=0.05)
    p.add_argument("--lambda-contrastive", type=float, default=0.1)
    p.add_argument("--aligner-warmup-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--vision-rank", type=int, default=32)
    p.add_argument("--text-rank", type=int, default=8)
    p.add_argument("--use-aligner", action="store_true")
    p.add_argument("--disable-text-prompts", action="store_true")
    p.add_argument("--disable-consistency", action="store_true")
    p.add_argument("--disable-contrastive", action="store_true")
    p.add_argument("--disable-channel-fusion", action="store_true")
    p.add_argument("--use-attn-reg", action="store_true")
    p.add_argument("--attn-reg-mode", default="binary", choices=["binary","gaussian","dilated"])
    p.add_argument("--attn-reg-sigma", type=float, default=3.0)
    p.add_argument("--attn-reg-radius", type=int, default=3)
    p.add_argument("--lambda-attn-reg", type=float, default=0.1)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--pos-weight", type=float, default=None)
    p.add_argument("--lora-dropout", type=float, default=0.25)
    p.add_argument("--device")
    args = p.parse_args()

    seed_everything(42)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    out_dir = ensure_dir(args.output_dir)

    print(f"Loading BiomedCLIP from {args.biomedclip_dir}..."); sys.stdout.flush()
    clip_model, preprocess, tokenizer = load_local_biomedclip(args.biomedclip_dir, device=device)
    if args.use_lora:
        report = inject_decoupled_lora(clip_model, vision_rank=args.vision_rank,
                                       text_rank=args.text_rank, dropout=args.lora_dropout)
        print(f"LoRA: vision={report.vision_modules}, text={report.text_modules}")

    single_ds = DDSMSingleViewDataset(args.single_csv, data_root=args.data_root, image_transform=preprocess)
    pair_ds = DDSMPairedViewDataset(args.pair_csv, data_root=args.data_root, image_transform=preprocess)
    print(f"Single: {len(single_ds)}, Paired: {len(pair_ds)}")
    single_loader = DataLoader(single_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    pair_loader = DataLoader(pair_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    val_loader = None
    if args.val_pair_csv:
        val_ds = DDSMPairedViewDataset(args.val_pair_csv, data_root=args.data_root, image_transform=preprocess)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = None
    if args.test_pair_csv:
        test_ds = DDSMPairedViewDataset(args.test_pair_csv, data_root=args.data_root, image_transform=preprocess)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    single_model = DDSMSingleViewPromptModelV31(
        clip_model=clip_model, tokenizer=tokenizer,
        freeze_backbone=not args.use_lora,
        use_text_prompts=not args.disable_text_prompts,
    ).to(device)
    pair_model = DDSMGeometryPromptModelV31(
        clip_model=clip_model, tokenizer=tokenizer,
        freeze_backbone=not args.use_lora,
        use_text_prompts=not args.disable_text_prompts,
        use_aligner=args.use_aligner,
        use_consistency=not args.disable_consistency,
        use_contrastive=not args.disable_contrastive,
        use_channel_fusion=not args.disable_channel_fusion,
        use_attn_reg=args.use_attn_reg,
        attn_reg_mode=args.attn_reg_mode,
        attn_reg_sigma=args.attn_reg_sigma,
        attn_reg_radius=args.attn_reg_radius,
    ).to(device)

    pos_weight = torch.tensor([args.pos_weight], device=device) if args.pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    params = [p for p in itertools.chain(single_model.parameters(), pair_model.parameters())
              if p.requires_grad and id(p) not in (seen := set()) and not seen.add(id(p))]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    print(f"Trainable params: {sum(p.numel() for p in params):,}")

    @torch.no_grad()
    def evaluate(loader, split_name):
        pair_model.eval()
        labels_all, probs_all, losses, rows = [], [], [], []
        for batch in tqdm(loader, desc=f"Eval {split_name}", leave=False):
            labels = batch["label"].to(device)
            out = _pair_forward(pair_model, batch, device)
            cls_loss = criterion(out["logits"], labels).mean()
            loss = (cls_loss + args.lambda_proj * out["projection_loss"]
                    + args.lambda_consistency * out.get("consistency_loss", 0.0)
                    + args.lambda_contrastive * out.get("contrastive_loss", 0.0)
                    + args.lambda_attn_reg * out.get("attn_reg_loss", 0.0))
            losses.append(float(loss.detach().cpu()))
            probs = torch.sigmoid(out["logits"])
            labels_np = labels.detach().cpu().numpy().tolist()
            probs_np = probs.detach().cpu().numpy().tolist()
            labels_all.extend(labels_np); probs_all.extend(probs_np)
            for ii, sid in enumerate(list(batch["side_id"])):
                rows.append({"split": split_name, "side_id": sid,
                             "label": labels_np[ii], "prob": probs_np[ii]})
        metrics = binary_metrics(labels_all, probs_all)
        metrics["loss"] = sum(losses) / max(len(losses), 1)
        return metrics, rows

    def _mem():
        p = psutil.Process(os.getpid())
        return f"RAM={p.memory_info().rss/(1024**3):.1f}G GPU={torch.cuda.memory_allocated()/(1024**3):.1f}G"

    history = []; best_metric = -1.0; best_state = None; best_epoch = 0; pc = 0
    for epoch in range(1, args.epochs + 1):
        wf = min(1.0, epoch / max(args.aligner_warmup_epochs, 1))
        eff_proj = args.lambda_proj * wf
        single_model.train(); pair_model.train()
        losses = []; s_iter = iter(single_loader); p_iter = iter(pair_loader)
        steps = max(len(pair_loader), len(single_loader))
        for step_idx in tqdm(range(steps), desc=f"Epoch {epoch:2d}/{args.epochs}", unit="step"):
            try: sb = next(s_iter)
            except StopIteration: s_iter = iter(single_loader); sb = next(s_iter)
            try: pb = next(p_iter)
            except StopIteration: p_iter = iter(pair_loader); pb = next(p_iter)
            pl = pb["label"].to(device)
            if args.mixup_alpha > 0:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                idx = torch.randperm(pl.size(0))
                for k in ("cc_image","cc_mask","mlo_image","mlo_mask"):
                    pb[k] = lam * pb[k] + (1-lam) * pb[k][idx]
                pl = lam * pl + (1-lam) * pl[idx]
            sl = sb["label"].to(device)
            so = _single_forward(single_model, sb, device)
            po = _pair_forward(pair_model, pb, device)
            st = _label_smoothing(sl, args.label_smoothing)
            pt = _label_smoothing(pl, args.label_smoothing)
            s_loss = criterion(so["logits"], st).mean()
            p_loss = criterion(po["logits"], pt).mean()
            loss = (args.lambda_single * s_loss + args.lambda_pair * p_loss
                    + eff_proj * po["projection_loss"]
                    + args.lambda_consistency * po.get("consistency_loss", 0.0)
                    + args.lambda_contrastive * po.get("contrastive_loss", 0.0)
                    + args.lambda_attn_reg * po.get("attn_reg_loss", 0.0))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if args.max_grad_norm > 0: nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            del so, po, s_loss, p_loss, loss, sb, pb
            if step_idx % 20 == 0: gc.collect(); torch.cuda.empty_cache()

        row = {"epoch": epoch, "loss": sum(losses)/max(len(losses),1), "lambda_proj": eff_proj}
        if val_loader:
            vm, _ = evaluate(val_loader, "val")
            row.update({f"val_{k}": v for k, v in vm.items()})
            score = vm.get("auroc", -row["loss"])
        else:
            score = -row["loss"]
        if score > best_metric:
            best_metric, best_epoch, pc = score, epoch, 0
            best_state = {
                "single": {k: v.detach().cpu().clone() for k, v in single_model.state_dict().items()},
                "pair": {k: v.detach().cpu().clone() for k, v in pair_model.state_dict().items()},
            }
        else:
            pc += 1
        history.append(row); print(row); sys.stdout.flush()
        if pc >= args.patience and epoch > args.aligner_warmup_epochs:
            print(f"Early stop epoch {epoch}"); break

    if best_state:
        single_model.load_state_dict(best_state["single"])
        pair_model.load_state_dict(best_state["pair"])

    preds = []; metrics = {"best_epoch": best_epoch, "best_metric": best_metric}
    tl = DataLoader(pair_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    tm, tr = evaluate(tl, "train"); metrics["train"] = tm; preds.extend(tr)
    if val_loader: vm, vr = evaluate(val_loader, "val"); metrics["val"] = vm; preds.extend(vr)
    if test_loader: tem, ter = evaluate(test_loader, "test"); metrics["test"] = tem; preds.extend(ter)
    pd.DataFrame(history).to_csv(out_dir/"history.csv", index=False)
    pd.DataFrame(preds).to_csv(out_dir/"predictions.csv", index=False)
    save_json(metrics, out_dir/"metrics.json")
    torch.save({"single": single_model.state_dict(), "pair": pair_model.state_dict(), "args": vars(args)}, out_dir/"model.pt")
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
