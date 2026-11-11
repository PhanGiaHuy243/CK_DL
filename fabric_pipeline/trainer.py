"""Pinned Ultralytics adapter: deterministic preprocessing and measured update budgets."""
from __future__ import annotations

import math
import time

import torch
from ultralytics.data.augment import Format, LetterBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer

from .common import write_csv, write_json


class PreprocessingOnlyTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        self.actual_steps = 0
        self.image_presentations = 0
        self.train_batches = 0
        self.step_trace = []
        self.epoch_trace = []
        self.transform_audit = {}
        super().__init__(*args, **kwargs)
        self.add_callback("on_fit_epoch_end", self.record_epoch)

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(self.model.stride.max()) if self.model is not None else 32, 32)
        # augment=False bypasses v8_transforms entirely, including automatic Albumentations.
        # This also disables augmentation-related resizing/interpolation in BaseDataset.
        dataset = YOLODataset(img_path=img_path, imgsz=self.args.imgsz, batch_size=batch or self.args.batch,
                              augment=False, hyp=self.args, rect=mode == "val", cache=self.args.cache,
                              single_cls=False, stride=stride, pad=0.5 if mode == "val" else 0.0,
                              prefix=f"{mode}: ", task="detect", data=self.data, fraction=1.0)
        transforms = list(dataset.transforms.transforms)
        if len(transforms) != 2 or not isinstance(transforms[0], LetterBox) or not isinstance(transforms[1], Format):
            raise RuntimeError(f"Unexpected transforms: {transforms}")
        if dataset.augment or transforms[1].bgr != 0:
            raise RuntimeError("Augmentation was not fully disabled")
        self.transform_audit[mode] = {"augment": dataset.augment, "transforms": [type(t).__name__ for t in transforms],
                                      "albumentations_present": False, "bgr_probability": transforms[1].bgr,
                                      "images": len(dataset)}
        write_json(self.save_dir / "transform_audit.json", self.transform_audit)
        return dataset

    def _setup_scheduler(self):
        if self.args.warmup_epochs != 0 or self.args.nbs != self.args.batch:
            raise ValueError("Budget adapter requires warmup_epochs=0 and nbs=batch")
        self.planned_steps = len(self.train_loader)*self.epochs
        # BaseTrainer expects an epoch scheduler. LR is assigned below at every update.
        self.lf = lambda _: 1.0
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lf)
        self.optimizer.register_step_post_hook(self._count_step)

    def _count_step(self, optimizer, args, kwargs):
        self.actual_steps += 1
        self.step_trace.append({"step": self.actual_steps, "epoch": self.epoch+1,
                                "lr": float(optimizer.param_groups[0]["lr"]),
                                "image_presentations": self.image_presentations})

    def preprocess_batch(self, batch):
        self.image_presentations += len(batch["img"])
        self.train_batches += 1
        return super().preprocess_batch(batch)

    def optimizer_step(self):
        progress = min(self.actual_steps/max(self.planned_steps-1, 1), 1.0)
        factor = ((1+math.cos(math.pi*progress))/2 if self.args.cos_lr else 1-progress)
        factor = self.args.lrf+(1-self.args.lrf)*factor
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"]*factor
        return super().optimizer_step()

    def _handle_nan_recovery(self, epoch):
        # Recovery could replay updates and violate the declared budget; fail clearly instead.
        if not torch.isfinite(self.loss).all():
            raise RuntimeError("Non-finite loss: experiment stopped; budget must not silently replay updates")
        return False

    @staticmethod
    def record_epoch(trainer):
        trainer.epoch_trace.append({"epoch": trainer.epoch+1, "optimizer_steps": trainer.actual_steps,
                                    "image_presentations": trainer.image_presentations,
                                    "elapsed_seconds": time.time()-trainer.train_time_start})
        write_csv(trainer.save_dir / "optimizer_steps.csv", trainer.step_trace)
        write_csv(trainer.save_dir / "epoch_budget.csv", trainer.epoch_trace)
