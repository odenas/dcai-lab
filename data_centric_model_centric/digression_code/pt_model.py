
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from transformers.models.vit.modeling_vit import ViTForImageClassification
from transformers import (TrainingArguments,
                          Trainer,
                          ViTImageProcessor)
import torch
from torchvision.transforms.v2 import (
                                    Compose,
                                    ColorJitter,
                                    ToImage,
                                    ToPILImage,
                                    RandomHorizontalFlip,
                                    RandomVerticalFlip)
import evaluate
from .tasks import Task


tillage_practice_transforms = Compose([
    ToImage(),
    RandomHorizontalFlip(0.5),
    RandomVerticalFlip(0.5),
    ToPILImage(),
])
tillage_practice_train_args = TrainingArguments(
    output_dir="./vit/tillage_practice",
    per_device_train_batch_size=128,
    eval_strategy="steps",
    num_train_epochs=10,
    fp16=True,
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)

cloud_cover_transforms = Compose([
    ToImage(),
    RandomHorizontalFlip(0.5),
    ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
    ToPILImage()
])
cloud_cover_train_args = TrainingArguments(
    output_dir="./vit/cloud_cover",
    per_device_train_batch_size=150,
    per_device_eval_batch_size=150,
    eval_strategy="steps",
    num_train_epochs=1,
    fp16=True,
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)

angle_to_row_transforms = Compose([
    ToImage(),
    RandomHorizontalFlip(0.5),
    ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
    ToPILImage()
])
angle_to_row_train_args = TrainingArguments(
    output_dir="./vit/angle_to_row",
    per_device_train_batch_size=150,
    per_device_eval_batch_size=150,
    eval_strategy="steps",
    num_train_epochs=2,
    fp16=True,
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)

crop_height_transforms = Compose([
    ToImage(),
    RandomHorizontalFlip(0.5),
    ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
    ToPILImage()
])
crop_height_train_args = TrainingArguments(
    output_dir="./vit/crop_height",
    per_device_train_batch_size=150,
    per_device_eval_batch_size=150,
    eval_strategy="steps",
    num_train_epochs=2,
    fp16=True,
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)

crop_name_transforms = Compose([
    ToImage(),
    RandomHorizontalFlip(0.5),
    ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
    ToPILImage()
])
crop_name_train_args = TrainingArguments(
    output_dir="./vit/crop_name",
    per_device_train_batch_size=150,
    per_device_eval_batch_size=150,
    eval_strategy="steps",
    num_train_epochs=2,
    fp16=True,
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)

crop_residue_transforms = Compose([
    ToImage(),
    RandomHorizontalFlip(0.5),
    ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
    ToPILImage()
])
crop_residue_train_args = TrainingArguments(
    output_dir="./vit/crop_residue",
    per_device_train_batch_size=150,
    per_device_eval_batch_size=150,
    eval_strategy="steps",
    num_train_epochs=2,
    fp16=True,
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)

_traning_args = {
    'angle_to_row': angle_to_row_train_args,
    'cloud_cover': cloud_cover_train_args,
    'crop_height': crop_height_train_args,
    'crop_name': crop_name_train_args,
    'tillage_practice': tillage_practice_train_args,
    'crop_residue': crop_residue_train_args
}
_transforms = {
    'angle_to_row': angle_to_row_transforms,
    'cloud_cover': cloud_cover_transforms,
    'crop_height': crop_height_transforms,
    'crop_name': crop_name_transforms,
    'tillage_practice': tillage_practice_transforms,
    'crop_residue': crop_residue_transforms
}


def _collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


@dataclass
class PTModel:
    model_name_or_path: Path
    task: Task
    processor: callable = field(init=False)
    pt_transform: callable = field(init=False)
    trainining_args: TrainingArguments = field(init=False)
    model: ViTForImageClassification = field(init=False)

    def __post_init__(self):
        self.processor = (ViTImageProcessor
                          .from_pretrained(self.model_name_or_path))
        self.pt_transform = _transforms[self.task.name]
        self.trainining_args = _traning_args[self.task.name]
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=len(self.task.labels.names),
            id2label={self.task.labels.str2int(c): c
                      for c in self.task.labels.names},
            label2id={c: self.task.labels.str2int(c)
                      for c in self.task.labels.names}
        )

    def transform(self, example_batch):
        # Take a list of PIL images and turn them to pixel values
        inputs = self.processor(
            [self.pt_transform(x) for x in example_batch['image']],
            return_tensors='pt'
        )

        # Don't forget to include the labels!
        inputs['labels'] = example_batch['labels']
        return inputs

    def compute_metrics(self, p, metric=evaluate.load("accuracy")):
        return metric.compute(
            predictions=np.argmax(p.predictions, axis=1),
            references=p.label_ids
        )

    def get_trainer(self, prepared_ds) -> Trainer:
        trainer = Trainer(
            model=self.model,
            args=self.trainining_args,
            data_collator=_collate_fn,
            compute_metrics=self.compute_metrics,
            train_dataset=prepared_ds["train"],
            eval_dataset=prepared_ds["test"],
            processing_class=self.processor,
        )
        return trainer
