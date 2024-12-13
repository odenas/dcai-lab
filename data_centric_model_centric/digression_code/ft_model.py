
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Iterable, Dict
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers.models.vit.modeling_vit import ViTForImageClassification
from transformers import (AutoModelForImageClassification,
                          ViTImageProcessor)
from .tasks import Task


@dataclass
class Answer:
    logits: np.array
    softmax: np.array
    predicted_class: int
    label: int

    @property
    def sm_text(self):
        return f"{["%.2f" % _ for _ in self.softmax]}"


def inference(model, processor, example) -> Answer:
    image = example['image']
    x = processor(image, return_tensors='pt')['pixel_values']
    with torch.no_grad():
        logits = model(x).logits[0]
        sm = softmax(logits, dim=0)
        ans = Answer(
            logits=logits.numpy(),
            softmax=sm.numpy(),
            predicted_class=int(logits.argmax().numpy()),
            label=example['labels']
        )
        return ans


@dataclass
class FTModel:
    checkpoint_path: Path
    model: ViTForImageClassification = field(init=False)
    processor: callable = field(init=False)

    def __post_init__(self):
        self.model = (AutoModelForImageClassification
                      .from_pretrained(self.checkpoint_path,
                                       device_map='auto'))
        self.processor = (ViTImageProcessor
                          .from_pretrained(self.checkpoint_path))

    def prediction_iterator(self, ds) -> Iterable[Tuple[int, Dict, Answer]]:
        processor = ViTImageProcessor.from_pretrained(self.checkpoint_path)

        for i, example in tqdm(enumerate(ds)):
            ans = inference(self.model, processor, example)
            yield i, example, ans

    def predictions_df(self, ds, task: Task,
                       sample: int = 0, seed: int = 1234):
        predictions = []

        if sample <= 0:
            ds_slice = ds
        else:
            ds_slice = ds.shuffle(seed).select(range(sample))

        for i, example, ans in self.prediction_iterator(ds_slice):
            predictions.append(
                (example['id'], ans.predicted_class, example['labels']) +
                tuple(ans.softmax)
            )
        return pd.DataFrame(
            predictions,
            columns=['image_id', "y_hat", "y_tilde"] + task.labels.names
        )
