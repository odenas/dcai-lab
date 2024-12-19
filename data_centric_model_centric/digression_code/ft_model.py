
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
                          AutoModel, ViTImageProcessor)
from annoy import AnnoyIndex

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


@dataclass
class FTFExtractor:
    checkpoint_path: Path
    device_map: str = 'auto'
    model: AutoModel = field(init=False)
    processor: callable = field(init=False)

    def __post_init__(self):
        self.model = AutoModel.from_pretrained(
            self.checkpoint_path,
            output_hidden_states=True,
            device_map=self.device_map
        )
        self.processor = (ViTImageProcessor
                          .from_pretrained(self.checkpoint_path))

    def embeddings_iterator(self, ds):
        data_iter = tqdm(enumerate(ds), desc='Embeddings', total=len(ds))
        for i, example in data_iter:
            output = self.model(
                **self.processor(example['image'], return_tensors='pt')
            ).last_hidden_state[:, 0]
            yield i, example, output

    def embeddings_array(self, ds):
        embeddings = []
        for i, example, emb in self.embeddings_iterator(ds):
            embeddings.append(emb.detach().numpy())
        return np.vstack(embeddings)

    @staticmethod
    def annoy_tree(embeddings, n_trees=100, metric='angular'):
        tree = AnnoyIndex(embeddings.shape[1], metric=metric)
        data_iter = tqdm(enumerate(embeddings),
                         desc='KNN Tree',
                         total=embeddings.shape[0])
        for i, emb in data_iter:
            tree.add_item(i, emb)
        tree.build(n_trees)
        return tree

    @staticmethod
    def pairwise_distances(annoy_tree: AnnoyIndex):
        size = annoy_tree.get_n_items()
        from itertools import product
        distances = np.zeros((size, size))
        idx_iter = tqdm(product(range(size), range(size)),
                        desc='Pairwise Distances',
                        total=size**2)
        for i, j in idx_iter:
            distances[i, j] = annoy_tree.get_distance(i, j)
        return distances

    @staticmethod
    def umap_projection(distances: np.array, n_components=2) -> np.array:
        from umap import UMAP
        return UMAP(n_components=n_components).fit_transform(distances)

    @staticmethod
    def umap_projection_df(umap_proj: np.array,
                           labels: np.array, image_ids: np.array):
        return pd.DataFrame(umap_proj, columns=['x', 'y']).assign(
            label=labels, image_id=image_ids
        )
