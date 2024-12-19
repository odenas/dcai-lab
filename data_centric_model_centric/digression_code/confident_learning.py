
from dataclasses import dataclass, field
from cleanlab.datalab.datalab import Datalab
import pandas as pd
from datasets import Features, Image as DsImage, load_dataset, Value
from pathlib import Path
from .tasks import Task


@dataclass
class CLIssues:
    task: Task
    split: str
    lab: Datalab
    predictions_df: pd.DataFrame
    issues_df: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.issues_df = pd.merge(
            self.predictions_df, self.lab.issues,
            left_index=True, right_index=True
        )

    def get_issue_df(self, issue_name: str):
        cols = (["image_id", "y_hat", "y_tilde"] +
                self.task.labels.names +
                [f"is_{issue_name}_issue", f"{issue_name}_score", "abs_path"])
        df_slice = (self.issues_df[cols]
                    .loc[lambda x: x[f"is_{issue_name}_issue"]])
        return df_slice

    @classmethod
    def find_issues(cls, dataset_path: Path, preds_path: Path, split: str,
                    task: Task, lab_features=None):
        ds = load_dataset(
            "imagefolder", data_dir=dataset_path, split=split,
            features=Features({'image': DsImage(),
                               'labels': task.labels,
                               'id': Value(dtype='string')}))

        def _path_func(row,
                       base_path=f"{dataset_path}/{split}",
                       i2s=task.labels.int2str):
            return f"{base_path}/{i2s(row.y_tilde)}/{row.image_id}.png"

        predictions_df = (
            pd.read_csv(preds_path)
            .assign(abs_path=lambda x: [_path_func(row)
                                        for _, row in x.iterrows()])
        )
        lab = Datalab(data=ds, label_name='labels')
        lab.find_issues(pred_probs=predictions_df[task.labels.names].values,
                        features=lab_features)

        return cls(
            task=task,
            split="test",
            lab=lab,
            predictions_df=predictions_df
        )
