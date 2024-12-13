
from dataclasses import dataclass, field
from datasets import ClassLabel


angle_to_row_labels = ClassLabel(names=['PARALLEL', 'PERPENDICULAR', 'ANGLED'])
cloud_cover_labels = ClassLabel(names=['OVERCAST', 'CLEAR'])
crop_height_labels = ClassLabel(names=['SMALL', 'LARGE'])
crop_name_labels = ClassLabel(names=['CORN', 'COTTON', 'RAPESEED', 'SOYBEANS',
                                     'SUGARBEETS', 'OTHER', 'WHEAT'])
crop_residue_labels = ClassLabel(names=['NONE', 'MEDIUM', 'HIGH'])
row_spacing_labels = ClassLabel(names=['SINGLE', 'TWIN', 'SKIP_ROW', 'OTHER'])
tillage_practice_labels = ClassLabel(names=['MINIMAL', 'CONVENTIONAL'])

_task_labels = {
    'angle_to_row': angle_to_row_labels,
    'cloud_cover': cloud_cover_labels,
    'crop_height': crop_height_labels,
    'crop_name': crop_name_labels,
    'crop_residue': crop_residue_labels,
    'row_spacing': row_spacing_labels,
    'tillage_practice': tillage_practice_labels
}


@dataclass
class Task:
    name: str
    labels: ClassLabel = field(init=False)

    def __post_init__(self):
        self.labels = _task_labels[self.name]
