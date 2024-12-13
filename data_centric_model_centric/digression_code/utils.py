from PIL import ImageDraw, ImageFont, Image

from .ft_model import FTModel

font_path = "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf"


def _paste_on_grid(grid, box, image, text, font_size,
                   font_color=(255, 255, 255)):
    grid.paste(image, box=box)
    draw = ImageDraw.Draw(grid)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(box, text, font_color, font=font)


def show_examples(ds, pre_sample: int = 200, seed: int = 1234,
                  image_key='image', label_key='labels',
                  examples_per_class: int = 3, size=(350, 350)):

    w, h = size
    labels = ds.features[label_key].names
    grid = Image.new('RGB', size=(examples_per_class * w, len(labels) * h))
    small_ds = (ds
                .shuffle(seed)
                .select(range(pre_sample)))

    for label_id, label in enumerate(labels):

        # Filter the dataset by a single label, shuffle it,
        # and grab a few samples
        ds_slice = (small_ds
                    .filter(lambda ex: ex[label_key] == label_id)
                    .shuffle(seed)
                    .select(range(examples_per_class)))

        # Plot this label's examples along a row
        for i, example in enumerate(ds_slice):
            idx = examples_per_class * label_id + i
            box = (idx % examples_per_class * w, idx // examples_per_class * h)
            _paste_on_grid(grid, box,
                           example[image_key].resize(size),
                           label, 20)

    return grid


def show_predictions(ds, model: FTModel, pre_sample: int = 100,
                     seed: int = 1234, examples_per_class: int = 3,
                     size=(350, 350), image_key='image', label_key='labels'):

    w, h = size
    labels = ds.features['labels']
    label_names = labels.names

    grid = Image.new('RGB', size=(examples_per_class * w,
                                  len(label_names) * h))
    small_ds = ds.shuffle(seed).select(range(pre_sample))
    for label_id, label in enumerate(label_names):
        # Filter the dataset by a single label, shuffle it,
        # and grab a few samples
        ds_slice = (small_ds
                    .filter(lambda ex: ex['labels'] == label_id)
                    .shuffle(seed).select(range(examples_per_class)))

        # Plot this label's examples along a row
        for i, example, ans in model.prediction_iterator(ds_slice):
            idx = examples_per_class * label_id + i
            box = (idx % examples_per_class * w, idx // examples_per_class * h)
            text = (f"L:{labels.int2str(ans.label)}\n"
                    f"P:{labels.int2str(ans.predicted_class)}\n"
                    f"{ans.sm_text}")
            _paste_on_grid(grid, box, example['image'].resize(size),
                           text, 10, font_color=(0, 0, 0))
    return grid
