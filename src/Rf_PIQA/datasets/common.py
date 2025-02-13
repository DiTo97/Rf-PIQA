from torch.utils.data import Subset, random_split

from Rf_PIQA.datasets.example import ExamplePIQADataset


def load_dataset(config: dict, split=True) -> tuple[Subset, Subset | None]:
    """
    Placeholder function to load a dataset from a given path.
    Extend this function to implement your own data loading logic.
    """
    if config["name_or_path"] == "example":
        dataset = ExamplePIQADataset(**config.get("args", {}))

        if not split:
            return dataset, None

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        raise ValueError("Unknown dataset: {}".format(config["name_or_path"]))
    
    return train_dataset, val_dataset
