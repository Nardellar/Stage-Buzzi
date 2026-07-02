from types import SimpleNamespace

# Re-export selected utilities from the project-level modules so that tests and
# example scripts can import them via `from Classificazione import dataset_organization`.
from common import dataset_organization as _dataset_org
from common.data_utils import balance_dataset, remap_labels

# Provide a namespace with the expected helpers
dataset_organization = SimpleNamespace(
    balance_dataset=balance_dataset,
    remap_labels=remap_labels,
    get_dataset=_dataset_org.get_dataset,
)