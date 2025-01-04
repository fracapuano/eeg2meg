"""Creates a dataset on hugging face contained the pre-processed and chunked data."""
from tqdm import tqdm
from utils import pickle_to_arrow

from huggingface_hub import Repository


repo_id = "fracapuano/eeg2meg-medium"
local_dir = "eeg2meg-medium"

repo = Repository(
    local_dir=local_dir,
    clone_from=repo_id,
    repo_type="dataset"
)
# sync with hub before modifying
repo.git_pull()

indices = ...
K = 8192

for i in tqdm(indices):
    pickle_to_arrow(
        idx=i,
        K=K
    )
    
    repo.git_push()

