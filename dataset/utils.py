"""Defines all the handles needed to build the dataset"""
import uuid
import numpy as np

from datasets.arrow_writer import ArrowWriter, Features, Repository
from datasets import Features, Sequence, Value
from huggingface_hub import Repository

import pickle
from preprocessing import SensorProcessing

def chunk_matrix(matrix: np.ndarray, K: int, pad_token=2048)->list[np.ndarray]:
    """Chunks a matrix column-wise, returning the chunks in a list."""
    H, W = matrix.shape
    num_chunks = -(-W // K)  # Ceiling division to ensure all parts are covered
    
    chunks = []
    for i in range(num_chunks):
        start = i * K
        end = min((i + 1) * K, W)
        chunk = matrix[:, start:end]
        
        # Pad if chunk width is less than K
        if chunk.shape[1] < K:
            pad_width = K - chunk.shape[1]
            pad_block = np.full((H, pad_width), pad_token)
            chunk = np.concatenate([chunk, pad_block], axis=1)
        
        chunks.append(chunk.astype(np.int8))
    
    return chunks

def prepare_data_dict(
        participant_id: str,
        fif_file: str,
        eeg_data:np.ndarray, 
        meg_data:np.ndarray,
        K: int=8192):
    keys = [
        "participant", 
        "fif_file",
        # data chunk specific from here onwards
        "id",
        "eeg_chunk",
        "meg_chunk",
        "chunk_index"
    ]

    data = {
        k: [] for k in keys
    }

    # split the data into chunks
    eeg_chunks, meg_chunks = chunk_matrix(eeg_data, K), chunk_matrix(meg_data, K)

    for i, (eeg_chunk, meg_chunk) in enumerate(zip(eeg_chunks, meg_chunks)):
        data["id"].append(str(uuid.uuid4()))
        data["chunk_index"].append(i)

        data["eeg_chunk"].append(eeg_chunk)
        data["meg_chunk"].append(meg_chunk)
        
        # broadcast the same participant and fif_file id across all chunks
        data["participant"].append(participant_id)
        data["fif_file"].append(fif_file)

    return data

get_path = lambda idx: ... # function returning a given instance of the dataset

def pickle_to_arrow(repo:Repository, idx:int=0, K:int=8192)->None:
    path = get_path(idx)
    with open(path, "rb") as data:
        X = pickle.load(data)

    processor = SensorProcessing()

    eeg = X["eeg_data"].reshape(X["eeg_shape"])
    eeg_discrete = processor.to_discrete(eeg)

    meg = X["meg_data"].reshape(X["meg_shape"])
    meg_discrete = processor.to_discrete(meg)

    data = prepare_data_dict(
        participant=X["participant"],
        fif_file=X["fif_file"],
        eeg_data=eeg_discrete,
        meg_data=meg_discrete,
        K=K
    )

    data_features = Features({
        "participant": Sequence(Value("string")),
        "fif_file": Sequence(Value("string")),
        "id": Sequence(Value("string")),
        "eeg_chunk": Sequence(Sequence(Sequence(Value("int8")))),  # a list of numpy arrays
        "meg_chunk": Sequence(Sequence(Sequence(Value("int8")))),
        "chunk_index": Sequence(Value("int64")),
    })


    del eeg, eeg_discrete, meg, meg_discrete
    # replace with split name for the data (typically, train, test or val)
    split_name = ...

    # where to find shard. Typically repo.local_dir/data/{split_name}-{idx:05d}.arrow
    shard_path = ...

    # 1) Create an ArrowWriter that saves to `shard_path`
    writer = ArrowWriter(features=data_features, path=shard_path)
    
    # 2) Write data to local file
    writer.write(data)

    del data

    # 3) Finalize the Arrow file
    writer.finalize()

    # 4) Add and commit, ready to push to the hub to sync data online
    repo.git_add(pattern="data/*.arrow")
    repo.git_commit(f"Add pickled-example no. {idx} for split '{split_name}'")

