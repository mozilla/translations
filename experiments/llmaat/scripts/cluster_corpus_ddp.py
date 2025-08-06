import os

import numpy as np
import toolz
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# run with torchrun --nproc_per_node=8


def main():
    # 1. Initialize process group
    # dist.init_process_group("nccl", timeout=timedelta(hours=3))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"processing local rank {local_rank}")

    # 2. Create model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-small").cuda(local_rank)

    # Wrap the model in DistributedDataParallel
    # model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    with open("distill_corpus/centroids/centroids_5k.npy", "rb") as f:
        raw_centroids = np.load(f)
        # normalize raw k-means centrods to calculate cosine similarity (k-mean was trained on normalized embeddings)
        centroids_norm = raw_centroids / np.linalg.norm(raw_centroids, axis=1, keepdims=True)
        centroids = torch.from_numpy(centroids_norm).cuda(local_rank)

    def process_emb(batch):
        try:
            batch_dict = tokenizer(
                batch, max_length=512, padding=True, truncation=True, return_tensors="pt"
            )
            batch_dict = {k: v.cuda(local_rank) for k, v in batch_dict.items()}
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            scores = embeddings @ centroids.T
            clusters = np.argmax(scores.to("cpu").detach().numpy(), axis=1)
            return clusters
        except Exception as ex:
            print(ex)
            return np.array([-1 for _ in range(len(batch))])

    all_clusters = []

    with tqdm() as pbar:
        # with open(f'distill_corpus/sample/sample.en.part0{local_rank}') as f:
        with open(f"distill_corpus/corpus/corpus.en.part0{local_rank}") as f:
            for part in toolz.partition_all(64000, f):
                maxi_batch = [f"query: {sent.rstrip()}" for sent in part]
                for mini_batch in toolz.partition_all(96, maxi_batch):
                    clusters = process_emb(mini_batch)
                    if clusters[0] == -1:
                        print("Batch failed, fall back to separate processing")
                        clusters = np.concatenate(
                            [process_emb([text]) for text in mini_batch], axis=0
                        )
                    all_clusters.append(clusters)
                    pbar.update(len(mini_batch))

    all_clusters = np.concatenate(all_clusters, axis=0)

    # Each rank writes to its own file
    # e.g. rank_0.pt, rank_1.pt, ...
    # save_path = f"distill_corpus/sample/sample_{local_rank}_clusters"
    save_path = f"distill_corpus/corpus/corpus_{local_rank}_clusters"
    np.save(save_path, all_clusters)
    print(f"{local_rank} Saved clusters to {save_path}")

    # dist.destroy_process_group()


if __name__ == "__main__":
    main()
