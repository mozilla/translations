import os

import toolz
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def main():
    # 1. Initialize process group
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    print(f"processing local rank {local_rank}")

    # 2. Create model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-small").cuda(local_rank)

    # Wrap the model in DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def process_emb(batch):
        batch_dict = tokenizer(
            batch, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        batch_dict = {k: v.cuda(local_rank) for k, v in batch_dict.items()}
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.to("cpu").detach()

    all_embs = []
    try:
        with tqdm() as pbar:
            with open(f"distill_corpus/sample/sample.en.part0{local_rank}") as f:
                for part in toolz.partition_all(64000, f):
                    maxi_batch = [f"query: {sent.rstrip()}" for sent in part]
                    for mini_batch in toolz.partition_all(64, maxi_batch):
                        emb = process_emb(mini_batch)
                        all_embs.append(emb.cpu())
                        pbar.update(len(mini_batch))

    finally:
        dist.destroy_process_group()
        all_embeddings = torch.cat(all_embs, dim=0)

        # Each rank writes to its own file
        # e.g. rank_0.pt, rank_1.pt, ...
        save_path = f"distill_corpus/sample/part_{local_rank}_embeddings.pt"
        torch.save(all_embeddings, save_path)
        print(f"{local_rank} Saved embeddings to {save_path}")


if __name__ == "__main__":
    main()
