import json
import os

import datasets
from metricx import models
import torch
import transformers
from transformers.data.data_collator import DataCollatorWithPadding

os.environ["WANDB_DISABLED"] = "true"


def get_dataset(input_file: str, tokenizer, max_input_length: int, device, is_qe: bool):
    def _make_input(example):
        if is_qe:
            example["input"] = (
                "source: " + example["source"] + " candidate: " + example["hypothesis"]
            )
        else:
            example["input"] = (
                "source: "
                + example["source"]
                + " candidate: "
                + example["hypothesis"]
                + " reference: "
                + example["reference"]
            )
        return example

    def _tokenize(example):
        return tokenizer(
            example["input"],
            max_length=max_input_length,
            truncation=True,
            padding=False,
            # We can skip padding here when using a data collator
            # truncation=False,
            # padding="longest",
        )

    def _remove_eos(example):
        example["input_ids"] = example["input_ids"][:-1]
        example["attention_mask"] = example["attention_mask"][:-1]
        return example

    ds = datasets.load_dataset("json", data_files={"test": input_file})
    ds = ds.map(_make_input)
    # !!! it's important to pad the whole batch to the same length otherwise it fails with:
    #         RuntimeError: stack expects each tensor to be equal size
    ds = ds.map(_tokenize, batched=True)
    ds = ds.map(_remove_eos)
    ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
        device=device,
        output_all_columns=True,
    )
    # Add a padding data collator for batching mode
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    return ds, data_collator


def predict(
    batch_size, tokenizer, model_name_or_path, input_file, output_file, max_input_length, qe
) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        per_device_batch_size = batch_size // torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        per_device_batch_size = batch_size

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)

    model = models.MT5ForRegression.from_pretrained(model_name_or_path, torch_dtype="auto")

    model.to(device)
    model.eval()

    ds, datacollator = get_dataset(
        input_file,
        tokenizer,
        max_input_length,
        device,
        qe,
    )

    training_args = transformers.TrainingArguments(
        output_dir=os.getcwd(),
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_pin_memory=False,
    )
    trainer = transformers.Trainer(model=model, args=training_args, data_collator=datacollator)
    predictions, _, _ = trainer.predict(test_dataset=ds["test"])

    dirname = os.path.dirname(output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(output_file, "w") as out:
        for pred, example in zip(predictions, ds["test"]):
            example["prediction"] = float(pred)
            del example["input"]
            del example["input_ids"]
            del example["attention_mask"]
            out.write(json.dumps(example) + "\n")
