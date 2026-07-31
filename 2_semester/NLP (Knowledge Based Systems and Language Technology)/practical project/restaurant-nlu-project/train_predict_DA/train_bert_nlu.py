"""
Train a JointBERT statistical NLU on the extracted DSTC2 dialogue acts.

we fine-tune a single shared (Distil)BERT encoder with one classification head
per intent-slot pair, trained jointly:
  * multi-value slots (e.g. inform(food)) -> softmax over <null> + observed values
  * valueless pairs   (e.g. request(phone), bye()) -> binary <null> / <present>

Targets are read from the DA strings with DA.parse_cambridge_da. We train on the
train/dev split only (results_nlu_dev.json); the test set is never touched.

Run with:  python -m train_predict_DA.train_bert_nlu
"""
#  Ciobanu Sergiu-Tudor

import sys
import json
import tqdm
from pathlib import Path
from collections import defaultdict

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
sys.path.append(str(BASE_DIR))

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from dialtask.da import DA
from dialtask.nlu.start_dstc_bert import (
    MODEL_DIR,
    NULL_LABEL,
    PRESENT_LABEL,
    JointBERTNLU,
    build_encoder_from_pretrained,
)

CHECKPOINT = "TODBERT/TOD-DistilBERT-JNT-V1"
TRAIN_DATA = BASE_DIR / "data" / "dstc_extracted_DA" / "results_nlu_dev.json"

MAX_LEN = 64
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 1e-5


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_fields(das):
    """One classification head per intent-slot pair seen in the training data."""
    values = defaultdict(set)
    for da in das:
        for dai in da.dais:
            values[(dai.intent, dai.slot)]  # register the pair
            if dai.value is not None:
                values[(dai.intent, dai.slot)].add(dai.value)

    fields = []
    for intent, slot in sorted(values):
        vals = values[(intent, slot)]
        classes = [NULL_LABEL] + sorted(vals) if vals else [NULL_LABEL, PRESENT_LABEL]
        fields.append({"intent": intent, "slot": slot, "valued": bool(vals), "classes": classes})
    return fields


def build_targets(das, fields):
    """Integer target matrix [num_instances, num_fields] (column 0 == <null>)."""
    label_to_idx = [{label: i for i, label in enumerate(f["classes"])} for f in fields]
    targets = torch.zeros(len(das), len(fields), dtype=torch.long)
    for row, da in enumerate(das):
        present = {}
        for dai in da.dais:
            present.setdefault((dai.intent, dai.slot), dai.value)
        for col, field in enumerate(fields):
            key = (field["intent"], field["slot"])
            if key not in present:
                continue  # leave as <null>
            if field["valued"]:
                value = present[key]
                if value in label_to_idx[col]:
                    targets[row, col] = label_to_idx[col][value]
            else:
                targets[row, col] = label_to_idx[col][PRESENT_LABEL]
    return targets


def train():
    device = get_device()
    print("Device:", device)

    print("Loading training data from:", TRAIN_DATA)

    with open(TRAIN_DATA, "r", encoding="utf-8") as datafile:
        data = json.load(datafile)

    texts = [inst.get("usr", "") or "" for inst in data]
    das = [DA.parse_cambridge_da(inst.get("DA", "") or "") for inst in data]

    fields = build_fields(das)
    targets = build_targets(das, fields)
    print(f"Training {len(fields)} intent-slot heads on {len(texts)} utterances.")


    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    encoded = tokenizer(texts, truncation=True, max_length=MAX_LEN,
                        padding="max_length", return_tensors="pt")
    
    loader = DataLoader(
        TensorDataset(encoded["input_ids"], encoded["attention_mask"], targets),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    model = JointBERTNLU(
        build_encoder_from_pretrained(CHECKPOINT),
        [len(f["classes"]) for f in fields],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in (range(1, EPOCHS + 1)):
        running_loss = 0.0
        for input_ids, attention_mask, y in  tqdm.tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            input_ids, attention_mask, y = input_ids.to(device), attention_mask.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = sum(loss_fn(logits[i], y[:, i]) for i in range(len(fields)))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch}/{EPOCHS}  loss={running_loss / len(loader):.4f}")

    # Save weights + encoder config + tokenizer + label metadata for offline inference.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(MODEL_DIR)
    model.encoder.config.save_pretrained(MODEL_DIR)
    torch.save(model.state_dict(), MODEL_DIR / "model.pt")
    with open(MODEL_DIR / "meta.json", "w", encoding="utf-8") as meta_file:
        json.dump({"max_len": MAX_LEN, "fields": fields}, meta_file, indent=2)
    print("Saved trained JointBERT model to:", MODEL_DIR)


if __name__ == "__main__":
    train()
