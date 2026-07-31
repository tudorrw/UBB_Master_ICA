#Ciobanu Sergiu-Tudor

"""
Statistical NLU based on JointBERT (BERT for Joint Intent Classification and
Slot Filling). A single shared (Distil)BERT encoder is fine-tuned with one
classification head per intent-slot pair:
  * multi-value slots (e.g. inform(food)) -> softmax over <null> + observed values
  * valueless pairs   (e.g. request(phone), bye()) -> binary <null> / <present>

At prediction time every head is evaluated and any head that predicts a non-<null>
class emits the corresponding dialogue-act item. The heads and their label
vocabularies are learnt by train_predict_DA/train_bert_nlu.py and stored next to
the weights, so this module is generic.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ..component import Component
from ..da import DA, DAI

# Directory holding the trained model (weights + tokenizer + label metadata).
# Relative to this source file so the solution works on any machine / OS.
MODEL_DIR = Path(__file__).resolve().parent / "bert_nlu_model"

# Sentinel labels used inside every classification head.
NULL_LABEL = "<null>"        # intent-slot pair not present in the utterance
PRESENT_LABEL = "<present>"  # valueless intent-slot pair is present


class JointBERTNLU(nn.Module):
    """Shared BERT encoder with one classification head per intent-slot pair."""

    def __init__(self, encoder, field_num_classes, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, n_classes) for n_classes in field_num_classes]
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation as the pooled sentence embedding.
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return [head(pooled) for head in self.heads]


def build_encoder_from_pretrained(checkpoint):
    """Create the encoder by downloading/loading pretrained weights (training)."""
    return AutoModel.from_pretrained(checkpoint)


def build_encoder_from_config(model_dir):
    """Create the encoder architecture from a saved config, with random weights.

    The real (fine-tuned) weights are loaded afterwards from the saved
    ``state_dict``, so no internet access is needed at inference time.
    """
    config = AutoConfig.from_pretrained(model_dir)
    return AutoModel.from_config(config)


class StatisticalNLUBert(Component):
    """Dialtask component that applies the trained JointBERT model to user input."""

    def __init__(self, config=None):
        super().__init__(config)
        self.config = config or {"model": str(MODEL_DIR)}

        model_dir = Path(self.config.get("model", MODEL_DIR))
        if not model_dir.is_absolute():
            # Resolve relative to the project root (parent of the dialtask package).
            base_dir = Path(__file__).resolve().parent.parent.parent
            model_dir = base_dir / model_dir

        with open(model_dir / "meta.json", "rt", encoding="utf-8") as meta_fd:
            meta = json.load(meta_fd)

        self.fields = meta["fields"]
        self.max_len = meta.get("max_len", 64)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        encoder = build_encoder_from_config(model_dir)
        field_num_classes = [len(f["classes"]) for f in self.fields]
        self.model = JointBERTNLU(encoder, field_num_classes)

        state_dict = torch.load(model_dir / "model.pt", map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_da(self, text):
        """Run the model on a single utterance and return a DA object."""
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        logits_per_field = self.model(input_ids, attention_mask)

        dais = []
        for field, logits in zip(self.fields, logits_per_field):
            pred_idx = int(torch.argmax(logits, dim=-1).item())
            pred_label = field["classes"][pred_idx]
            if pred_label == NULL_LABEL:
                continue
            slot = field["slot"]  # may be None
            if field["valued"]:
                dais.append(DAI(intent=field["intent"], slot=slot, value=pred_label))
            else:
                dais.append(DAI(intent=field["intent"], slot=slot))

        return DA(dais)

    def __call__(self, dial, logger):
        user_text = (dial.user or "").strip()
        if not user_text:
            dial.nlu = DA()
            logger.info("NLU: %s", str(dial.nlu))
            return dial

        dial.nlu = self.predict_da(user_text)
        logger.info("NLU: %s", str(dial.nlu))
        return dial
