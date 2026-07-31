# Moga Patricia 

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
TRAIN_DATA = BASE_DIR / "data" / "dstc_extracted_DA" / "results_nlu_dev.json"
MODEL_PATH = BASE_DIR / "dialtask" / "nlu" / "trained_models_lr.pkl"

BINARY_INTENTS = [
    "greet", "goodbye", "thankyou", "affirm", "negate",
    "request_addr", "request_phone", "request_postcode", 
    "request_pricerange", "request_area", "request_food",
    "dontcare", "task_find", "type_restaurant"
]

def parse_extracted_da(da_string):
    da_string = da_string.strip()
    parsed = {
        "food": "null",
        "pricerange": "null",
        "area": "null",
        "intents": set()
    }
    
    if da_string == "null()":
        return parsed

    if da_string.startswith("inform("):
        content = da_string[7:-1]
        elements = content.split(",")
        for elem in elements:
            if "=" in elem:
                slot, val = elem.split("=")
                if slot == "price":
                    slot = "pricerange"
                if slot in ["food", "pricerange", "area"]:
                    parsed[slot] = val
                elif slot == "task" and val == "find":
                    parsed["intents"].add("task_find")
                elif slot == "type" and val == "restaurant":
                    parsed["intents"].add("type_restaurant")
                elif slot == "this" and val == "dontcare":
                    parsed["intents"].add("dontcare")
                
    elif da_string.startswith("request("):
        content = da_string[8:-1]
        slots = content.split(",")
        for slot in slots:
            slot = slot.strip()
            if slot:
                parsed["intents"].add(f"request_{slot}")
                
    elif da_string.startswith("greet"):
        parsed["intents"].add("greet")
    elif da_string.startswith("bye") or "bye" in da_string:
        parsed["intents"].add("goodbye")
    elif da_string.startswith("thankyou"):
        parsed["intents"].add("thankyou")
    elif da_string.startswith("affirm"):
        parsed["intents"].add("affirm")
    elif da_string.startswith("negate"):
        parsed["intents"].add("negate")
        
    return parsed

def train():
    print(f"Loading data from: {TRAIN_DATA}")
    with open(TRAIN_DATA, "r", encoding="utf-8") as datafile:
        data = json.load(datafile)

    texts = []
    labels_food = []
    labels_price = []
    labels_area = []
    labels_binary = {intent: [] for intent in BINARY_INTENTS}

    for inst in data:
        user_text = inst.get("usr", "")
        da_target = inst.get("DA", "null()")
        parsed_da = parse_extracted_da(da_target)
        
        texts.append(user_text)
        labels_food.append(parsed_da["food"])
        labels_price.append(parsed_da["pricerange"])
        labels_area.append(parsed_da["area"])
        
        for intent in BINARY_INTENTS:
            if intent in parsed_da["intents"]:
                labels_binary[intent].append(1)
            else:
                labels_binary[intent].append(0)

    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1, binary=True)
    X_train = vectorizer.fit_transform(texts)

    models = {}
    
    models["food"] = LogisticRegression(C=10.0, max_iter=1000, class_weight='balanced', random_state=42)
    models["food"].fit(X_train, labels_food)
    
    models["pricerange"] = LogisticRegression(C=10.0, max_iter=1000, class_weight='balanced', random_state=42)
    models["pricerange"].fit(X_train, labels_price)
    
    models["area"] = LogisticRegression(C=10.0, max_iter=1000, class_weight='balanced', random_state=42)
    models["area"].fit(X_train, labels_area)

    for intent in BINARY_INTENTS:
        unique_classes = np.unique(labels_binary[intent])
        if len(unique_classes) > 1:
            models[intent] = LogisticRegression(C=10.0, max_iter=1000, class_weight='balanced', random_state=42)
            models[intent].fit(X_train, labels_binary[intent])
        else:
            models[intent] = "dummy_zero"

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "vectorizer": vectorizer,
        "models": models,
        "binary_intents": BINARY_INTENTS
    }
    
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
        
    print("Baseline model trained successfully with CountVectorizer!")

if __name__ == '__main__':
    train()