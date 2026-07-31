# Moga Patricia - Ciobanu Sergiu-Tudor
import os
import sys
import json
import logging
import re

current_script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = current_script_dir
while base_dir and not os.path.exists(os.path.join(base_dir, "data")):
    parent = os.path.dirname(base_dir)
    if parent == base_dir:
        break
    base_dir = parent

sys.path.append(base_dir)

from dialtask.dialogue import Dialogue
from dialtask.nlu.restaurant import RestaurantNLU

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("TestNLU")

nlu_rules = RestaurantNLU(config={})

output_folder = os.path.join(base_dir, "data", "dstc_extracted_DA")
os.makedirs(output_folder, exist_ok=True)

datasets = [
    {
        "root": "data/dstc2_traindev",
        "txt_out": os.path.join(output_folder, "results_nlu_dev.txt"),
        "json_out": os.path.join(output_folder, "results_nlu_dev.json"),
        "label": "DEVELOPMENT (DEV)"
    },
    {
        "root": "data/dstc2_test",
        "txt_out": os.path.join(output_folder, "results_nlu_test.txt"),
        "json_out": os.path.join(output_folder, "results_nlu_test.json"),
        "label": "TEST"
    }
]

def clean_and_format_da(da_string, user_text):
    da_string = da_string.strip()
    
    if da_string == "" or da_string == "[]":
        return "null()"
        
    if "greet" in da_string:
        return "greet()"
        
    if "goodbye" in da_string:
        return "thankyou()|bye()" if "thank" in user_text.lower() else "bye()"

    if "inform" in da_string:
        slots = re.findall(r'inform\(([^)]+)\)', da_string)
        if slots:
            cleaned_slots = []
            for slot in slots:
                if slot.startswith("price="):
                    slot = slot.replace("price=", "pricerange=")
                cleaned_slots.append(slot)
            
            if "restaurant" in user_text.lower() and "type=restaurant" not in cleaned_slots:
                if len(cleaned_slots) > 1:
                    cleaned_slots.insert(1, "type=restaurant")
                else:
                    cleaned_slots.append("type=restaurant")
                    
            return f"inform({','.join(cleaned_slots)})"

    if "request" in da_string:
        slots = re.findall(r'request\(([^)]+)\)', da_string)
        cleaned_slots = [s.replace("slot=", "") for s in slots]
        return f"request({','.join(cleaned_slots)})"

    return da_string

for dataset in datasets:
    root_path = os.path.join(base_dir, dataset["root"])
    
    if not os.path.exists(root_path):
        print(f"Warning: Could not find the folder at path: {root_path}. Skipping...")
        continue
        
    print(f"\nProcessing {dataset['label']} dataset from path: {root_path}...")
    
    contor = 0
    json_data_list = []
    
    with open(dataset["txt_out"], "w", encoding="utf-8") as f_out:
        f_out.write(f"RESULTS OBTAINED VIA RULE-BASED NLU ({dataset['label']})\n")
        f_out.write("=" * 70 + "\n\n")
        
        for root, dirs, files in os.walk(root_path):
            if "label.json" in files:
                with open(os.path.join(root, "label.json"), "r", encoding="utf-8") as f:
                    date_dialog = json.load(f)
                    
                    for turn in date_dialog.get("turns", []):
                        text_utilizator = turn.get("transcription", "")
                        
                        dial = Dialogue()
                        dial.user = text_utilizator
                        
                        dial = nlu_rules(dial, logger)

                        raw_da = str(dial.nlu).strip()
                        da_string = clean_and_format_da(raw_da, text_utilizator)

                        f_out.write(f"User: {text_utilizator}\n")
                        f_out.write(f"NLU Result: {da_string}\n")
                        f_out.write("-" * 50 + "\n")
                        
                        json_data_list.append({
                            "usr": text_utilizator,
                            "DA": da_string
                        })
                        
                        if contor < 3:
                            print(f"[{dataset['label']}] Processed: '{text_utilizator}' -> {da_string}")
                        
                        contor += 1

    with open(dataset["json_out"], "w", encoding="utf-8") as f_json:
        json.dump(json_data_list, f_json, indent=4, ensure_ascii=False)
                        
    print("=" * 50)
    print(f"Done for {dataset['label']}! Processed {contor} utterances.")
    print(f"TXT saved to: '{dataset['txt_out']}'")
    print(f"JSON saved to: '{dataset['json_out']}'")
    print("=" * 50)