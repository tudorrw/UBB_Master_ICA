# Moga Patricia 

import pickle
from pathlib import Path
from ..component import Component
from ..da import DA, DAI

MODEL = Path(__file__).resolve().parent / "trained_models_lr.pkl"

class StatisticalNLUProb(Component):

    def __init__(self, config=None):
        super().__init__(config)
        self.config = config or {'model': MODEL}
        
        model_path = Path(self.config['model'])
        if not model_path.is_absolute():
            base_dir = Path(__file__).resolve().parent.parent.parent
            model_path = base_dir / model_path

        with open(model_path, "rb") as f:
            payload = pickle.load(f)
            
        self.vectorizer = payload["vectorizer"]
        self.models = payload["models"]
        self.binary_intents = payload["binary_intents"]

    def __call__(self, dial, logger):
        user_text = dial.user.strip()
        
        if not user_text:
            dial.nlu = DA([DAI(intent="null")])
            logger.info('NLU: %s', str(dial.nlu))
            return dial
            
        features = self.vectorizer.transform([user_text])
        dais = []
        
        # Extragere Sloturi (food, pricerange, area) cu PROCENTE
        for slot in ["food", "pricerange", "area"]:
            model = self.models[slot]
            probs = model.predict_proba(features)[0]
            classes = model.classes_
            
            for cls_name, prob_val in zip(classes, probs):
                if prob_val >= 0.01 and cls_name != "null":
                    dais.append(DAI(intent="inform", slot=slot, value=cls_name, confidence=prob_val))
                    
        # Extragere Intenții Binare
        for intent in self.binary_intents:
            model = self.models[intent]
            if model == "dummy_zero":
                continue
                
            probs = model.predict_proba(features)[0]
            classes = model.classes_
            
            idx_1 = list(classes).index(1) if 1 in classes else -1
            if idx_1 != -1 and probs[idx_1] >= 0.5:
                prob_val = probs[idx_1]
                if intent.startswith("request_"):
                    slot_requested = intent.replace("request_", "")
                    dais.append(DAI(intent="request", slot=slot_requested, confidence=prob_val))
                elif intent == "dontcare":
                    dais.append(DAI(intent="inform", slot="this", value="dontcare", confidence=prob_val))
                elif intent == "task_find":
                    dais.append(DAI(intent="inform", slot="task", value="find", confidence=prob_val))
                elif intent == "type_restaurant":
                    dais.append(DAI(intent="inform", slot="type", value="restaurant", confidence=prob_val))
                elif intent == "goodbye":
                    dais.append(DAI(intent="bye", confidence=prob_val))
                else:
                    dais.append(DAI(intent=intent, confidence=prob_val))
                    
        if not dais:
            dais.append(DAI(intent="null"))
            
        dial.nlu = DA(dais)
        logger.info('NLU: %s', str(dial.nlu))
        return dial