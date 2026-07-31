# Moga Patricia

import pickle
from pathlib import Path
from ..component import Component
from ..da import DA, DAI

MODEL = Path(__file__).resolve().parent / "trained_models_lr.pkl"

class StatisticalNLU(Component):

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
        inform_slots = []
        
        for slot in ["food", "pricerange", "area"]:
            pred_val = self.models[slot].predict(features)[0]
            if pred_val != "null":
                inform_slots.append((slot, pred_val))
                
        for intent in self.binary_intents:
            model = self.models[intent]
            if model == "dummy_zero":
                continue
                
            if model.predict(features)[0] == 1:
                if intent.startswith("request_"):
                    slot_requested = intent.replace("request_", "")
                    dais.append(DAI(intent="request", slot=slot_requested))
                elif intent == "dontcare":
                    dais.append(DAI(intent="inform", slot="this", value="dontcare"))
                elif intent == "task_find":
                    dais.append(DAI(intent="inform", slot="task", value="find"))
                elif intent == "type_restaurant":
                    dais.append(DAI(intent="inform", slot="type", value="restaurant"))
                elif intent == "goodbye":
                    dais.append(DAI(intent="bye"))
                else:
                    dais.append(DAI(intent=intent))
                                  
        if inform_slots:
            for slot_name, val in inform_slots:
                dais.append(DAI(intent="inform", slot=slot_name, value=val))
                
        if not dais:
            dais.append(DAI(intent="null"))
            
        dial.nlu = DA(dais)
        logger.info('NLU: %s', str(dial.nlu))
        return dial