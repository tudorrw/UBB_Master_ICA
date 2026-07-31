from ..component import Component

# Moga Patricia

class DialogueStateTracker(Component):

    def __init__(self, config=None):
        super().__init__(config)
        self.threshold = float(self.config.get("threshold", 0.01))
        self.slots = ["food", "pricerange", "area"]

    def __call__(self, dial, logger):
        if not hasattr(dial, 'state') or dial.state is None:
            dial.state = {}
            
        for slot in self.slots:
            if slot not in dial.state:
                dial.state[slot] = {None: 1.0}

        current_nlu_mentions = {slot: {} for slot in self.slots}
        
        if hasattr(dial, 'nlu') and dial.nlu:
            for dai in dial.nlu:
                if dai.intent == "inform" and dai.slot in self.slots and dai.value != "null":
                    confidence = getattr(dai, 'confidence', 1.0)
                    if confidence >= self.threshold:
                        current_nlu_mentions[dai.slot][dai.value] = confidence

        for slot in self.slots:
            mentions = current_nlu_mentions[slot]
            
            total_nlu_prob = sum(mentions.values())
            none_prob = max(0.0, 1.0 - total_nlu_prob)
            
            current_distribution = dial.state[slot]
            updated_distribution = {}
            
            for val, old_prob in current_distribution.items():
                new_prob = old_prob * none_prob
                if new_prob > 0.0:
                    updated_distribution[val] = new_prob
                    
            for val, nlu_prob in mentions.items():
                if val in updated_distribution:
                    updated_distribution[val] += nlu_prob
                else:
                    updated_distribution[val] = nlu_prob
                    
            if None not in updated_distribution:
                updated_distribution[None] = 0.0
                
            cleaned_distribution = {}
            for val, prob in updated_distribution.items():
                if prob >= 0.001 or val is None:
                    cleaned_distribution[val] = round(prob, 4)
                    
            dial.state[slot] = cleaned_distribution

        logger.info('DST State: %s', str(dial.state))
        return dial