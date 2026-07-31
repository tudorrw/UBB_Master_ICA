import re
from ..component import Component
from ..da import DAI
from dialtask.dialogue import DA

# Moga Patricia - Ciobanu Sergiu-Tudor

class RestaurantNLU(Component):
    """An optimized rule-based NLU for the DSTC2 restaurant domain."""

    PHRASE_TO_SLOT = {
        'address|addr|location': 'addr',
        'phone number|phone|telephone': 'phone',
        'postcode|post code': 'postcode',
        'food|cuisine': 'food',
        'price|price range|pricerange': 'pricerange',
        'area|neighborhood': 'area',
    }

    SLOT_VALUES = {
        'pricerange': ['cheap', 'moderate', 'expensive'],
        'area': ['north', 'south', 'east', 'west', 'centre', 'center', 'middle'],
        'food': [
            'italian', 'mexican', 'chinese', 'indian', 'thai', 'french', 'singaporean',
            'japanese', 'spanish', 'korean', 'lebanese', 'cantonese', 'malaysian',
            'persian', 'european', 'international', 'vegetarian', 'north american', 'welsh',
            'romanian', 'greek', 'vietnamese', 'turkish', 'mediterranean', 
            'caribbean', 'german', 'african', 'brazilian', 'argentinian', 'polish',
            'seafood', 'steak house', 'steakhouse', 'barbecue', 'hungarian', 
            'australian', 'fast food', 'pub'
        ],
    }

    def __call__(self, dial, logger):
        local_results = []

        if re.search(r'\b(hello|hey|hi|ola|ahoj)\b', dial.user, re.IGNORECASE):
            local_results.append(DAI('greet'))
        if re.search(r'\b(bye|goodbye|good bye|see ya|see you)\b', dial.user, re.IGNORECASE):
            local_results.append(DAI('goodbye'))
            
        if re.search(r'\b(thank you|thanks|thank)\b', dial.user, re.IGNORECASE):
            local_results.append(DAI('thankyou'))

        if re.match(r'^(no|I don\'t)\b', dial.user, re.IGNORECASE):
            local_results.append(DAI('negate'))
        if re.match(r'^(yes|I do|yea|yeah|yep|correct)\b', dial.user, re.IGNORECASE):
            local_results.append(DAI('affirm'))

        for phrase, slot in self.PHRASE_TO_SLOT.items():
            if re.search(r'\b(' + phrase + r')\b', dial.user, re.IGNORECASE):
                local_results.append(DAI('request', slot))

        if re.search(r'\bnorth american\b', dial.user, re.IGNORECASE):
            local_results.append(DAI('inform', 'food', 'north american'))
        else:
            for slot, values in self.SLOT_VALUES.items():
                m = re.search(r'\b(' + '|'.join(values) + r')\b', dial.user, re.IGNORECASE)
                if m:
                    val = m.group(0).lower()
                    if val == 'center' or val == 'middle':
                        val = 'centre'

                    local_results.append(DAI('inform', slot, val))

        if re.search(r'\bmoderately\b', dial.user, re.IGNORECASE):
            local_results.append(DAI('inform', 'pricerange', 'moderate'))
            
        if re.search(r'\btailand\b', dial.user, re.IGNORECASE):
            local_results.append(DAI('inform', 'food', 'thai'))

        if re.search(r'\b(any|dont care|doesnt matter)\b', dial.user, re.IGNORECASE):
            local_results.append(DAI('inform', 'this', 'dontcare'))


        if re.search(r"i'?\s*m looking for|i'?\s*m search(ing)?|search(ing)?|find", dial.user, re.IGNORECASE):
            local_results.append(DAI('inform', 'task', 'find'))


        dial.nlu = DA(local_results)

        logger.info('NLU: %s', str(dial.nlu))
        return dial
    