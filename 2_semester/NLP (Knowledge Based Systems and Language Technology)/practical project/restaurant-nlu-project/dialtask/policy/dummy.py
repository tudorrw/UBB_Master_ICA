from ..component import Component
from ..utils import choose_one
from ..da import DA

# Moga Patricia - Ciobanu Sergiu-Tudor


class ReplyWithNLU(Component):
    """A "policy" good for testing a NLU system in a stand-alone setting -- it will only
    copy a textual representation of the NLU result into the system reply."""

    def __call__(self, dial, logger):
        if len(dial.user) == 0:
            dial.end_dialogue()
        dial.set_system_response(dial.nlu.to_cambridge_da_string() or '<EMPTY>')
        return dial


class ReplyWithState(Component):

    def __call__(self, dial, logger):
        if not hasattr(dial, 'state') or dial.state is None:
            dial.response = "State is empty!"
            dial.system = "State is empty!"
            return dial

        lines = ["\n====== DIALOGUE STATE ======"]
        for slot, distribution in dial.state.items():
            lines.append(f" Slot: {slot.upper()}")
            
            sorted_dist = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
            for val, prob in sorted_dist:
                val_clean = "None" if val is None else str(val)
                prob_clean = float(prob)
                
                if prob_clean > 0.0:
                    lines.append(f"   -> {val_clean:<12} : {prob_clean * 100:>6.2f}%")
        lines.append("============================\n")
        
        print("\n".join(lines))
        
        dial.response = "."
        dial.system = "." 
        
        return dial
