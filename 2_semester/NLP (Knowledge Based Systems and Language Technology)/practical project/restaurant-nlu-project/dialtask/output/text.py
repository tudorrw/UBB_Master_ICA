#!/usr/bin/env python3

import sys
from ..component import Component

# Moga Patricia

class ConsoleOutput(Component):
    """Print the output to the console, following a 'SYSTEM:' prompt, one utterance per line."""

    def __init__(self, *args):
        super(ConsoleOutput, self).__init__()

    def __call__(self, utterance, *args, **kwargs):
        print('SYSTEM:', utterance)


