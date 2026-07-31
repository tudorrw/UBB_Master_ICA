#!/usr/bin/env python3

import sys
import time
import json
import tqdm
from abc import ABC

from ..component import Component

# Moga Patricia

class ConsoleInput(Component):
    """Input from the console, following a text prompt."""

    def __call__(self, *args, **kwargs):
        time.sleep(.05)
        return input('USER INPUT> ').strip().lower()
