"""
Custom JSON encoder.
Used to serialise numpy arrays and other special data types
"""
import json

import numpy as np


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that supports numpy arrays and other special data types"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)

        # Call the default method of the parent class
        return super().default(obj)