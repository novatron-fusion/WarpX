#!/usr/bin/env python3
#
# Checks nearwall.json of inputs_test_rz_electrostatic_eb_nearwall_picmi.py:
# with warpx.eb_e_field = conforming the field gathered by particles closer to
# the wall than 0.6 cells and by particles farther away is accurate to 15 %.
# At --n 32 the largest relative errors are 9.8 % and 10.3 % with particle
# shape 1, and 9.3 % and 2.6 % with shape 3. Without ghost values on the
# covered nodes that have no uncovered neighbor along an edge but are within
# reach of the particles next to the wall, those particles interpolate partly
# from edges inside the EB: 38 % with either shape.

import json
import sys

with open("nearwall.json") as f:
    c = json.load(f)
print(json.dumps(c))
if c["err_max_far"] > 0.15:
    sys.exit(f"field away from the wall inaccurate: {c['err_max_far']}")
if c["err_max_near"] > 0.15:
    sys.exit(f"field next to the wall inaccurate: {c['err_max_near']}")
print("nearwall OK")
