#!/usr/bin/env python3
#
# Checks reciprocity.json of inputs_test_{rz,3d}_electrostatic_eb_reciprocity_picmi.py:
# with the symmetric EB stencil the potential at b of a charge at a equals the
# potential at a of the same charge at b (Shortley-Weller: their ratio is
# (1 + h) / 2 for a charge on the node next to the wall, h its distance to
# the wall in cells).

import json
import sys

with open("reciprocity.json") as f:
    r = json.load(f)
print(json.dumps(r))
if abs(r["asymmetry"]) > 1e-9:
    sys.exit(f"the EB Poisson solve is not reciprocal: asymmetry {r['asymmetry']:.3e}")
print("reciprocity OK")
