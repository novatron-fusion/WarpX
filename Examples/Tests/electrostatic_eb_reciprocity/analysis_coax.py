#!/usr/bin/env python3
#
# Checks coax.json of inputs_test_rz_electrostatic_eb_coax_picmi.py: the
# vacuum potential and field between the electrodes are second-order
# accurate (4.5e-4 / 1.3e-4 and 1.9e-3 / 5.0e-4 relative at --n 16 / 32), and
# with warpx.eb_e_field = conforming E_r on the edges cut by the electrodes is
# the field at the wall to first order (3.7 % and 2.0 % at --n 32; centered
# gives 56 % and 97 %).

import json
import sys

with open("coax.json") as f:
    c = json.load(f)
print(json.dumps(c))
if c["phi_rel_max"] > 1e-3 or c["Er_rel_max"] > 4e-3:
    sys.exit("vacuum potential or field between the electrodes inaccurate")
if c["eb_e_field"] == "conforming":
    for cut, wall in zip(c["Er_cut"], c["Er_surface"]):
        if abs(cut / wall - 1) > 0.05:
            sys.exit(f"E_r {cut} on a cut edge, {wall} at the wall")
print("coax OK")
