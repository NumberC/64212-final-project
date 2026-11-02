# config file and interpretor
"""
The config file specifies what block (what size) should be placed where and in what orientation.
If you are looking at the lego board, a 2x4 (WxL) block is the equivalent of a 4x2 block rotated 90 degrees.
To reduce ambiguity, W will always be >= L. so if you are placing a 2x4, you would say you are rotating a 4x2.
There are only 2 rotations that are possible. 0 or 90 degrees. The lego platform is a grid, so only multiples of 90 are allowed.
Due to legos being rectangles there is no difference from 90 and 270 or 0 and 180.

This means the format is:
W, L, x, y, rotation(only 2 options, 0, 90; 0 for 0 and 1 for 90)

The pivot point for location is the bottom-left corner regardless of vertical or horizontal placemntn 

Legos are placed out in the order they appear in the file. So line 1 will be placed before line 2. A series of lines will build our structure.
"""

import csv

# TODO
def writeLegoSDF(W, L):
    # Code that returns a string corresponding to a lego block's SDF (3D model description)
    # pick_and_place.py has this in PS2
    raise NotImplementedError

# TODO
def generateLegoDirective(W, L, x, y, rotation):
    # returns a string for the directive (position) of the lego. pi
    # look at table_directive() function in pick_and_play.py in PS2
    # use writeLegoSDF()
    raise NotImplementedError

def config_interpretor(file_path: str):
    # returns a list of which legos need to be placed in order
    pythonFile = open(file_path, "r")
    configFile = csv.reader(pythonFile)

    # W, L, x, y, is0degrees (0/1)
    return [tuple(map(int, row)) for row in configFile]

print(config_interpretor("./sample_configs/4by4Square.csv"))