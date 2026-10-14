# -*- coding: utf-8 -*-
"""OD-100K labels, indices 260-415 (assigned by reading each entry)."""
from od100k_labels_part1 import (geo, his, rel, lit, sci, art, war, phi,
                                 myt, law, com, lan)

LABELS2 = [
    rel, lit, lit, law, lit, geo, lan, geo, geo, geo,          # 260-269
    geo, rel, myt, lit, lit, his, law, rel, lit, sci,          # 270-279
    his, his, his, rel, myt, art, geo, lit, sci, his,          # 280-289
    rel, sci, geo, geo, com, geo, geo, geo, geo, lit,          # 290-299
    phi, rel, art, rel, war, sci, lit, geo, his, com,          # 300-309
    his, art, his, his, rel, his, law, his, law, geo,          # 310-319
    sci, lit, myt, rel, rel, his, rel, art, lit, art,          # 320-329
    his, rel, lit, sci, art, law, sci, lan, lit, myt,          # 330-339
    rel, rel, lit, rel, lan, war, sci, lit, his, lit,          # 340-349
    lit, lit, com, rel, lit, myt, lit, lit, rel, lit,          # 350-359
    lit, com, art, art, lit, geo, war, his, law, lit,          # 360-369
    art, art, rel, com, law, lan, lit, rel, art, law,          # 370-379
    lit, his, com, rel, geo, lan, geo, art, lan, sci,          # 380-389
    com, sci, rel, geo, art, phi, rel, geo, phi, sci,          # 390-399
    geo, geo, com, geo, rel, lit, his, his, rel, his,          # 400-409
    his, rel, sci, lit, his, phi,                              # 410-415
]
