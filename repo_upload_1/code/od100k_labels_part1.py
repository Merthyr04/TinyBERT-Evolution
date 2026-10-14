geo = "geography_places"
his = "history_politics"
rel = "religion_theology"
lit = "literature_writing"
sci = "science_nature"
art = "art_music"
war = "war_military"
phi = "philosophy_scholarship"
myt = "myth_legend"
law = "law_education_institutions"
com = "commerce_industry"
lan = "peoples_languages"

# idx 0..259 of data/organic_entries.jsonl, assigned by reading each entry.
LABELS = [
    rel, his, rel, rel, rel, his, rel, myt, his, his,          #   0-  9
    sci, sci, sci, sci, geo, geo, his, his, rel, rel,          #  10- 19
    rel, lit, geo, sci, rel, war, his, rel, rel, his,          #  20- 29
    law, rel, rel, his, phi, phi, his, lit, geo, lit,          #  30- 39
    rel, his, geo, lit, lit, sci, geo, geo, lit, lit,          #  40- 49
    rel, war, lit, his, lit, lit, war, geo, sci, sci,          #  50- 59
    his, his, his, his, his, sci, sci, lit, com, myt,          #  60- 69
    com, rel, rel, his, his, his, his, war, rel, rel,          #  70- 79
    his, his, sci, his, his, rel, sci, lan, sci, phi,          #  80- 89
    his, sci, art, art, lit, his, lit, sci, art, rel,          #  90- 99
    sci, lit, sci, geo, law, com, his, war, lit, his,          # 100-109
    geo, sci, rel, sci, law, war, art, his, war, myt,          # 110-119
    sci, sci, phi, his, law, com, geo, geo, geo, geo,          # 120-129
    lan, war, lit, sci, lit, rel, rel, rel, phi, lit,          # 130-139
    lit, lit, law, geo, his, geo, war, rel, sci, art,          # 140-149
    sci, war, war, art, phi, his, his, his, geo, sci,          # 150-159
    sci, rel, his, geo, phi, rel, geo, rel, rel, his,          # 160-169
    myt, rel, myt, phi, rel, rel, rel, myt, sci, his,          # 170-179
    art, geo, lit, his, lit, rel, geo, lit, his, his,          # 180-189
    his, his, sci, phi, rel, geo, lan, rel, lit, com,          # 190-199
    rel, rel, rel, art, lit, his, rel, lit, law, rel,          # 200-209
    war, his, rel, art, his, rel, art, geo, lit, his,          # 210-219
    rel, sci, geo, law, rel, his, rel, law, rel, phi,          # 220-229
    myt, com, his, phi, sci, sci, his, rel, rel, phi,          # 230-239
    phi, lit, geo, rel, his, war, his, sci, law, art,          # 240-249
    rel, rel, sci, rel, rel, his, law, geo, his, war,          # 250-259
]
KEYS = {"geo": geo, "his": his, "rel": rel, "lit": lit, "sci": sci,
        "art": art, "war": war, "phi": phi, "myt": myt, "law": law,
        "com": com, "lan": lan}
