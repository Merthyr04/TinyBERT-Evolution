# OD-100K second-annotation passes (round 75)

Stratified sample: 119 entries, 10 per class, seed 20261011.  The first label is the author's, made while reading the whole encyclopaedia.

| second pass | observed agreement | expected agreement | Cohen's kappa |
|---|---|---|---|
| keyword rule | 49.6% | 8.3% | 0.45 |
| model annotation | 84.0% | 8.3% | 0.83 |
| rule vs model | 51.3% | 9.5% | 0.46 |

Per-class agreement of the model pass with the first annotation:

* geography_places: 90.0%
* history_politics: 90.0%
* religion_theology: 100.0%
* literature_writing: 100.0%
* science_nature: 70.0%
* art_music: 90.0%
* war_military: 80.0%
* philosophy_scholarship: 70.0%
* myth_legend: 80.0%
* law_education_institutions: 70.0%
* commerce_industry: 70.0%
* peoples_languages: 100.0%

19 of 119 entries disagree with the first annotation; in 16 of them the keyword rule disagrees as well, so those two passes do not corroborate each other.

## Entries where the model pass disagrees

| idx | headword | first label | model | keyword rule |
|---|---|---|---|---|
| 33 | AGOUST | history_politics | war_military | war_military |
| 46 | AJALON | geography_places | religion_theology | war_military |
| 68 | ALDINE EDITIONS | commerce_industry | literature_writing | history_politics |
| 105 | ALMANACH DE GOTHA | commerce_industry | history_politics | peoples_languages |
| 146 | ANDREOSSY | war_military | history_politics | war_military |
| 216 | ARUNDEL MARBLES | art_music | philosophy_scholarship | law_education_institutions |
| 227 | ASSEMANI | law_education_institutions | philosophy_scholarship | literature_writing |
| 235 | ASTRUC | science_nature | philosophy_scholarship | law_education_institutions |
| 239 | ATHEISM | philosophy_scholarship | religion_theology | myth_legend |
| 240 | ATHEISM | philosophy_scholarship | religion_theology | religion_theology |
| 252 | AUGUST | science_nature | history_politics | history_politics |
| 256 | AULIC COUNCIL | law_education_institutions | history_politics | law_education_institutions |
| 259 | AUMALE | war_military | history_politics | history_politics |
| 272 | AYLMER | myth_legend | religion_theology | religion_theology |
| 276 | AZEGLIO | law_education_institutions | history_politics | law_education_institutions |
| 284 | BABYLONISH CAPTIVITY | myth_legend | religion_theology | peoples_languages |
| 291 | BADRINATH | science_nature | religion_theology | history_politics |
| 294 | BAFFIN BAY | commerce_industry | geography_places | science_nature |
| 398 | BAZARD | philosophy_scholarship | history_politics | religion_theology |
