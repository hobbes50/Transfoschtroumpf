from enum import Enum, auto
from typing import Dict, List
import sys

class FrenchTense(Enum):
    PRESENT = auto()
    IMPARFAIT = auto()
    PASSE_SIMPLE = auto()
    FUTUR = auto()
    SUBJ_PRESENT = auto()
    SUBJ_IMPARFAIT = auto()
    COND_PRESENT = auto()
    IMPERATIF = auto()
    INFINITIF = auto()
    GERONDIF = auto()
    PARTICIPE_PASSE = auto()


First_group_suffix: Dict[FrenchTense, List[str]] = {}
First_group_suffix[FrenchTense.PRESENT] = ["e", "es", "e", "ons", "ez", "ent"]
First_group_suffix[FrenchTense.IMPARFAIT] = ["ais", "ais", "ait", "ions", "iez", "aient"]
First_group_suffix[FrenchTense.PASSE_SIMPLE] = ["ai", "as", "a", "âmes", "âtes", "èrent"]
First_group_suffix[FrenchTense.FUTUR] = ["erai", "eras", "era", "erons", "erez", "eront"]
First_group_suffix[FrenchTense.SUBJ_PRESENT] = ["e", "es", "e", "ions", "iez", "ent"]
First_group_suffix[FrenchTense.SUBJ_IMPARFAIT] = ["asse", "asses", "ât", "assions", "assiez", "assent"]
First_group_suffix[FrenchTense.COND_PRESENT] = ["erais", "erais", "erait", "erions", "eriez", "eraient"]
First_group_suffix[FrenchTense.IMPERATIF] = ["<?>", "es", "<?>", "ons", "ez", "<?>"]

First_group_suffix[FrenchTense.GERONDIF] = ["ant"]
First_group_suffix[FrenchTense.INFINITIF] = ["er"]
First_group_suffix[FrenchTense.PARTICIPE_PASSE] = ["é", "ée", "és", "ées"]

def conjugate_1st_group_verb(radical: str,
                             tense: FrenchTense,
                             person: int,  # 1 = 1st person
                             feminine: bool = False,
                             plural: bool = False):
    suffix = ""

    if tense not in First_group_suffix:
        print(f"Invalid tense: {tense.name}", file=sys.stderr)
        suffix = First_group_suffix[FrenchTense.INFINITIF][0]
    if tense == FrenchTense.PARTICIPE_PASSE:
        index = feminine + 2*plural
        suffix = First_group_suffix[tense][index]
    else:
        person_suffixes = First_group_suffix[tense]
        if len(person_suffixes) == 1:
            suffix = person_suffixes[0]
        else:
            index = person - 1 + 3*plural
            suffix = person_suffixes[index]

    return radical + suffix






# tu schtroumpfes
# il schtroumpfe
# nous schtroumpfons
# vous schtroumpfez
# ils schtroumpfent
#"]

# Nom : schtroumpf
# -tion: schtroumpf
# -teur: schtroumpfeur

# Pronom, Article : jamais
# Auxiliaire être, avoir: jamais

# Adjectif : schtroumf, schtroumpfant
# Adverbe : schtroumpfement
# Interjection : Schtroumpf !

# Verbe
#re-<verb> = re-schtroumpf
#dé-<verb> = dé-schtroumpf
#en-<verb> = en-schtroumpf
