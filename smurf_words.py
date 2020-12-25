from enum import Enum, auto
from typing import Dict, List
from dataclasses import dataclass
from spacy.tokens import Token as SpacyToken
import re
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
    NONE = auto()


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
    if tense not in First_group_suffix:
        print(f"Invalid tense: {tense.name}", file=sys.stderr)
        suffix = First_group_suffix[FrenchTense.INFINITIF][0]
    elif tense == FrenchTense.PARTICIPE_PASSE:
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

class BasicPOS(Enum):
    VERB = auto()
    NOUN = auto()
    ADVERB = auto()
    ADJECTIVE = auto()
    AUXILIARY = auto()
    INTERJECTION = auto()
    OTHER = auto()


SCHTROUMPF_STR="schtroumpf"
UNTOUCHED_VERB_PREFIXS = re.compile(r"^(dé|re|en)")

class FrenchWord:
    def text(self) -> str:
        raise NotImplementedError()

    def pos(self) -> BasicPOS:
        raise NotImplementedError()

    def tense(self) -> FrenchTense:
        raise NotImplementedError()

    def person(self) -> int:  # 1..3
        raise NotImplementedError()

    def is_plural(self) -> bool:
        raise NotImplementedError()

    def plural_suffix(self) -> str:
        return "s" if self.is_plural() else ""

    def is_feminine(self) -> bool:
        raise NotImplementedError()

    def to_smurf(self) -> str:
        pos = self.pos()

        if pos == BasicPOS.NOUN:
            m = re.search(r"tion(s)?$", self.text())
            if m:
                new_text = self.text()[:m.span()[0]] + SCHTROUMPF_STR
            else:
                m = re.search(r"teur(s)?$", self.text())
                if m:
                    new_text = self.text()[:m.span()[0]] + SCHTROUMPF_STR + "eur"
                else:
                    new_text = SCHTROUMPF_STR

            return new_text + self.plural_suffix()
        elif pos == BasicPOS.VERB:
            prefix = ""
            m = re.match(UNTOUCHED_VERB_PREFIXS, self.text())
            if m:
                prefix = self.text()[:m.span()[1]]
            return prefix + conjugate_1st_group_verb(SCHTROUMPF_STR,
                                                     self.tense(),
                                                     self.person(),
                                                     self.is_feminine(),
                                                     self.is_plural())
        elif pos == BasicPOS.ADVERB:
            return SCHTROUMPF_STR + "ement"
        elif pos == BasicPOS.ADJECTIVE:
            return SCHTROUMPF_STR + self.plural_suffix()
        elif pos == BasicPOS.INTERJECTION:
            return SCHTROUMPF_STR
        else:
            return self.text()

@dataclass
class FrenchWordTest(FrenchWord):
    _text: str
    _pos: BasicPOS
    _tense: FrenchTense = FrenchTense.NONE
    _person: int = 3
    _plural: bool = False
    _feminine: bool = False

    def text(self) -> str:
        return self._text

    def pos(self) -> BasicPOS:
        return self._pos

    def tense(self) -> FrenchTense:
        return self._tense

    def person(self) -> int:  # 1..3
        return self._person

    def is_feminine(self) -> bool:
        return self._feminine

    def is_plural(self) -> bool:
        return self._plural


SpacyTag_to_BasicPOS: Dict[str, BasicPOS] = {"NOUN": BasicPOS.NOUN,
                                             "ADJ": BasicPOS.ADJECTIVE,
                                             "ADV": BasicPOS.ADVERB,
                                             "AUX": BasicPOS.AUXILIARY,
                                             "VERB": BasicPOS.VERB,
                                             "INTJ": BasicPOS.INTERJECTION}


class FrenchWordSpacy(FrenchWord):
    def __init__(self, token: SpacyToken):
        self.token = token
        features = token.tag_.split("__")
        self.pos = features[0]
        self.features = dict(map(lambda x: x.split("="), features[1].split("|")))

    def text(self) -> str:
        return self.token.string

    def pos(self) -> BasicPOS:
        try:
            return SpacyTag_to_BasicPOS[self.pos]
        except KeyError:
            return BasicPOS.OTHER

    def tense(self) -> FrenchTense:
        try:
            verbform = self.features["VerbForm"]
        except KeyError:
            verbform = ""

        if verbform == "Inf":
            return FrenchTense.INFINITIF

        try:
            tense = self.features["Tense"]
        except KeyError:
            tense = ""

        if verbform == "Part":
            if tense == "Pres":
                return FrenchTense.GERONDIF
            else:
                return FrenchTense.PARTICIPE_PASSE

        try:
            mood = self.features["Mood"]
        except KeyError:
            mood = ""

        if mood == "Ind":
            if tense == "Pres":
                return FrenchTense.PRESENT
            elif tense == "Imp":
                return FrenchTense.SUBJ_IMPARFAIT
            elif tense == "Past":
                return FrenchTense.PASSE_SIMPLE
            elif tense == "Fut":
                return FrenchTense.FUTUR
        elif mood == "Sub":
            if tense == "Pres":
                return FrenchTense.SUBJ_PRESENT
            elif tense == "Past":
                return FrenchTense.SUBJ_IMPARFAIT
        elif mood == "Cnd":
            if tense == "Pres":
                return FrenchTense.COND_PRESENT
        elif mood == "Imp":
            return FrenchTense.IMPERATIF

        return FrenchTense.NONE

    def person(self) -> int:  # 1..3
        try:
            return int(self.features["Person"])
        except (KeyError, ValueError):
            return 0

    def is_feminine(self) -> bool:
        try:
            return self.features["Gender"] == "Fem"
        except KeyError:
            return False

    def is_plural(self) -> bool:
        try:
            return self.features["Number"] == "Plur"
        except KeyError:
            return False
