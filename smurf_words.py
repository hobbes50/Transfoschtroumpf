from enum import Enum, auto
from typing import Dict, List, Optional
from dataclasses import dataclass
import spacy
from spacy.tokens import Token as SpacyToken
import stanza
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

Non_smurf_verbs = {"être", "avoir", "pouvoir", "devoir", "falloir"}

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

    def lemma(self) -> str:
        raise NotImplementedError()

    def can_smurf(self) -> bool:
        pos = self.pos()
        return (pos == BasicPOS.NOUN
                or
                pos == BasicPOS.ADJECTIVE
                or
                pos == BasicPOS.ADVERB and self.text().endswith("ment")
                or
                pos == BasicPOS.VERB and self.lemma() not in Non_smurf_verbs
                or
                pos == BasicPOS.INTERJECTION)

    def to_smurf(self) -> str:
        pos = self.pos()
        if not self.can_smurf():
            return self.text()

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

            new_text += self.plural_suffix()
        elif pos == BasicPOS.VERB:
            prefix = ""
            m = re.match(UNTOUCHED_VERB_PREFIXS, self.text())
            if m:
                prefix = self.text()[:m.span()[1]]
            new_text = prefix + conjugate_1st_group_verb(SCHTROUMPF_STR,
                                                         self.tense(),
                                                         self.person(),
                                                         self.is_feminine(),
                                                         self.is_plural())
        elif pos == BasicPOS.ADVERB:
            new_text = SCHTROUMPF_STR + "ement"
        elif pos == BasicPOS.ADJECTIVE:
            new_text = SCHTROUMPF_STR + self.plural_suffix()
        elif pos == BasicPOS.INTERJECTION:
            new_text = SCHTROUMPF_STR
        else:
            new_text = self.text()

        if self.text() and self.text()[0].isupper():
            new_text = new_text[0].upper() + (new_text[1:] if len(new_text) > 1 else "")

        return new_text

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


UTag_to_BasicPOS: Dict[str, BasicPOS] = {"NOUN": BasicPOS.NOUN,
                                         "ADJ": BasicPOS.ADJECTIVE,
                                         "ADV": BasicPOS.ADVERB,
                                         "AUX": BasicPOS.AUXILIARY,
                                         "VERB": BasicPOS.VERB,
                                         "INTJ": BasicPOS.INTERJECTION}

def ufeats_to_fr_tense(features: Dict[str, str]) -> FrenchTense:
    try:
        verbform = features["VerbForm"]
    except KeyError:
        verbform = ""

    if verbform == "Inf":
        return FrenchTense.INFINITIF

    try:
        tense = features["Tense"]
    except KeyError:
        tense = ""

    if verbform == "Part":
        if tense == "Pres":
            return FrenchTense.GERONDIF
        else:
            return FrenchTense.PARTICIPE_PASSE

    try:
        mood = features["Mood"]
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


class FrenchWordSpacy(FrenchWord):
    def __init__(self, token: SpacyToken):
        self.token = token
        features = token.tag_.split("__")
        self.spacy_tag = token.pos_
        self.features = dict([x.split("=") for x in features[-1].split("|") if "=" in x])

    def text(self) -> str:
        return self.token.text

    def pos(self) -> BasicPOS:
        try:
            return UTag_to_BasicPOS[self.spacy_tag]
        except KeyError:
            return BasicPOS.OTHER

    def tense(self) -> FrenchTense:
        return ufeats_to_fr_tense(self.features)

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

    def lemma(self) -> str:
        return self.token.lemma_

def smurf_spacy(text: str, smurf_indexes: Optional[List[int]] = None):
    global Spacy_fr_model
    if Spacy_fr_model is None:
        snlp = stanza.Pipeline(lang="fr")
        Spacy_fr_model = stanza.StanzaLanguage(snlp)
    doc = Spacy_fr_model(text)
    smurf_text = ""
    last_token_end_idx = 0
    for n, token in enumerate(doc):
        smurf_text += text[last_token_end_idx:token.idx]
        last_token_end_idx = token.idx + len(token)
        if smurf_indexes is None or n in smurf_indexes:
            smurf_word = FrenchWordSpacy(token).to_smurf()
            smurf_text += smurf_word
        else:
            smurf_text += text[token.idx:last_token_end_idx]
    smurf_text += text[last_token_end_idx:]

    return smurf_text

from stanza.models.common.doc import Token as StanzaToken, Document as StanzaDoc

class FrenchWordStanza(FrenchWord):
    def __init__(self, token: StanzaToken):
        self.token = token
        self.upos = token.words[0].upos
        features = token.words[0].feats
        if features:
            self.features = dict([x.split("=") for x in features.split("|") if "=" in x])
        else:
            self.features = {}

    def text(self) -> str:
        return self.token.text

    def pos(self) -> BasicPOS:
        try:
            return UTag_to_BasicPOS[self.upos]
        except KeyError:
            return BasicPOS.OTHER

    def tense(self) -> FrenchTense:
        return ufeats_to_fr_tense(self.features)

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

    def lemma(self) -> str:
        return self.token.words[0].lemma


Stanza_fr_model = None


def smurf_stanza(text: str, smurf_indexes: Optional[List[int]] = None):
    global Stanza_fr_model
    if Stanza_fr_model is None:
        Stanza_fr_model = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos,lemma')
    doc: StanzaDoc = Stanza_fr_model(text)
    smurf_text = ""
    last_token_end_char = 0
    for sentence in doc.sentences:
        for n, token in enumerate(sentence.tokens):
            smurf_text += text[last_token_end_char:token.start_char]
            last_token_end_char = token.end_char
            if smurf_indexes is None or n in smurf_indexes:
                smurf_word = FrenchWordStanza(token).to_smurf()
                smurf_text += smurf_word
            else:
                smurf_text += text[token.start_char:last_token_end_char]
    smurf_text += text[last_token_end_char:]

    return smurf_text

