from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import spacy
from spacy.tokens import Token as SpacyToken
from datasets import load_dataset
import stanza
import re
import sys
import os

SCHTROUMPF_STR="schtroumpf"

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

SPECIAL_SMURF_NOUNS: Dict[str, str] = {
    "bouchon":f"bou{SCHTROUMPF_STR}",
    "fratricide":f"{SCHTROUMPF_STR}icide",
    "montgolfière":f"mont{SCHTROUMPF_STR}ière",
    "nitroglycérine":f"nitroglycé{SCHTROUMPF_STR}",
    "parapluie":f"para{SCHTROUMPF_STR}",
    "pistolet":f"pisto{SCHTROUMPF_STR}",
    "pneumonie":f"pneumo{SCHTROUMPF_STR}",
    "soporifique":f"sopori{SCHTROUMPF_STR}",
    "symphonie":f"{SCHTROUMPF_STR}onie",
    "trombone":f"{SCHTROUMPF_STR}bone"}

SPECIAL_SMURF_ADJECTIVES: Dict[str, str] = {
    "formidable":f"formi{SCHTROUMPF_STR}",
    "esthétique":f"esthéti{SCHTROUMPF_STR}",
    "universel":f"univer{SCHTROUMPF_STR}"
}

SPECIAL_SMURF_VERBS: Dict[str, str] = {
    "défaire":f"dé{SCHTROUMPF_STR}er",
    "entraider":f"en{SCHTROUMPF_STR}er",
    "emmerder":f"en{SCHTROUMPF_STR}er",
    "rammener":f"re{SCHTROUMPF_STR}er",
    "recommencer":f"re{SCHTROUMPF_STR}er",
    "redonner":f"re{SCHTROUMPF_STR}er",
    "regagner":f"re{SCHTROUMPF_STR}er",
    "remettre":f"re{SCHTROUMPF_STR}er",
    "retourner":f"re{SCHTROUMPF_STR}er",
    "revenir":f"re{SCHTROUMPF_STR}er",
    "retrouver":f"re{SCHTROUMPF_STR}er"}

SPECIAL_SMURF_INTERJ: Dict[str, str] = {
    "atchoum":f"a{SCHTROUMPF_STR}",
    "cocorico":f"cocori{SCHTROUMPF_STR}",
    "bonjour":f"bon{SCHTROUMPF_STR}",
    "sapristi":f"sapri{SCHTROUMPF_STR}"}

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
        # Already "smurfed" words must be left untouched (often proper nouns, e.g. "Grand Schtroumpf")
        if SCHTROUMPF_STR in self.text().lower():
            return False

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

    def word_to_smurf(self, word_compound_index=-1) -> str:
        pos = self.pos()
        if not self.can_smurf():
            return self.text()

        if pos == BasicPOS.NOUN:
            if self.lemma() in SPECIAL_SMURF_NOUNS:
                new_text = SPECIAL_SMURF_NOUNS[self.lemma()]
            else:
                m = re.search(r"tion(s)?$", self.text())
                if m:
                    new_text = self.text()[:m.span()[0]] + SCHTROUMPF_STR
                else:
                    #m = re.search(r"teur(s)?$", self.text())
                    #if False and m:
                    #    new_text = self.text()[:m.span()[0]] + SCHTROUMPF_STR + "eur"
                    #else:
                    new_text = SCHTROUMPF_STR

            new_text += self.plural_suffix()
        elif pos == BasicPOS.VERB:
            prefix = ""
            #m = re.match(UNTOUCHED_VERB_PREFIXS, self.text())
            #if m:
            #    prefix = self.text()[:m.span()[1]]
            verb_stem = SCHTROUMPF_STR
            if self.lemma() in SPECIAL_SMURF_VERBS:
                verb_stem = SPECIAL_SMURF_VERBS[self.lemma()][:-2]
            new_text = prefix + conjugate_1st_group_verb(verb_stem,
                                                         self.tense(),
                                                         self.person(),
                                                         self.is_feminine(),
                                                         self.is_plural())
        elif pos == BasicPOS.ADVERB:
            new_text = SCHTROUMPF_STR + "ement"
        elif pos == BasicPOS.ADJECTIVE:
            if self.lemma() in SPECIAL_SMURF_ADJECTIVES:
                new_text = SPECIAL_SMURF_ADJECTIVES[self.lemma()]
            else:
                new_text = SCHTROUMPF_STR \
                           + ("ant" if self.text().endswith("ant") or self.text().endswith("ants") else "")

            new_text += self.plural_suffix()
        elif pos == BasicPOS.INTERJECTION:
            if self.text() in SPECIAL_SMURF_INTERJ:
                new_text = SPECIAL_SMURF_INTERJ[self.text()]
            else:
                new_text = SCHTROUMPF_STR
        else:
            new_text = self.text()

        if self.text() and self.text().isupper():
            new_text = new_text.upper()
        elif self.text() and self.text()[0].isupper():
            new_text = new_text[0].upper() + (new_text[1:] if len(new_text) > 1 else "")

        return new_text

#Based on Stanza structures:
#- doc must contains a list of sentences in .sentences
#- each sentence must contain a list of tokens
#- each token must provide start_char and end_char methods

WHOLE_WORD=-1
def text_to_smurf(text, nlp, adapter_cls, smurf_indexes: Optional[Dict[Tuple[int, int], int]] = None):
    doc = nlp(text)
    smurf_text = ""
    last_token_end_char = 0
    for s, sentence in enumerate(doc.sentences):
        for n, token in enumerate(sentence.tokens):
            smurf_text += text[last_token_end_char:token.start_char]
            last_token_end_char = token.end_char
            if smurf_indexes is None or (s, n) in smurf_indexes:
                word = adapter_cls(token)
                #Handle compound words
                index_in_compound_word = WHOLE_WORD
                if smurf_indexes:
                    index_in_compound_word = smurf_indexes[(s,n)]

                if index_in_compound_word == WHOLE_WORD:
                    smurf_word = word.word_to_smurf()
                else:
                    parts = word.text().split("-")
                    subtext = parts[index_in_compound_word]
                    subword = adapter_cls(nlp(subtext).sentences[0].tokens[0])
                    smurf_subword = subword.word_to_smurf()
                    parts[index_in_compound_word] = smurf_subword
                    smurf_word = "-".join(parts)

                # Quick and dirty fix of apostrophe (') issues (me vs m' etc.)
                # Proper handling should use linguistic features
                # (e.g. "l'alouette" = "la alouette" while "l'oiseau" = "le oiseau")
                if smurf_text.lower().endswith("'") and not re.match("[aeiou]", smurf_word.lower()):
                    smurf_text = smurf_text[:-1] + "e "
                smurf_text += smurf_word
            else:
                smurf_text += text[token.start_char:last_token_end_char]
    smurf_text += text[last_token_end_char:]

    return smurf_text


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
def get_fr_model():
    global Stanza_fr_model
    if Stanza_fr_model is None:
        Stanza_fr_model = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos,lemma')
    return Stanza_fr_model

def smurf_stanza(text: str, smurf_indexes: Optional[Dict[Tuple[int, int], int]] = None) -> str:
    nlp = get_fr_model()
    return text_to_smurf(text, nlp, FrenchWordStanza, smurf_indexes)

def test_smurf_dataset(data_dir="./data"):
    files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".txt")]
    dataset = load_dataset('csv',
                           data_files=files,
                           column_names=["smurf", "french"],
                           delimiter=";",
                           quote_char=None)
    nlp = stanza.Pipeline(lang='fr', processors="tokenize")
    nbr_of_examples = 0
    nbr_of_correct_examples = 0
    for example in dataset["train"]:
        nbr_of_examples += 1
        doc_smurf = nlp(example["smurf"])
        doc_fr = nlp(example["french"])
        smurf_indexes = {}
        for s, sentence in enumerate(doc_smurf.sentences):
            for n, token in enumerate(sentence.tokens):
                if SCHTROUMPF_STR in token.text.lower():
                    if "-" in token.text:
                        for i, subword in enumerate(token.text.split("-")):
                            if SCHTROUMPF_STR in subword:
                                smurf_indexes[(s, n)] = i
                                break
                    else:
                        smurf_indexes[(s, n)] = WHOLE_WORD

        to_smurf = smurf_stanza(doc_fr.text, smurf_indexes)
        if to_smurf == doc_smurf.text:
            nbr_of_correct_examples += 1
        else:
            print("==================ERROR==================")
            print("FRENCH  : " + doc_fr.text)
            print("TO_SMURF: " + to_smurf)
            print("LABEL   : " + doc_smurf.text)
            print("=========================================")

    print(f"\nTotal correct = {nbr_of_correct_examples}/{nbr_of_examples} "
          f"({nbr_of_correct_examples/nbr_of_examples}%)")

test_smurf_dataset()
