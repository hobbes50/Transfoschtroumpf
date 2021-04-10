from enum import Enum, auto
import random
from typing import Dict, List, Optional, Set, Tuple, Callable, Any, AnyStr, Pattern
from dataclasses import dataclass
import spacy
#import spacy_udpipe
import ufal.udpipe as udpipe
from spacy.tokens import Token as SpacyToken, Doc as SpacyDoc
import datasets
import stanza
from stanza.models.common.doc import Token as StanzaToken, Document as StanzaDoc
import time
import re
import sys
import os
import shutil
import copy
import pandas
import csv

Lexicon = Dict[str, List[Dict[str, str]]]

def remove_prefix(string: str, prefix: str) -> str:
    if string.startswith(prefix):
        return string[len(prefix):]
    else:
        return string


def remove_suffix(string: str, suffix: str) -> str:
    if string.endswith(suffix):
        return string[:-len(suffix)]
    else:
        return string

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
    "circonstance":f"{SCHTROUMPF_STR}onstance",
    "collimateur":f"collima{SCHTROUMPF_STR}eur",
    "connerie":f"{SCHTROUMPF_STR}erie",
    "barbecue":f"barbe{SCHTROUMPF_STR}",
    "bouchon":f"bou{SCHTROUMPF_STR}",
    "fratricide":f"{SCHTROUMPF_STR}icide",
    "flatteur":f"{SCHTROUMPF_STR}eur",
    "forgeron":f"{SCHTROUMPF_STR}eron",
    "menteur":f"{SCHTROUMPF_STR}eur",
    "montgolfière":f"mont{SCHTROUMPF_STR}ière",
    "nitroglycérine":f"nitroglycé{SCHTROUMPF_STR}",
    "parapluie":f"para{SCHTROUMPF_STR}",
    "pistolet":f"pisto{SCHTROUMPF_STR}",
    "plumage":f"{SCHTROUMPF_STR}age",
    "pneumonie":f"pneumo{SCHTROUMPF_STR}",
    "question":f"{SCHTROUMPF_STR}", #To prevent "quesschtrompf"
    "soporifique":f"sopori{SCHTROUMPF_STR}",
    "symphonie":f"{SCHTROUMPF_STR}onie",
    "trombone":f"{SCHTROUMPF_STR}bone"}

SPECIAL_SMURF_ADJECTIVES: Dict[str, str] = {
    "esthétique":f"esthéti{SCHTROUMPF_STR}",
    "formidable":f"formi{SCHTROUMPF_STR}",
    "fratricide":f"{SCHTROUMPF_STR}icide",
    "universel":f"univer{SCHTROUMPF_STR}"
}

SPECIAL_SMURF_VERBS: Dict[str, str] = {
    "défaire":f"dé{SCHTROUMPF_STR}er",
    "démonter":f"dé{SCHTROUMPF_STR}er",
    "démolir":f"dé{SCHTROUMPF_STR}er",
    "entraider":f"entre{SCHTROUMPF_STR}er",
    "emmerder":f"en{SCHTROUMPF_STR}er",
    "rammener":f"re{SCHTROUMPF_STR}er",
    "recommencer":f"re{SCHTROUMPF_STR}er",
    "redire":f"re{SCHTROUMPF_STR}er",
    "redonner":f"re{SCHTROUMPF_STR}er",
    "regagner":f"re{SCHTROUMPF_STR}er",
    "remettre":f"re{SCHTROUMPF_STR}er",
    "retourner":f"re{SCHTROUMPF_STR}er",
    "revenir":f"re{SCHTROUMPF_STR}er",
    "retrouver":f"re{SCHTROUMPF_STR}er"}

SPECIAL_SMURF_WORDS: Dict[str, str] = {
    "atchoum":f"a{SCHTROUMPF_STR}",
    "bonjour":f"bon{SCHTROUMPF_STR}",
    "cocorico":f"cocori{SCHTROUMPF_STR}",
    "eureka":f"eure{SCHTROUMPF_STR}",
    "eurêka":f"eurê{SCHTROUMPF_STR}",
    "messieurs":f"mes{SCHTROUMPF_STR}s",
    "ramage":f"{SCHTROUMPF_STR}age",
    "sapristi":f"sapri{SCHTROUMPF_STR}",
    "tronçonneuse":f"{SCHTROUMPF_STR}onneuse"}


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
Ajd_ant_ends = re.compile("ant[e]?[s]?$")

# Bases classes used by "to_smurf" functions to ba able to change backing NLP library easily (spacy, stanza..)
class TokenAdapter:
    @property
    def text(self) -> str:
        raise NotImplementedError()

    @property
    def pos(self) -> BasicPOS:
        raise NotImplementedError()

    @property
    def tense(self) -> FrenchTense:
        raise NotImplementedError()

    @property
    def person(self) -> int:  # 1..3
        raise NotImplementedError()

    def is_plural(self) -> bool:
        raise NotImplementedError()

    def plural_suffix(self) -> str:
        return "s" if self.is_plural() else ""

    def is_feminine(self) -> bool:
        raise NotImplementedError()

    @property
    def lemma(self) -> str:
        raise NotImplementedError()

    @property
    def start_char(self) -> int:
        raise NotImplementedError()

    @property
    def end_char(self) -> int:
        raise NotImplementedError()

    def __str__(self):
        string = f"{self.text}:{remove_prefix(str(self.pos), 'BasicPOS.')},"
        string += (remove_prefix(str(self.tense), "FrenchTense.") + ",") if self.tense != FrenchTense.NONE else ""
        if self.person != 0:
            string += str(self.person)
        string += "f" if self.is_feminine() else "m"
        string += "p" if self.is_plural() else "s"
        if self.lemma:
            string += f"[{self.lemma}]"
        return string


    def can_smurf(self) -> bool:
        # Already "smurfed" words must be left untouched (often proper nouns, e.g. "Grand Schtroumpf")
        if SCHTROUMPF_STR in self.text.lower():
            return False

        if self.text.lower() in SPECIAL_SMURF_WORDS:
            return True

        pos = self.pos
        return (pos == BasicPOS.NOUN
                or
                pos == BasicPOS.ADJECTIVE
                or
                pos == BasicPOS.ADVERB and self.text.endswith("ment")
                or
                pos == BasicPOS.VERB and self.lemma not in Non_smurf_verbs
                or
                pos == BasicPOS.INTERJECTION)

    def to_smurf(self, word_compound_index=-1) -> str:
        pos = self.pos
        if not self.can_smurf():
            return self.text

        if self.text.lower() in SPECIAL_SMURF_WORDS:
            new_text = SPECIAL_SMURF_WORDS[self.text.lower()]
        elif pos == BasicPOS.NOUN:
            if self.lemma in SPECIAL_SMURF_NOUNS:
                new_text = SPECIAL_SMURF_NOUNS[self.lemma]
            else:
                m = re.search(r"tion(s)?$", self.text)
                if m:
                    new_text = self.text[:m.span()[0]] + SCHTROUMPF_STR
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
            if self.lemma in SPECIAL_SMURF_VERBS:
                verb_stem = SPECIAL_SMURF_VERBS[self.lemma][:-2]
            new_text = prefix + conjugate_1st_group_verb(verb_stem,
                                                         self.tense,
                                                         self.person,
                                                         self.is_feminine(),
                                                         self.is_plural())
        elif pos == BasicPOS.ADVERB:
            new_text = SCHTROUMPF_STR + "ement"
        elif pos == BasicPOS.ADJECTIVE:
            if self.lemma in SPECIAL_SMURF_ADJECTIVES:
                new_text = SPECIAL_SMURF_ADJECTIVES[self.lemma]
            else:
                m = re.search(Ajd_ant_ends, self.text)
                suffix = m.group(0) if m else self.plural_suffix()

                new_text = SCHTROUMPF_STR + suffix

        elif pos == BasicPOS.INTERJECTION:
            new_text = SCHTROUMPF_STR
        else:
            new_text = self.text

        if self.text and self.text.isupper():
            new_text = new_text.upper()
        elif self.text and self.text[0].isupper():
            new_text = new_text[0].upper() + (new_text[1:] if len(new_text) > 1 else "")

        return new_text

class SentenceAdapter:
    def __init__(self, tokens):
        self._tokens = tokens

    @property
    def tokens(self) -> List[TokenAdapter]:
        return self._tokens

    def add_leff_decorator(self, lefff: Lexicon):
        self._tokens = list(map(lambda token: LefffTokenAdapter(token, lefff), self._tokens))

class DocAdapter:
    @property
    def text(self) -> str:
        raise NotImplementedError()

    @property
    def sentences(self) -> List[SentenceAdapter]:
        raise NotImplementedError()

    def add_leff_decorator(self, lefff: Lexicon):
        for sentence in self.sentences:
            sentence.add_leff_decorator(lefff)



WHOLE_WORD=-1
def doc_to_smurf(doc : DocAdapter,
                 nlp,
                 adapter_doc_cls: Callable[[Any], DocAdapter],
                 smurf_indexes: Optional[Dict[Tuple[int, int], int]] = None,
                 tagger_details: List[str] = None):
    text = doc.text
    smurf_text = ""
    last_token_end_char = 0
    for s, sentence in enumerate(doc.sentences):
        for n, token in enumerate(sentence.tokens):
            smurf_text += text[last_token_end_char:token.start_char]
            last_token_end_char = token.end_char
            if smurf_indexes is None or (s, n) in smurf_indexes:
                #Handle compound words
                index_in_compound_word = WHOLE_WORD
                if smurf_indexes:
                    index_in_compound_word = smurf_indexes[(s,n)]

                if index_in_compound_word == WHOLE_WORD:
                    smurf_word = token.to_smurf()
                    if tagger_details is not None:
                        tagger_details.append(str(token))
                else:
                    parts = token.text.split("-")
                    subtext = parts[index_in_compound_word]
                    subtoken = adapter_doc_cls(nlp(subtext)).sentences[0].tokens[0]
                    smurf_subword = subtoken.to_smurf()
                    if tagger_details is not None:
                        tagger_details.append(str(subtoken))
                    parts[index_in_compound_word] = smurf_subword
                    smurf_word = "-".join(parts)

                # Handling apostrophes (') issues (me vs m' etc.)
                # (e.g. "l'alouette" = "la alouette" while "l'oiseau" = "le oiseau")
                if re.search("['’]$", smurf_text) and not re.match("[aeiou]", smurf_word.lower()):
                    suffix = "e "
                    if n > 0:
                        previous_token = sentence.tokens[n - 1]
                        if previous_token.is_feminine():
                            suffix = "a "
                    smurf_text = smurf_text[:-1] + suffix
                smurf_text += smurf_word
            else:
                smurf_text += text[token.start_char:last_token_end_char]
    smurf_text += text[last_token_end_char:]

    return smurf_text

Lefff_to_BasicPOS: Dict[str, BasicPOS] = {"nc": BasicPOS.NOUN,
                                          "n": BasicPOS.NOUN,
                                          "adj": BasicPOS.ADJECTIVE,
                                          "adv": BasicPOS.ADVERB,
                                          "auxAvoir": BasicPOS.AUXILIARY,
                                          "auxEtre": BasicPOS.AUXILIARY,
                                          "v": BasicPOS.VERB}

Lefff_to_fr_tense: Dict[str, FrenchTense] = {"P": FrenchTense.PRESENT,
                                             "F": FrenchTense.FUTUR,
                                             "I": FrenchTense.IMPARFAIT,
                                             "J": FrenchTense.PASSE_SIMPLE,
                                             "C": FrenchTense.COND_PRESENT,
                                             "Y": FrenchTense.IMPERATIF,
                                             "S": FrenchTense.SUBJ_PRESENT,
                                             "T": FrenchTense.SUBJ_IMPARFAIT,
                                             "K": FrenchTense.PARTICIPE_PASSE,
                                             "G": FrenchTense.GERONDIF,
                                             "W": FrenchTense.INFINITIF}

@dataclass
class TokenFeatures:
    pos: BasicPOS = BasicPOS.OTHER
    tense: FrenchTense = FrenchTense.NONE
    is_feminine: bool = False
    is_plural: bool = False
    person: int = 0
    lemma: str = ""


lefff_feats_regex: Pattern[AnyStr] = re.compile(r"^(?:.*_)?([PFIJCYSTKGW]*)([123]*)?([mf])?([sp])?$")


def lefff_code_to_feats(lefff_code: str) -> List[TokenFeatures]:
    m = re.match(lefff_feats_regex, lefff_code)
    if not m:
        print("Invalid lefff code: " + lefff_code)
        return []

    tenses = m.group(1)
    persons = m.group(2)
    feminine = m.group(3)
    plural = m.group(4)

    feats = []
    for tense in tenses:
        feats.append(TokenFeatures(tense=Lefff_to_fr_tense[tense]))

    if not feats:
        feats.append(TokenFeatures())

    if feminine and "f" in feminine:
        for feat in feats:
            feat.is_feminine = True

    if not plural and not tenses:
        feats_copy = list(map(copy.copy, feats))
        for feat in feats_copy:
            feat.is_plural = True
        feats += feats_copy
    elif plural and "p" in plural:
        for feat in feats:
            feat.is_plural = True

    if persons:
        additional_persons_feats = []
        for n, p in enumerate(persons):
            if n == 0:
                for feat in feats:
                    feat.person = int(p)
            else:
                feats_copy = list(map(copy.copy, feats))
                for feat in feats_copy:
                    feat.person = int(p)
                additional_persons_feats += feats_copy

        feats += additional_persons_feats

    return feats


def read_lefff_dict(path) -> Lexicon:
    result: Lexicon = {}
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile,
                                fieldnames=["form", "pos", "lemma", "feats"],
                                delimiter='\t',
                                quoting=csv.QUOTE_NONE)
        for row in reader:
            try:
                result[row["form"]].append(row)
            except KeyError:
                result[row["form"]] = [row]
    return result


def get_token_features_from_lefff(token: str, lefff: Lexicon) -> List[TokenFeatures]:
    token_features = []
    token_lower = token.lower()
    try:
        matching_rows = lefff[token_lower]
    except KeyError:
        return []
    for row in matching_rows:
        row_features: List[TokenFeatures] = []
        feats = row["feats"]
        if feats and pandas.notna(feats) and feats != "e":
            row_features = lefff_code_to_feats(feats)
        else:
            row_features.append(TokenFeatures())

        for feat in row_features:
            feat.lemma = row["lemma"]
            if row["pos"] in Lefff_to_BasicPOS:
                feat.pos = Lefff_to_BasicPOS[row["pos"]]
            else:
                feat.pos = BasicPOS.OTHER
        token_features += row_features

    return token_features

def get_token_feature_default_sort_key(feat: TokenFeatures) -> Tuple[int, int]:
    return (feat.tense != FrenchTense.PRESENT, # Priority to present
            feat.person != 3) # Priority to 3rd person

def get_token_feature_sort_key_with_token(feat: TokenFeatures, token: TokenAdapter) -> Tuple[int, int, Tuple[int, int]]:
    # Score possible feats using the following key:
    # (pos_match_p,
    #  tense_match_p + person_match_p ^ is_plural_match_p ^ is_feminine_match_p + lemma_match_p
    #  get_token_feature_sort_key)
    tense_match_p = (feat.tense == token.tense)
    person_match_p = (feat.person == token.person)
    plural_match_p = (feat.is_plural == token.is_plural())
    fem_match_p = (feat.is_feminine == token.is_feminine())
    full_person_match_p = person_match_p and plural_match_p and fem_match_p
    lemma_match_p = (feat.lemma == token.lemma)

    return (feat.pos != token.pos,
            -(tense_match_p + full_person_match_p + lemma_match_p),
            get_token_feature_default_sort_key(feat)) # Priority to 3rd person


class LefffTokenAdapter(TokenAdapter):
    def __init__(self, token: TokenAdapter, lefff: Lexicon):
        self.token = token
        possible_feats = get_token_features_from_lefff(token.text, lefff)
        if len(possible_feats) == 1:
            self.chosen_feat = possible_feats[0]
        elif len(possible_feats) == 0:
            self.chosen_feat = TokenFeatures(pos=token.pos,
                                             tense=token.tense,
                                             person=token.person,
                                             lemma=token.lemma,
                                             is_plural=token.is_plural(),
                                             is_feminine=token.is_feminine())
        else:
            self.chosen_feat = min(possible_feats, key=lambda feat: get_token_feature_sort_key_with_token(feat, token))

    @property
    def text(self) -> str:
        return self.token.text

    @property
    def pos(self) -> BasicPOS:
        return self.chosen_feat.pos

    @property
    def tense(self) -> FrenchTense:
        return self.chosen_feat.tense

    @property
    def person(self) -> int:  # 1..3
        return self.chosen_feat.person

    def is_plural(self) -> bool:
        return self.chosen_feat.is_plural

    def is_feminine(self) -> bool:
        return self.chosen_feat.is_feminine

    @property
    def lemma(self) -> str:
        return self.chosen_feat.lemma

    @property
    def start_char(self) -> int:
        return self.token.start_char

    @property
    def end_char(self) -> int:
        return self.token.end_char

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
            return FrenchTense.IMPARFAIT
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


class SpacyTokenAdapter(TokenAdapter):
    def __init__(self, token: SpacyToken):
        self.token = token
        features = str(token.morph)
        self.spacy_tag = token.pos_
        self.features = dict([x.split("=") for x in features.split("|") if "=" in x])

    @property
    def text(self) -> str:
        return self.token.text

    @property
    def pos(self) -> BasicPOS:
        try:
            return UTag_to_BasicPOS[self.spacy_tag]
        except KeyError:
            return BasicPOS.OTHER

    @property
    def tense(self) -> FrenchTense:
        return ufeats_to_fr_tense(self.features)

    @property
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

    @property
    def lemma(self) -> str:
        return self.token.lemma_

    @property
    def start_char(self) -> int:
        return self.token.idx

    @property
    def end_char(self) -> int:
        return self.token.idx + len(self.token)


class SpacyDocAdapter(DocAdapter):

    def __init__(self, doc: SpacyDoc):
        self.doc = doc
        self._sentences = []
        current_sentence = None
        for token in doc:
            if token.is_sent_start or current_sentence is None:
                current_sentence = SentenceAdapter([])
                self._sentences.append(current_sentence)
            current_sentence.tokens.append(SpacyTokenAdapter(token))

    @property
    def sentences(self):
        return self._sentences

    @property
    def text(self):
        return self.doc.text


class StanzaTokenAdapter(TokenAdapter):
    def __init__(self, token: StanzaToken):
        self.token = token
        self.upos = token.words[0].upos
        features = token.words[0].feats
        if features:
            self.features = dict([x.split("=") for x in features.split("|") if "=" in x])
        else:
            self.features = {}

    @property
    def text(self) -> str:
        return self.token.text

    @property
    def pos(self) -> BasicPOS:
        try:
            return UTag_to_BasicPOS[self.upos]
        except KeyError:
            return BasicPOS.OTHER

    @property
    def tense(self) -> FrenchTense:
        return ufeats_to_fr_tense(self.features)

    @property
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

    @property
    def lemma(self) -> str:
        return self.token.words[0].lemma

    @property
    def start_char(self) -> int:
        return self.token.start_char

    @property
    def end_char(self) -> int:
        return self.token.end_char


class StanzaDocAdapter(DocAdapter):
    def __init__(self, doc: StanzaDoc):
        self.doc = doc
        self._sentences = []
        for sentence in doc.sentences:
            current_sentence = SentenceAdapter([])
            self._sentences.append(current_sentence)
            for token in sentence.tokens:
                current_sentence.tokens.append(StanzaTokenAdapter(token))

    @property
    def sentences(self):
        return self._sentences

    @property
    def text(self):
        return self.doc.text

@dataclass()
class UDPipeDoc:
    text: str
    sentences: List[udpipe.Sentence]

def udpipe_process_text(model: udpipe.Model, text: str):
    tok = model.newTokenizer(model.DEFAULT)
    tok.setText(text)

    sentences: List[udpipe.Sentence] = []
    sentence = udpipe.Sentence()
    error = udpipe.ProcessingError()
    while tok.nextSentence(sentence, error):
        sentences.append(sentence)
        sentence = udpipe.Sentence()
    for sentence in sentences:
        model.tag(sentence, model.DEFAULT, error)

    return UDPipeDoc(text, sentences)

class UDPipeTokenAdapter(TokenAdapter):
    def __init__(self, token: udpipe.Word, char_offset=0):
        self.token = token
        self.upos = token.upostag
        self._start_char = char_offset
        features = token.feats
        if features:
            self.features = dict([x.split("=") for x in features.split("|") if "=" in x])
        else:
            self.features = {}

    @property
    def text(self) -> str:
        return self.token.form

    @property
    def pos(self) -> BasicPOS:
        try:
            return UTag_to_BasicPOS[self.upos]
        except KeyError:
            return BasicPOS.OTHER

    @property
    def tense(self) -> FrenchTense:
        return ufeats_to_fr_tense(self.features)

    @property
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

    @property
    def lemma(self) -> str:
        return self.token.lemma

    @property
    def start_char(self) -> int:
        return self._start_char

    @property
    def end_char(self) -> int:
        return self._start_char + len(self.token.form)

class UDPipeDocAdapter(DocAdapter):
    def __init__(self, doc: UDPipeDoc):
        self.doc = doc
        self._sentences = []
        offset = 0
        for sentence in doc.sentences:
            current_sentence = SentenceAdapter([])
            self._sentences.append(current_sentence)
            i = 0
            multi_word_iter = iter(sentence.multiwordTokens)
            def get_next_multi_word():
                try:
                    return next(multi_word_iter)
                except StopIteration:
                    return None

            next_multi_word = get_next_multi_word()
            i = 1  # Skip "root" word
            while i < len(sentence.words):
                word = sentence.words[i]
                if next_multi_word and i == next_multi_word.idFirst:
                    token = next_multi_word
                    #  Just replace multi-words by their first word for tagging, they never can_smurf() in French
                    i = next_multi_word.idLast + 1
                    next_multi_word = get_next_multi_word()
                else:
                    token = word
                    i += 1

                offset += len(token.getSpacesBefore())
                current_sentence.tokens.append(UDPipeTokenAdapter(word, offset))
                offset += len(token.form) + len(token.getSpacesAfter())

    @property
    def sentences(self):
        return self._sentences

    @property
    def text(self):
        return self.doc.text


Cached_fr_models: Dict[str, Any] = {}
def get_fr_nlp_model(name: str):
    name = remove_suffix(name, "^lefff")
    global Cached_fr_models
    try:
        return Cached_fr_models[name]
    except KeyError:
        model = None
        if name == "stanza":
            model = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos,lemma') #, dir="/media/simon/DATA/stanza_resources")
        elif name == "spacy" or name == "spacy/large":
            model = spacy.load("fr_core_news_lg")
        elif name == "spacy" or name == "spacy/trf":
            model = spacy.load("fr_dep_news_trf")
        elif name == "spacy/medium":
            model = spacy.load("fr_core_news_md")
        elif name == "udpipe":
            udpipe_model = udpipe.Model.load("./french-gsd-ud-2.5-191206.udpipe")
            model = lambda text:udpipe_process_text(udpipe_model, text)

        if model is None:
            raise Exception(f"ERROR: model '{name}' not found")
        Cached_fr_models[name] = model

        return model


def get_doc_adapter_class(model_name: str, lefff: Lexicon=None):
    if model_name.startswith("stanza"):
        cls = StanzaDocAdapter
    elif model_name.startswith("spacy"):
        cls = SpacyDocAdapter
    elif model_name.startswith("udpipe"):
        cls = UDPipeDocAdapter
    else:
        raise Exception(f"ERROR: unknown model '{model_name}'")

    if lefff is None or not model_name.endswith("^lefff"):
        return cls
    else:
        def create_doc_adapter_and_apply_leff(adapter, doc, lefff: Lexicon):
            doc_adapt: DocAdapter = adapter(doc)
            doc_adapt.add_leff_decorator(lefff)
            return doc_adapt
        return lambda doc: create_doc_adapter_and_apply_leff(cls, doc, lefff)

def smurf_stanza(text: str, smurf_indexes: Optional[Dict[Tuple[int, int], int]] = None) -> str:
    nlp = get_fr_nlp_model("stanza")
    return doc_to_smurf(StanzaDocAdapter(nlp(text)), nlp, StanzaDocAdapter, smurf_indexes)


def load_smurf_dataset(data_dir="./data"):
    files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".txt")]
    return datasets.load_dataset('csv',
                                 data_files=files,
                                 column_names=["smurf", "french"],
                                 delimiter=";",
                                 quote_char=None,
                                 split="train")


def find_smurf_indexes(doc: DocAdapter):
    smurf_indexes = {}
    for s, sentence in enumerate(doc.sentences):
        for n, token in enumerate(sentence.tokens):
            if SCHTROUMPF_STR in token.text.lower():
                if "-" in token.text:
                    for i, subword in enumerate(token.text.split("-")):
                        if SCHTROUMPF_STR in subword:
                            smurf_indexes[(s, n)] = i
                            break
                else:
                    smurf_indexes[(s, n)] = WHOLE_WORD
    return smurf_indexes


def add_smurf_for_model_and_compare_label(row, doc_adapter, nlp, model_name):
    doc_smurf = doc_adapter(nlp(row["smurf"])) # label
    doc_fr = doc_adapter(nlp(row["french"]))
    smurf_indexes = find_smurf_indexes(doc_smurf)

    tagger_details = []
    to_smurf = doc_to_smurf(doc_fr, nlp, doc_adapter, smurf_indexes, tagger_details)
    row[model_name] = to_smurf
    row[model_name + "_ok"] = (to_smurf == doc_smurf.text)
    row[model_name + "_details"] = " ; ".join(tagger_details)

    return row


def test_models_on_smurf_dataset(model_names=["stanza"],
                                 data_dir="./data",
                                 lefff: Optional[Lexicon]=None,
                                 output_csv="test_smurf_dataset.csv"):
    dataset = load_smurf_dataset(data_dir)

    for model_name in model_names:
        start = time.process_time()
        nlp = get_fr_nlp_model(model_name)
        doc_adapter = get_doc_adapter_class(model_name, lefff)

        print("Evaluating " + model_name)
        dataset = dataset.map(lambda row: add_smurf_for_model_and_compare_label(row, doc_adapter, nlp, model_name),
                              load_from_cache_file=False)

        total_correct = len(dataset.filter(lambda row: row[model_name + '_ok'], load_from_cache_file=False))
        total_examples = len(dataset)
        print(f"\nTotal correct ({model_name}) = {total_correct}/{total_examples} "
              f"({total_correct/total_examples}%) "
              f"(time = {time.process_time() - start:.2f}s)")

    dataset.to_csv(output_csv)


def smurf_dataset_stats(model_name="stanza", data_dir="./data", lefff: Optional[Lexicon]=None):
    dataset = load_smurf_dataset()
    nlp = get_fr_nlp_model(model_name)
    doc_adapter = get_doc_adapter_class(model_name, lefff)
    nbr_of_smurf_words = 0
    nbr_of_can_smurf_words = 0
    nbr_of_words = 0
    for example in dataset["train"]:
        doc_smurf = doc_adapter(nlp(example["smurf"]))
        for sentence in doc_smurf.sentences:
            for token in sentence.tokens:
                nbr_of_words += 1
                if SCHTROUMPF_STR in token.text.lower():
                    nbr_of_smurf_words += 1
                elif token.can_smurf():
                    nbr_of_can_smurf_words += 1

    print(f"\nTotal words = {nbr_of_words}, smurf words = {nbr_of_smurf_words} ({nbr_of_smurf_words/nbr_of_words}%), "
          f"can_smurf = {nbr_of_can_smurf_words}, "
          f"smurf/(smurf + can) = {nbr_of_smurf_words/(nbr_of_can_smurf_words + nbr_of_smurf_words)}")


random.seed(42)
SMURF_VS_CAN_SMURF_RATIO=0.33


def random_smurf(text: str, model_name="stanza", lefff: Optional[Lexicon]=None):
    nlp = get_fr_nlp_model(model_name)
    doc_adapter = get_doc_adapter_class(model_name, lefff)
    doc = doc_adapter(nlp(text))

    smurf_indexes = {}
    for s, sentence in enumerate(doc.sentences):
        for n, token in enumerate(sentence.tokens):
            if token.can_smurf() and random.random() <= SMURF_VS_CAN_SMURF_RATIO:
                if "-" in token.text and token.pos == BasicPOS.NOUN:
                    subwords = token.text.split("-")
                    can_smurf_subwords = []
                    for i, subword in enumerate(subwords):
                        if subword and (doc_adapter(nlp(subword)).sentences[0].tokens[0]).can_smurf():
                            can_smurf_subwords.append(i)
                    if can_smurf_subwords:
                        smurf_indexes[(s, n)] = random.choice(can_smurf_subwords)
                    else:
                        smurf_indexes[(s, n)] = WHOLE_WORD
                else:
                    smurf_indexes[(s, n)] = WHOLE_WORD

    return doc_to_smurf(doc, nlp, doc_adapter, smurf_indexes)


def add_smurf(row, model_name, lefff):
    row['smurf'] = random_smurf(row["text"], model_name, lefff)
    return row


def convert_text_file_to_hf_dataset(files, result_path, model_name="stanza", lefff: Lexicon=None, checkpoint_steps=10000):
    first_index_to_process = 0
    input_dataset = datasets.load_dataset('text', data_files=files, split='train')
    processed_dataset = None
    if os.path.isdir(result_path):
        processed_dataset = datasets.load_from_disk(result_path)
        first_index_to_process = len(processed_dataset)

    previous_checkpoints = (first_index_to_process + checkpoint_steps - 1) // checkpoint_steps
    next_checkpoints = (len(input_dataset) - first_index_to_process + checkpoint_steps - 1) // checkpoint_steps
    total_checkpoints = previous_checkpoints + next_checkpoints

    print(f"Checkpoint 0/{total_checkpoints}")
    for n, i in enumerate(range(first_index_to_process, len(input_dataset), checkpoint_steps)):
        end_index = min(i + checkpoint_steps, len(input_dataset))
        dataset_split = input_dataset.select(range(i, end_index))
        processed_split = dataset_split.map(lambda row: add_smurf(row, model_name, lefff))
        if processed_dataset is None:
            processed_dataset = processed_split
        else:
            processed_dataset = datasets.concatenate_datasets([processed_dataset, processed_split])

        saved_old_results = ""
        if os.path.isdir(result_path):
            saved_old_results = result_path + ".saved"
            shutil.move(result_path, saved_old_results)

        print(f"Checkpoint {previous_checkpoints + n + 1}/{total_checkpoints}, saving in {result_path}... ", end="")
        processed_dataset.save_to_disk(result_path)
        print("done")

        if saved_old_results:
            shutil.rmtree(saved_old_results)



