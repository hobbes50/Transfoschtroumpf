from enum import Enum, auto
import random
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from dataclasses import dataclass
import spacy
from spacy.tokens import Token as SpacyToken, Doc as SpacyDoc
from datasets import load_dataset
import stanza
from stanza.models.common.doc import Token as StanzaToken, Document as StanzaDoc
import time
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
    "question":f"{SCHTROUMPF_STR}", #To prevent "quesschtrompf"
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

    def can_smurf(self) -> bool:
        # Already "smurfed" words must be left untouched (often proper nouns, e.g. "Grand Schtroumpf")
        if SCHTROUMPF_STR in self.text.lower():
            return False

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

        if pos == BasicPOS.NOUN:
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
                new_text = SCHTROUMPF_STR \
                           + ("ant" if self.text.endswith("ant") or self.text.endswith("ants") else "")

            new_text += self.plural_suffix()
        elif pos == BasicPOS.INTERJECTION:
            if self.text in SPECIAL_SMURF_INTERJ:
                new_text = SPECIAL_SMURF_INTERJ[self.text]
            else:
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

class DocAdapter:
    @property
    def text(self) -> str:
        raise NotImplementedError()

    @property
    def sentences(self) -> List[SentenceAdapter]:
        raise NotImplementedError()


WHOLE_WORD=-1
def doc_to_smurf(doc : DocAdapter,
                 nlp,
                 adapter_doc_cls: Callable[[Any], DocAdapter],
                 smurf_indexes: Optional[Dict[Tuple[int, int], int]] = None):
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
                else:
                    parts = token.text.split("-")
                    subtext = parts[index_in_compound_word]
                    subtoken = adapter_doc_cls(nlp(subtext)).sentences[0].tokens[0]
                    smurf_subword = subtoken.to_smurf()
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
        features = token.tag_.split("__")
        self.spacy_tag = token.pos_
        self.features = dict([x.split("=") for x in features[-1].split("|") if "=" in x])

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


Cached_fr_models: Dict[str, Any] = {}
def get_fr_nlp_model(name: str):
    global Cached_fr_models
    try:
        return Cached_fr_models[name]
    except KeyError:
        model = None
        if name == "stanza":
            model = stanza.Pipeline(lang='fr', processors='tokenize,mwt,pos,lemma')
        elif name == "spacy" or name == "spacy/large":
            model = spacy.load("fr_core_news_lg")
        elif name == "spacy" or name == "spacy/medium":
            model = spacy.load("fr_core_news_md")

        if model is None:
            raise Exception(f"ERROR: model '{name}' not found")
        Cached_fr_models[name] = model

        return model

def get_doc_adapter_class(model_name: str):
    if model_name.startswith("stanza"):
        return StanzaDocAdapter
    elif model_name.startswith("spacy"):
        return SpacyDocAdapter
    else:
        raise Exception(f"ERROR: unknown model '{model_name}'")

def smurf_stanza(text: str, smurf_indexes: Optional[Dict[Tuple[int, int], int]] = None) -> str:
    nlp = get_fr_nlp_model("stanza")
    return doc_to_smurf(StanzaDocAdapter(nlp(text)), nlp, StanzaDocAdapter, smurf_indexes)

def load_smurf_dataset(data_dir="./data"):
    files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".txt")]
    return load_dataset('csv',
                        data_files=files,
                        column_names=["smurf", "french"],
                        delimiter=";",
                        quote_char=None)

def test_smurf_dataset(model_name="stanza", data_dir="./data"):
    dataset = load_smurf_dataset()
    start = time.process_time()
    nlp = get_fr_nlp_model(model_name)
    doc_adapter = get_doc_adapter_class(model_name)
    nbr_of_examples = 0
    nbr_of_correct_examples = 0
    for example in dataset["train"]:
        nbr_of_examples += 1
        doc_smurf = doc_adapter(nlp(example["smurf"]))
        doc_fr = doc_adapter(nlp(example["french"]))
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

        to_smurf = doc_to_smurf(doc_fr, nlp, doc_adapter, smurf_indexes)
        if to_smurf == doc_smurf.text:
            nbr_of_correct_examples += 1
        else:
            print("==================ERROR==================")
            print("FRENCH  : " + doc_fr.text)
            print("TO_SMURF: " + to_smurf)
            print("LABEL   : " + doc_smurf.text)
            print("=========================================")

    print(f"\nTotal correct = {nbr_of_correct_examples}/{nbr_of_examples} "
          f"({nbr_of_correct_examples/nbr_of_examples}%) "
          f"(time = {time.process_time() - start:.2f}s)"
          )

def smurf_dataset_stats(model_name="stanza", data_dir="./data"):
    dataset = load_smurf_dataset()
    nlp = get_fr_nlp_model(model_name)
    doc_adapter = get_doc_adapter_class(model_name)
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
def random_smurf(text: str, model_name="stanza"):
    nlp = get_fr_nlp_model(model_name)
    doc_adapter = get_doc_adapter_class(model_name)
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

OSCAR_SMURF_CSV_SEPARATOR="ༀ"
def convert_oscar_file(filepath, start_line=1, model_name="stanza"):
    with open(filepath, 'r') as input_file:
        nbr_of_lines = 0
        while (input_file.readline()):
            nbr_of_lines += 1
    print(f"{nbr_of_lines} lines to process in {filepath}")

    with open(filepath, 'r') as input_file:
        output_filename = os.path.join(os.path.dirname(filepath),
                                       "smurf_" + os.path.basename(filepath))
        line_number = 0
        with open(output_filename, 'w' if start_line <= 1 else "a") as output_file:
            while True:
                line = input_file.readline().rstrip('\n')
                if not line:
                    break
                line_number += 1
                if line_number < start_line:
                    continue

                if (line_number % 1000) == 0:
                    print(f"{line_number} lines processed {100*line_number/nbr_of_lines}%")
                try:
                    smurf_line = random_smurf(line, model_name)
                except:
                    print(f"ERROR converting sentence {line}")
                    continue
                output_file.write(str(line_number) + OSCAR_SMURF_CSV_SEPARATOR
                                  + line + OSCAR_SMURF_CSV_SEPARATOR
                                  + smurf_line + "\n")
