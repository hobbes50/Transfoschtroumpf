from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, Trainer, TrainingArguments, PreTrainedModel,\
    EncoderDecoderModel, EncoderDecoderConfig, DataCollatorWithPadding
from transformers import RobertaConfig, RobertaForCausalLM, CamembertModel, modeling_outputs
import transformers
from datasets import load_dataset
from torch import nn, utils
import torch
from typing import List, Set, Dict, Tuple, Pattern, Optional, Union
import os
from dataclasses import dataclass

SENTENCE_PIECE_SPACE="▁"



NBR_OF_SMURF_TOKENS=3  #_schtroumpf, _Schtroumpf, ##schtroumpf (in-word)
class SmurfTokenizer(PreTrainedTokenizer):
    # vocab_files_names = tokenization_camembert.VOCAB_FILES_NAMES
    # pretrained_vocab_files_map = tokenization_camembert.PRETRAINED_VOCAB_FILES_MAP
    # max_model_input_sizes = tokenization_camembert.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # model_input_names = ["attention_mask"]

    def __init__(self, model_name="camembert-base", smurf_base_token="schtroumpf", space_symbol=SENTENCE_PIECE_SPACE,
                 **kwargs):
        super().__init__()
        self.base_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.smurf_base_token = smurf_base_token
        self.space_symbol = space_symbol
        self.all_smurf_tokens = [smurf_base_token, self.space_symbol + self.smurf_base_token,
                                 self.space_symbol + self.smurf_base_token.capitalize()]
        self.pad_token = self.base_tokenizer.pad_token
        self.mask_token = self.base_tokenizer.mask_token
        self.eos_token = self.base_tokenizer.eos_token
        self.bos_token = self.base_tokenizer.bos_token
        self.sep_token = self.base_tokenizer.sep_token
        self.cls_token = self.base_tokenizer.cls_token
        self.unk_token = self.base_tokenizer.unk_token
        self.model_max_length = self.base_tokenizer.model_max_length

    def build_inputs_with_special_tokens(self, token_ids_0: List, token_ids_1: Optional[List] = None) -> List:
        return self.base_tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1)

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        return self.base_tokenizer.get_special_tokens_mask(token_ids_0, token_ids_1)

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return self.base_tokenizer.create_token_type_ids_from_sequences(token_ids_0, token_ids_1)

    @property
    def vocab_size(self):
        return self.base_tokenizer.vocab_size() + len(self.all_smurf_tokens)

    def _convert_token_to_id(self, token):
        if token in self.all_smurf_tokens:
            return self.base_tokenizer.vocab_size + self.all_smurf_tokens.index(token)
        else:
            return self.base_tokenizer._convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        if index < self.base_tokenizer.vocab_size:
            return self.base_tokenizer._convert_id_to_token(index)
        else:
            return self.all_smurf_tokens[index - self.base_tokenizer.vocab_size]

    def convert_tokens_to_string(self, tokens):
        return self.base_tokenizer.convert_tokens_to_string(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        return self.base_tokenizer.save_vocabulary(save_directory)

    def remove_spaces_in_first_token(self, tokens):
        if (tokens
                and
                tokens[0]
                and
                tokens[0][0] == self.space_symbol):
            tokens[0] = tokens[0][1:]
            if tokens[0] == "":
                tokens.pop(0)

    def _tokenize(self, text) -> List[str]:
        tokenized_text = []
        offset = 0
        inside_smurf_word = False
        while offset < len(text):
            smurf_index = text.lower().find(self.smurf_base_token, offset)
            if smurf_index < 0:
                smurf_index = len(text)
                end_smurf_index = len(text)
                smurf_token = None
                after_space = False
            else:
                end_smurf_index = smurf_index + len(self.smurf_base_token)
                is_first_word = smurf_index == 0
                after_space = smurf_index > 0 and text[smurf_index - 1] == " "
                smurf_token = text[smurf_index:end_smurf_index]
                if is_first_word or after_space:
                    smurf_token = self.space_symbol + smurf_token

            previous_tokens = self.base_tokenizer.tokenize(text[offset:smurf_index - int(after_space)])

            if inside_smurf_word:
                self.remove_spaces_in_first_token(previous_tokens)
                inside_smurf_word = False

            if end_smurf_index < len(text) and text[end_smurf_index] != " ":
                inside_smurf_word = True

            tokenized_text += previous_tokens
            if smurf_token is not None:
                tokenized_text.append(smurf_token)
            offset = end_smurf_index

        return tokenized_text


class TransfoSchtroumpf(EncoderDecoderModel):

    def __init__(self, tokenizer: SmurfTokenizer, config=None):

        encoder: CamembertModel = AutoModel.from_pretrained("camembert-base", add_pooling_layer=False)
        encoder.config.vocab_size += 3
        encoder.resize_token_embeddings(encoder.config.vocab_size)
        self.config = encoder.config
        self.config.is_encoder_decoder = True

        super().__init__(self.config)
        self.encoder = encoder

        # Add an encoder from a pretrained Camembert

        # Adjust embedding sizes
        self.encoder.config.vocab_size = self.encoder.config.vocab_size + 3
        self.encoder.resize_token_embeddings(self.encoder.config.vocab_size)

        # Add a lightweight decoder with a LM head
        config_decoder = RobertaConfig.from_pretrained("roberta-base")
        config_decoder.vocab_size = self.encoder.config.vocab_size
        config_decoder.is_decoder = True
        config_decoder.add_cross_attention = True
        config_decoder.num_hidden_layers = 1
        config_decoder.num_attention_heads = 1
        self.decoder = RobertaForCausalLM(config_decoder)
        self.decoder.roberta.embeddings = self.encoder.embeddings

    def forward(self, input_ids=None, decoder_input_ids=None, labels=None, return_dict=None, **kwargs):
        if decoder_input_ids is None:
            if labels is not None:
                decoder_input_ids = labels

        outputs = self.decoder.roberta(decoder_input_ids, encoder_hidden_states=self.encoder(input_ids)[0])
        prediction_scores = self.decoder.lm_head(outputs[0])

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.encoder.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return modeling_outputs.CausalLMOutput(
            loss=lm_loss,
            logits=prediction_scores,
            hidden_states=outputs[0])

    def get_input_embeddings(self):
        return self.encoder.embeddings.word_embeddings

    def get_output_embeddings(self):
        return self.decoder.lm_head.decoder

def get_transfoschtroumpf(tokenizer, base_model="camembert-base",
                          decoder_nbr_of_hidden_layers=1,
                          freeze_encoder=True,
                          decoder_nbr_of_heads=1):
    #config = EncoderDecoderConfig.from_encoder_decoder_configs(base_model, base_model)
    model : EncoderDecoderModel = EncoderDecoderModel.from_encoder_decoder_pretrained(base_model, base_model)

    model.encoder.config.vocab_size += NBR_OF_SMURF_TOKENS
    model.encoder.resize_token_embeddings(model.encoder.config.vocab_size)
    model.decoder.config.vocab_size += NBR_OF_SMURF_TOKENS
    model.decoder.resize_token_embeddings(model.decoder.config.vocab_size)

    encoder: CamembertModel = model.encoder
    decoder: CamembertModel = model.decoder
    mask_token_id = tokenizer.base_tokenizer.mask_token_id

    # Initial embedding of smurf word = <mask> embedding
    for i in range(encoder.config.vocab_size - NBR_OF_SMURF_TOKENS, encoder.config.vocab_size):
        encoder.get_input_embeddings().weight[i].data.copy_(encoder.get_input_embeddings().weight[mask_token_id].data)
        decoder.get_input_embeddings().weight[i].data.copy_(decoder.get_input_embeddings().weight[mask_token_id].data)

    if freeze_encoder:
        for name, param in encoder.named_parameters():
            print(f"freezing {name}")
            param.requires_grad = False

    # Add a lightweight decoder with a LM head
    #config.decoder.is_decoder = True
    #config.decoder.add_cross_attention = True
    #config.decoder.num_hidden_layers = 1
    #config.decoder.num_attention_heads = 1
    old_nbr_of_hidden_layers = model.decoder.config.num_hidden_layers
    if decoder_nbr_of_hidden_layers != old_nbr_of_hidden_layers:
        step = old_nbr_of_hidden_layers // decoder_nbr_of_hidden_layers

        model.decoder.config.num_hidden_layers = decoder_nbr_of_hidden_layers
        old_hidden_layers = model.decoder.roberta.encoder.layer
        new_hidden_layers = nn.ModuleList()
        layer_to_keep = 0
        for i in range(decoder_nbr_of_hidden_layers - 1):
            new_hidden_layers.append(old_hidden_layers[layer_to_keep])
            layer_to_keep += step

        new_hidden_layers.append(old_hidden_layers[-1])

        model.decoder.roberta.encoder.layer = nn.ModuleList(new_hidden_layers)

    #old_nbr_of_attention_heads = model.decoder.config.num_attention_heads
    #if decoder_nbr_of_heads != old_nbr_of_attention_heads:
    #    model.decoder.prune_heads({i: list(range(old_nbr_of_attention_heads - decoder_nbr_of_heads))
    #                               for i in range(decoder_nbr_of_hidden_layers)})
    #    model.decoder.config.num_attention_heads = decoder_nbr_of_heads

    #print(model)

    # Needed by generation step
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.config.encoder = model.encoder.config
    model.config.decoder = model.decoder.config
    model.config.vocab_size = model.config.encoder.vocab_size

    return model

def prepare_sentence(tokenizer, data_sample, from_language="smurf", to_language="french"):
    item = tokenizer(data_sample[from_language], max_length=256, truncation=True)
    item['labels'] = tokenizer(data_sample[to_language], max_length=256, truncation=True)["input_ids"]
    #item["decoder_input_ids"] = tokenizer(data_sample[to_language])["input_ids"]
    return item

import csv
csv.field_size_limit(2 << 30)

def get_dataset(tokenizer, data_dir="./data"):
    #files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".txt")]
    files = ["/home/simon/Downloads/OSCAR/smurf_fr_part_1.txt"]
    dataset = load_dataset('csv',
                           data_files=files,
                           column_names=["index", "french", "smurf"],
                           delimiter="ༀ",
                           quoting=3,  # Disable quoting
                           decimal=",",
                           doublequote=False,
                           error_bad_lines=False)
    dataset = dataset.map(lambda x: prepare_sentence(tokenizer, x))
    dataset = dataset["train"].train_test_split(test_size=0.1)
    dataset.set_format(type='torch')
    dataset.save_to_disk("./data/dataset_oscar_1_tok")
    return dataset

class SmurfTrainer(Trainer):
    def log(self, logs: Dict[str, float]) -> None:
        test_sentence = "Je me schtroumpferai jusqu'à la mort !"
        test_ids = self.tokenizer(test_sentence, return_tensors="pt")["input_ids"]
        generated = self.model.generate(input_ids=test_ids)
        result_sentence = self.tokenizer.decode(generated[0])
        logs = {**logs, f"'{test_sentence}'": result_sentence}
        super().log(logs)


@dataclass
class EncoderDecoderCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        features_decoder = [{"input_ids":sample["labels"]} for sample in features]
        for sample in features:
            del sample["labels"]
        batch = self.tokenizer.pad(features,
                                   padding=self.padding,
                                   max_length=self.max_length,
                                   pad_to_multiple_of=self.pad_to_multiple_of,
                                   return_tensors="pt")

        batch_labels = self.tokenizer.pad(features_decoder,
                                          padding=self.padding,
                                          max_length=self.max_length,
                                          pad_to_multiple_of=self.pad_to_multiple_of,
                                          return_tensors="pt")

        batch["labels"] = batch_labels["input_ids"].copy()
        # -100: special value to ignore <pad> tokens when computing the loss
        batch["labels"] = [[-100 if token == self.tokenizer.pad_token_id else token for token in labels]
                           for labels in batch["labels"]]

        batch["decoder_input_ids"] = batch_labels["input_ids"]

        return batch

def train_transmoschtroumpf():
    smurf_tok = SmurfTokenizer()

    #model = get_transfoschtroumpf(smurf_tok, decoder_nbr_of_hidden_layers=3)
    transformers.logging.set_verbosity_info()
    model = EncoderDecoderModel.from_pretrained("./results/checkpoint-15000")
    model.train()
    #dataset = datasets.load_from_disk("./data/dataset_oscar_1_tok")
    dataset = get_dataset(smurf_tok)

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=1,              # total # of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=8,   # batch size for evaluation
        warmup_steps=256,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_steps=10,
        logging_dir='./logs'            # directory for storing logs
    )

    trainer = SmurfTrainer(
        tokenizer=smurf_tok,
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=dataset["train"],         # training dataset
        eval_dataset=dataset["test"],
        data_collator=EncoderDecoderCollator(smurf_tok))

    print("Training...")
    trainer.train("./results/checkpoint-15000")
    print("Saving model...")
    model.save_pretrained("saved_models")
    print("Evaluating...")
    print(trainer.evaluate())

if __name__ == '__main__':
    train_transmoschtroumpf()
