from transformers import AutoTokenizer, PreTrainedTokenizer, CamembertTokenizer
from typing import List, Set, Dict, Tuple, Pattern, Optional

SENTENCE_PIECE_SPACE="▁"


class SmurfTokenizer(PreTrainedTokenizer):
    # vocab_files_names = tokenization_camembert.VOCAB_FILES_NAMES
    # pretrained_vocab_files_map = tokenization_camembert.PRETRAINED_VOCAB_FILES_MAP
    # max_model_input_sizes = tokenization_camembert.PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    # model_input_names = ["attention_mask"]

    def __init__(self, model_name="camembert-base", smurf_base_token="schtroumpf", space_symbol=SENTENCE_PIECE_SPACE,
                 **kwargs):
        super().__init__()
        self.base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.smurf_base_token = smurf_base_token
        self.space_symbol = space_symbol
        self.all_smurf_tokens = [smurf_base_token, self.space_symbol + self.smurf_base_token,
                                 self.space_symbol + self.smurf_base_token.capitalize()]
        self.pad_token = self.base_tokenizer.pad_token
        self.mask_token = self.base_tokenizer.mask_token
        self.eos_token = self.base_tokenizer.eos_token
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
            return self.all_smurf_tokens[index - self.base_tokenizer.vocab_size()]

    def convert_tokens_to_string(self, tokens):
        return self.base_tokenizer.convert_tokens_to_string(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        return self.base_tokenizer.save_vocabulary(save_directory)

    def _tokenize(self, text):
        smurf_index = text.lower().find(self.smurf_base_token)
        if (smurf_index < 0):
            return self.base_tokenizer.tokenize(text)
        else:
            end_smurf_index = smurf_index + len(self.smurf_base_token)
            is_first_word = smurf_index == 0
            after_space = smurf_index > 0 and text[smurf_index - 1] == " "
            previous_tokens = self.base_tokenizer.tokenize(text[:smurf_index - int(after_space)])
            smurf_token = text[smurf_index:end_smurf_index]
            if (is_first_word or after_space):
                smurf_token = self.space_symbol + smurf_token

            next_tokens = self.tokenize(text[end_smurf_index:])
            if (end_smurf_index + 1 < len(text)
                    and
                    text[end_smurf_index] != " "
                    and
                    next_tokens
                    and
                    next_tokens[0]
                    and
                    next_tokens[0][0] == self.space_symbol):
                next_tokens[0] = next_tokens[0][1:]
                if (next_tokens[0] == ""):
                    next_tokens.pop(0)
            return previous_tokens + [smurf_token] + next_tokens


from transformers import AutoModel, RobertaConfig, RobertaForCausalLM, modeling_outputs, Trainer, TrainingArguments

from torch import Tensor, nn, utils
class TransfoSchtroumpf(nn.Module):
      def __init__(self):
        super().__init__()

        #Add an encoder from a pretrained Camembert
        self.encoder = AutoModel.from_pretrained("camembert-base", add_pooling_layer=False)

        #Adjust embedding sizes
        self.encoder.config.vocab_size = self.encoder.config.vocab_size + 3
        self.encoder.resize_token_embeddings(self.encoder.config.vocab_size)

        #Add a lightweight decoder with a LM head
        config_decoder = RobertaConfig.from_pretrained("roberta-base")
        config_decoder.vocab_size = self.encoder.config.vocab_size
        config_decoder.is_decoder = True
        config_decoder.add_cross_attention = True
        config_decoder.num_hidden_layers = 1
        config_decoder.num_attention_heads = 1
        self.decoder = RobertaForCausalLM(config_decoder)
        self.decoder.roberta.embeddings = self.encoder.embeddings

      def forward(self, input_ids=None, labels=None, return_dict=None, **kwargs):
        outputs = self.decoder.roberta(input_ids, encoder_hidden_states=self.encoder(input_ids)[0])
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
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)


transfoschtroumpf = TransfoSchtroumpf()
#print(test)

smurf_tok = SmurfTokenizer()
#data = smurf_tok("Je me schtroumpferai jusqu'à la mort !", return_tensors="pt")
#print(transfoschtroumpf(**data, labels=smurf_tok("Je me battrai jusqu'à la mort !", return_tensors="pt")["input_ids"]))

import torch
class SmurfDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels['input_ids'][idx]
        return item

    def __len__(self):
        return len(self.labels['input_ids'])

all_raw_data = [("Je me schtroumpferai jusqu'à la mort !",
                   "Je me battrai jusqu'à la mort !"),
                  ("Qu'est-ce qu'il fait schtroumpf ! J'aurais dû prendre une écharpe !",
                   "Qu'est-ce qu'il fait froid ! J'aurais dû prendre une écharpe !"),
                  ("Il va falloir schtroumpfer par le gué !",
                   "Il va falloir passer par le gué !"),
                  ("Schtroumpfons une petite chanson de marche pour nous schtroumpfer du courage !",
                   "Chantons une petite chanson de marche pour nous donner du courage !"),
                  ("Trois petites lieues à pied, ça schtroumpfe, ça schtroumpfe !",
                   "Trois petites lieues à pied, ça use, ça use !"),
                  ("Il va nous schtroumpfer comme des souris !",
                   "Il va nous croquer comme des souris !"),
                  ("Pfff ! On l'a schtroumpfé belle !",
                   "Pfff ! On l'a échappé belle !")]

dataset_size = len(all_raw_data)
train_size = int(dataset_size*0.8)

#train_raw_data = all_raw_data[:train_size]
#eval_raw_data = all_raw_data[train_size:]
train_raw_data = all_raw_data
eval_raw_data = all_raw_data

train_encoded_inputs = smurf_tok([x for (x, y) in train_raw_data], padding=True, truncation=True)
train_encoded_labels = smurf_tok([y for (x, y) in train_raw_data], padding=True, truncation=True)

eval_encoded_inputs = smurf_tok([x for (x, y) in eval_raw_data], padding=True, truncation=True)
eval_encoded_labels = smurf_tok([y for (x, y) in eval_raw_data], padding=True, truncation=True)

train_dataset = SmurfDataset(train_encoded_inputs, train_encoded_labels)
eval_dataset = SmurfDataset(eval_encoded_inputs, eval_encoded_labels)

model = TransfoSchtroumpf()


from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1024,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_steps=10,
    logging_dir='./logs',            # directory for storing logs
)

trainer = Trainer(
    tokenizer=smurf_tok,
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=eval_dataset            # evaluation dataset
)

print("Training...")
trainer.train()
print("Evaluating...")
print(trainer.evaluate())
#print("Saving model...")
#model.save_pretrained("saved_models")
