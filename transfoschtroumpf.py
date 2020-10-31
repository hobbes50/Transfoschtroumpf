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
        self.base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.smurf_base_token = smurf_base_token
        self.space_symbol = space_symbol
        self.all_smurf_tokens = [smurf_base_token, self.space_symbol + self.smurf_base_token,
                                 self.space_symbol + self.smurf_base_token.capitalize()]
        super().__init__()

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

from transformers import AutoModel, RobertaConfig, RobertaModel, RobertaForCausalLM, RobertaForCausalLM, modeling_outputs

from torch import Tensor, nn
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
data = smurf_tok("Je me schtroumpferai jusqu'à la mort !", return_tensors="pt")
print(transfoschtroumpf(**data, labels=smurf_tok("Je me battrai jusqu'à la mort !", return_tensors="pt")["input_ids"]))
