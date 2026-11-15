import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5TokenizerFast
import torch

from sql_processing import encode_sql, validate_sql_preprocessing

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split, sql_preprocessing = "none"):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output.
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

        # SQL processing: retain the selected mode for every split of this dataset
        self.sql_preprocessing = validate_sql_preprocessing(sql_preprocessing)

        self.encoder_ids, self.target_ids = self.process_data(data_folder, split, self.tokenizer, sql_preprocessing = self.sql_preprocessing)

    def process_data(self, data_folder, split, tokenizer, sql_preprocessing = "none"):
        # TODO
        # SQL processing: also validate test datasets, which have no SQL targets.
        validate_sql_preprocessing(sql_preprocessing)

        # Read the natural language inputs:
        input_path = os.path.join(data_folder, f"{split}.nl")
        inputs = load_lines(input_path)

        # Read the SQL targeted outputs:
        # train/dev set have SQL targets, but test does not
        if split == "test":
            targets = None
        else:
            target_path = os.path.join(data_folder, f"{split}.sql")
            targets = load_lines(target_path)

            if len(inputs) != len(targets):
                raise ValueError(f"{split}: the numbers of inputs and targets differ!")

        # Tokenize inputs without padding or truncation:
        encoder_ids = tokenizer(
            inputs,
            add_special_tokens = True,
            padding = False,
            truncation = False
        )["input_ids"]

        # SQL processing: encode train/dev targets before tokenization
        if targets is not None:
            targets = [encode_sql(sql, sql_preprocessing) for sql in targets]

        if targets is None:
            target_ids = None
        else:
            target_ids = tokenizer(
                targets,
                add_special_tokens = True,
                padding = False,
                truncation = False
            )["input_ids"]

        return encoder_ids, target_ids

    def __len__(self):
        # TODO
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        # TODO
        encoder_ids = torch.tensor(self.encoder_ids[idx], dtype = torch.long)

        if self.target_ids is None:
            return encoder_ids

        target_ids = torch.tensor(self.target_ids[idx], dtype = torch.long)

        return encoder_ids, target_ids


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    # Each example contains (encoder_ids, target_ids):
    encoder_sequences, target_sequences = zip(*batch)

    encoder_ids = pad_sequence(encoder_sequences, batch_first = True, padding_value = PAD_IDX)

    encoder_mask = (encoder_ids != PAD_IDX).long()

    decoder_targets = pad_sequence(target_sequences, batch_first = True, padding_value = PAD_IDX)

    # Apply teacher-forcing during training (shift-by-one)
    # Shift targets right, beginning with T5's start token:
    decoder_inputs = torch.full_like(decoder_targets, PAD_IDX)
    decoder_inputs[:, 1:] = decoder_targets[:, :-1]

    initial_decoder_inputs = decoder_inputs[:, :1].clone()

    return (
        encoder_ids,
        encoder_mask,
        decoder_inputs,
        decoder_targets,
        initial_decoder_inputs
    )

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns:
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    # Each test example contains only encoder_ids:
    encoder_ids = pad_sequence(batch, batch_first = True, padding_value = PAD_IDX)

    encoder_mask = (encoder_ids != PAD_IDX).long()

    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype = torch.long)

    return (
        encoder_ids,
        encoder_mask,
        initial_decoder_inputs
    )

def get_dataloader(batch_size, split, sql_preprocessing = "none"):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split, sql_preprocessing = sql_preprocessing)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size, sql_preprocessing = "none"):
    train_loader = get_dataloader(batch_size, "train", sql_preprocessing = sql_preprocessing)
    dev_loader = get_dataloader(test_batch_size, "dev", sql_preprocessing = sql_preprocessing)
    test_loader = get_dataloader(test_batch_size, "test", sql_preprocessing = sql_preprocessing)

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x
