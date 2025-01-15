from typing import Dict, List, Tuple
import os
from tqdm import tqdm
from abc import ABC, abstractmethod
import copy
from transformers import AutoTokenizer
PROJECT_ROOT= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class PrepareDataABC(ABC):
    @abstractmethod
    def prepare_conll_data_format(
        self,
        path: str,
        sep: str = "\t",
        lower: bool = True,
        verbose: bool = True,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        pass 
class BertTokenPrepareData(PrepareDataABC): 
    name='BertTokenPrepareData'
    def __init__(self):
        super().__init__()
        self.auto_tokenizer = AutoTokenizer.from_pretrained(os.path.join(PROJECT_ROOT,'bert-base-chinese'))
    def prepare_conll_data_format(
        self,
        path: str,
        sep: str = "\t",
        lower: bool = True,
        verbose: bool = True,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        token_seq = []
        label_seq = []
        with open(path, mode="r") as fp:
            tokens = []
            labels = []
            if verbose:
                fp = tqdm(fp)
            for line in fp:
                if line != "\n":
                    token, label = line.strip().split(sep)
                    if lower:
                        token = token.lower()
                    # 这里要把token进行特殊化处理
                    
                    bert_token_ids=self.auto_tokenizer(token,add_special_tokens=False)['input_ids']
                    bert_token_list=self.auto_tokenizer.convert_ids_to_tokens(bert_token_ids)
                    tokens.append(bert_token_list[0])
                    labels.append(label)
                    new_label=copy.copy(label)
                    # 分词把token和标签数据进行对齐操作。
                    if len(label)>1 and label[0]=='B':
                        new_label='I'+label[1:]
                    for xtoken in bert_token_list[1:]:
                        tokens.append(xtoken)
                        labels.append(new_label)
                else:
                    if len(tokens) > 0:
                        token_seq.append(tokens)
                        label_seq.append(labels)
                    tokens = []
                    labels = [] 
        return token_seq,label_seq

class DefaultPrepareData(PrepareDataABC):
    name='DefaultPrepareData'

    def prepare_conll_data_format(
        self,
        path: str,
        sep: str = "\t",
        lower: bool = True,
        verbose: bool = True,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Prepare data in CoNNL like format.
        Tokens and labels separated on each line.
        Sentences are separated by empty line.
        Labels should already be in necessary format, e.g. IO, BIO, BILUO, ...

        Data example:
        token_11    label_11
        token_12    label_12

        token_21    label_21
        token_22    label_22
        token_23    label_23

        ...
        """

        token_seq = []
        label_seq = []
        with open(path, mode="r") as fp:
            tokens = []
            labels = []
            if verbose:
                fp = tqdm(fp)
            for line in fp:
                if line != "\n":
                    token, label = line.strip().split(sep)
                    if lower:
                        token = token.lower()
                    tokens.append(token)
                    labels.append(label)
                else:
                    if len(tokens) > 0:
                        token_seq.append(tokens)
                        label_seq.append(labels)
                    tokens = []
                    labels = []
        return token_seq, label_seq   


class PrepareDataFactory:
    @staticmethod
    def create(prepareDataName:str)->PrepareDataABC:
        if(prepareDataName==BertTokenPrepareData.name):
            return BertTokenPrepareData()
        if(prepareDataName==DefaultPrepareData.name):
            return DefaultPrepareData()
        '''有个bug默认的数据预处理，不支持bert的词向量，先这样吧'''
        return DefaultPrepareData()
        
        
    
def prepare_conll_data_format(
    path: str,
    sep: str = "\t",
    lower: bool = True,
    verbose: bool = True,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Prepare data in CoNNL like format.
    Tokens and labels separated on each line.
    Sentences are separated by empty line.
    Labels should already be in necessary format, e.g. IO, BIO, BILUO, ...

    Data example:
    token_11    label_11
    token_12    label_12

    token_21    label_21
    token_22    label_22
    token_23    label_23

    ...
    """

    token_seq = []
    label_seq = []
    with open(path, mode="r") as fp:
        tokens = []
        labels = []
        if verbose:
            fp = tqdm(fp)
        for line in fp:
            if line != "\n":
                token, label = line.strip().split(sep)
                if lower:
                    token = token.lower()
                tokens.append(token)
                labels.append(label)
            else:
                if len(tokens) > 0:
                    token_seq.append(tokens)
                    label_seq.append(labels)
                tokens = []
                labels = []

    return token_seq, label_seq


def get_token2idx(
    token2cnt: Dict[str, int],
    min_count: int = 1,
    add_pad: bool = True,
    add_unk: bool = True,
) -> Dict[str, int]:
    """
    Get mapping from tokens to indices to use with Embedding layer.
    """

    token2idx: Dict[str, int] = {}

    if add_pad:
        token2idx["<PAD>"] = len(token2idx)
    if add_unk:
        token2idx["<UNK>"] = len(token2idx)

    for token, cnt in token2cnt.items():
        if cnt >= min_count:
            token2idx[token] = len(token2idx)

    return token2idx


def get_label2idx(label_set: List[str]) -> Dict[str, int]:
    """
    Get mapping from labels to indices.
    """

    label2idx: Dict[str, int] = {}

    for label in label_set:
        label2idx[label] = len(label2idx)

    return label2idx


def process_tokens(
    tokens: List[str], token2idx: Dict[str, int], unk: str = "<UNK>"
) -> List[int]:
    """
    Transform list of tokens into list of tokens' indices.
    """

    processed_tokens = [token2idx.get(token, token2idx[unk]) for token in tokens]
    return processed_tokens


def process_labels(labels: List[str], label2idx: Dict[str, int]) -> List[int]:
    """
    Transform list of labels into list of labels' indices.
    """

    processed_labels = [label2idx[label] for label in labels]
    return processed_labels
