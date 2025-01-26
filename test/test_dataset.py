from tuwnlp.dataset import PropagandaDataset
from tuwnlp.utils import Language, LabelLevel
from transformers import BertTokenizer


def test_dataset_len():
    dataset = PropagandaDataset("data")
    assert len(dataset) == 200

    dataset = PropagandaDataset("data", languages=[Language.EN, Language.BG, Language.HI, Language.PT])
    assert len(dataset) == 726

def test_dataset_indexing():
    dataset = PropagandaDataset("data")
    x, y= dataset[0:10]
    assert len(x) == 10
    assert y.shape == (10, 22)

    dataset = PropagandaDataset("data", tokenizer = BertTokenizer.from_pretrained('distilbert-base-multilingual-cased'))
    x, y = dataset[0:10]
    assert isinstance(x, dict)
    assert y.shape == (10, 22)

def test_naratives():
    dataset = PropagandaDataset("data", labelLevel=LabelLevel.SUBNARATIVES)
    assert dataset.y.shape == (200,96)



