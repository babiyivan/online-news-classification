from pathlib import Path
from enum import Enum
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import numpy as np

class Language(Enum):
    BG = "BG"
    EN = "EN"
    HI = "HI"
    PT = "PT"

class TokenType(Enum):
    DEFAULT = 1
    LEMMA = 3

class LabelLevel(Enum):
    NARATIVES = "NARATIVES"
    SUBNARATIVES = "SUBNARATIVES"

class Topic(Enum):
    CLIMATE_CHANGE = "cc"
    UKRAINE = "ua"

def read_file_as_string(filepath: Path, tokenType = TokenType.DEFAULT) -> str:
    lines = open(filepath).readlines()[1:-1]
    res = []
    for line in lines:
        res.append(line.split("\t")[tokenType.value])
    return " ".join(res)

def read_files_to_df(basedir: Path, language: Language) -> DataFrame:
    dir = basedir / language.value
    res = []
    for filepath in tqdm(dir.iterdir()):
        res.append({
            "file path": filepath,
            "file name": filepath.name.split(".")[0],
            "text": read_file_as_string(filepath),
        })
    return pd.DataFrame(data=res)


def get_top_lvl_label_mappings(labelDir: Path):
    lines = open(labelDir).readlines()
    top_lvl_labels = [
        line.strip()
        for line in lines
        if line[0] != "-"
    ]
    top_lvl_labels_to_index = {
        label:indx
        for indx, label in enumerate(top_lvl_labels)
    }
    top_lvl_indx_to_labels = {
        indx:label
        for indx, label in enumerate(top_lvl_labels)
    }
    return top_lvl_labels_to_index, top_lvl_indx_to_labels 

def get_low_lvl_label_mappings(labelDir: Path):
    lines = open(labelDir).readlines()
    low_lvl_labels = []
    prefix = ""
    for line in lines:
        if line.strip() == "Other":
            low_lvl_labels.append("Other")
        elif line[0] != "-":
            prefix = f"{line.strip()}: "
            low_lvl_labels.append(prefix + "Other")
        else:
            low_lvl_labels.append(prefix + line[1:].strip())

    low_lvl_labels_to_index = {
        label:indx
        for indx, label in enumerate(low_lvl_labels)
    }
    low_lvl_indx_to_labels = {
        indx:label
        for indx, label in enumerate(low_lvl_labels)
    }
    return low_lvl_labels_to_index, low_lvl_indx_to_labels


def get_file_labels_dataframe(
        data_dir: Path,
        language: Language,
        labelLevel: LabelLevel
) -> tuple[DataFrame, dict, dict]:    

    annotation_file = data_dir / "training_data_16_October_release"/ language.value /"subtask-2-annotations.txt"
    ccLabelDir = data_dir / "labels" / "cc-labels.txt"
    uaLabelDir = data_dir / "labels" / "ua-labels.txt"

    #Get label mapper
    label_to_indx = {}
    if labelLevel == LabelLevel.NARATIVES:
        labels_to_indx_cc, _ = get_top_lvl_label_mappings(ccLabelDir)
        labels_to_indx_ua, _ = get_top_lvl_label_mappings(uaLabelDir)

    else:
        labels_to_indx_cc, _ = get_low_lvl_label_mappings(ccLabelDir)
        labels_to_indx_ua, _ = get_low_lvl_label_mappings(uaLabelDir)
    
    all_labels = list(set([f"CC: {el}" if el != "Other" else el for el in labels_to_indx_cc.keys() ] + [f"URW: {el}" if el != "Other" else el for el in labels_to_indx_ua.keys() ]))
    all_labels = sorted(all_labels)
    label_to_indx = {label:i for i, label in enumerate(all_labels)}

    # Read file annotations and generate labels
    lines = open(annotation_file).readlines()
    filenames = []
    label_indxs = []
    for line in lines:
        filename, naratives, subnaratives = line.strip().split("\t")
        to_labels = naratives if labelLevel == LabelLevel.NARATIVES else subnaratives
        labels = set(to_labels.split(";"))
        maped_labels = list(map(lambda x: label_to_indx.get(x, None), labels))
        filenames.append(filename.strip().split(".")[0])
        label_indxs.append(maped_labels)

    n = len(filenames)
    m = len(all_labels)
    x = np.zeros((n,m))
    for i, entry in enumerate(label_indxs):
        x[i][entry] = 1
    df = pd.DataFrame(columns=all_labels, data=x, dtype=np.bool_)
    df.index = filenames
    return df