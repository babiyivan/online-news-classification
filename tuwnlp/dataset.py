from tuwnlp.utils import Language, LabelLevel
from tuwnlp.utils import read_files_to_df
from tuwnlp.utils import get_file_labels_dataframe


from pathlib import Path
import pandas as pd

from torch.utils.data import Dataset

class PropagandaDataset(Dataset):
    def __init__(self, data_dir:str, languages = [Language.EN], labelLevel = LabelLevel.NARATIVES, tokenizer = None): 
        self.data_dir = Path(data_dir)
        self.languages = languages
        self.labelLevel = labelLevel
        self.tokenizer = tokenizer

        all_dfs = []
        for language in self.languages:
            labels = get_file_labels_dataframe(
                self.data_dir,
                language,
                self.labelLevel
            )

            texts = read_files_to_df(self.data_dir / "tmp", language)
            texts.index = texts["file name"].values
            texts = texts.drop(columns = ["file name", "file path"])
            texts["text"] = texts["text"].apply(lambda x: " ".join(x.split("\t")[1::3]))
            df = pd.merge(texts, labels, left_index=True, right_index=True)
            print(language, self.labelLevel, df.shape)
            all_dfs.append(df)
        self.data = pd.concat(all_dfs)
        print(self.data.shape)

        self.x = self.data["text"].values
        self.y = self.data.drop(columns=["text"]).values
        if tokenizer is not None:
            res = self.tokenizer(self.x.tolist(), return_tensors='pt', padding=True, truncation=True)
            self.attention_mask = res["attention_mask"]
            self.token_type_ids = res["token_type_ids"]
            self.input_ids = res["input_ids"]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        if self.tokenizer is None:
            return self.x[index], self.y[index]
        
        return_dict = {
            "token_type_ids": self.token_type_ids[index,:],
            "attention_mask": self.attention_mask[index,:],
            "input_ids": self.input_ids[index,:],
        }
        return return_dict, self.y[index]