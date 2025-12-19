import pandas as pd
import torch


class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.df = self.preprocess(df)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = ["asthma", "smoking", "COPD", "hay_fever", "dysp", "cough", "pain", "nasal", "antibiotics", "season", "fever"]
        self.diagnosis = "diagnosis"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        input_ids = self.tokenizer.encode(row["text"], max_length=self.max_length, padding="max_length")
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        labels = row[self.labels].tolist()
        labels = torch.tensor(labels, dtype=torch.float32)

        diagnosis = torch.tensor(row[self.diagnosis], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'diagnosis': diagnosis
        }

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        # to binary
        binary_columns = ["asthma", "smoking", "COPD","hay_fever", "dysp", "cough", "pain", "nasal", "antibiotics"]
        for col in binary_columns:
            df[col] = df[col].apply(lambda x: 0 if x == "no" else 1)

        # make binary
        df["season"] = df["season"].apply(lambda x: 0 if x in ["summer", "spring"] else 1)
        df["fever"] = df["fever"].apply(lambda x: 0 if x == "none" else 1)

        # clean text column
        df["text"] = df["text"].apply(
            lambda x: (
                x
                .lower()
                .replace("**History**", "")
                .replace("**Physical Examination**", "")
            )
        )

        # create diagnosis column
        mapping = {
            ("no", "no"): 0,
            ("yes", "no"): 1,
            ("no", "yes"): 2,
            ("yes", "yes"): 3,
        }

        df["diagnosis"] = list(zip(df["pneu"], df["common_cold"]))
        df["diagnosis"] = df["diagnosis"].map(mapping)
        return  df
