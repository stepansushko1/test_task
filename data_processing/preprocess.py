# here will be removing duplicates, balance dataset
from dataset_airship import DataSet
import pandas as pd
import os

from constansts import *

from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self,
        raw_data_csv: str,
        train_csv: str,
        validation_csv: str,
        save_intermediate_results = False
        ) -> None:
        
        self.raw_data_csv = pd.read_csv(raw_data_csv)
        self.train_csv = train_csv
        self.validation_csv = validation_csv
        self.save_intermediate_results = save_intermediate_results

    def helper_join(self, mask):
        try:
            return ' '.join(mask)
        except Exception:
            return mask
    

    def concantenate_multiple_entries(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        # Join masks for images with more than 1 mask into a single entry
        mask_df = raw_df.groupby('ImageId')['EncodedPixels'].apply(lambda x: ' '.join(x.dropna()) if len(x.dropna()) > 1 else ''.join(x.dropna())).reset_index()

        # Merge the mask_df with the raw_df based on 'ImageId'
        merged_df = raw_df[['ImageId', 'count_ships']].merge(mask_df, on='ImageId')
        if self.save_intermediate_results:
            self.save_intermediate_results(merged_df)
        return merged_df


    def add_count_ships_column(self, raw_df: pd.DataFrame) -> pd.DataFrame:

        # Create a new column 'count_ships' with the count of ships
        raw_df['count_ships'] = raw_df.groupby('ImageId')['EncodedPixels'].transform(lambda x: x.notna().sum())

        # Replace NaN values in 'count_ships' with 0
        raw_df['count_ships'].fillna(0, inplace=True)

        if self.save_intermediate_results:
            self.save_result(raw_df, CLEANED_CSV_PATH)
        return raw_df


    def get_n_samples(self, group, n: int, random_state: int = 1):
        """
        Returns n random samples from group. If group has less values that n,
        returns the whole group.
        """
        if len(group) <= n:
            return group
        return group.sample(n, random_state=random_state)

    def get_balanced_subset(self, df: pd.DataFrame, samples_per_group: int,
         random_state: int = 1) -> pd.DataFrame:
        """
        Returns a dataframe where each class occurs at most samples_per_group times
        """
        balanced_df = df.groupby('count_ships', as_index=False).apply(
            self.get_n_samples, 3000, 42
        ) # can problem be here

        if self.save_intermediate_results:
            self.save_result(balanced_df.reset_index(drop=True), BALANCED_CSV_PATH)

        return balanced_df.reset_index(drop=True)

    def split_train_val(self,data: pd.DataFrame, validation_amount: int | float,
        random_state: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data into train and validation via sklearn function
        """
        train_df, val_df = train_test_split(
            data,
            test_size=validation_amount,
            random_state=random_state,
            stratify=data.count_ships,
        )
        train_df.reset_index(inplace=True, drop=True)
        val_df.reset_index(inplace=True, drop=True)


        self.save_result(train_df, self.train_csv)
        self.save_result(val_df, self.validation_csv)

        return train_df, val_df

    def save_result(self, data: pd.DataFrame, path:str) -> None:
        try:
            curr_dir = os.path.dirname(os.path.abspath(__file__)) 
            path = os.path.join(curr_dir, path)
            data.to_csv(path, columns=['ImageId', 'EncodedPixels', 'count_ships'])
        except Exception:
            print("Some error ocurred!")

    def executor(self) -> None:
        """
        Execute preprocessing pipeline
        """
        raw_df_with_ships_num = self.add_count_ships_column(self.raw_data_csv)
        df_without_duplicates = self.concantenate_multiple_entries(raw_df_with_ships_num)
        balanced_df = self.get_balanced_subset(df_without_duplicates, 2500)
        train_df, val_df = self.split_train_val(
                                balanced_df, 2000, RANDOM_STATE
                                )


if __name__ == "__main__": 

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    data_preparation = Preprocessor(os.path.join(curr_dir, RAW_CVS_PATH),
                                    TRAIN_DF, 
                                    VALIDATION_DF, 
                                    False)
    
    data_preparation.executor()