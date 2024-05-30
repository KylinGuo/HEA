import multiprocessing
import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from my_composition_featurizer import MyCompositionFeaturizer


def composition_featurize(df: pd.DataFrame) -> pd.DataFrame:
    source_col_id = 'composition'
    target_col_id = 'composition_pmg'

    ignore_errors = True
    n_job = 24  # multiprocessing.cpu_count()

    str_to_comp = StrToComposition(target_col_id=target_col_id)
    df_pmg: pd.DataFrame = str_to_comp.featurize_dataframe(df=df, col_id=source_col_id, ignore_errors=ignore_errors, pbar=True)

    composition_featurizer = MyCompositionFeaturizer()
    composition_featurizer.set_n_jobs(n_jobs=n_job)

    df_featurized: pd.DataFrame = composition_featurizer.featurize_dataframe(df=df_pmg, col_id=target_col_id, ignore_errors=ignore_errors, pbar=True)
    df_featurized.drop(labels=[target_col_id, ], axis=1)
    return df_featurized


def test_composition_featurize():
    data_file_input = '../data/data.json'
    data_file_output = '../data/data_featurized.json'
    df = pd.read_json(data_file_input)
    df_featurized = composition_featurize(df=df)
    print(df_featurized)
    df_featurized.to_json(path_or_buf=data_file_output, orient='records', indent=4)


def main():
    test_composition_featurize()


if __name__ == '__main__':
    main()
