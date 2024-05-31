import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from my_composition_featurizer import MyCompositionFeaturizer


def composition_featurize(df: pd.DataFrame) -> pd.DataFrame:
    source_col_id = 'composition'
    target_col_id = 'composition_pmg'

    ignore_errors = True
    n_job = 24

    str_to_comp = StrToComposition(target_col_id=target_col_id)
    df_pmg: pd.DataFrame = str_to_comp.featurize_dataframe(df=df, col_id=source_col_id, ignore_errors=ignore_errors, pbar=True)

    composition_featurizer = MyCompositionFeaturizer()
    composition_featurizer.set_n_jobs(n_jobs=n_job)

    df_featurized: pd.DataFrame = composition_featurizer.featurize_dataframe(df=df_pmg, col_id=target_col_id, ignore_errors=ignore_errors, pbar=True)
    df_featurized.drop(labels=[target_col_id, ], axis=1, inplace=True)
    return df_featurized


def test_composition_featurize():
    data_file_input = 'data/data.json'
    data_file_output_json = 'data/data_featurized.json'
    data_file_output_xlsx = 'data/data_featurized.xlsx'

    df = pd.read_json(data_file_input)
    df_featurized = composition_featurize(df=df)

    df_featurized.to_excel(excel_writer=data_file_output_xlsx, index=False)
    df_featurized.to_json(path_or_buf=data_file_output_json, orient='records', indent=4)  # Aborted (core dumped)


def main():
    test_composition_featurize()


if __name__ == '__main__':
    main()
