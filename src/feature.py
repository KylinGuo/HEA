import pandas as pd
from matminer.featurizers.conversions import StrToComposition
# from matminer.featurizers.composition import ElementProperty
# from matminer.featurizers.composition.alloy import Miedema
# from matminer.featurizers.composition.tests import test_alloy
from matminer_base_featurizer_demo import CompoundFeaturizer


def composition_featurize(df: pd.DataFrame) -> pd.DataFrame:
    str_to_comp = StrToComposition(target_col_id='composition_pmg')
    df_comp: pd.DataFrame = str_to_comp.featurize_dataframe(df=df, col_id='composition', ignore_errors=False, pbar=True)

    df_features: pd.DataFrame = df_comp.apply(
        func=lambda row: pd.Series(CompoundFeaturizer.featurize(composition_pmg=row['composition_pmg'])), axis=1)
    df_features.columns = CompoundFeaturizer.feature_labels()

    df_featurized = pd.concat(objs=[df, df_features], axis=1, ignore_index=False)
    return df_featurized


def test_composition_featurize():
    data_file_input = 'data/data.json'
    data_file_output = 'data/featurized.json'
    df = pd.read_json(data_file_input)
    df_featurized = composition_featurize(df=df)
    df_featurized.to_json(path_or_buf=data_file_output, orient='records', indent=4)


def main():
    test_composition_featurize()


if __name__ == '__main__':
    main()
