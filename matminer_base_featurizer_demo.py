"""
File: matminer_base_featurizer_demo.py
Author: GUO Qilin
Email: kylinguo@icloud.com
Date: 2024/05/17 14:53:43
Description: 
Copyright (c) 2024 GUO Qilin
"""
from abc import ABC
import numpy as np
from pymatgen.core.composition import Composition
from matminer.featurizers.base import BaseFeaturizer
from mendeleev import element


class CompoundFeaturizer(BaseFeaturizer, ABC):
    def __init__(self):
        super().__init__()

    def featurize(self, composition: Composition):
        composition_dict = composition.fractional_composition.get_el_amt_dict()
        atomic_fractions = list(composition_dict.values())
        atomic_symbols = list(composition_dict.keys())
        element_list = [element(symbol) for symbol in atomic_symbols]

        property_list = [
            'nvalence',
            'atomic_radius',
            'covalent_radius',
            'electronegativity_pauling',
            'density',
            'thermal_conductivity',
            'melting_point',
        ]

        feature_dict: dict[str: float] = dict()
        for prop in property_list:
            value_list = [
                getattr(elem, prop)()
                if callable(getattr(elem, prop))
                else getattr(elem, prop) for elem in element_list]
            feature_dict.update({prop: np.dot(atomic_fractions, value_list)})

        for i in range(3):  # the first three IE (eV)
            value_list = [getattr(elem, '_ionization_energies')[i].energy for elem in element_list]
            feature_dict.update({f'ionization_energy_{i + 1}': np.dot(atomic_fractions, value_list)})

        feature_list = [feature_dict[key] for key in property_list]
        for i in range(3):
            feature_list.append(feature_dict[f'ionization_energy_{i + 1}'])
        return feature_list

    def feature_labels(self):
        labels: list[str] = [
            'valence electron concentration',
            'atomic radius (empirical)',
            'covalent radius',
            'Pauling electronegativity',
            'mass density',
            'thermal conductivity',
            'melting point',
            'first ionization energy',
            'second ionization energy',
            'third ionization energy',
        ]
        return labels

    def citations(self):
        bib_file = 'Rao_2022.bib'
        with open(file=bib_file, mode='r', encoding='utf-8') as file:
            lines = file.readlines()
        lines = [s.strip() for s in lines]
        return lines

    def implementors(self):
        return ['Zhang Hongyu', 'Wang Xiaobing', 'GUO Qilin', ]


def test_compound_featurizer():
    compound_featurizer = CompoundFeaturizer()

    composition_str = 'Fe1.00Ni0.00'
    composition_pmg = Composition(composition_str)

    feature_labels = compound_featurizer.feature_labels()
    feature_list = compound_featurizer.featurize(composition_pmg)
    print('\n'.join([f'{key:<30s}: {value:10.6f}' for key, value in zip(feature_labels, feature_list)]))


def main():
    test_compound_featurizer()


if __name__ == '__main__':
    main()

