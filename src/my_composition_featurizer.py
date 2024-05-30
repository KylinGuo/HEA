"""
File: matminer_base_featurizer_demo.py
Author: GUO Qilin
Email: kylinguo@icloud.com
Date: 2024/05/17 14:53:43
Description: 
Copyright (c) 2024 GUO Qilin
"""
import os
import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition
from mendeleev.fetch import fetch_table
from mendeleev import element
from matminer.featurizers.base import BaseFeaturizer


def safe_get_ionization_energy(elem, index, default_value=0.0):
    try:
        return getattr(elem, '_ionization_energies')[index].energy
    except IndexError:
        return default_value
            

class MyCompositionFeaturizer(BaseFeaturizer):
    def __init__(self):
        self.dataset_elemental = fetch_table(table='elements')

    @staticmethod
    def featurize(composition_pmg: Composition):
        composition_dict = composition_pmg.fractional_composition.get_el_amt_dict()
        atomic_symbols = list(composition_dict.keys())
        atomic_fractions = list(composition_dict.values())
        atomic_numbers = [elem.number for elem in composition_pmg.elements]  # map()
        element_list = [element(symbol) for symbol in atomic_symbols]

        # TODO (GUO Qilin): Not all elemental properties are aviable!
        property_list = [
            'nvalence',
            'atomic_radius',
            'covalent_radius',
            'electronegativity_pauling',
            'density',
            # 'thermal_conductivity',
            # 'melting_point',
        ]

        feature_dict: dict[str: float] = {}
        for prop in property_list:
            value_list = [
                getattr(elem, prop)()
                if callable(getattr(elem, prop))
                else getattr(elem, prop) for elem in element_list]
            feature_dict.update({prop: np.dot(atomic_fractions, value_list)})

        for i in range(3):  # the first three IE (eV)
            # value_list = [getattr(elem, '_ionization_energies')[i].energy for elem in element_list]  # IndexError: out of range
            value_list = [safe_get_ionization_energy(elem, i) for elem in element_list]
            feature_dict.update({f'ionization_energy_{i + 1}': np.dot(atomic_fractions, value_list)})

        feature_list = [feature_dict[key] for key in property_list]
        for i in range(3):
            feature_list.append(feature_dict[f'ionization_energy_{i + 1}'])

        feature_list.append(-np.dot(a=atomic_fractions, b=np.log(atomic_fractions)))
        feature_list.append(np.sqrt(np.dot(a=atomic_fractions, b=(1 - np.array(atomic_fractions) / np.mean(atomic_fractions)) ** 2)))
        
        return feature_list
    
    @staticmethod
    def feature_labels():
        labels: list[str] = [
            'valence electron concentration',
            'atomic radius (empirical)',
            'covalent radius',
            'Pauling electronegativity',
            'mass density',
            # 'thermal conductivity',
            # 'melting point',
            'first ionization energy',
            'second ionization energy',
            'third ionization energy',
            'mixing entropy', 
            'atomic size difference',
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


def test_my_composition_featurizer():
    composition_featurizer = MyCompositionFeaturizer()

    composition_str = 'Fe2.00Ni4.00'
    composition_pmg = Composition(composition_str)

    feature_labels = composition_featurizer.feature_labels()
    feature_list = composition_featurizer.featurize(composition_pmg)
    print('\n'.join([f'{key:>30s}: {value:12.6f}' for key, value in zip(feature_labels, feature_list)]))


def main():
    test_my_composition_featurizer()


if __name__ == '__main__':
    main()

