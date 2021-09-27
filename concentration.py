#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-
import re

__author__ = 'Weiguo Jing'

kb = 8.617e-5  # unit eV / K


class ReadInput(object):
    def __init__(self, filename='input parameters.txt'):
        self.restrict = False
        self.chg_step = 1
        self.miu_step = 1
        with open(filename, 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            else:
                key, value = line.split('=')
                if re.search('host', key.lower()):
                    self.host = float(value)
                elif re.search('vbm', key.lower()):
                    self.vbm = float(value)
                elif re.search('formula units', key.lower()):
                    self.weight = float(value)
                elif re.search('temperature', key.lower()):
                    self.temperature = float(value)
                elif re.search('potential', key.lower()):
                    tmp = value.strip().split()
                    self.potential = np.array([float(x) for x in tmp])
                elif re.match('concentration restrict method', key.lower()):
                    self.restrict = True
                elif re.match('concentration restrict element', key.lower()):
                    tmp = value.strip().split()
                    self.element = [int(x) for x in tmp]
                elif re.match('restrict concentration', key.lower()):
                    tmp = value.strip().split()
                    self.concentration = [float(x) for x in tmp]
                else:
                    print('\nERROR: {0} tag is not found\n'.format(key))
                    exit(1)
        if self.restrict:
            if not hasattr(self, 'element') or not hasattr(self, 'concentration'):
                print('\nERROR: The restrict concentration calculation need *CONCENTRATION RESTRICT ELEMENT*'
                      ' and *RESTRICT CONCENTRATION (at.%) * tag\n')
                exit(1)
            elif len(self.element) != len(self.concentration):
                print('\nERROR: The number of element is mismatch between *CONCENTRATION RESTRICT ELEMENT*'
                      ' and *RESTRICT CONCENTRATION (at.%) * tag\n')
                exit(1)


class Data(object):
    def __init__(self, filename='defects data.txt'):
        raw_data = np.loadtxt(filename, comments='N')
        with open(filename, 'r') as fp:
            self.header = fp.readline().strip().split()
        num = len(self.header)
        if num < 7:
            print('\nERROR: The information for defect calculation is not complete.\n')
            exit(1)
        self.no = raw_data[:, 0].T
        self.charge = raw_data[:, 1].T
        self.etot = raw_data[:, 2].T
        self.weight = raw_data[:, 3].T
        self.iic = raw_data[:, 4].T
        self.apv = raw_data[:, 5].T
        self.delta_n = raw_data[:, 6:num].T


def self_consist_calculation_for_fermi_level(miu, order, data, fermi0=0, tolerance=1e-5):
    fermi = fermi0
    fermi_min = fermi
    fermi_max = fermi
    step = order.chg_step
    const0 = (data.etot + data.iic) - order.host + data.charge * data.apv - np.dot(miu, data.delta_n)
    const = const0 + data.charge * fermi
    concentration = data.weight * np.exp(-const / (kb * order.temperature))
    charge = data.charge * concentration
    flag = np.sign(np.sum(charge))
    dichotomy_method = False
    if abs(np.sum(charge)/max(np.max(charge), np.min(charge))) < tolerance:
        return concentration, fermi
    fermi += flag * step
    while True:
        const = const0 + data.charge * fermi
        concentration = data.weight * np.exp(-const / (kb * order.temperature))
        charge = data.charge * concentration
        if abs(np.sum(charge)/max(np.max(charge), np.min(charge))) < tolerance:
            return concentration, fermi
        if dichotomy_method:
            if np.sum(charge) * flag < 0:
                fermi_max = fermi
            else:
                fermi_min = fermi
            fermi = (fermi_max + fermi_min) / 2
        elif np.sum(charge) * flag < 0:
            dichotomy_method = True
            fermi_min = fermi - flag * step
            fermi_max = fermi
            fermi = (fermi_max + fermi_min) / 2
        else:
            fermi += flag * step
            flag = np.sign(np.sum(charge))


def self_consist_calculation_for_chemical_potential(miu, element, restrict_c, order, data, tolerance=1e-5):
    from copy import deepcopy
    coefficient = data.delta_n[element]
    w_tot = order.weight
    step = order.miu_step
    concentration, fermi = self_consist_calculation_for_fermi_level(miu, order, data, tolerance=tolerance)
    atom_weight = concentration * coefficient / w_tot
    delta = (restrict_c * 0.01 - np.sum(atom_weight)) / (restrict_c * 0.01)
    if abs(delta) < tolerance:
        return miu, fermi, concentration
    flag = np.sign(delta)
    miu[element] += flag * step
    dichotomy_method = False
    miu_min = deepcopy(miu)
    miu_max = deepcopy(miu)
    while True:
        concentration, fermi = self_consist_calculation_for_fermi_level(miu, order, data,
                                                                        fermi0=fermi, tolerance=tolerance)
        atom_weight = concentration * coefficient / w_tot
        delta = (restrict_c * 0.01 - np.sum(atom_weight)) / (restrict_c * 0.01)
        if abs(delta) < tolerance:
            return miu, fermi, concentration
        if dichotomy_method:
            if delta * flag < 0:
                miu_max = miu
            else:
                miu_min = miu
            miu = (miu_max + miu_min) / 2
        elif delta * flag < 0:
            dichotomy_method = True
            miu_min[element] = miu[element] - flag * step
            miu_max = miu
            miu = (miu_max + miu_min) / 2
        else:
            miu[element] += flag * step
            flag = np.sign(delta)


if __name__ == "__main__":
    from math import log
    import numpy as np
    order = ReadInput()
    data = Data()
    miu0 = order.potential
    index = order.element.index(1)
    restrict_concentration = order.concentration[index]
    tol = 1e-5
    delta = 0.5 * kb * order.temperature * log(0.1)
    delta_miu = [0, -1.5 * delta, delta]
    miu_O_min = -9.788598833
    tmp = []
    c = []
    f = []
    while miu0[2] > miu_O_min:
        miu, fermi, concentration = self_consist_calculation_for_chemical_potential(miu0, index, restrict_concentration,
                                                                                    order, data, tol)
        tmp.append(miu[::-1])
        f.append(fermi)
        c.append(concentration)
        miu0 += delta_miu
    tmp = np.array(tmp).T
    f = np.array(f)
    tmp = np.vstack((tmp, f))
    data = np.array(c).T
    np.savetxt('chemical potential.txt', tmp)
    np.savetxt('concentration.txt', data)
