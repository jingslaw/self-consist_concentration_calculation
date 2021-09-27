#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import re
import numpy as np
__author__ = 'Weiguo Jing'

kb = 8.617e-5  # unit eV / K


class ReadInput(object):
    def __init__(self, filename='formation energy input.txt'):
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
                elif re.search('cbm', key.lower()):
                    self.cbm = float(value)
                elif re.search('temperature', key.lower()):
                    self.temperature = float(value)
                elif re.search('potential', key.lower()):
                    tmp = value.strip().split()
                    self.potential = np.array([float(x) for x in tmp])
                else:
                    print('\nERROR: {0} tag is not found\n'.format(key))
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


if __name__ == "__main__":
    order = ReadInput()
    data = Data()
    points = 100
    defects = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12, 90, 91], [13, 14, 15, 92, 93], [16, 17, 18, 19],
               [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31], [40, 41, 42, 43], [44, 45, 46, 47],
               [52, 53, 54, 55], [94, 56, 57, 58, 59], [95, 60, 61, 62, 63], [86, 87, 88]]
    fermi_level = np.linspace(order.vbm, order.cbm, num=points)
    formation_energy = (data.etot+data.iic)-order.host-np.dot(order.potential, data.delta_n)+data.charge*data.apv
    formation_energy = formation_energy.reshape(formation_energy.shape[0], 1) + 0 * fermi_level
    charge = data.charge
    charge = charge.reshape(charge.shape[0], 1)
    x = charge * fermi_level
    formation_energy += charge * fermi_level
    min_formation_energy = []
    for defect in defects:
        tmp = []
        for index in defect:
            tmp.append(formation_energy[index])
        tmp = np.array(tmp)
        min_formation_energy.append(tmp.min(axis=0))
    result = np.vstack((fermi_level, min_formation_energy))
    result = result.T
    np.savetxt('formation energy.txt', result)

