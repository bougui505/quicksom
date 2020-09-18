#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-06-16 15:41:44 (UTC+0200)

from scipy.io import FortranFile


def get_nframes(dcdfilename):
    """
    Get the number of frames in the dcd file
    """
    with FortranFile(dcdfilename, 'r') as f:
        nframes = f.read_ints()[1]
    return nframes
