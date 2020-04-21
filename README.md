# tf-mpc [![Build Status](https://travis-ci.org/thiagopbueno/tf-mpc.svg?branch=master)](https://travis-ci.org/thiagopbueno/tf-mpc) [![Documentation Status](https://readthedocs.org/projects/tfmpc/badge/?version=latest)](https://tfmpc.readthedocs.io/en/latest/?badge=latest) [![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/thiagopbueno/tf-mpc/blob/master/LICENSE)


# Quickstart

**tfmpc** is a Python3.6+ package available in PyPI.

```text
$ pip3 install -U tfmpc
```


# Usage

## LQR

```text
$ tfmpc lqr --help
Usage: tfmpc lqr [OPTIONS] INITIAL_STATE

  Generate and solve a randomly-created LQR problem.

  Args:

      initial_state: list of floats.

Options:
  -a, --action-size INTEGER RANGE
                                  The number of action variables.
  -hr, --horizon INTEGER RANGE    The number of timesteps.
  --help                          Show this message and exit.
```

```text
$ tfmpc lqr -a 2 -hr 10 -- "-1.0 0.5 3.6"

Trajectory(init=[-1.   0.5  3.6], final=[-6.8887715 -5.8231974 -2.4906292], total=-22.6460)

Steps |             States             |       Actions        |  Costs  
===== | ============================== | ==================== | ========
  0   | [  2.4519,  -3.4247,   1.5683] | [ -2.4000,  -1.9967] |   1.7630
  1   | [ -1.3597,   0.6466,   0.4730] | [ -0.4974,  -2.2108] |  -4.9768
  2   | [  1.2518,  -2.4087,   1.8576] | [ -0.8572,  -1.7336] |   0.6628
  3   | [ -0.5029,  -0.3449,   1.0460] | [ -0.9881,  -2.3027] |  -3.9363
  4   | [  0.7103,  -1.8426,   1.4427] | [ -0.9374,  -1.8516] |  -1.2144
  5   | [ -0.2330,  -0.6179,   1.4067] | [ -0.9244,  -2.2985] |  -3.2234
  6   | [  0.5808,  -1.8719,   1.0914] | [ -1.0021,  -1.7919] |  -1.7650
  7   | [ -0.5810,  -0.5750,   1.7039] | [ -0.7238,  -2.5045] |  -3.1283
  8   | [ -0.1008,  -2.8592,   0.9244] | [ -0.8470,  -2.0201] |  -1.7682
  9   | [ -6.8888,  -5.8232,  -2.4906] | [  2.1113,  -2.2470] |  -5.0595

```

## Linear Navigation

```text
$ tfmpc navlin --help
Usage: tfmpc navlin [OPTIONS] INITIAL_STATE GOAL

  Generate and solve the linear navigation LQR problem.

  Args:

      initial_state: list of floats.

      goal: list of floats.

Options:
  -b, --beta FLOAT              The weight of the action cost.
  -hr, --horizon INTEGER RANGE  The number of timesteps.
  --help                        Show this message and exit.
```

```text
$ tfmpc navlin -b 5.0 -hr 10 -- "0.0 0.0" "8.0 -9.0"

Trajectory(init=[0. 0.], final=[ 7.757592 -8.727291], total=-1045.4086)

Steps |        States        |       Actions        |   Costs  
===== | ==================== | ==================== | =========
  0   | [  2.8645,  -3.2225] | [  2.8645,  -3.2225] |   92.9486
  1   | [  4.7018,  -5.2895] | [  1.8373,  -2.0670] |  -47.0048
  2   | [  5.8795,  -6.6145] | [  1.1777,  -1.3249] | -104.6422
  3   | [  6.6331,  -7.4623] | [  0.7536,  -0.8478] | -128.3791
  4   | [  7.1134,  -8.0025] | [  0.4802,  -0.5403] | -138.1544
  5   | [  7.4163,  -8.3433] | [  0.3029,  -0.3408] | -142.1795
  6   | [  7.6025,  -8.5528] | [  0.1862,  -0.2094] | -143.8354
  7   | [  7.7091,  -8.6727] | [  0.1067,  -0.1200] | -144.5131
  8   | [  7.7576,  -8.7273] | [  0.0485,  -0.0545] | -144.7817
  9   | [  7.7576,  -8.7273] | [  0.0000,   0.0000] | -144.8669

```

# Documentation

Please refer to [https://tfmpc.readthedocs.io/](https://tfmpc.readthedocs.io/) for the code documentation.


# License

Copyright (c) 2020- Thiago P. Bueno All Rights Reserved.

tfmpc is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

tfmpc is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with tfmpc. If not, see http://www.gnu.org/licenses/.
