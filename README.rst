SCABox Attack Demo
***************************************************************

.. image:: https://api.travis-ci.com/emse-sas/sca-automation.svg?token=LqpGzZ56omzjYoep5ESp&branch=master
    :target: https://travis-ci.com/emse-sas/sca-automation
    :alt: tests

.. image:: https://img.shields.io/github/license/emse-sas/sca-automation
    :target: https://choosealicense.com/licenses/mit/
    :alt: license

.. image:: https://img.shields.io/github/deployments/emse-sas/sca-automation/github-pages
    :target: https://emse-sas.github.io/sca-automation/
    :alt: pages

`Website <https://emse-sas-lab.github.io/SCAbox/>`_

Overview
===============================================================

This repository contains a Python GUI application and a library to retrieve side-channel acquisition data from serial
port and perform an attack. It is part of the `SCABox <https://emse-sas-lab.github.io/SCAbox/>`_ project.

- Application : attack from serial port to key guess with exports
- Library : tools to perform attack, serial communication and correlation

The application is based on the library and this least is intended to work for any kind of side-channel leakage data and crypto-algorithm.

Features
===============================================================

Library
---------------------------------------------------------------

- Deserialization of acquisition data
- Fast acquisition data exports and imports
- Fast CPA computation and statistics
- Leakage model hypothesis generation
- Leakage signal processing
- Advanced Encryption Standard (AES)

Application
---------------------------------------------------------------

- Automate acquisition and attack
- Provide correlation and leakage visualization
- Export attack and acquisition results and images
- Parametrize the acquisition and visualize performances

Install
===============================================================

To install the repository you must clone the sources from GitHub and install the pip requirements

.. code-block:: shell

    $ git clone https://github.com/emse-sas-lab/SCAbox-automation
    $ cd sca-automation
    $ pip3 install -r requirements.txt

You might alternatively create a venv and install the requirements inside to use the project. 

Compatibility
---------------------------------------------------------------

The project is compatible with Python 3.8 and latter. It is platform independent.

However the it requires Tkinter to be installed in order to use the GUI application.
The instructions my varies according to which system your are using and we encourage
to visit the Tkinter documentation to install it. 

Usage
===============================================================

Library
---------------------------------------------------------------

The library provides a complete API to develop your own application.
In order to get started you can take a look at the examples and the reference.

Application
---------------------------------------------------------------

The GUI application can be started by running the ``main.py`` script

.. code-block:: shell

    $ cd sca-automation/app 
    $ sudo python3 main.py --target /dev/ttyUSB1

You might pass arguments to the ``main.py`` script in order parametrize the acquisition from shell.

.. code-block:: shell

    $ sudo python3 main.py --iteration 1024 --target /dev/ttyUSB1 --chunks 8

To get an exhaustive list, please visit the reference

Documentation
===============================================================

The complete documentation of the project is available `here <https://emse-sas-lab.github.io/SCAbox-automation/>`_.

Build
---------------------------------------------------------------

You can build the documentation of the library and application using sphinx and autodoc

.. code-block:: shell

    $ cd sca-automation/docs/sources
    $ make html

More
===============================================================

SCABox is a project on the topic of side-channel analysis.
The goal of SCABox is to provide an efficient test-bench for FPGA-based side-channel analysis.

To know more about SCABox please visit our `website <https://emse-sas-lab.github.io/SCAbox/>`_.
It provides a tutorials and a wiki about side-channel analysis.

SCABox is an open-source project, all the sources are hosted on GitHub

- `IP repository <https://github.com/emse-sas-lab/SCAbox-ip/>`_
- `Acquisition demo <https://github.com/emse-sas-lab/SCAbox-demo/>`_
- `Attack demo <https://github.com/emse-sas-lab/SCAbox-automation/>`_
- `SCAbox website  <https://github.com/emse-sas-lab/SCAbox/>`_

Contributing
---------------------------------------------------------------

Please feel free to take part into SCABox project, all kind of contributions are welcomed.

The project aims at gathering a significant number of IP cores, crypto-algorithms and attack models 
in order to provide an exhaustive view of today's remote SCA threat.

Software and embedded improvements are also greatly welcomed. Since the project is quite vast and invovles
a very heterogeneous technical stack, it is difficult to maintain the quality with a reduced size team.  

License
---------------------------------------------------------------

All the contents of the SCABox project are licensed under the `MIT license <https://choosealicense.com/licenses/mit/>`_ provided in each GitHub repository.

Copyright (c) 2020 Anonymous
