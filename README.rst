******
bpwave
******

.. image:: https://www.mypy-lang.org/static/mypy_badge.svg
    :target: https://mypy-lang.org/
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
    :target: https://pycqa.github.io/isort/

Blood Pressure Waveform Processing Toolbox // Vérnyomáshullám-feldolgozó eszközök.

.. warning::

    Work in progress. Any modules are subject to breaking changes.
    When using this package, please always specify the exact version number.

Installation
============

::

    $ cd <project_root>
    $ pip install -U .


Generating documentation
========================

::

    $ cd <project_root>
    $ pip install -U .[docs]
    $ cd <project_root>/docs
    $ make clean
    $ make html

This creates ``<project_root>/docs/build/html/index.html``,
that can be opened in a browser.


Tests
=====

::

    $ cd <project_root>
    $ pip install -U .[tests]
    $ python -m pytest
    $ python -m pytest --cov --cov-report=html

Tests requiring human evaluation (like visualisations) can be skipped or run
separately (these are marked with the ``@pytest.mark.human`` decorator)::

    $ python -m pytest -m human
    $ python -m pytest -m "not human"

BP waveforms used for tests and examples were downloaded from the
*Autonomic Aging: A dataset to quantify changes of cardiovascular autonomic function during healthy aging* [AAC]_
dataset of [PhysioNet]_.

Development
===========

This project uses some automated QA and source formatting tools, such as
isort_, Flake8_, Black_ and mypy_::

    $ cd <project_root>
    $ pip install -U .[dev]
    $ isort .
    $ flake8 .
    $ black .
    $ mypy .

.. _isort: https://pycqa.github.io/isort/
.. _Flake8: https://flake8.pycqa.org/en/latest/
.. _Black: https://black.readthedocs.io/en/stable/index.html
.. _mypy: https://mypy-lang.org/

References
==========

.. [AAC] Schumann, A., & Bär, K. (2021).
    Autonomic Aging: A dataset to quantify changes of cardiovascular autonomic
    function during healthy aging (version 1.0.0). PhysioNet.
    https://doi.org/10.13026/2hsy-t491.
.. [PhysioNet] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C.,
    Mark, R., ... & Stanley, H. E. (2000).
    PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research
    resource for complex physiologic signals. Circulation [Online]. 101 (23),
    pp. e215–e220.
