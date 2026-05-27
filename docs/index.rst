.. figure::
   :alt: MieAI logo
   :align: center

Welcome to MieAI
=================

.. image:: https://codecov.io/gh/d-attaway/mieai/graph/badge.svg?token=TJK7ELIKE5
.. image:: https://github.com/d-attaway/mieai/blob/documentation/.github/pylint.svg
.. image:: https://github.com/d-attaway/mieai/blob/documentation/.github/python.svg

MieAi is a software package to calculate the opacities of heterogenouse cloud particles. To accelerate the otherwise slow calcualtions, it provides three methods:

- 'Full': Perform effective refractive indice and Mie theory calculation.
- 'Grid': Precalculate opacity grids provide fast approximation to cloud particle opacities.
- 'AI': Convoluted Neural Networks trained to deliver high accuracy at a fraction of the computation time.

Fully trained models are provided on `Zenodo <https://zenodo.org/records/20346256>`_, or can be trained by yourself according to your needs. MieAi is under active development and contributions are welcommed. If you want to run MieAI checkout the `quick start guide <Install_And_Quick_Start.ipynb>`_.

Credit
------
If you use MieAI, please cite the following papers:

- `Attaway et al. (2026) <LINK TO PAPER>`_
- `Kiefer et al. (2026) <LINK TO PAPER>`_

Also consider citing the softwares MieAi is based on:

- `Prahl et al. (2026) <https://doi.org/10.5281/zenodo.7949263>`_
- `Tensorflow <https://zenodo.org/records/18894642>`_

.. toctree::
   :maxdepth: 1
   :caption: Contents

   Install_And_Quick_Start.ipynb
   Tutorial.ipynb
   API.ipynb


