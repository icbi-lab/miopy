MIOPY: Python tool to study miRNA-mRNA relationships. 
====================================================================================================



MIOPY is a Python3 tool that allows studying the miRNA/mRNA interaction in several different ways. MIOPY includes the state-of-the-art method to predict miRNA target through expression data. MIOPY includes 40 pre-process prediction tools 
to improve the results. Finally, MIOPY allows to use the of state-of-the-art machine learning methods to apply a feature selection in order to identify prognostic biomarker signatures.
 
The four main types of analyses available in MIOPY are:

* Correlation and regularized regression analysis
* microRNA target prediction
* Identification of prognostic biomarkers and survival analyses
* Identification of predictive biomarkers and classification

MIOPY was developed to handle the analysis in `MIO <http://mio.icbi.at>`_. Please check the MIO `repository <http://github.com/icbi-lab/mio>`_to run MIO locally. MIOPY includes all the functions available in MIO.

**We are happy about feedback and welcome contributions!**

Getting started
^^^^^^^^^^^^^^^
Please check the jupyter-notebook with that same Use cases example present in the *P. Monfort-Lanzas et al.* publication.

-  `Use cases <./test/test.ipynb`_

Installation
^^^^^^^^^^^^
MIOPY was developed using Python 3.8.8.

There are several alternative options to install MIOPY:

.. 1) Install the latest development version:

.. code-block::

  git clone git@github.com:icbi-lab/miopy.git && cd miopy && python3 setup.py install

.. 2) Install the latest development version:

.. code-block::
 python3 -m pip install git+https://github.com/icbi-lab/miopy.git


Release notes
^^^^^^^^^^^^^
See the `release section <https://github.com/icbi-lab/miopy/releases>`_.

Contact
^^^^^^^
Please use the `issue tracker <https://github.com/icbi-lab/miopy/issues>`_.

Citation
^^^^^^^^
