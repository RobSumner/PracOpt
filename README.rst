PracOpt
=======

PracOpt ("Practical" + "Optimisation") is a set of tools for solving
practical optimisation problems, based upon the 4M17 lecture series
given by Dr Geoff Parks at the University of Cambridge.


Getting Started:
----------------

The easiest way to get started with PracOpt is to create a Conda (or
Miniconda) virtual environment. Instructions for doing this can be found
`here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/
manage-environments.html>`_.

.. note::

    PracOpt requires Python 3.7 or higher. This can be requested when
    creating a conda environment with the flag *python=3.7*.

Once a conda virtual environment has been created, navigate to the pracopt
folder and run the command:

.. code-block::

    $ conda install --file requirements.txt

To install the required packages. PracOpt can then be installed in
development mode using the command:

.. code-block::

    $ conda develop .

To check the installation was sucess you can start up a Python interpreter:

.. code-block::

	$ python

followed by:

.. code-block::

	>>> import pracopt

This should successfully import PracOpt into the interpreter.

.. note::

    When using PracOpt, the conda virtual environment created for using
    PracOpt must be activated.


Running the tests:
------------------

Tests for PracOpt can be found in the *pracopt/tests* folder. These can be run
using pytest (from terminal) with the command:

.. code-block::

    $ pytest -v path_to_tests_folder

Or, for a specific test file:

.. code-block::

    $ pytest -v path_to_tests_folder/test_file.py


Documentation:
--------------
PracOpt is a work in progress, and as such does yet not have any official
documentation. Docstrings are provided where possible in the code to provide
some explanation.


Authors:
--------
* **Rob Sumner** (rob_sumner1@hotmail.co.uk)


License:
--------
This project is copywrited - see `license <LICENSE.rst>`_ file for more
details.