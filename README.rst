=====
Zanes
=====


.. image:: https://img.shields.io/pypi/v/zanes.svg
        :target: https://pypi.python.org/pypi/zanes

.. image:: https://readthedocs.org/projects/zanes/badge/?version=latest
        :target: https://zanes.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/kaifengZheng/zanes/shield.svg
     :target: https://pyup.io/repos/github/kaifengZheng/zanes/
     :alt: Updates



A wrapper for XAS analysis and fitting

* Free software: MIT license
* Documentation: https://kaifengzheng.github.io/zanes

Introduction
------------
Installation
------------

.. code-block:: shell

   git clone https://github.com/kaifengZheng/zanes.git

Usage
-----
organize your data into following format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   * Athena project file
   * csv(set Energy at the first column, and all spectra list subsequently(interpolate to the same Energy grid)(with column name))
   * ProQEXAFS(a plain text, first column is the Energy and all spectra list subsequently(no column name))
   * others are under development(coming soon...)
Data processing(back ground removal, flattening and plotting)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   * import data

     .. code-block:: python

        import numpy as np
        import pandas as pd
        from larch import Group
        from zanes.zanes.data_analysis import data_processing as dp
        from zanes.zanes.data_analysis import zanes_data_analysis as zda
        from glob import glob
        import toml
        import os
        filename='test_data/pd.prj'
        exp_data=zda(filename,datatype='Athena',batch_size=None,read_range=None)


   * Data processing and plotting
     To process the data, one requires to specify parameters, which are given by a dictionary:

     .. code-block:: JSON

        data_param={'pre_start':-350,
               'pre_end':-70,
               'post_start':71,
               'post_end':1450,
               'kweight':2,'rbkg':1,
               'E0':11563.5,
               'krange':[2,15]}

    - process one data with plotting

          .. code-block:: python

              dp(exp_data.data[0],pre_start=data_param['pre_start'],
                      pre_end=data_param['pre_end'],
               post_start=data_param['post_start'],
               post_end=data_param['post_end'],
               kweight=data_param['kweight'],rbkg=data_param['rbkg'],
               krange=data_param['krange'],plot=True)

          .. image:: https://github.com/kaifengZheng/zanes/blob/distortion/docs/_satic/data_analysis.png
    - wavelet transform

          .. code-block:: python

              from Zanes.zanes.data_analysis import wavelet_transform
              wavelet_transform(exp_data.data[0],plot=True)

          .. image:: https://github.com/kaifengZheng/zanes/blob/distortion/docs/_satic/wavelet_transform.png



    - process all data

          .. code-block:: python

             exp_data.process_data(data_param)
Fitting data
~~~~~~~~~~~~~

    Two parameter dictionaries are used in the fitting process:

    - multiple-shell fitting parameters:

    .. code-block:: JSON

       param_dict={
          'SO2':{"SO2":{'initial':0.84,'vary':False,'global':True}},
          'dele':{ 'dele':{'initial':3,'vary':True,'global':True}},
                  #'dele_Pt':{'initial':0,'vary':True,'global':True}},
          'ss2':{'ss2_O':{'initial':0.003,'vary':True,'global':False,
                        'thermal':{'type':False,
                        'theta':{'initial':400,'vary':True,'global':True}}},
                 'ss2_Pt':{'initial':0.003,'vary':True,'global':False,
                        'thermal':{'type':False,
                        'theta':{'initial':400,'vary':True,'global':True}}}},
          'N':{'N_O':{'initial':6.0,'vary':True,'global':False},
               'N_Pt':{'initial':12.0,'vary':True,'global':False},
             },
          'delr':{'delr_O':{'initial':0.0,'vary':True,'global':False},
                  'delr_Pt':{'initial':0.0,'vary':True,'global':False},
                }}

    - fitting range parameters

    .. code-block:: JSON

        fit_range_param={'kmin':2,
                         'kmax':15,
                         'kw':2,
                         'dk':2,
                         'window':'hanning',
                         'rmin':1,
                         'rmax':3.292}

    - run fitting and write reports (mpi version is under development, use mpi=False for now)

    .. code-block:: python

         output_path='results'
         surfix='_rbkg1' #surfix can be used to describe the special conditions to run the fitting. It is useful when one wants to run multiple fittings.
         file_name=os.path.basename(file) # file is the path pointing direct to the datatable to process
         foldername=os.path.join('results',file_name[:-4]+surfix)
         if not os.path.exists(foldername):
            os.makedirr(foldername)
         exp_data=zda()
         exp_data.read_datacollection(file,datatype='ProQEXAFS')
         exp_data.process_data(data_param)
         dset,out,report,path=exp_data.run_fit_batch(param_dict,fit_range_param,[[0],[0]],['feff_PtO','feff'],save_name=os.path.join(foldername,file_name[:-4]+'_rbkg1_CN_ss.txt'),write=True,mpi=False,core=10,batch_size=100) # We can use paths from different feff calculations.
         report_sort=exp_data.write_sorted_report(out,report,param_dict,path)                                                 # In this example, there are two paths used in the fitting;
         exp_data.write_fitted_data(dset,file,foldername,suffix='_rbkg1_CN_ss')                                               # The first one comes from the first path in the 'feff_PtO' calculation
         output_fit = os.path.join(foldername,file_name[:-4]+'_rbkg1_CN_ss.toml')                                                     # and the second one comes from the first path in the 'feff' calculation.
         with open(output_fit, "w") as f:
             toml.dump(report_sort,f)

Codes run on windows 10 system(support python >3.10, <3.10 requires to change the typing styles)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There is an option to run the fitting on a Linux system using the bash command. The codes are in the **codes_for_linux** folder. **run.slurm**, the slurm script for submitting jobs on supercluster, contains the whole bash script for running the fitting.
To utilize this code, XAS spectra should be stored in the **input** folder, and either the FEFF input file or the FEFF results should stored in the FEFF folder. There also requires an output folder created in the directory.

.. code-block:: bash

    python fitting.py --input "${files[0]}" -o output/ -dt ProQEXAFS -bs -1 -os _rbkg1_ss



Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
