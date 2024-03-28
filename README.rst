=====
Zanes
=====


.. image:: https://img.shields.io/pypi/v/zanes.svg
        :target: https://pypi.python.org/pypi/zanes

.. image:: https://img.shields.io/travis/kaifengZheng/zanes.svg
        :target: https://travis-ci.com/kaifengZheng/zanes

.. image:: https://readthedocs.org/projects/zanes/badge/?version=latest
        :target: https://zanes.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/kaifengZheng/zanes/shield.svg
     :target: https://pyup.io/repos/github/kaifengZheng/zanes/
     :alt: Updates



A wrapper for XAS analysis and fitting

* Free software: MIT license
* Documentation: https://zanes.readthedocs.io.

Introduction
------------
Installation
------------

.. code-block:: shell

   git clone https://github.com/kaifengZheng/zanes.git

Usage
-----
1. organize your data into following format
   * Athena project file
   * csv(set Energy at the first column, and all spectra list subsequently(interpolate to the same Energy grid)(with column name))
   * ProQEXAFS(a plain text, first column is the Energy and all spectra list subsequently(no column name))
   * other are under development(coming soon...)
2. Data processing(back ground removal, flattening and plotting)
   * import data

     .. code-block:: python

        from zanes.data_analysis import zanes_data_analysis as zanes_data_analysis
        from zanes.data_analysis import data_processing as dp
        filename='test_data/pd.prj'
        exp_data=zda(filename,datatype='Athena',batch_size=None,read_range=None)


   * Data processing and plotting
     To process the data, one requires to specify parameters, which are given by a dictionary:

     .. code-block:: JSON

        data_param={'pre_start':-350,
                    'pre_end':-70,
                    'post_start':71,
                    'post_end':1450,
                    'kweight':2,'rbkg':1.5,
                    'E0':11563.5,
                    'krange':[2,15]}

    - process one data with plotting

          .. code-block:: python

              dp(exp_data.data[1],pre_start=data_param['pre_start'],
                      pre_end=data_param['pre_end'],
               post_start=data_param['post_start'],
               post_end=data_param['post_end'],
               kweight=data_param['kweight'],rbkg=data_param['rbkg'],
               krange=data_param['krange'],plot=True)
           
    - process all data

          .. code-block:: python

             exp_data.process_data(data_param)
* fitting data
Two parameter dictionaries are used in fitting process:
  - fitting parameters:

  .. code-block:: JSON

        data_param={'pre_start':-350,
                    'pre_end':-70,
                    'post_start':71,
                    'post_end':1450,
                    'kweight':2,'rbkg':1.5,
                    'E0':11563.5,
                    'krange':[2,15]}
  - fitting range parameters
  
  .. code-block:: JSON

        fit_range_param={'kmin':2,
                         'kmax':15,
                         'kw':2,
                         'dk':2,
                         'window':'hanning',
                         'rmin':1.646,
                         'rmax':3.292}
 - running and write fitting(mpi version is under development, use mpi=False for now)
 
 .. code-block:: python

      surfix='_rbkg1'
      foldername='results'
      save_path=join(foldername,'pd'+surfix+'.txt')
      dset,out,report,path=exp_data.run_fit_batch(param_dict=param_dict,fit_range_param=fit_range_param,fitpath_num=[0],feff_folder='feff',save_name=save_path,mpi=False,core=10,write=True)
      report_sort=exp_data.write_sorted_report(out,report,param_dict,path[0])
      exp_data.write_fitted_data(dset,'Pd',foldername,suffix='_rbkg1')
      output_fit = join(foldername,file_name[:-4]+'_rbkg1.toml')
      with open(output_fit, "w") as f:
           toml.dump(report_sort,f)

 

Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
