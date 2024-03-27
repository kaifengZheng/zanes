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
This package serves as a wrapper for processing X-ray Absorption Spectroscopy (XAS) data and conducting data fitting procedures using X-raylarch. It offers capabilities for multiset-multipath fitting, comprising three primary classes:
* ```data_processing_multiOnerun()```:  This class facilitates multiset-multipath fitting and enables the manipulation of global variables.
* ```data_processing_multiloop()```: Designed to support multiset fitting, this class orchestrates fitting operations sequentially in a queue, processing one dataset at a time.
* ```data_processing_minibatch()```: Tailored for handling large datasets that necessitate extensive computational resources, particularly those impractical for processing with  ```data_processing_multiOnerun()```. Minibatch fitting divides the dataset into several groups, allowing fitting operations on each group individually.
    * Note: function ```check_convergence_batch``` is provided to assess the effectiveness of different batch sizes, identifying the optimal size that minimizes fitting uncertainties while maintaining stability as more data is incorporated.
The package supports Athena file, ProQEXAFS(txt), and CSV file as the input
* csv(format...)
* ProQEXAFS(format...)
* ATHENA




Usage
-----
Multiset-multipath fitting--data_processing_multiOnerun()
=========================================================
!. clone this repo:
``` python
from Zanes.Zanes import data_analysis as da
```
2. XAS data normalization
Select an Athena project file containing the spectra(At least E, mu).
``` python
project=da.data_processing_multiOnerun(filename=filename[0], datatype='Athena',read_range=[0,None]) #read_range[1]=None will considering the last spectrum.
data_param={'pre_start':-350,
               'pre_end':-70,
               'post_start':71,
               'post_end':1450,
               'kweight':3,'rbkg':1.3,
               'E0':11564,
               'krange':[2,15]}
project.process_data(data_param,plot=True)
```
![image](https://github.com/kaifengZheng/Zanes/assets/48105165/b7b2f159-857b-4322-a198-aedd77297495)
Note: if there is no E0 in data param, the program will run the ```find_E0()``` function and automatically find the E0 for you.

3. fitting the data
``` python
parm_dict={
    'SO2':{'initial':0.76,'vary':False,'global':True},
    'dele':{'dele_O':{'initial':-1.02,'vary':True,'global':True},
            'dele_Ce':{'initial':0.0,'vary':True,'global':True}},
    'ss2':{'ss2_O':{'initial':0.0024,'vary':True,'global':True,
                    'thermal':{'type':"Einstein",
                     'theta':{'initial':400,'vary':True,'global':True}}},
           'ss2_Ce':{'initial':0.003,'vary':True,'global':True,
                     'thermal':{'type':"Einstein",
                     'theta':{'initial':300,'vary':True,'global':True}}}},
    'N':{'N_O':{'initial':8,'vary':False,'global':True},
          'N_Ce':{'initial':12,'vary':False,'global':True}},
    'delr':{'delr_O':{'initial':-0.15,'vary':True,'global':True},
            'delr_Ce':{'initial':0.0,'vary':True,'global':True}},
}
temp=[83.15,173.15,298.15,373.15,473.15,573.15]

fit_range_param={'kmin':3,
                 'kmax':17,
                 'kw':2,
                 'dk':1,
                 'window':'hanning',
                 'rmin':1.31,
                 'rmax':3.85}
```
* initial: initial guess
* vary:     vary or fix (True/False)
* global:   set the variable as one global variable for all groups (True/False)
* thermal:  The methods for calculating DW factor<br>
  * Einstein: using Einstein mode(still working on):
    * theta:          Einstein temperature. ```vary: True``` ```global:True```
    * temp:           Experimental temperatures for all datasets in the study
-

4. Run the fit
```python
dset,out,report,paths=project.run_fit(parm_dict,'feff',fitpath_num=[0,2],fit_range_param=fit_range_param,write=True,save_name='test_100_multi.txt',temp=temp)
```
5. plot fitting result
```python
project.plot_fitting_results(dset,klim=[0,20],rlim=[0.5,3.85])
```
<img width="498" alt="image" src="https://github.com/kaifengZheng/Zanes/assets/48105165/871c1d92-b820-4127-96bd-c456178dc3b2">


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
