import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
import matplotlib.pyplot as plt
# outputfolder=glob('results/*')
input_file_name1 = "results_1_2\\Deglitched_Amm_-0.5-0.19V_PtOPtPt_ss_N\\Deglitched_Amm_-0.5-0.19V_PtOPtPt_ss_N.toml"
with open(input_file_name1) as toml_file:
    toml_dict1 = toml.load(toml_file)

r1_O=np.double(pd.DataFrame(toml_dict1['r_O']).loc['value'])
r1_Pt=np.double(pd.DataFrame(toml_dict1['r_Pt']).loc['value'])
ss1_O=np.double(pd.DataFrame(toml_dict1['ss2_O']).loc['value'])
ss1_Pt=np.double(pd.DataFrame(toml_dict1['ss2_Pt']).loc['value'])
cn1_O=np.double(pd.DataFrame(toml_dict1['N_O']).loc['value'])
cn1_Pt=np.double(pd.DataFrame(toml_dict1['N_Pt']).loc['value'])
# chi_square1=np.double(toml_dict1['reduced chi_square'])
# r_factor1=  np.double(toml_dict1['r_factor'])

fig,ax=plt.subplots(3,2,figsize=(8,6))
time1=np.arange(0,len(r1_O))
time2=np.arange(0,len(r1_Pt))
time3=np.arange(0,len(ss1_O))
time4=np.arange(0,len(ss1_Pt))
time5=np.arange(0,len(cn1_O))
time6=np.arange(0,len(cn1_Pt))
ax[0,0].plot(time1,r1_O,label='$r_{Pt-O}$')
ax[0,1].plot(time2,r1_Pt,label='$r_{Pt-Pt}$')
ax[1,0].plot(time3,ss1_O,label='$\sigma^2_{Pt-O}$')
ax[1,1].plot(time4,ss1_Pt,label='$\sigma^2_{Pt-Pt}$')
ax[2,0].plot(time5,cn1_O,label='$CN_{Pt-O}$')
ax[2,1].plot(time6,cn1_Pt,label='$CN_{Pt-Pt}$')
ax[0,0].legend(frameon=False)
ax[0,1].legend(frameon=False)
ax[1,0].legend(frameon=False)
ax[1,1].legend(frameon=False)
ax[2,0].legend(frameon=False)
ax[2,1].legend(frameon=False)
ax[0,0].set_xlabel('Time')
ax[0,1].set_xlabel('Time')
ax[1,0].set_xlabel('Time')
ax[1,1].set_xlabel('Time')
ax[0,0].set_ylabel('r($\AA$)')
ax[0,1].set_ylabel('r($\AA$)')
ax[1,0].set_ylabel('$\sigma^2$($\AA^2$)')
ax[1,1].set_ylabel('$\sigma^2$($\AA^2$)')
ax[2,0].set_ylabel('CN')
ax[2,1].set_ylabel('CN')
ax[2,0].set_ylim(-1,2)
ax[2,1].set_ylim(8,11)
plt.tight_layout()



