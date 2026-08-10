from copy import deepcopy
from larch.io import read_athena
from larch.xafs import find_e0, pre_edge, autobk, xftf, xftr
from larch.xafs import cauchy_wavelet
from larch import Group
from os.path import basename, join, exists
from sys import platform
from larch.xafs.feffit import feffit_transform, feffit_dataset, feffit, feffit_report
from larch.fitting import param, param_group
from larch.xafs import feffpath, feffrunner
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist
import pandas as pd
import concurrent.futures as confu
from functools import partial
from tqdm import tqdm
from warnings import warn
from re import search, match, split
from typing import List


def data_processing(
    group,
    e0=None,
    pre_start=-30,
    pre_end=-10,
    post_start=20,
    post_end=900,
    kweight=2,
    rbkg=1,
    plot=False,
    kwin="hanning",
    dk=2,
    krange=[2, 8],
    nnorm=3,
):
    """
    Process the data for X-ray absorption spectroscopy analysis.

    Args:
    - group: dictionary containing energy and mu data
    - e0: edge energy NOTE: e0 can be slightly different from given value,
          since the algorithm in the pre-edge map the given e0 to the nearest energy point.
    - pre_start: start of the pre-edge region
    - pre_end: end of the pre-edge region
    - post_start: start of the post-edge region
    - post_end: end of the post-edge region
    - kweight: k-weight for the Fourier transform
    - rbkg: rbkg for autobk function to remove background
    - plot: boolean flag to indicate whether to display plots
    - kwin: window function for Fourier transform
    - dk: k-space step size
    - krange: range of k values for Fourier transform
    - nnorm: number of normalization functions

    Returns:
    - Processed X-ray absorption spectroscopy data

    """
    # print(f"e0={e0}")
    if "e0" not in group.keys() and e0 is None:
        find_e0(group)
        e0 = group.e0
    # print(f"e0={e0}")
    if e0 is not None:
        group.e0 = e0
    interfunc = interp1d(group.energy, group.mu)
    # print(f"group.e0={group.e0}")
    pre_edge(
        group,
        pre1=pre_start,
        pre2=pre_end,
        norm1=post_start,
        norm2=post_end,
        nnorm=nnorm,
    )
    # print(f"after pre_edge group.e0={group.e0}")
    autobk(energy=group.energy, mu=group.mu, group=group, rbkg=rbkg, kweight=2)
    # print(f"after autobk group.e0={group.e0}")
    xftf(
        k=group.k,
        chi=group.chi,
        dk=dk,
        kweight=kweight,
        group=group,
        kmin=krange[0],
        kmax=krange[1],
        kstep=group.k[2] - group.k[1],
        window=kwin,
    )
    if plot:
        k_inten_unit = -kweight
        r_inten_unit = -kweight - 1
        fig, ax = plt.subplots(2, 2, figsize=(8, 5))
        # Plot data and calculated values
        ax[0, 0].plot(group.energy, group.mu, label=group.label)
        ax[0, 0].plot(group.energy, group.bkg)
        ax[0, 0].plot(group.energy, group.pre_edge)
        ax[0, 0].plot(group.energy, group.post_edge)
        # Add vertical lines for edge and normalization regions
        ax[0, 0].vlines(
            pre_start + group.e0, 0, np.max(group.mu), "b", alpha=0.3, linestyles="--"
        )
        ax[0, 0].vlines(
            pre_end + group.e0, 0, np.max(group.mu), "b", alpha=0.3, linestyles="--"
        )
        ax[0, 0].vlines(
            post_start + group.e0, 0, np.max(group.mu), "b", alpha=0.3, linestyles="--"
        )
        ax[0, 0].vlines(
            post_end + group.e0, 0, np.max(group.mu), "b", alpha=0.3, linestyles="--"
        )
        ax[0, 0].scatter([group.e0], [float(interfunc(group.e0))], c="r")
        ax[0, 0].set_xlabel("Energy(eV)")
        ax[0, 0].set_ylabel("$\mu(E)$")
        ax[0, 1].plot(group.energy, group.flat, label="flat")
        ax[0, 1].set_xlabel("Energy(eV)")
        ax[0, 1].set_ylabel("$\mu(E)$")
        ax[0, 0].legend(frameon=False)
        ax[0, 1].legend(frameon=False)
        chi_k = group.k**kweight * group.chi
        ax[1, 0].plot(group.k, chi_k, label="chi")
        ax[1, 0].plot(group.k, group.kwin * np.max(chi_k) * 1.05, label="kwin")
        ax[1, 0].set_xlabel("k(A$^{-1}$)")
        ax[1, 0].set_ylabel(f"$k^{kweight}$ $\chi(\AA^{k_inten_unit})$")
        ax[1, 0].set_xlim(0, krange[1] + 2)
        ax[1, 0].set_ylim(-np.max(chi_k) * 1.1, np.max(chi_k) * 1.1)
        ax[1, 1].plot(group.r, group.chir_mag, label="chir_mag")
        ax[1, 1].set_xlabel("r(A)")
        ax[1, 1].set_ylabel(f"$|\chi(R)|$ $(\AA^{r_inten_unit})$")
        ax[1, 1].set_xlim([0, 6])
        plt.tight_layout()
        print(f"E0={group.e0}")


def data_analysis_xanes(
    group,
    pre_start=-30,
    pre_end=-10,
    post_start=20,
    post_end=900,
    e0=None,
    plot=False,
    nnorm=3,
    energy_range=[],
    **kwargs,
):
    """
    Perform XANES data analysis on the given group.

    Args:
    - group: XANES data group
    - pre_start: start of pre-edge region
    - pre_end: end of pre-edge region
    - post_start: start of post-edge region
    - post_end: end of post-edge region
    - e0: edge jump energy
    - plot: flag to enable/disable plotting
    - nnorm: number of normalization points
    - energy_range: energy range for plotting
    - **kwargs: additional keyword arguments

    Returns:
    - None
    """

    if "e0" not in group.keys() or group.e0 is None or e0 is None:
        find_e0(group)  # Find the edge jump energy if not given
        e0 = group.e0
    interfunc = interp1d(group.energy, group.mu)  # Interpolate mu vs energy
    pre_edge(
        group,
        e0=e0,
        pre1=pre_start,
        pre2=pre_end,
        norm1=post_start,
        norm2=post_end,
        nnorm=nnorm,
        **kwargs,
    )  # Calculate pre-edge features
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 5))  # Create a figure for plotting
        ax[0].plot(group.energy, group.mu, label=group.label)  # Plot mu vs energy
        ax[0].plot(group.energy, group.pre_edge)  # Plot pre-edge
        ax[0].plot(group.energy, group.post_edge)  # Plot post-edge
        # Add vertical lines for specific energy points
        ax[0].vlines(
            pre_start + group.e0, 0, np.max(group.mu), "b", alpha=0.3, linestyles="--"
        )
        # Add more vertical lines for specific energy points
        ax[0].vlines(
            pre_end + group.e0, 0, np.max(group.mu), "b", alpha=0.3, linestyles="--"
        )
        ax[0].scatter([group.e0], [float(interfunc(group.e0))], c="r")  # Scatter plot
        ax[0].set_xlabel("Energy(eV)")  # Set x-axis label
        ax[0].set_ylabel("$\mu(E)$")  # Set y-axis label
        ax[1].plot(group.energy, group.flat, label="flat")  # Plot flat energy
        ax[1].set_xlabel("Energy(eV)")  # Set x-axis label
        ax[1].set_ylabel("$\mu(E)$")  # Set y-axis label
        ax[0].legend(frameon=False)  # Add legend without frame
        ax[1].legend(frameon=False)  # Add legend without frame
        ax[0].set_xlim([energy_range[0], energy_range[1]])  # Set x-axis limits
        ax[1].set_xlim([energy_range[0], energy_range[1]])  # Set x-axis limits
        plt.tight_layout()  # Adjust subplot parameters to give specified padding
        print(f"E0={group.e0}")  # Print edge jump energy


def wavelet_transform(
    group,
    kmin=0,
    kmax=20,
    rmin=0,
    rmax=20,
    kweight=0,
    rweight=0,
    dk=1,
    dr=0,
    windows="hanning",
    original=True,
    plot_spec=dict(),
):
    """
    plot_spec: {rrange=[],krange=[]}
    """
    print(group)
    xftf(
        group.k,
        group.chi,
        kmin=kmin,
        kmax=kmax,
        dk=dk,
        window=windows,
        kweight=kweight,
        group=group,
    )
    xftr(group.r, group.chir, rmin=rmin, rmax=rmax, dr=dr, window=windows, group=group)
    fig, ax = plt.subplots(
        2,
        2,
        figsize=(8, 8),
        gridspec_kw={"width_ratios": [1, 4], "height_ratios": [4, 1]},
    )
    k2chi = group.k * group.k * group.chi
    # morlet_wavelet(k=group.q,chi=group.chiq,group=group,kweight=0)
    if original == True:
        X, Y = np.meshgrid(group.k, group.r)
        cauchy_wavelet(
            k=group.k,
            chi=group.chi * group.k**kweight * group.kwin,
            group=group,
        )
    else:
        X, Y = np.meshgrid(group.q, group.r)
        cauchy_wavelet(k=group.q, chi=group.chiq, group=group)

    kwinmax = np.max(group.kwin)
    kmax = np.max(group.chi * group.k**kweight)
    rwinmax = np.max(group.rwin)
    rmax = np.max(group.chir_mag)

    # ax[0,1].contourf(, Y/2,coef,cmap='jet',levels=200, vmax=abs(coef).max(), vmin=-abs(coef).min())#,vmin=a0bs(coef).min(),vmax=abs(coef).max())
    # plt.imshow(back_f.wcauchy_mag,label='Wavelet Transform: Magnitude')
    cf = ax[0, 1].contourf(
        X,
        Y,
        group.wcauchy_mag,
        cmap="jet",
        levels=200,
        vmax=abs(group.wcauchy_re).max(),
        vmin=-abs(group.wcauchy_re).min(),
    )  # ,vmin=a0bs(coef).min(),vmax=abs(coef).max())
    cbar = fig.colorbar(
        cf, ax=ax[0, 1], orientation="vertical", fraction=0.05, pad=0.04
    )
    ax[0, 0].plot(group.chir_mag, group.r)
    if original == False:
        ax[0, 0].plot(group.rwin * rmax / rwinmax, group.r)
    # ax[0,0].invert_xaxis()
    ax[0, 0].set_ylabel("r")
    ax[0, 0].sharey(ax[0, 1])
    ax[1, 1].set_xlabel("k")
    ax[1, 1].sharex(ax[0, 1])
    if original == True:
        ax[1, 1].plot(group.k, group.k**kweight * group.chi)
        ax[1, 1].plot(group.k, group.kwin * kmax / kwinmax, "r-")
        ax[1, 1].plot(group.k, group.kwin * group.chi * group.k**kweight, "g--")
    if original == False:
        ax[1, 1].plot(group.q, group.chiq)
    #  ax[1,1].plot(group.k,group.kwin*kmax/kwinmax)
    ax[0, 1].set_ylim(plot_spec["rrange"][0], plot_spec["rrange"][1])
    ax[0, 1].set_xlim(plot_spec["krange"][0], plot_spec["krange"][1])

    ax[1, 0].remove()
    ax[1, 1].set_xlim(plot_spec["krange"][0], plot_spec["krange"][1])
    ax[1, 1].set_ylim(-kmax * 1.1, kmax * 1.1)


def plot_multi_spectrum(groupset, krange=[2, 12], kweight=2):
    # Create subplots with 2 rows and 2 columns
    fig, ax = plt.subplots(2, 2, figsize=(8, 5))

    # Iterate over each group in the groupset
    for group in groupset:
        # Interpolate the data
        interfunc = interp1d(group.energy, group.mu)

        # Plot various spectra and related data on the first subplot
        ax[0, 0].plot(group.energy, group.mu, label=group.label)
        ax[0, 0].plot(group.energy, group.bkg)
        ax[0, 0].plot(group.energy, group.pre_edge)
        ax[0, 0].plot(group.energy, group.post_edge)
        ax[0, 0].scatter([group.e0], [float(interfunc(group.e0))], c="r")
        ax[0, 0].set_xlabel("Energy(eV)")
        ax[0, 0].set_ylabel("$\mu(E)$")

        # Plot the flat spectrum on the second subplot
        ax[0, 1].plot(group.energy, group.flat, label="flat")
        ax[0, 1].set_xlabel("Energy(eV)")
        ax[0, 1].set_ylabel("$\mu(E)$")

        # Plot chi(k) and related data on the third subplot
        chi_k = group.k**kweight * group.chi
        ax[1, 0].plot(group.k, chi_k, label="chi")
        ax[1, 0].plot(group.k, group.kwin * np.max(chi_k) * 1.05, label="kwin")
        ax[1, 0].set_xlabel("k(A$^{-1}$)")
        ax[1, 0].set_ylabel(f"$k^{kweight}$ $\chi(\AA^-{kweight})$")
        ax[1, 0].set_xlim(0, krange[1] + 2)
        ax[1, 0].set_ylim(-np.max(chi_k) * 1.1, np.max(chi_k) * 1.1)

        # Plot chir_mag on the fourth subplot
        ax[1, 1].plot(group.r, group.chir_mag, label="chir_mag")
        ax[1, 1].set_xlabel("r(A)")
        ax[1, 1].set_ylabel("$|\chi(R)|$")
        ax[1, 1].set_xlim([0, 6])

    # Adjust the layout to prevent overlapping
    plt.tight_layout()


def BF_analysis(group, kweight=3, rrange=[], krange=[]):
    """
    Perform Back Fourier Analysis on the input group data
    with optional kweight, rrange, and krange parameters
    Perform Fourier Transform
    """
    xftr(
        r=group.r,
        chir=group.chir,
        rmin=rrange[0],
        rmax=rrange[1],
        dr=0,
        group=group,
        window="hanning",
        kstep=group.k[2] - group.k[1],
    )

    # Create a deep copy of the input group
    BF_group = deepcopy(group)

    # Perform Inverse Fourier Transform
    xftf(
        k=BF_group.q,
        chi=BF_group.chiq,
        dk=2,
        group=BF_group,
        kweight=0,
        kmin=krange[0],
        kmax=krange[1],
        kstep=group.k[2] - group.k[1],
        window="hanning",
    )

    # Create another deep copy of the transformed BF group
    BF_group_2 = deepcopy(BF_group)

    # Perform Fourier Transform again
    xftr(
        r=BF_group_2.r,
        chir=BF_group_2.chir,
        dr=0,
        group=BF_group_2,
        kweight=0,
        rmin=rrange[0],
        rmax=rrange[1],
        kstep=group.k[2] - group.k[1],
        window="hanning",
    )

    # Plot the results in a 2x1 grid of subplots
    fig, ax = plt.subplots(2, 1, figsize=(8, 5))
    ax[0].plot(group.k, group.k**kweight * group.chi, color="b", label="chi")
    ax[0].plot(group.q, group.chiq, color="r", label="Back Fourier Transform")
    ax[0].plot(
        BF_group_2.q, BF_group_2.chiq, color="g", label="Fourier transform again"
    )
    ax[0].set_xlabel("k(A$^{-1}$)")
    ax[0].set_ylabel(f"$k^{kweight}\chi(\AA^{-kweight})$")
    ax[0].set_xlim(0, 12)
    ax[0].legend(frameon=False)
    ax[1].plot(group.r, group.chir_mag, color="b", label="chir_mag")
    ax[1].plot(BF_group.r, BF_group.chir_mag, color="r", label="Back Fourier Transform")
    ax[1].legend(frameon=False)
    ax[1].set_xlabel("r(A)")
    ax[1].set_ylabel("$|\chi(R)|")


def deglitch_Sfilter(E, mu, act_range=[], window_length=10, polyorder=3):
    min_point = find_nearest_idx(E, act_range[0])
    if act_range[1] is None:
        mu_act = mu[min_point:]
    else:
        max_point = find_nearest_idx(E, act_range[1])
        mu_act = mu[min_point:max_point]
    mu_filter = savgol_filter(mu_act, window_length, polyorder)
    if act_range[1] is None:
        mu[min_point:] = mu_filter
    else:
        mu[min_point:max_point] = mu_filter
    return mu


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class zanes_data_analysis:

    def __init__(self, data: None | list[Group] = None):
        if data is None:
            self.data = []
        else:
            self.data = data

    def read_datacollection(
        self, filename, datatype: str = "Athena", read_range: int | list[int] = None
    ) -> None:
        if isinstance(read_range, int):
            scan = 0
        elif isinstance(read_range, list) and len(read_range) == 2:
            scan_begin = read_range[0]
            scan_end = read_range[1]
        elif read_range is None:
            pass
        else:
            raise ValueError("read_range is wrong!")
        if datatype == "Athena":
            project_group = read_athena(filename)
            if isinstance(read_range, (int, np.int16, np.int32, np.int64)):
                for key in project_group:
                    self.data.append(project_group[key])
                    scan += 1
                    if scan == read_range:
                        break
            if isinstance(read_range, (list, np.ndarray)) and read_range != []:
                keys = project_group.keys()
                for key in keys[scan_begin:scan_end]:
                    self.data.append(project_group[key])
            if read_range is None:
                keys = project_group.keys()
                for key in keys:
                    self.data.append(project_group[key])

        if datatype == "csv":
            data_get = pd.read_csv(filename)
            keys = data_get.keys()
            if (
                not isinstance(read_range, (list, np.ndarray))
                and read_range is not None
            ):
                for i in range(1, len(keys)):

                    self.data.append(
                        Group(
                            energy=np.array(data_get["E"]),
                            mu=np.array(data_get[keys[i]]),
                            label=keys[i],
                        )
                    )
                    scan += 1
                    if scan == read_range:
                        break
            if isinstance(read_range, (list, np.ndarray)) and read_range != []:
                for i in range(scan_begin, scan_end):
                    self.data.append(
                        Group(
                            energy=np.array(data_get["E"]),
                            mu=np.array(data_get[keys[i]]),
                            label=keys[i],
                        )
                    )
            if read_range is None:
                for i in range(1, len(keys)):
                    self.data.append(
                        Group(
                            energy=np.array(data_get["E"]),
                            mu=np.array(data_get[keys[i]]),
                            label=keys[i],
                        )
                    )
        if datatype == "ProQEXAFS":
            data_get = self.read_data_raw(filename)
            keys = data_get.keys()
            if (
                not isinstance(read_range, (list, np.ndarray))
                and read_range is not None
            ):
                for i in range(1, len(keys)):

                    self.data.append(
                        Group(
                            energy=np.array(data_get["E"]),
                            mu=np.array(data_get[keys[i]]),
                            label=keys[i],
                        )
                    )
                    scan += 1
                    if scan == read_range:
                        break
            if isinstance(read_range, (list, np.ndarray)) and read_range != []:
                for i in range(scan_begin, scan_end):
                    self.data.append(
                        Group(
                            energy=np.array(data_get["E"]),
                            mu=np.array(data_get[keys[i]]),
                            label=keys[i],
                        )
                    )
            if read_range is None:
                for i in range(1, len(keys)):
                    self.data.append(
                        Group(
                            energy=np.array(data_get["E"]),
                            mu=np.array(data_get[keys[i]]),
                            label=keys[i],
                        )
                    )

    def read_from_table(self, table, read_range):
        if isinstance(read_range, int):
            scan = 0
        elif isinstance(read_range, list) and len(read_range) == 2:
            scan_begin = read_range[0]
            scan_end = read_range[1]
        elif read_range is None:
            pass
        else:
            raise ValueError("read_range is wrong!")
        keys = table.keys()
        if not isinstance(read_range, (list, np.ndarray)) and read_range is not None:
            for i in range(1, len(keys)):

                self.data.append(
                    Group(
                        energy=np.array(table["E"]),
                        mu=np.array(table[keys[i]]),
                        label=keys[i],
                    )
                )
                scan += 1
                if scan == read_range:
                    break
        if isinstance(read_range, (list, np.ndarray)) and read_range != []:
            for i in range(scan_begin, scan_end):
                self.data.append(
                    Group(
                        energy=np.array(table["E"]),
                        mu=np.array(table[keys[i]]),
                        label=keys[i],
                    )
                )
        if read_range is None:
            for i in range(1, len(keys)):
                self.data.append(
                    Group(
                        energy=np.array(table["E"]),
                        mu=np.array(table[keys[i]]),
                        label=keys[i],
                    )
                )

    def read_data_raw(self, filename: str) -> pd.DataFrame:
        """
        Read raw data from ProQEXAFS data file, using a very greedy way!
        """
        with open(filename) as file1:
            data = file1.readlines()
        dict_data = dict()
        for i in range(len(data)):
            data_lines = data[i].split()
            for j in range(len(data_lines)):
                if j == 0:
                    if "E" not in dict_data.keys():
                        dict_data["E"] = []
                    dict_data["E"].append(float(data_lines[j]))
                else:
                    if j not in dict_data.keys():
                        dict_data[j] = []
                    dict_data[j].append(float(data_lines[j]))
        return pd.DataFrame(dict_data)

    def deglitch_k(
        self, act_range, data_dict, kweight=0, window_length=10, polyorder=3
    ):
        for i in range(len(self.data)):
            k = self.data[i].k
            chi = self.data[i].chi
            chi_de = deglitch_Sfilter(
                k, k**kweight * chi, act_range, window_length, polyorder
            )
            self.data[i].chi = chi_de
            xftf(
                k=self.data[i].k,
                chi=self.data[i].chi,
                dk=2,
                kweight=data_dict["kweight"],
                group=self.data[i],
                kmin=data_dict["krange"][0],
                kmax=data_dict["krange"][1],
                kstep=self.data[i].k[2] - self.data[i].k[1],
                window="hanning",
            )

    def remove_outliers(
        self,
        rrange=[],
        pop=False,
        plot=False,
        method={"name": "max", "quantile": 0.95},
        name="",
        save_plot=False,
        save_plot_format="eps",
    ):
        """
        method={'name':'running_mean','num':5,'delta':0}
        method={'name':'max',"quantile":0.95}

        """

        def running_mean(dist, num):
            mean_array = []
            std_array = []
            for i in range(len(dist)):

                if num + i + 1 <= len(dist):
                    limit = np.quantile(dist[i : i + num + 1], 0.95)
                    # print(dist[i:i+num+1])
                    # calculate mean and std dev. no influence from outliers
                    mean_array.append(
                        np.mean(
                            [
                                dist[i : i + num + 1][j]
                                for j in range(num + 1)
                                if dist[i : i + num + 1][j] < limit
                            ]
                        )
                    )
                    std_array.append(
                        np.std(
                            [
                                dist[i : i + num + 1][j]
                                for j in range(num + 1)
                                if dist[i : i + num + 1][j] < limit
                            ]
                        )
                    )
                else:
                    limit = np.quantile(dist[i : len(dist)], 0.95)
                    mean_array.append(
                        np.mean(
                            [
                                dist[i : len(dist)][j]
                                for j in range(len(dist) - i)
                                if dist[i : len(dist)][j] < limit
                            ]
                        )
                    )
                    std_array.append(
                        np.std(
                            [
                                dist[i : len(dist)][j]
                                for j in range(len(dist) - i)
                                if dist[i : len(dist)][j] < limit
                            ]
                        )
                    )
            return mean_array, std_array

        rspace = []
        for i in range(len(self.data)):
            index = np.where(
                (self.data[i].r < rrange[1]) & (self.data[i].r > rrange[0])
            )[0]
            rspace.append(self.data[i].chir_mag[index[0] : index[-1]])

        rspace = np.array(rspace)
        dist_matrix = cdist(rspace, rspace)
        dist = np.mean(dist_matrix, axis=0)
        if method["name"] == "max":
            outlier_index = np.where(dist > np.quantile(dist, method["quantile"]))[0]
            print(f"num of outliers:{len(outlier_index)}")
            data_remain = [
                self.data[i] for i in range(len(self.data)) if i not in outlier_index
            ]
        if method["name"] == "running_mean":
            mean_array, std_array = running_mean(dist, method["num"])
            mean_array = np.array(mean_array)
            std_array = np.array(std_array)
            outlier_index = []

            for i in range(len(mean_array)):
                if (
                    dist[i] > mean_array[i] + method["delta"] * std_array[i]
                    or dist[i] < mean_array[i] - method["delta"] * std_array[i]
                ):
                    outlier_index.append(i)
            outlier_index = np.array(outlier_index)
            print(f"num of outliers:{len(outlier_index)}")
            data_remain = [
                self.data[i] for i in range(len(self.data)) if i not in outlier_index
            ]
        if plot == True:
            plt.figure()
            plt.plot(np.arange(len(dist)), dist, label="data", color="cyan", alpha=0.7)
            plt.plot(outlier_index, dist[outlier_index], ".", label="outlier")
            if method["name"] == "running_mean":
                plt.plot(
                    np.arange(len(self.data)),
                    mean_array,
                    "--",
                    color="red",
                    alpha=0.6,
                    linewidth=2,
                    label="mean",
                )
                delta = method["delta"]
                plt.fill_between(
                    np.arange(len(self.data)),
                    mean_array - method["delta"] * std_array,
                    mean_array + method["delta"] * std_array,
                    color="orange",
                    alpha=0.5,
                    label=f"mean$\pm${delta}$\sigma$",
                )
            plt.legend(frameon=False)
            plt.ylabel("distance")
            plt.xlabel("data index")
            plt.title(f"outlier detection: {name}")
            if save_plot == True:
                plt.savefig(
                    name + f"_outlier_detection.{save_plot_format}",
                    format=save_plot_format,
                )

        if pop == True:
            self.data = data_remain
            print(f"remain data={len(self.data)}")
        return dist

    def pop_data(self, index):
        self.data.pop(index)

    def process_data(
        self, data_dict: dict, plot: bool = False, group_plot: bool = False
    ):
        self.data_processing_params = data_dict
        for d in self.data:
            if "e0" in data_dict.keys():
                e0 = data_dict["e0"]
            else:
                e0 = None
            data_processing(
                d,
                e0=e0,
                pre_start=data_dict["pre_start"],
                pre_end=data_dict["pre_end"],
                post_start=data_dict["post_start"],
                post_end=data_dict["post_end"],
                kweight=data_dict["kweight"],
                rbkg=data_dict["rbkg"],
                krange=data_dict["krange"],
                plot=plot,
            )
        if group_plot:
            plot_multi_spectrum(
                self.data, krange=data_dict["krange"], kweight=data_dict["kweight"]
            )

    def fit_param_batch(self, batch_size: int, **p) -> param_group:
        # keys = p.keys()
        fit_param_get = {}
        for k_SO2 in p["SO2"].keys():
            if p["SO2"][k_SO2]["global"]:
                fit_param_get[k_SO2] = param(
                    p["SO2"][k_SO2]["initial"], vary=p["SO2"][k_SO2]["vary"]
                )
            else:
                for i in range(batch_size):
                    fit_param_get[f"{k_SO2}_{i}"] = param(
                        p["SO2"][k_SO2]["initial"], vary=p["SO2"][k_SO2]["vary"]
                    )
        for k_N in p["N"].keys():
            if p["N"][k_N]["global"]:
                fit_param_get[k_N] = param(
                    p["N"][k_N]["initial"], vary=p["N"][k_N]["vary"]
                )
            else:
                for i in range(batch_size):
                    fit_param_get[f"{k_N}_{i}"] = param(
                        p["N"][k_N]["initial"], vary=p["N"][k_N]["vary"]
                    )
        for k_dele in p["dele"].keys():
            if p["dele"][k_dele]["global"]:
                fit_param_get[k_dele] = param(
                    p["dele"][k_dele]["initial"], vary=p["dele"][k_dele]["vary"]
                )
            else:
                for i in range(batch_size):
                    fit_param_get[f"{k_dele}_{i}"] = param(
                        p["dele"][k_dele]["initial"], vary=p["dele"][k_dele]["vary"]
                    )
        for k_ss2 in p["ss2"].keys():
            if not p["ss2"][k_ss2]["thermal"]["type"]:
                if p["ss2"][k_ss2]["global"]:
                    fit_param_get[k_ss2] = param(
                        p["ss2"][k_ss2]["initial"], vary=p["ss2"][k_ss2]["vary"]
                    )
                else:
                    for i in range(batch_size):
                        fit_param_get[f"{k_ss2}_{i}"] = param(
                            p["ss2"][k_ss2]["initial"], vary=p["ss2"][k_ss2]["vary"]
                        )
            if p["ss2"][k_ss2]["thermal"]["type"] == "Einstein":
                if p["ss2"][k_ss2]["global"]:
                    suffix = k_ss2.split("_")[1]
                    fit_param_get[k_ss2] = param(
                        p["ss2"][k_ss2]["initial"], vary=p["ss2"][k_ss2]["vary"]
                    )
                    fit_param_get[f"theta_{suffix}"] = param(
                        p["ss2"][k_ss2]["thermal"]["theta"]["initial"],
                        vary=p["ss2"][k_ss2]["thermal"]["theta"]["vary"],
                    )
                else:
                    """
                    only change static disorder
                    """
                    for i in range(batch_size):
                        suffix = k_ss2.split("_")[1]
                        fit_param_get[f"{k_ss2}_{i}"] = param(
                            p["ss2"][k_ss2]["initial"], vary=p["ss2"][k_ss2]["vary"]
                        )
                        fit_param_get[f"theta_{suffix}"] = param(
                            p["ss2"][k_ss2]["thermal"]["theta"]["initial"],
                            vary=p["ss2"][k_ss2]["thermal"]["theta"]["vary"],
                        )
            if p["ss2"][k_ss2]["thermal"]["type"] == "Debye":
                if p["ss2"][k_ss2]["global"]:
                    suffix = k_ss2.split("_")[1]
                    fit_param_get[k_ss2] = param(
                        p["ss2"][k_ss2]["initial"], vary=p["ss2"][k_ss2]["vary"]
                    )
                    fit_param_get[f"theta_{suffix}"] = param(
                        p["ss2"][k_ss2]["thermal"]["theta"]["initial"],
                        vary=p["ss2"][k_ss2]["thermal"]["theta"]["vary"],
                    )
                else:
                    for i in range(batch_size):
                        suffix = k_ss2.split("_")[1]
                        fit_param_get[f"{k_ss2}_{i}"] = param(
                            p["ss2"][k_ss2]["initial"], vary=p["ss2"][k_ss2]["vary"]
                        )
                        fit_param_get[f"theta_{suffix}"] = param(
                            p["ss2"][k_ss2]["thermal"]["theta"]["initial"],
                            vary=p["ss2"][k_ss2]["thermal"]["theta"]["vary"],
                        )
            for dr_k in p["delr"].keys():
                if p["delr"][dr_k]["global"]:
                    fit_param_get[dr_k] = param(
                        p["delr"][dr_k]["initial"], vary=p["delr"][dr_k]["vary"]
                    )
                else:
                    for i in range(batch_size):
                        fit_param_get[f"{dr_k}_{i}"] = param(
                            p["delr"][dr_k]["initial"], vary=p["delr"][dr_k]["vary"]
                        )
        return param_group(**fit_param_get)

    def path_param_batch(
        self,
        feff_folder: list[str],
        fitpath_num: list[list[int]],  # feff path from different file
        rules: dict,
        temp=[],
        batch_size: int = 0,
    ) -> list:
        # for now, only consider the first path
        keys = rules.keys()
        paths = []
        # iteration over different data sets
        for i in range(batch_size):
            # iteration over different feff files
            for f in range(len(feff_folder)):
                # read all paths in the fth folder
                feff_paths = self.feff_path(feff_folder[f])
                N_k = list(rules["N"].keys())
                SO2_k = list(rules["SO2"].keys())
                dele_k = list(rules["dele"].keys())
                ss2_k = list(rules["ss2"].keys())
                dr_k = list(rules["delr"].keys())
                # print(N_k, SO2_k, dele_k, ss2_k, dr_k)

                total_path_num = int(
                    np.sum([len(fitpath_num[ff]) for ff in range(len(fitpath_num))])
                )
                # print(total_path_num)
                # iteration over different paths in the fth folder
                for j in range(len(fitpath_num[f])):
                    s02 = None
                    dele = None
                    ss2 = None
                    delr = None

                    if len(N_k) == total_path_num:
                        # print('pass_1')
                        if (
                            rules["SO2"][SO2_k[0]]["global"]
                            and rules["N"][N_k[f + j]]["global"]
                        ):
                            s02 = f"{N_k[f+j]}*SO2"
                        elif (
                            not rules["SO2"][SO2_k[0]]["global"]
                            and not rules["N"][N_k[f + j]]["global"]
                        ):
                            s02 = f"{N_k[f+j]}_{i}*SO2_{i}"
                        elif (
                            rules["SO2"][SO2_k[0]]["global"]
                            and not rules["N"][N_k[f + j]]["global"]
                        ):
                            s02 = f"{N_k[f+j]}_{i}*SO2"
                        else:
                            s02 = f"N*SO2_{i}"
                    # share path parameter
                    if len(N_k) < total_path_num:
                        # print('pass_2')
                        if (
                            rules["SO2"][SO2_k[0]]["global"]
                            and rules["N"][N_k[0]]["global"]
                        ):
                            s02 = f"{N_k[0]}*SO2"
                        elif (
                            not rules["SO2"][SO2_k[0]]["global"]
                            and not rules["N"][N_k[0]]["global"]
                        ):
                            s02 = f"{N_k[0]}_{i}*SO2_{i}"
                        elif (
                            rules["SO2"][SO2_k[0]]["global"]
                            and not rules["N"][N_k[0]]["global"]
                        ):
                            s02 = f"{N_k[0]}_{i}*SO2"
                        else:
                            s02 = f"N*SO2_{i}"
                    if len(dele_k) == total_path_num:
                        if rules["dele"][dele_k[f + j]]["global"]:
                            dele = dele_k[f + j]
                        else:
                            dele = f"{dele_k[f+j]}_{i}"

                    if (
                        len(dele_k) < total_path_num
                    ):  # share variables for different paths

                        if rules["dele"][dele_k[0]]["global"]:
                            dele = dele_k[0]
                        else:
                            dele = f"{dele_k[0]}_{i}"
                    if len(ss2_k) == total_path_num:
                        if not rules["ss2"][ss2_k[f + j]]["thermal"]:
                            if rules["ss2"][ss2_k[f + j]]["global"]:
                                ss2 = ss2_k[f + j]
                            else:
                                ss2 = f"{ss2_k[f+j]}_{i}"
                        else:
                            suffix = ss2_k[f + j].split("_")[1]
                            if rules["ss2"][ss2_k[f + j]]["thermal"] == "Einstein":
                                ss2 = f"{ss2_k[f+j]}+sigma2_eins({temp[i]},theta_{suffix})"
                            elif rules["ss2"][ss2_k[f + j]]["thermal"] == "Debye":
                                ss2 = f"{ss2_k[f+j]}+sigma2_debye({temp[i]},theta_{suffix})"
                    if len(ss2_k) < total_path_num:
                        if not rules["ss2"][ss2_k[0]]["thermal"]:
                            if rules["ss2"][ss2_k[0]]["global"]:
                                ss2 = ss2_k[0]
                            else:
                                ss2 = f"{ss2_k[0]}_{i}"
                        else:
                            suffix = ss2_k[0].split("_")[1]
                            if rules["ss2"][ss2_k[0]]["thermal"] == "Einstein":
                                ss2 = (
                                    f"{ss2_k[0]}+sigma2_eins({temp[i]},theta_{suffix})"
                                )
                            elif rules["ss2"][ss2_k[0]]["thermal"] == "Debye":
                                ss2 = (
                                    f"{ss2_k[0]}+sigma2_debye({temp[i]},theta_{suffix})"
                                )
                    if len(dr_k) == total_path_num:
                        if rules["delr"][dr_k[f + j]]["global"]:
                            delr = dr_k[f + j]
                        if not rules["delr"][dr_k[f + j]]["global"]:
                            delr = f"{dr_k[f+j]}_{i}"
                    if len(dr_k) < total_path_num:
                        if rules["delr"][dr_k[0]]["global"]:
                            delr = dr_k[0]
                        if not rules["delr"][dr_k[0]]["global"]:
                            delr = f"{dr_k[0]}_{i}"

                    # print(rules)
                    paths.append(
                        feffpath(
                            f"{feff_folder[f]}/{feff_paths['file'][fitpath_num[f][j]]}",
                            s02=s02,  # abs(N*SO2)
                            degen=1,
                            e0=dele,
                            sigma2="abs(" + ss2 + ")",
                            deltar=delr,
                        )
                    )
        return paths

    def feff_call(self, path: str) -> None:
        feff = feffrunner(folder=path, feffinp="feff.inp")
        feff.run()

    def feff_path(self, feff_folder: str) -> pd.DataFrame:
        if not exists(join(feff_folder, "files.dat")):
            if exists(join(feff_folder, "feff.inp")):
                self.feff_call(feff_folder)
            else:
                raise ValueError("feff.inp is not in the folder")
        with open(join(feff_folder, "files.dat"), "r") as f:
            lines = f.readlines()
        num = 0
        for i in range(len(lines)):
            if search("^(\s+[A-Za-z]+\d*){8}$", lines[i]):
                num = i
                break
        return pd.read_csv(
            join(feff_folder, "files.dat"),
            sep=" +",
            skiprows=num + 1,
            names=["file", "sig2", "amp ratio", "deg", "nlegs", "r effective"],
        )

    def feff_rules(self, **param_dict):
        rules = dict()
        rules["SO2"] = {}
        for k_SO2 in param_dict["SO2"].keys():
            rules["SO2"][k_SO2] = {
                "vary": param_dict["SO2"][k_SO2]["vary"],
                "global": param_dict["SO2"][k_SO2]["global"],
            }
        rules["N"] = {}
        for k_N in param_dict["N"].keys():
            rules["N"][k_N] = {
                "vary": param_dict["N"][k_N]["vary"],
                "global": param_dict["N"][k_N]["global"],
            }

        rules["ss2"] = {}
        for k_ss2 in param_dict["ss2"].keys():

            if not param_dict["ss2"][k_ss2]["thermal"]["type"]:
                rules["ss2"][k_ss2] = {
                    "vary": param_dict["ss2"][k_ss2]["vary"],
                    "global": param_dict["ss2"][k_ss2]["global"],
                    "thermal": False,
                }
            else:
                rules["ss2"][k_ss2] = {
                    "vary": param_dict["ss2"][k_ss2]["vary"],
                    "global": param_dict["ss2"][k_ss2]["global"],
                    "thermal": param_dict["ss2"][k_ss2]["thermal"]["type"],
                }
                theta = param_dict["ss2"][k_ss2]["thermal"]["theta"]
                rules["ss2"][k_ss2].update(
                    {"theta": {"vary": theta["vary"], "global": theta["global"]}}
                )
        rules["delr"] = {}
        for dr_k in param_dict["delr"].keys():
            rules["delr"][dr_k] = {
                "vary": param_dict["delr"][dr_k]["vary"],
                "global": param_dict["delr"][dr_k]["global"],
            }
        rules["dele"] = {}
        for dele in param_dict["dele"].keys():
            rules["dele"][dele] = {
                "vary": param_dict["dele"][dele]["vary"],
                "global": param_dict["dele"][dele]["global"],
            }
        return rules

    def fit_one_batch(
        self,
        parm_dict: dict,
        feff_folder: list[str],
        fitpath_num: list[list[int]],
        fit_range_param: dict,
        batch_size: int,
        batch_index: int,
    ):
        dset = []
        true_batch_size = 0
        batch_index = batch_index + 1
        rules = self.feff_rules(**parm_dict)
        batch_num = len(self.data) // batch_size
        if batch_index == batch_num:
            batch_size_last = len(self.data) - batch_size * (batch_index - 1)
            pars = self.fit_param_batch(batch_size_last, **parm_dict)
            paths = self.path_param_batch(
                feff_folder, fitpath_num, rules=rules, batch_size=batch_size_last
            )
        else:
            pars = self.fit_param_batch(batch_size, **parm_dict)
            paths = self.path_param_batch(
                feff_folder, fitpath_num, rules=rules, batch_size=batch_size
            )
        trans = feffit_transform(**fit_range_param)
        # print(paths)
        # sa_check=[]
        # sa_check2=[]
        special_paths = len(paths) // batch_size
        # print(f"special_paths={special_paths}")
        # number of specific paths for each data
        if batch_index * batch_size <= len(self.data) and batch_index < batch_num:
            batch_add_i = 0
            for i in range((batch_index - 1) * batch_size, batch_index * batch_size):
                # sa_check.append(i)
                # sa_check2.append(i-(batch_index-1)*batch_size)
                # print(paths[batch_add_i*special_paths:(batch_add_i+1)*special_paths])
                dset.append(
                    feffit_dataset(
                        data=self.data[i],
                        pathlist=paths[
                            batch_add_i
                            * special_paths : (batch_add_i + 1)
                            * special_paths
                        ],  # [i - (batch_index - 1) * batch_size]]
                        transform=trans,
                    )
                )
                batch_add_i += 1
            # batch_add_i: ith data in the batch
            true_batch_size = batch_size

            # linux has problem with global constraints, so need to set fix_unused_variables=False
            if platform == "linux":
                out = feffit(pars, dset, fix_unused_variables=False)
            else:
                out = feffit(pars, dset)
            report = feffit_report(out)

        if batch_index == batch_num:
            batch_add_i = 0
            for i in range((batch_index - 1) * batch_size, len(self.data)):
                # last iteration
                # print(batch_add_i * special_paths, (batch_add_i + 1) * special_paths)
                dset.append(
                    feffit_dataset(
                        data=self.data[i],
                        pathlist=paths[
                            batch_add_i
                            * special_paths : (batch_add_i + 1)
                            * special_paths
                        ],  # [i - (batch_index - 1) * batch_size]
                        transform=trans,
                    )
                )
                # batch_add_i: ith data in the batch
                batch_add_i += 1

            true_batch_size = len(self.data) - batch_size * (batch_index - 1)

            # linux has problem with global constraints, so need to set fix_unused_variables=False
            if platform == "linux":
                out = feffit(pars, dset, fix_unused_variables=False)
            else:
                out = feffit(pars, dset)
            report = feffit_report(out)
        # print(sa_check)
        # print(sa_check2)
        return dset, out, report, true_batch_size, paths

    def run_fit_batch(
        self,
        param_dict: dict,
        fit_range_param: dict,
        fitpath_num: list[list[int]],
        feff_folder: list[str],
        save_name: str = "fit.out",
        write: bool = False,
        mpi: bool = False,
        core=1,
        batch_size: int | None = None,
        batch_num: int | None = None,
    ):
        if batch_size is None:
            batch_size = len(self.data)
        else:
            batch_size = batch_size
        if batch_num is None:
            batch_num = len(self.data) // batch_size
        elif batch_num != None and batch_num <= len(self.data) // batch_size:
            batch_num = batch_num
        else:
            batch_num = len(self.data) // batch_size
            warn(
                f"The batch_num exceed the maximum value of batch number, It changed to the maximum batch number {batch_num} automatically."
            )
        self.batch_size = batch_size
        dsets, outs, reports, batch_nums = [], [], [], []
        # print(fitpath_num)
        if mpi == False:
            for i in tqdm(range(batch_num), total=batch_num):
                # print(i)
                dset, out, report, batch_num, paths = self.fit_one_batch(
                    param_dict,
                    feff_folder,
                    fitpath_num=fitpath_num,
                    fit_range_param=fit_range_param,
                    batch_size=batch_size,
                    batch_index=i,
                )
                dsets.append(dset)
                outs.append(out)
                reports.append(report)
                batch_nums.append(batch_num)
        if mpi == True and core > 1:
            run = partial(
                self.fit_one_batch,
                param_dict,
                feff_folder,
                fitpath_num,
                fit_range_param,
                batch_size,
            )
            my_iter = range(batch_num)
            with confu.ProcessPoolExecutor(max_workers=core) as executor:
                # jobs=tqdm([executor.submit(run,i) for i in range(batch_num)],total=batch_num)
                # for job in confu.as_completed(jobs):
                #     dset,out,report,batch_num=job.result()
                #     dsets.append(dset)
                #     outs.append(out)
                #     reports.append(report)
                #     batch_nums.append(batch_num)
                jobs = tqdm(executor.map(run, my_iter), total=batch_num)
            for job in jobs:
                dset, out, report, batch_num = job
                dsets.append(dset)
                outs.append(out)
                reports.append(report)
                batch_nums.append(batch_num)
        if write == True:
            self.write_report_mini_batch(save_name, reports, batch_nums)
        return dsets, outs, reports, paths

    def write_report_mini_batch(self, filename: str, report: list, batch_size: int):
        try:
            f = open(filename, "w")
            for i in range(len(report)):
                f.write(f"============batch {i}============\n")
                f.write(f"batch size: {batch_size[i]}\n")
                f.write(report[i])
                f.write("\n")
                f.write("\n")
            f.close()
        except:
            print("could not write %s" % filename)

    def write_sorted_report(
        self, out: list, report: list, param_dict, path, batch_size=None, batch_num=None
    ):
        if batch_size == None and batch_num == None:
            num = len(self.data)
            batch_size = self.batch_size
            batch_num = num // batch_size
        elif batch_size != None and batch_num != None:
            num = batch_size
            batch_num = batch_num
        if batch_size > len(self.data):
            raise ValueError("batch_size is larger than the data size")
        reff = []
        for i in range(len(path)):
            reff.append(path[i].reff)  # add reffs for different paths
        if isinstance(reff, float):
            reff = [reff]
        params = {}
        evaluation = self.get_chi_square_rfactor(report)
        k_eval = list(evaluation.keys())
        for k in k_eval:
            if k not in params.keys():
                params[k] = evaluation[k]
        for i in range(batch_num):
            for k in param_dict.keys():
                for k2 in param_dict[k].keys():
                    if param_dict[k][k2]["global"]:
                        if k2 not in params.keys():
                            params[k2] = {
                                "value": [out[i].params[k2].value],
                                "stderr": [out[i].params[k2].stderr],
                            }
                        else:
                            params[k2]["value"].append(out[i].params[k2].value)
                            params[k2]["stderr"].append(out[i].params[k2].stderr)
                    else:
                        num_k = len([key for key in out[i].params.keys() if k2 in key])
                        for j in range(num_k):
                            if k2 not in params.keys():
                                params[k2] = {
                                    f"{k2}_{i*batch_size+j}": {
                                        "value": out[i].params[f"{k2}_{j}"].value,
                                        "stderr": out[i].params[f"{k2}_{j}"].stderr,
                                    }
                                }
                            else:
                                params[k2][f"{k2}_{i*batch_size+j}"] = {
                                    "value": out[i].params[f"{k2}_{j}"].value,
                                    "stderr": out[i].params[f"{k2}_{j}"].stderr,
                                }

        for i in range(num):
            k2 = list(param_dict["delr"].keys())
            for k2i in range(len(k2)):
                if param_dict["delr"][k2[k2i]]["global"]:
                    ele = k2[k2i].split("_")[1]
                    params["r"] = {
                        f"r_{ele}": {
                            "value": params[f"delr_{ele}"]["value"] + reff[k2i],
                            "stderr": params["delr"][k2[k2i]]["stderr"],
                        }
                    }
                else:
                    ele = k2[k2i].split("_")[1]
                    if f"r_{ele}" not in params.keys():
                        # print(reff)
                        params[f"r_{ele}"] = {
                            f"r_{ele}_{i}": {
                                "value": params[f"delr_{ele}"][f"delr_{ele}_{i}"][
                                    "value"
                                ]
                                + reff[k2i],
                                "stderr": params[f"delr_{ele}"][f"delr_{ele}_{i}"][
                                    "stderr"
                                ],
                            }
                        }
                    else:
                        ele = k2[k2i].split("_")[1]
                        params[f"r_{ele}"].update(
                            {
                                f"r_{ele}_{i}": {
                                    "value": params[f"delr_{ele}"][f"delr_{ele}_{i}"][
                                        "value"
                                    ]
                                    + reff[k2i],
                                    "stderr": params[f"delr_{ele}"][f"delr_{ele}_{i}"][
                                        "stderr"
                                    ],
                                }
                            }
                        )
        return params

    def write_fitted_data(
        self, dset, filename: str, foldername: str, suffix: str
    ) -> None:
        data_k = {}
        data_r = {}
        fitted_k = {}
        fitted_r = {}
        kdata = dset[0][0].data.k
        min_k = []
        max_k = []
        min_r = []
        max_r = []
        file = basename(filename)
        print(file)
        for i in range(len(dset)):
            for j in range(len(dset[i])):
                min_k.append(min(dset[i][j].data.k))
                min_k.append(min(dset[i][j].model.k))
                max_k.append(max(dset[i][j].data.k))
                max_k.append(max(dset[i][j].model.k))
                min_r.append(min(dset[i][j].data.r))
                min_r.append(min(dset[i][j].model.r))
                max_r.append(max(dset[i][j].data.r))
                max_r.append(max(dset[i][j].model.r))
        max_min = max(min_k)
        min_max = min(max_k)
        max_min_r = max(min_r)
        min_max_r = min(max_r)
        kdata = np.round(np.linspace(max_min + 0.001, min_max - 0.001, 500), 4)
        rdata = np.round(np.linspace(max_min_r + 0.001, min_max_r - 0.001, 500), 4)
        np.savetxt(join(foldername, "kdata.txt"), kdata)
        np.savetxt(join(foldername, "rdata.txt"), rdata)
        for i in range(len(dset)):
            for j in range(len(dset[i])):
                # print((i,len(dset[i]),j))
                # warning the last dset[i] may have different length than others
                data_k[self.data[i * len(dset[0]) + j].label] = interp1d(
                    dset[i][j].data.k,
                    dset[i][j].data.chi
                    * dset[i][j].data.k ** dset[i][j].transform.kweight,
                )(kdata)
                data_r[self.data[i * len(dset[0]) + j].label] = interp1d(
                    dset[i][j].data.r, dset[i][j].data.chir_mag
                )(rdata)
                fitted_k[self.data[i * len(dset[0]) + j].label] = interp1d(
                    dset[i][j].model.k,
                    dset[i][j].model.chi
                    * dset[i][j].model.k ** dset[i][j].transform.kweight,
                )(kdata)
                fitted_r[self.data[i * len(dset[0]) + j].label] = interp1d(
                    dset[i][j].model.r, dset[i][j].model.chir_mag
                )(rdata)
        data_k_table = pd.DataFrame(data_k)
        data_r_table = pd.DataFrame(data_r)
        fitted_k_table = pd.DataFrame(fitted_k)
        fitted_r_table = pd.DataFrame(fitted_r)
        data_k_table.to_csv(
            join(foldername, f"{file[:-4]}{suffix}_kdata.csv"), index=False
        )
        data_r_table.to_csv(
            join(foldername, f"{file[:-4]}{suffix}_rdata.csv"), index=False
        )
        fitted_k_table.to_csv(
            join(foldername, f"{file[:-4]}{suffix}_kfitted.csv"), index=False
        )
        fitted_r_table.to_csv(
            join(foldername, f"{file[:-4]}{suffix}_rfitted.csv"), index=False
        )

    def get_chi_square_rfactor(self, report):
        evaluation = dict()
        evaluation["chi_square"] = []
        evaluation["reduced chi_square"] = []
        evaluation["r_factor"] = []
        for rep in report:
            lines = rep.split("\n")
            for i in range(len(lines)):
                if match("\ +r-factor\ +=\ +(\d+.?\d+)", lines[i]):
                    evaluation["r_factor"].append(
                        np.round(
                            np.float32(
                                match("\ +r-factor\ +=\ +(\d+.?\d+)", lines[i]).group(1)
                            ),
                            6,
                        )
                    )
                if match("\ +chi_square\ +=\ +(\d+.?\d+)", lines[i]):
                    evaluation["chi_square"].append(
                        np.round(
                            np.float32(
                                match("\ +chi_square\ +=\ +(\d+.?\d+)", lines[i]).group(
                                    1
                                )
                            ),
                            6,
                        )
                    )
                if match("\ +reduced chi_square\ +=\ +(\d+.?\d+)", lines[i]):
                    evaluation["reduced chi_square"].append(
                        np.round(
                            np.float32(
                                match(
                                    "\ +reduced chi_square\ +=\ +(\d+.?\d+)", lines[i]
                                ).group(1)
                            ),
                            6,
                        )
                    )
        return evaluation

    def clean_fit_param(self):
        self.fit_param_get = {}

    def plot_fitting_batch_results(self, batch_index, dset, d_range=[0, -1]):
        batch_size = len(dset[batch_index])
        for i in range(len(dset[batch_index][d_range[0] : d_range[1]])):
            mod = dset[batch_index][i].model
            dat = dset[batch_index][i].data
            kweight = dset[batch_index][i].transform.kweight
            data_chik = dat.chi * dat.k**kweight
            model_chik = mod.chi * mod.k**kweight
            fig, ax = plt.subplots(1, 2)
            ax[0].plot(
                dat.k,
                data_chik,
                color="cornflowerblue",
                linestyle="-",
                label=self.data[batch_size * batch_index + d_range[0] + i].label,
            )
            ax[0].plot(mod.k, model_chik, color="orange", linestyle="-", label="fit")
            ax[0].plot(dat.k, dat.kwin, color="lightcoral", linestyle="-", label="kwin")
            ax[0].legend(frameon=False)
            ax[1].plot(
                dat.r,
                dat.chir_mag,
                color="cornflowerblue",
                linestyle="-",
                label=self.data[batch_size * batch_index + d_range[0] + i].label,
            )
            ax[1].plot(mod.r, mod.chir_mag, color="orange", linestyle="-", label="fit")
            ax[1].plot(mod.r, mod.rwin, color="lightcoral", linestyle="-", label="rwin")
            ax[0].set_xlim(0, 20)
            ax[0].legend(frameon=False)
            ax[0].set_xlabel(f"k($\AA^{-1}$)")
            ax[1].set_xlabel("r(A)")
            ax[0].set_ylabel(f"$k^{-kweight}\chi(k)$ $(\AA^{-kweight})$")
            ax[1].set_ylabel(f"$|\chi(R)|$ ($\AA^{-(kweight+1)}$)")
            plt.tight_layout()
            plt.show()


# def check_convergence_batch(filename,datatype,data_dict,
#                             parm_dict,feff_folder,
#                             fitpath_num,fit_range_param,
#                             min_batch=1,max_batch=100,
#                             space=5):

#     batch_sizes=np.int32(np.linspace(min_batch,max_batch,num=(max_batch-min_batch+1)//space))
#     evaluation=[]

#     for batch in tqdm(batch_sizes,total=len(batch_sizes)):
#         exp_data=data_processing_multiOnerun(filename,datatype,read_range=batch)
#         exp_data.process_data(data_dict)
#         dset,out,report,paths=exp_data.run_fit(parm_dict,feff_folder,fitpath_num,
#                     fit_range_param)
#         lines=report.split('\n')
#         for i in range(len(lines)):
#             if match('\ +r-factor\ +=\ +(\d+.?\d+)',lines[i]):
#                     evaluation.append(np.float32(match('\ +r-factor\ +=\ +(\d+.?\d+)',lines[i]).group(1)))

#     return evaluation,batch_sizes
