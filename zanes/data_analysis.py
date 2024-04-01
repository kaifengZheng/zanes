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
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import pandas as pd
import concurrent.futures as confu
from functools import partial
from tqdm import tqdm
from warnings import warn
from re import search, match, split


def data_processing(
    group,
    e0=None,
    pre_start=-30,
    pre_end=-10,
    post_start=20,
    post_end=900,
    kweight=2,
    rbkg=1.5,
    plot=False,
    kwin="hanning",
    krange=[2, 8],
    nnorm=3,
    **kwargs,
):

    if "e0" not in group.keys() and group.e0 is None and e0 is None:
        find_e0(group)
        e0 = group.e0
    interfunc = interp1d(group.energy, group.mu)
    pre_edge(
        group,
        e0=e0,
        pre1=pre_start,
        pre2=pre_end,
        norm1=post_start,
        norm2=post_end,
        nnorm=nnorm,
        **kwargs,
    )
    autobk(energy=group.energy, mu=group.mu, group=group, e0=e0, rbkg=rbkg)
    xftf(
        k=group.k,
        chi=group.chi,
        dk=2,
        kweight=kweight,
        group=group,
        kmin=krange[0],
        kmax=krange[1],
        kstep=group.k[2] - group.k[1],
        window=kwin,
    )
    if plot:
        fig, ax = plt.subplots(2, 2, figsize=(8, 5))
        ax[0, 0].plot(group.energy, group.mu, label=group.label)
        ax[0, 0].plot(group.energy, group.bkg)
        ax[0, 0].plot(group.energy, group.pre_edge)
        ax[0, 0].plot(group.energy, group.post_edge)
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
        ax[1, 0].set_ylabel(f"$k^{kweight}$ $\chi(\AA^-{kweight})$")
        ax[1, 0].set_xlim(0, krange[1] + 2)
        ax[1, 0].set_ylim(-np.max(chi_k) * 1.1, np.max(chi_k) * 1.1)
        ax[1, 1].plot(group.r, group.chir_mag, label="chir_mag")
        ax[1, 1].set_xlabel("r(A)")
        ax[1, 1].set_ylabel("$|\chi(R)|$")
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

    if "e0" not in group.keys() and group.e0 is None and e0 is None:
        find_e0(group)
        e0 = group.e0
    interfunc = interp1d(group.energy, group.mu)
    pre_edge(
        group,
        e0=e0,
        pre1=pre_start,
        pre2=pre_end,
        norm1=post_start,
        norm2=post_end,
        nnorm=nnorm,
        **kwargs,
    )
    # autobk(energy=group.energy,mu=group.mu,group=group,e0=e0,rbkg=rbkg,nknots=nknots)
    # xftf(k=group.k,chi=group.chi, dk=2,kweight=kweight,group=group,kmin=krange[0],kmax=krange[1],kstep=group.k[2]-group.k[1],window=kwin)
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(8, 5))
        ax[0].plot(group.energy, group.mu, label=group.label)
        # ax[0].plot(group.energy,group.bkg)
        ax[0].plot(group.energy, group.pre_edge)
        ax[0].plot(group.energy, group.post_edge)
        ax[0].vlines(
            pre_start + group.e0, 0, np.max(group.mu), "b", alpha=0.3, linestyles="--"
        )
        ax[0].vlines(
            pre_end + group.e0, 0, np.max(group.mu), "b", alpha=0.3, linestyles="--"
        )
        ax[0].vlines(
            post_start + group.e0, 0, np.max(group.mu), "b", alpha=0.3, linestyles="--"
        )
        ax[0].vlines(
            post_end + group.e0, 0, np.max(group.mu), "b", alpha=0.3, linestyles="--"
        )
        ax[0].scatter([group.e0], [float(interfunc(group.e0))], c="r")
        ax[0].set_xlabel("Energy(eV)")
        ax[0].set_ylabel("$\mu(E)$")
        ax[1].plot(group.energy, group.flat, label="flat")
        ax[1].set_xlabel("Energy(eV)")
        ax[1].set_ylabel("$\mu(E)$")
        ax[0].legend(frameon=False)
        ax[1].legend(frameon=False)
        ax[0].set_xlim([energy_range[0], energy_range[1]])
        ax[1].set_xlim([energy_range[0], energy_range[1]])
        plt.tight_layout()
        print(f"E0={group.e0}")


def wavelet_transform(group, kweight=2, plot=False):
    """
    The group should contains chi and k
    Note: Cauchy_wavelet acts on the k^kweight*chi*win, not the k^weight*chi
    """

    cauchy_wavelet(
        group.k, group.k**kweight * group.chi * group.kwin, group=group, kweight=0
    )
    if plot == True:
        fig, ax = plt.subplots(
            2,
            2,
            figsize=(8, 8),
            gridspec_kw={"width_ratios": [1, 4], "height_ratios": [4, 1]},
        )
        # imopts = {'x': spectra_UF4.k, 'y': spectra_UF4.wcauchy_r}
        X, Y = np.meshgrid(group.wcauchy_r, group.k)
        contour = ax[0, 1].contourf(
            X,
            Y,
            group.wcauchy_mag.T,
            cmap="jet",
            levels=100,
            vmax=group.wcauchy_mag.max(),
            vmin=group.wcauchy_mag.min(),
        )  # ,vmin=a0bs(coef).min(),vmax=abs(coef).max())
        scale_win = np.max(group.k**kweight * group.chi * group.kwin) * 1.5
        ax[0, 0].set_ylabel("$k(\AA^{-1})$")
        ax[0, 0].set_xlabel("$k^3\chi(\AA^{-3})$")
        ax[0, 0].invert_xaxis()
        ax[1, 1].set_xlabel("$R(\AA)$")
        ax[0, 1].set_title("Cauchy Wavelet Transform for UF4")
        ax[0, 1].set_xlim([1, 5])
        ax[1, 1].plot(group.r, group.chir_mag, label="FFT")
        ax[1, 1].sharex(ax[0, 1])
        ax[0, 0].plot(group.k**3 * group.chi, group.k)
        ax[0, 0].plot(group.kwin * scale_win, group.k, "r")
        ax[0, 0].sharey(ax[0, 1])
        ax[1, 0].remove()
        plt.show()


def plot_multi_spectrum(groupset, krange=[2, 12]):
    fig, ax = plt.subplots(2, 2, figsize=(8, 5))
    for group in groupset:
        interfunc = interp1d(group.energy, group.mu)
        ax[0, 0].plot(group.energy, group.mu, label=group.label)
        ax[0, 0].plot(group.energy, group.bkg)
        ax[0, 0].plot(group.energy, group.pre_edge)
        ax[0, 0].plot(group.energy, group.post_edge)
        ax[0, 0].scatter([group.e0], [float(interfunc(group.e0))], c="r")
        ax[0, 0].set_xlabel("Energy(eV)")
        ax[0, 0].set_ylabel("$\mu(E)$")
        ax[0, 1].plot(group.energy, group.flat, label="flat")
        ax[0, 1].set_xlabel("Energy(eV)")
        ax[0, 1].set_ylabel("$\mu(E)$")
        ax[0, 0].legend(frameon=False)
        ax[0, 1].legend(frameon=False)
        chi_k = group.k**group.kweight * group.chi
        ax[1, 0].plot(group.k, chi_k, label="chi")
        ax[1, 0].plot(group.k, group.kwin * np.max(chi_k) * 1.05, label="kwin")
        ax[1, 0].set_xlabel("k(A$^{-1}$)")
        ax[1, 0].set_ylabel(f"$k^{group.kweight}$ $\chi(\AA^-{group.kweight})$")
        ax[1, 0].set_xlim(0, krange[1] + 2)
        ax[1, 0].set_ylim(-np.max(chi_k) * 1.1, np.max(chi_k) * 1.1)
        ax[1, 1].plot(group.r, group.chir_mag, label="chir_mag")
        ax[1, 1].set_xlabel("r(A)")
        ax[1, 1].set_ylabel("$|\chi(R)|$")
        ax[1, 1].set_xlim([0, 6])
    plt.tight_layout()


def BF_analysis(group, kweight=3, rrange=[], krange=[]):
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
    # chiq=interp1d(group.q,group.chiq)(group.k)
    BF_group = deepcopy(group)
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
    # # transform again
    BF_group_2 = deepcopy(BF_group)
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
    # chiq2=interp1d(BF_group.q,BF_group.chiq)(BF_group.k)
    # BF_group2=Group(k=BF_group.k,chi=chiq)
    # xftf(k=BF_group2.k,chi=BF_group2.chi, dk=2,group=BF_group2,kweight=0,kmin=krange[0],kmax=krange[1],window='hanning')

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
    ax[1].set_ylabel("$|\chi(R)|$")


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

    def __init__(
        self,
        filename: str,
        datatype: str,
        batch_size: int | None = None,
        read_range: int | list[int] | np.ndarray | None = None,
    ):
        self.data = []
        self.filename = filename
        self.datatype = datatype
        self.read_range = read_range
        self.read_data(self.filename, self.datatype, read_range=self.read_range)
        self.batch_size = batch_size
        if batch_size == None:
            batch_size = len(self.data)
        elif type(read_range) == int and read_range != None:
            self.batch_num = read_range // batch_size
        elif type(read_range) == list and read_range != [] and len(read_range) != 1:
            self.batch_num = (read_range[1] - read_range[0] + 1) // batch_size
            self.scan_begin = read_range[0]
            self.scan_end = read_range[1]
        elif read_range == None:
            self.batch_num = len(self.data) // batch_size
        else:
            raise ValueError("read_range is not correct")

    def read_data(
        self, filename, datatype: str = "Athena", read_range: int | list[int] = None
    ) -> None:
        if isinstance(read_range, int) and read_range is not None:
            scan = 0
        elif (
            isinstance(read_range, (list, np.ndarray))
            and read_range != []
            and len(read_range) != 1
        ):
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

    def read_data_raw(self, filename: str) -> pd.DataFrame:
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

    def process_data(self, data_dict: dict, plot: bool = False):
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

    def fit_param_batch(self, batch_size: int, **p) -> param_group:
        keys = p.keys()
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
        feff_folder: str,
        fitpath_num: list[int],
        rules: dict,
        temp=[],
        batch_size: int = 0,
    ) -> list:
        # for now, only consider the first path
        keys = rules.keys()
        feff_pathes = self.feff_path(feff_folder)
        paths = []

        for i in range(batch_size):
            N_k = list(rules["N"].keys())
            SO2_k = list(rules["SO2"].keys())
            dele_k = list(rules["dele"].keys())
            ss2_k = list(rules["ss2"].keys())
            dr_k = list(rules["delr"].keys())
            for j in range(len(fitpath_num)):
                if rules["SO2"][SO2_k[0]]["global"] and rules["N"][N_k[j]]["global"]:
                    s02 = f"{N_k[j]}*SO2"
                elif (
                    not rules["SO2"][SO2_k[0]]["global"]
                    and not rules["N"][N_k[j]]["global"]
                ):
                    s02 = f"{N_k[j]}_{i}*SO2_{i}"
                elif (
                    rules["SO2"][SO2_k[0]]["global"]
                    and not rules["N"][N_k[j]]["global"]
                ):
                    s02 = f"{N_k[j]}_{i}*SO2"
                else:
                    s02 = f"N*SO2_{i}"
                if rules["dele"][dele_k[j]]["global"]:
                    dele = dele_k[j]
                else:
                    dele = f"{dele_k[j]}_{i}"
                if not rules["ss2"][ss2_k[j]]["thermal"]:
                    if rules["ss2"][ss2_k[j]]["global"]:
                        ss2 = ss2_k[j]
                    else:
                        ss2 = f"{ss2_k[j]}_{i}"
                else:
                    suffix = ss2_k[j].split("_")[1]
                    if rules["ss2"][ss2_k[j]]["thermal"] == "Einstein":
                        ss2 = f"{ss2_k[j]}+sigma2_eins({temp[i]},theta_{suffix})"
                    elif rules["ss2"][ss2_k[j]]["thermal"] == "Debye":
                        ss2 = f"{ss2_k[j]}+sigma2_debye({temp[i]},theta_{suffix})"
                if rules["delr"][dr_k[j]]["global"]:
                    delr = dr_k[j]
                if not rules["delr"][dr_k[j]]["global"]:
                    delr = f"{dr_k[j]}_{i}"

                paths.append(
                    feffpath(
                        f"{feff_folder}/{feff_pathes['file'][fitpath_num[j]]}",
                        s02=s02,
                        degen=1,
                        e0=dele,
                        sigma2=ss2,
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

    def feff_rules(self, **parm_dict):
        rules = dict()
        rules["SO2"] = {}
        for k_SO2 in parm_dict["SO2"].keys():
            rules["SO2"][k_SO2] = {
                "vary": parm_dict["SO2"][k_SO2]["vary"],
                "global": parm_dict["SO2"][k_SO2]["global"],
            }
        rules["N"] = {}
        for k_N in parm_dict["N"].keys():
            rules["N"][k_N] = {
                "vary": parm_dict["N"][k_N]["vary"],
                "global": parm_dict["N"][k_N]["global"],
            }

        rules["ss2"] = {}
        for k_ss2 in parm_dict["ss2"].keys():

            if parm_dict["ss2"][k_ss2]["thermal"]["type"] == False:
                rules["ss2"][k_ss2] = {
                    "vary": parm_dict["ss2"][k_ss2]["vary"],
                    "global": parm_dict["ss2"][k_ss2]["global"],
                    "thermal": False,
                }
            else:
                rules["ss2"][k_ss2] = {
                    "vary": parm_dict["ss2"][k_ss2]["vary"],
                    "global": parm_dict["ss2"][k_ss2]["global"],
                    "thermal": parm_dict["ss2"][k_ss2]["thermal"]["type"],
                }
                theta = parm_dict["ss2"][k_ss2]["thermal"]["theta"]
                rules["ss2"][k_ss2].update(
                    {"theta": {"vary": theta["vary"], "global": theta["global"]}}
                )
            rules["delr"] = {}
            for dr_k in parm_dict["delr"].keys():
                rules["delr"][dr_k] = {
                    "vary": parm_dict["delr"][dr_k]["vary"],
                    "global": parm_dict["delr"][dr_k]["global"],
                }
        rules["dele"] = {}
        for dele in parm_dict["dele"].keys():
            rules["dele"][dele] = {
                "vary": parm_dict["dele"][dele]["vary"],
                "global": parm_dict["dele"][dele]["global"],
            }
        return rules

    def fit_one_batch(
        self,
        parm_dict: dict,
        feff_folder: str,
        fitpath_num: int,
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

        # sa_check=[]
        # sa_check2=[]
        if batch_index * batch_size <= len(self.data) and batch_index < batch_num:
            for i in range((batch_index - 1) * batch_size, batch_index * batch_size):
                # sa_check.append(i)
                # sa_check2.append(i-(batch_index-1)*batch_size)
                dset.append(
                    feffit_dataset(
                        data=self.data[i],
                        pathlist=[paths[i - (batch_index - 1) * batch_size]],
                        transform=trans,
                    )
                )
            true_batch_size = batch_size

            # linux has problem with global constraints, so need to set fix_unused_variables=False
            if platform == "linux":
                out = feffit(pars, dset, fix_unused_variables=False)
            else:
                out = feffit(pars, dset)
            report = feffit_report(out)
        if batch_index == batch_num:
            for i in range((batch_index - 1) * batch_size, len(self.data)):

                dset.append(
                    feffit_dataset(
                        data=self.data[i],
                        pathlist=[paths[i - (batch_index - 1) * batch_size]],
                        transform=trans,
                    )
                )
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
        fitpath_num: list[int],
        feff_folder: str,
        save_name: str = "fit.out",
        write: bool = False,
        mpi: bool = False,
        core=1,
        batch_size: int | None = None,
        batch_num: int | None = None,
    ):
        if batch_size is None:
            batch_size = self.batch_size
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

        reff = path.reff
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
        kdata = np.round(np.linspace(max_min, min_max, 500), 4)
        rdata = np.round(np.linspace(max_min_r, min_max_r, 500), 4)
        np.savetxt(join(foldername, "kdata.txt"), kdata)
        np.savetxt(join(foldername, "rdata.txt"), rdata)
        for i in range(len(dset)):
            for j in range(len(dset[i])):
                data_k[f"{i*j+j}"] = np.round(
                    interp1d(
                        dset[i][j].data.k,
                        dset[i][j].data.chi
                        * dset[i][j].data.k ** dset[i][j].transform.kweight,
                    )(kdata),
                    6,
                )
                data_r[f"{i*j+j}"] = np.round(dset[i][j].model.r, 6)
                fitted_k[f"{i*j+j}"] = np.round(
                    interp1d(
                        dset[i][j].model.k,
                        dset[i][j].model.chi
                        * dset[i][j].model.k ** dset[i][j].transform.kweight,
                    )(kdata),
                    6,
                )
                fitted_r[f"{i*j+j}"] = np.round(dset[i][j].model.chir_mag, 6)
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
