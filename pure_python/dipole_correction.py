from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc

from .grid import TPI
from .solute_potential import FELECT


ARGMR = 4.0
ARGMQ = 4.0
DIS_DIPOLE = 1.0e-1


def reciprocal_rows(cell: np.ndarray) -> np.ndarray:
    return np.linalg.inv(cell).T


def cart_to_direct(cart: np.ndarray, cell: np.ndarray) -> np.ndarray:
    return np.asarray(cart) @ np.linalg.inv(cell)


def direct_to_cart(frac: np.ndarray, cell: np.ndarray) -> np.ndarray:
    return np.asarray(frac) @ cell


def fewald_forces(pos_direct: np.ndarray, zval: np.ndarray, cell: np.ndarray) -> tuple[np.ndarray, float]:
    nions = len(zval)
    b = reciprocal_rows(cell)
    anorm = np.linalg.norm(cell, axis=1)
    bnorm = np.linalg.norm(b, axis=1)
    omega = abs(float(np.linalg.det(cell)))
    scale = np.sqrt(np.pi) / omega ** (1.0 / 3.0)
    pidsca = np.pi / scale
    rcut = ARGMR / scale
    maxc = np.asarray(rcut * bnorm + 0.99, dtype=int)
    qcut = ARGMQ / pidsca
    maxgp = np.asarray(qcut * anorm + 0.99, dtype=int)
    rscale = scale**3 * 4.0 * FELECT / np.sqrt(np.pi)
    rensca = scale * 2.0 * FELECT / np.sqrt(np.pi)
    gscale = TPI**2 * FELECT / (omega * scale**2)
    gensca = TPI * FELECT / (omega * scale**2)

    force_pair = np.zeros((nions, nions, 3), dtype=float)
    rewen = 0.0
    gewen = 0.0 + 0.0j

    for n1 in range(-maxc[0], maxc[0] + 1):
        for n2 in range(-maxc[1], maxc[1] + 1):
            for n3 in range(-maxc[2], maxc[2] + 1):
                shift = np.array([n1, n2, n3], dtype=float)
                for ni in range(nions):
                    for nni in range(ni, nions):
                        zz = zval[ni] * zval[nni]
                        dcount = 0.5 if nni == ni else 1.0
                        xfrac = np.mod(pos_direct[ni] - pos_direct[nni] + 100.5, 1.0) - 0.5 + shift
                        r = direct_to_cart(xfrac, cell)
                        dist = float(np.linalg.norm(r))
                        arg = dist * scale
                        if arg < ARGMR and abs(arg) > 1.0e-10:
                            ew = (np.sqrt(np.pi) / 2.0) * erfc(arg) / arg * zz
                            erd = -np.exp(-(arg * arg)) / arg - np.sqrt(np.pi) * erfc(arg) / (2.0 * arg * arg)
                            ewd = -erd / (2.0 * arg) * zz
                            force_pair[nni, ni] += r * ewd * rscale
                            rewen += ew * dcount

    for n1 in range(-maxgp[0], maxgp[0] + 1):
        for n2 in range(-maxgp[1], maxgp[1] + 1):
            for n3 in range(-maxgp[2], maxgp[2] + 1):
                n = np.array([n1, n2, n3], dtype=float)
                g = direct_to_cart(n, b)
                garg = float(np.linalg.norm(g) * pidsca)
                if garg < ARGMQ and abs(garg) > 1.0e-10:
                    ew = np.exp(-(garg * garg)) / (2.0 * garg * garg)
                    forg = g * ew
                    cphase = np.exp(-1j * TPI * (pos_direct @ n))
                    sfact = 0.0 + 0.0j
                    for ni in range(nions):
                        for nni in range(ni, nions):
                            zz = zval[ni] * zval[nni]
                            dcount = 0.5 if nni == ni else 1.0
                            cexphf = cphase[ni] * np.conj(cphase[nni])
                            force_pair[nni, ni] -= forg * (np.imag(cexphf) * zz) * gscale
                            sfact += cexphf * dcount * zz
                    gewen += ew * sfact

    for ni in range(nions):
        for nni in range(ni):
            force_pair[nni, ni] = -force_pair[ni, nni]
    forces = force_pair.sum(axis=0)
    zion = float(zval.sum())
    zzion = float(np.dot(zval, zval))
    energy = rensca * rewen + float(np.real(gewen)) * gensca - (zion**2 * gensca / 4.0) - (
        scale * FELECT * zzion / np.sqrt(np.pi)
    )
    return forces, energy


@dataclass
class EwaldDipoleMixer:
    dipolc_tmp: np.ndarray
    res_old: float = 0.0

    @classmethod
    def fresh(cls) -> "EwaldDipoleMixer":
        return cls(np.zeros(3, dtype=float), 0.0)

    def ewald_dipol(self, dipolc_cart: np.ndarray, cell: np.ndarray, idipol: int = 3) -> tuple[float, np.ndarray]:
        idir = 2 if idipol == 4 else idipol - 1
        dipolc_in = np.zeros(3, dtype=float)
        dipolc_in[idir] = np.clip(dipolc_cart[idir], -20.0, 20.0)
        res = dipolc_in[idir] - self.dipolc_tmp[idir]
        alpha = 0.6
        if abs(res) > 1.0:
            alpha = alpha / abs(res) / abs(res)
        if res * self.res_old < 0.0 and abs(res) > 0.7 * abs(self.res_old):
            alpha *= 0.5
        d_mix = (1.0 - alpha) * self.dipolc_tmp[idir] + alpha * dipolc_in[idir]
        self.res_old = float(res)
        self.dipolc_tmp[:] = 0.0
        self.dipolc_tmp[idir] = d_mix

        dip = float(np.linalg.norm(self.dipolc_tmp))
        if dip < 1.0e-12:
            return 0.0, np.zeros(3, dtype=float)
        pos_cart = np.zeros((2, 3), dtype=float)
        pos_cart[1] = DIS_DIPOLE * self.dipolc_tmp / dip
        pos_direct = cart_to_direct(pos_cart, cell)
        zval = np.array([-dip / DIS_DIPOLE, dip / DIS_DIPOLE], dtype=float)
        forces, energy = fewald_forces(pos_direct, zval, cell)

        if idipol == 4:
            edipol = FELECT * zval[0] * zval[1] / DIS_DIPOLE - energy
            force0 = -FELECT * zval[0] * zval[1] * self.dipolc_tmp / dip / DIS_DIPOLE**2 - forces[0]
            ef_cart = force0 / zval[0]
        else:
            cell2 = cell.copy()
            cell2[idir] *= 2.0
            pos_direct2 = cart_to_direct(pos_cart, cell2)
            forces2, energy2 = fewald_forces(pos_direct2, zval, cell2)
            edipol = 2.0 * (energy2 - energy)
            ef_cart = 2.0 * (forces2[0] - forces[0]) / zval[0]
        ef_direct = cart_to_direct(ef_cart, cell)
        return float(edipol), ef_direct


def cdipol_potential_1d(nz: int, length: float, ef_direct_z: float, indmin: int = 1, width: float = 4.0) -> np.ndarray:
    nouth = nz // 2
    indices = np.arange(1, nz + 1, dtype=int)
    ii = np.mod(indices - indmin + nz, nz) - nouth
    xx = np.abs(np.abs(ii) - nouth)
    cutoff = np.where(xx > width, 1.0, np.abs(np.sin(np.pi * xx / width / 2.0)))
    e_compensate = ef_direct_z * length
    dipfac = -e_compensate * length / nz
    return dipfac * ii * cutoff


def cdipol_indmin_from_center(nout: int, poscen: float = 0.5) -> int:
    nouth = nout // 2
    ipos = int(poscen * nout)
    return int((nouth + ipos + 10 * nout) % nout + 1)


def valence_ion_dipole_cart(
    valence_values: np.ndarray,
    positions_direct: np.ndarray,
    zvals_by_type: list[float],
    counts: list[int],
    cell: np.ndarray,
    poscen: tuple[float, float, float] = (0.5, 0.5, 0.5),
    idipol: int = 3,
    width: float = 4.0,
) -> np.ndarray:
    nx, ny, nz = valence_values.shape
    nout = [nx, ny, nz][idipol - 1]
    nouth = nout // 2
    indmin = cdipol_indmin_from_center(nout, poscen[idipol - 1])
    plane = valence_values.mean(axis=tuple(i for i in range(3) if i != idipol - 1))
    denlin = plane / nout
    indices = np.arange(1, nout + 1, dtype=int)
    ii = np.mod(indices - indmin + nout, nout) - nouth
    xx = np.abs(np.abs(ii) - nouth)
    cutoff = np.where(xx > width, 1.0, np.abs(np.sin(np.pi * xx / width / 2.0)))
    direct = np.zeros(3, dtype=float)
    direct[idipol - 1] = float(np.sum(denlin * ii * (1.0 / nout) * cutoff))

    start = 0
    anorm = np.linalg.norm(cell, axis=1)
    poscen_arr = np.asarray(poscen, dtype=float)
    for zval, count in zip(zvals_by_type, counts):
        stop = start + count
        disp = np.mod(positions_direct[start:stop] - poscen_arr + 10.5, 1.0) - 0.5
        tiny = 1.0e-2
        for idir in range(3):
            mask = np.abs(np.abs(disp[:, idir]) - 0.5) < tiny / anorm[idir]
            disp[mask, idir] = 0.0
        direct -= float(zval) * disp.sum(axis=0)
        start = stop
    return direct_to_cart(direct, cell)


def solvent_moments(charge_values: np.ndarray, cell: np.ndarray) -> tuple[float, np.ndarray]:
    nx, ny, nz = charge_values.shape
    fx = np.arange(nx, dtype=float)[:, None, None] / nx
    fy = np.arange(ny, dtype=float)[None, :, None] / ny
    fz = np.arange(nz, dtype=float)[None, None, :] / nz
    x = fx * cell[0, 0] + fy * cell[1, 0] + fz * cell[2, 0]
    y = fx * cell[0, 1] + fy * cell[1, 1] + fz * cell[2, 1]
    z = fx * cell[0, 2] + fy * cell[1, 2] + fz * cell[2, 2]
    inv_ngrid = 1.0 / float(nx * ny * nz)
    q = float(charge_values.sum() * inv_ngrid)
    d = np.array(
        [
            float(np.sum(charge_values * x) * inv_ngrid),
            float(np.sum(charge_values * y) * inv_ngrid),
            float(np.sum(charge_values * z) * inv_ngrid),
        ]
    )
    return q, d
