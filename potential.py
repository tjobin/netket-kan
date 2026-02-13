import jax.numpy as jnp
import numpy as np
from utils import get_full_distance_matrix, get_el_ion_distance_matrix
from pyscf_molecule import Molecule


REG_EPS = 1e-12

def get_el_el_potential_energy(
        r_el: jnp.ndarray
        ) -> jnp.ndarray:
    """
    Coulomb interaction between electrons
    Args:
        r_el: shape [n_el x 3], contains positions of electrons
    Returns:
        E_pot: scalar, potential energy of electron-electron interactions
    """

    assert r_el.ndim == 2
    n_el = r_el.shape[-2]
    eye = jnp.eye(n_el)
    dist_matrix = get_full_distance_matrix(r_el)
    # add eye to diagonal to prevent div/0
    E_pot = jnp.triu(1.0 / (dist_matrix + eye + REG_EPS), k=1)
    return jnp.sum(E_pot, axis=[-2, -1])

def get_ion_ion_potential_energy(
        R: jnp.ndarray,
        Z: jnp.ndarray
        ) -> jnp.ndarray:
    """
    Coulomb interaction between nucleons
    Args:
        R: shape [n_ions x 3], contains positions of ions
        Z: shape [n_ions], contains charges of ions
    Returns:
        E_pot: scalar, potential energy of ion-ion interactions
    """
    assert R.ndim == 2
    n_ions = R.shape[-2]
    eye = jnp.eye(n_ions)
    dist_matrix = get_full_distance_matrix(R)
    charge_matrix = jnp.expand_dims(Z, -1) * jnp.expand_dims(Z, -2)
    # add eye to diagonal to prevent div/0
    E_pot = jnp.triu(charge_matrix / (dist_matrix + eye + REG_EPS), k=1)
    return jnp.sum(E_pot, axis=[-2, -1])

def get_potential_energy(
        r: jnp.ndarray,
        R: jnp.ndarray,
        Z: jnp.ndarray
        ) -> jnp.ndarray:
    """
    Total Coulomb interaction energy
    Args:
        r: shape [n_el x 3], contains positions of electrons
        R: shape [n_ions x 3], contains positions of ions
        Z: shape [n_ions], contains charges of ions
    Returns:
        E_pot: scalar, total potential energy of the system
    """
    r = r.reshape(-1,3)
    assert r.ndim == 2
    assert R.ndim == 2
    _, dist_el_ion = get_el_ion_distance_matrix(r, R)
    E_pot_el_ions = -jnp.sum(Z / (dist_el_ion + REG_EPS), axis=[-2, -1])
    E_pot_el_el = get_el_el_potential_energy(r)
    E_pot_ion_ion = get_ion_ion_potential_energy(R, Z)
    E_pot = E_pot_el_el + E_pot_el_ions + E_pot_ion_ion
    return E_pot
