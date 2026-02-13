import jax
import jax.numpy as jnp
import numpy as np

geometries = {
    'LiH' : [('Li', (0,0,0)),
             ('H', (0,0,3.015))],
    'Li2' : [('Li', (0, 0, 0)),
             ('Li', (0, 0, 5.0512))],
    'Be2' : [('Be', (0,0,0)),
             ('Be', (0,0,4.6487))],
    'N2' : [('N', (0, 0, 1.0371)),
            ('N', (0, 0, -1.0371))],
    'NH3' : [('N', (0, 0, 0)),
             ('H', (0, -1.7720, -0.7211)),
             ('H', (1.5346, 0.8861, -0.7211)),
             ('H', (-1.5346, 0.8861, -0.7211))],
    'CH4' : [('C', (0,0,0)),
             ('H', (1.18886,1.18886,1.18886)),
             ('H', (-1.18886,-1.18886,1.18886)),
             ('H', (1.18886,-1.18886,-1.18886)),
             ('H', (-1.18886,1.18886,-1.18886))],
    }

def get_el_ion_distance_matrix(
        r_el: jnp.ndarray, # shape [N_batch x n_el x 3]
        R_ion: jnp.ndarray  # shape [N_ion x 3]
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes distance vectors and their norm between inputs
    Args:
        r_el: shape [N_batch x n_el x 3]
        R_ion: shape [N_ion x 3]
    Returns:
        diff: shape [N_batch x n_el x N_ion x 3]
        dist: shape [N_batch x n_el x N_ion]
    """
    diff = r_el[..., None, :] - R_ion[..., None, :, :]
    dist = jnp.linalg.norm(diff, axis=-1)
    return diff, dist

def get_full_distance_matrix(
        r_el: jnp.ndarray
        ) -> jnp.ndarray:
    """
    Computes distance vectors between inputs
    Args:
        r_el: jnp.array of shape [n_el x 3], contains 
    Returns:
        dist: shape [n_el x n_el], distances between electrons
    """
    diff = jnp.expand_dims(r_el, -2) - jnp.expand_dims(r_el, -3)
    dist = jnp.linalg.norm(diff, axis=-1)
    return dist

def dists_from_diffs_matrix(
        r_el_diff: jnp.ndarray
        ) -> jnp.ndarray:
    n_el = r_el_diff.shape[-2]
    diff_padded = r_el_diff + jnp.eye(n_el)[..., None]
    dist = jnp.linalg.norm(diff_padded, axis=-1) * (1 - jnp.eye(n_el))
    return dist

def get_distance_matrix(
        r_el: jnp.ndarray
        ) -> tuple[jnp.array, jnp.array]: #  stable!
    """
    Compute distance matrix omitting the main diagonal (i.e. distance to the particle itself)
    Args:
        r_el: [batch_dims x n_electrons x 3]
    Returns:
        diff: jnp.array of shape [batch_dims, n_el, n_el, 3], 
        dist: jnp.array of shape [batch_dims, n_el, n_el], distances
    """
    diff = r_el[..., :, None, :] - r_el[..., None, :, :]
    dist = dists_from_diffs_matrix(diff)
    return diff, dist

def calculate_dist(r1, r2 = None):
    """
    Calculates all interparticle distances between all particles contained in r1 and r2
    or to the ones contained within r1 if r1 is the only input
    
    inputs:
        - r1: jnp.array of size (Ns, d*N1), contains the xyz coordinates of N1 particles
        - r2: jnp.array of size (Ns, d*N2), contains the xyz coordinates of N2 particles (optional)
    returns:
        dist: **either**  jnp.array of size (Ns, N1, N2), contains all interparticle distances between r1 and r2, if r2 != None
              **or**      jnp.array of size (Ns, N1*(N1-1)/2) contains all interparticle distances within r1, if r2 == None"""
    
    if r1.ndim != 2:
        r1 = jnp.reshape(r1, (1, -1))           # if r1 is of the form (d*N1), reshape (1, d*N1)
    r1 = jnp.reshape(r1, (jnp.shape(r1)[0], -1, 3))
    
    # if r2 == None:
    #     N1 = r1.shape[1]
    #     dist = (-r1[:, jnp.newaxis, :, :] + r1[:, :, jnp.newaxis, :])       # array of size (Ns, N, N, 3)
    #     dist = jnp.linalg.norm(dist + jnp.eye(N1)[...,None], axis = -1)                             # array of size (Ns, N, N)
    #     dist *= (1. - jnp.eye(N1))
    #     indy, indz = jnp.triu_indices(N1, k=1)
    #     dist = dist[:, indy, indz]                             # array of size (Ns, N(N-1)/2)
        

    # else:
    if r2.ndim != 2:
        r2 = jnp.reshape(r2, (1, -1))
    r2 = jnp.reshape(r2, (jnp.shape(r2)[0], -1, 3))
    dist = (-r1[:, :, jnp.newaxis, :] + r2[:, jnp.newaxis, :, :])
    dist = jnp.linalg.norm(dist, axis = -1)     # array of shape (Ns, N1, N2)
    
    return dist


REG_EPS = 1e-12


def v(r, Rn, Zn):
    """
    Calculates the Coulomb potential energy in H2

    Args:
        r: jnp.array of shape (N*d,), contains the xyz coordinates of the 2 electrons
        Rn: jnp.array of shape (Nn*d,), contains the xyz coordinates of the two nuclei
        Zn: jnp.array of shape (Nn,), contains the electric charges of the two nuclei
    Returns:
        e_pot: jnp.float, Coulomb potential energy 
    """


    dist_ee = calculate_dist(r)             # jnp.array of size (1, N*(N-1)/2), first axis due to calculate_dist
    dist_nn = calculate_dist(Rn)            # jnp.array of size (1, Nn*(Nn-1)/2)
    dist_en = calculate_dist(r, Rn)         # jnp.array of size (1, N, Nn)
    Nn = len(Rn) // 3 

    Znn = jnp.einsum("i,j->ij", Zn, Zn)[jnp.triu_indices(Nn, k=1)]     # (Nn*(Nn-1)/2,), products of nucleus charges

    # calculate the three Coulomb potential energy terms
    arg_pot_ee = 1 / (dist_ee + REG_EPS)                                #  (N*(N-1)/2,)
    pot_ee = jnp.sum(arg_pot_ee)

    arg_pot_nn = jnp.einsum('i,...i->...i', Znn, 1/(dist_nn + REG_EPS))       # (Nn*(Nn-1)/2,), (1,Nn*(Nn-1)/2) -> (1,Nn*(Nn-1)/2)
    pot_nn = jnp.sum(arg_pot_nn)

    arg_pot_en = jnp.einsum('j,...ij->...ij', Zn, 1/(dist_en + REG_EPS))      # (Nn,), (N,Nn) -> (N, Nn)
    pot_en = -jnp.sum(arg_pot_en)

    e_pot = pot_ee + pot_nn + pot_en
    
    return e_pot

def create_Phi(dist_en_sigma, pi_sigma, c_sigma):
    """
    Creates an array Phi_up or Phi_down

    inputs:
        dist_rR_sigma : jnp.array of shape (Ns, N_sigma, Nn)
        pi_sigma : jnp.array of shape (N_states_sigma,), decay parameters 
        c_sigma : jnp.array of shape (N_states_sigma,), weight parameters
    returns:
        Phi_sigma : jnp.array of shape (Ns, N_states_sigma, N_sigma)
    """
    exp_arg_sigma = -jnp.einsum("i,...jk->...ijk", pi_sigma, dist_en_sigma) 
    exp_sigma = jnp.einsum("i, ...ijk->...ijk", c_sigma, jnp.exp(exp_arg_sigma))
    Phi_sigma = jnp.sum(exp_sigma, axis = -1)

    return Phi_sigma

    

def pseudopotential(r, params):
    """
    Returns a pseudo-potential of the form ar / (1 + br)

    inputs:
        r : jnp.array of size (Ns, N_dist), contains all interparticle distances of interest
        params : jnp.array of size (2, 1), contains the jastrow params a and b
    returns:
        u : jnp.array of size (Ns, N_dist), contains the pseudo-potentials
    """

    a, b = params[0], params[1]

    #print(f'in pseudo potential, \
    #    a = {a} \
    #    b = {b}')
    u = a*r / (1 + b*r)

    return u

