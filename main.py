
import jax.numpy as jnp
from pyscf_molecule import Molecule
from trial_wavefunction import Minimalist
import netket as nk
import netket_extensions as nkext
from optax._src import linear_algebra
from potential import get_potential_energy
from utils import geometries

molecule = 'LiH'  #  'LiH', 'Be2', 'CH4', 'C2H4'
geometry = geometries[molecule]
basis_set = 'sto-3g'
charge = 0
mol = Molecule(geometry, run_fci=False, basis=basis_set, unit='Bohr', charge=charge)

## Ansatz parameters
N_orbitals=8        # Number of orbitals in the Slater determinant
intermediate_dim=8      # Dimension of the intermediate layers in the MPNN and MLPs
mlp_output_dim=intermediate_dim     # Dimension of the output of the MLP used for the Jastrow term
kan_output_dim=intermediate_dim     # Dimension of the output of the KAN used for the backflow transformation
mlp_layers=2        # Number of layers in the MLP used for the Jastrow term
kan_layers=2        # Number of layers in the KAN used for the backflow transformation
attention_dim=intermediate_dim      # Dimension of the attention mechanism in the KAN
n_features=8        # Number of features in the MPNN embeddings
n_interactions=1        # Number of interaction rounds in the MPNN (i.e., how many times the node and edge features are updated based on their neighbors)

## Sampler parameters
n_samples = 256     # Total number of samples to draw from the sampler (across all chains and iterations)
n_chains = 128      # Number of parallel Markov chains to run in the sampler. 
sweep_size = 32     # Number of proposed moves per chain before updating the model parameters. 
n_discard_per_chain = 32        # Number of initial samples to discard from each chain to allow for thermalization 
chunk_size = 256        # Number of samples to process in each chunk when computing the loss and gradients. This can help manage memory usage during training

## Optimizer parameters
opt_name = 'Sgd'
lr = 0.005      # Learning rate
diag_shift = 0.001      # Diagonal shift for the Stochastic Reconfiguration preconditioner

## Output filename
scratch_path = ''
filename_head = 'LiH_MinimalistAnsatz'
output_sample_filename = scratch_path + 'samples/mc_samples_' + filename_head
logfile_name = scratch_path + 'data_log/' + filename_head

hilb = nk.hilbert.Particle(N=mol.n_electrons, L=(jnp.inf,jnp.inf,jnp.inf), pbc=False)

sampler = nkext.sampler.MetropolisGaussAdaptive(
    hilb,
    initial_sigma=0.05,
    target_acceptance=0.5,
    n_chains=32,
    sweep_size=sweep_size
    )


model = Minimalist(
    mol=mol,
)

potential = lambda x: get_potential_energy(x, mol.coordinates, mol.nuclear_charges)
epot = nk.operator.PotentialEnergy(hilb, potential)
ekin = nk.operator.KineticEnergy(hilb, mass=1.)
ham = ekin + epot
vs = nk.vqs.MCState(
    sampler,
    model,
    n_samples=n_samples,
    n_discard_per_chain=n_discard_per_chain,
    chunk_size=chunk_size
    )

# # Load optimized parameters
# mpack_filename = sctach_path + f'data_log/' + filename_head + '.mpack'
# with open(mpack_filename, 'rb') as file:
#     vs.variables = flax.serialization.from_bytes(vs.variables, file.read())

op = nk.optimizer.Sgd(lr)
sr = nk.optimizer.SR(diag_shift=diag_shift)

def mycb(step, logged_data, driver):
    logged_data["acceptance"] = float(driver.state.sampler_state.acceptance)
    logged_data["globalnorm"] = float(linear_algebra.global_norm(driver._loss_grad))
    return True

log = nk.logging.JsonLog(logfile_name, save_params_every=25)
gs = nk.VMC(ham, op, variational_state=vs, preconditioner=sr)
gs.run(n_iter=20000, callback=mycb, out=log)

