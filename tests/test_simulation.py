"""Simulation-based testing for each experiment.

These are designed to be smoke-tests to make sure the
experiment objects can be created and and at least simulated.

The setup was taken from the notebooks. Some parameters
(shot_count, seq_reps) were changed to make the tests
run faster, and should not be copied into running code.
"""

from pathlib import Path

import pytest

from selene_sim import DepolarizingErrorModel, Stim, Quest

from circuit_benchmarks_guppy.benchmarks import BinaryRB_Experiment
from circuit_benchmarks_guppy.benchmarks import CB_Experiment
from circuit_benchmarks_guppy.benchmarks import FullyRandomBinaryRB_Experiment
from circuit_benchmarks_guppy.benchmarks import FullyRandomMB_Experiment
from circuit_benchmarks_guppy.benchmarks import FullyRandomSQRB_Experiment
from circuit_benchmarks_guppy.benchmarks import GHZ_Experiment
from circuit_benchmarks_guppy.benchmarks import MCMR_Crosstalk_Experiment
from circuit_benchmarks_guppy.benchmarks import MB_Experiment
from circuit_benchmarks_guppy.benchmarks import SPAM_Experiment
from circuit_benchmarks_guppy.benchmarks import SQRB_Experiment
from circuit_benchmarks_guppy.benchmarks import TQRB_Experiment
from circuit_benchmarks_guppy.benchmarks import Transport_SQRB_Experiment



@pytest.fixture
def depolarizing_error_model() -> DepolarizingErrorModel:
    error_model = DepolarizingErrorModel(
        random_seed=1234,

        # single qubit gate error rate
        p_1q=3e-5,

        # two qubit gate error rate
        p_2q=1e-3,

        # set state preparation and measurement error rates to 0
        p_meas=1.5e-3,
        p_init=0,
    )
    return error_model



@pytest.mark.slow_sim
def test_binary_rb_simulation_smoke_test(
    tmp_path: Path,
    depolarizing_error_model: DepolarizingErrorModel
):

    # select parameters and build experiment
    n_qubits = 98
    seq_lengths = [4, 12, 20]
    seq_reps = 2
    n_meas_per_layer = [0, 8, 16]
    filename = str(tmp_path)

    exp = BinaryRB_Experiment(
        n_qubits, 
        seq_lengths, 
        seq_reps,
        n_meas_per_layer=n_meas_per_layer,
        filename=filename
    )
    exp.add_settings()

    # simulate experiment
    shots = 2
    simulator = Stim()

    exp.sim(
        shots, 
        error_model=depolarizing_error_model, 
        simulator=simulator
    )


@pytest.mark.slow_sim
def test_cycle_simulation_smoke_test(
    tmp_path: Path,
    depolarizing_error_model: DepolarizingErrorModel
):

    # select parameters and build experiment
    qubits = [(0,1), (2,3), (4,5), (6,7)]
    seq_lengths = [4, 100, 200]
    filename = str(tmp_path)

    exp = CB_Experiment(qubits, seq_lengths, filename=filename)
    exp.add_settings()

    # simulate experiment
    shots = 2
    simulator = Stim()

    exp.sim(shots, error_model=depolarizing_error_model, simulator=simulator)


@pytest.mark.slow_sim
def test_fully_random_binary_rb_simulation_smoke_test(
    tmp_path: Path,
    depolarizing_error_model: DepolarizingErrorModel
):

    # select parameters and build experiment

    n_qubits = 98
    n_meas_per_layer = [0, 6, 12]
    seq_lengths = [4, 10, 16]
    seq_reps = 1
    filename = str(tmp_path)

    exp = FullyRandomBinaryRB_Experiment(
        n_qubits, 
        seq_lengths, 
        seq_reps=seq_reps, 
        n_meas_per_layer=n_meas_per_layer, 
        filename=filename
    )
    exp.add_settings()

    shots = 2
    simulator = Stim()

    exp.sim(
        shots, 
        error_model=depolarizing_error_model, 
        simulator=simulator
    )


@pytest.mark.slow_sim
def test_fully_random_mb_simulation_smoke_test(
    tmp_path: Path,
    depolarizing_error_model: DepolarizingErrorModel
):

    # select parameters and build experiment
    n_qubits = 98
    seq_lengths = [4, 8, 12] # half of circuit depth
    seq_reps = 1 # number of jobs for each sequence length. Default is 1. Increasing allows for interleaving jobs of different sequence lengths
    filename = str(tmp_path)

    exp = FullyRandomMB_Experiment(
        n_qubits, 
        seq_lengths, 
        seq_reps=seq_reps, 
        filename=filename
    )
    exp.options['SQ_type'] = 'Clifford' # or 'Clifford+T'
    exp.add_settings()

    shots = 2
    simulator = Stim()

    exp.sim(shots, error_model=depolarizing_error_model, simulator=simulator)


@pytest.mark.slow_sim
def test_fully_random_sqrb_simulation_smoke_test(
    tmp_path: Path,
    depolarizing_error_model: DepolarizingErrorModel
):

    n_qubits = 8
    seq_lengths = [6, 24]
    seq_reps = 2
    filename = 'exp_fully_random_SQRB_example.p'
    qubit_length_groups = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 3,
        5: 3,
        6: 6,
        7: 6
    }
    interleave_operation = 'transport'

    exp = FullyRandomSQRB_Experiment(
        n_qubits, 
        seq_lengths, 
        seq_reps, 
        qubit_length_groups,
        interleave_operation,
        filename=filename
    )
    exp.add_settings()

    shots = 2
    simulator = Stim()

    exp.sim(shots, error_model=depolarizing_error_model, simulator=simulator)


@pytest.mark.slow_sim
def test_ghz_simulation_smoke_test(
    tmp_path: Path,
    depolarizing_error_model: DepolarizingErrorModel
):

    n_qubits = 10
    filename = str(tmp_path)

    exp = GHZ_Experiment(n_qubits, filename=filename)
    exp.add_settings()

    shots = 2
    simulator = Quest() # GHZ fidelity protocol requires statevector simulator

    exp.sim(shots, error_model=depolarizing_error_model, simulator=simulator)


@pytest.mark.slow_sim
def test_mcmr_crosstalk_simulation_smoke_test(
    tmp_path: Path,
    depolarizing_error_model: DepolarizingErrorModel
):

    probe_qubits = [q for q in range(6)]#range(56)]
    focus_qubits = [0,1]
    for focus_qubit in focus_qubits:
        probe_qubits.remove(focus_qubit)

    seq_lengths = [10, 500, 1000] #, 200, 400, 600]
    seq_reps = 1
    reset = True
    measure = True
    # filename = f'exp_MCMR_Crosstalk.p' #_Reset_{reset}_Measure_{measure}_q{focus_qubits}.p'
    filename = str(tmp_path)

    exp = MCMR_Crosstalk_Experiment(
        focus_qubits, 
        probe_qubits, 
        seq_lengths, 
        seq_reps, 
        filename=filename, 
        measure = measure, 
        reset = reset
    )
    exp.add_settings()

    shots = 2
    simulator = Quest()

    exp.sim(shots, error_model = depolarizing_error_model, simulator=simulator)


@pytest.mark.slow_sim
def test_mirror_simulation_smoke_test(
    tmp_path: Path,
    depolarizing_error_model: DepolarizingErrorModel
):

    # select parameters and build experiment
    n_qubits = 98
    seq_lengths = [6, 12, 18] # half of circuit depth
    seq_reps = 10 # number of repetitions of each sequence length

    exp = MB_Experiment(
        n_qubits, 
        seq_lengths, 
        seq_reps
    )

    # additional options
    #exp.options['SQ_type'] = 'Clifford' # Clifford by default or 'Clifford+T'
    #exp.options['Pauli_twirl'] = True # True by default
    #exp.options['permute'] = True # True by default

    exp.add_settings()

    shots = 2
    simulator = Stim() # use Quest() if SQ_type = 'Clifford+T'

    exp.sim(shots, error_model=depolarizing_error_model, simulator=simulator)


@pytest.mark.slow_sim
def test_spam_simulation_smoke_test(
    tmp_path: Path,
    depolarizing_error_model: DepolarizingErrorModel
):

    n_qubits = 16
    rounds = 5
    filename = 'exp_spam_example.py'

    exp = SPAM_Experiment(n_qubits, rounds=rounds, filename=filename)
    exp.add_settings()

    shots = 2
    simulator = Stim()

    exp.sim(shots, error_model=depolarizing_error_model, simulator=simulator)


@pytest.mark.slow_sim
def test_sqrb_simulation_smoke_test(
    tmp_path: Path,
    depolarizing_error_model: DepolarizingErrorModel
):

    # select parameters and build experiment
    n_qubits = 10
    seq_lengths = [10, 500, 1000]
    seq_reps = 2
    filename = str(tmp_path)

    exp = SQRB_Experiment(
        n_qubits, 
        seq_lengths, 
        seq_reps, 
        filename=filename
    )
    exp.add_settings()
    #exp.options['measure_leaked'] = True # False by default

    # simulate experiment
    shots = 2
    simulator = Stim()

    exp.sim(shots, error_model=depolarizing_error_model, simulator=simulator)


@pytest.mark.slow_sim
def test_tqrb_simulation_smoke_test(
    tmp_path: Path,
    depolarizing_error_model: DepolarizingErrorModel
):

    # select parameters and build experiment
    qubits = [(0,1),(2,3),(4,5),(6,7)]
    seq_lengths = [10, 200, 400]
    seq_reps = 2
    filename = str(tmp_path)

    exp = TQRB_Experiment(
        qubits, 
        seq_lengths, 
        seq_reps, 
        filename=filename
    )
    exp.add_settings()

    # simulate experiment
    shots = 2
    simulator = Stim()

    exp.sim(shots, error_model=depolarizing_error_model, simulator=simulator)

@pytest.mark.slow_sim
def test_transport_sqrb_simulation_smoke_test(
    tmp_path: Path,
    depolarizing_error_model: DepolarizingErrorModel
):

    # select parameters and build experiment
    n_qubits = 16
    seq_lengths = [8, 32, 128]
    seq_reps = 2
    filename = str(tmp_path)

    qubit_transport_depths = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 4,
        9: 4,
        10: 4,
        11: 4,
        12: 8,
        13: 8,
        14: 8,
        15: 8
    }

    exp = Transport_SQRB_Experiment(
        n_qubits, 
        seq_lengths, 
        seq_reps, 
        qubit_transport_depths, 
        filename=filename
    )
    exp.add_settings()
    #exp.options['measure_leaked'] = True # False default

    # simulate experiment
    shots = 2
    simulator = Stim()

    exp.sim(shots, error_model=depolarizing_error_model, simulator=simulator)