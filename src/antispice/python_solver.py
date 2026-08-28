"""Optional NumPy/SciPy runtime for generated Antispice equation systems."""

import dataclasses
import importlib
import math
from typing import Any, Literal, Protocol

from .circuit import Circuit
from .compiler import REFERENCE_NODE, EquationSystem, StateLayout, compile_circuit, differential_state_indices
from .python_codegen import PythonKernels, generate_python_kernels


class PythonBackendUnavailable(ImportError):
    """A requested optional Python numerical dependency is unavailable."""


class LinearSolver(Protocol):
    """Numerical linear-system strategy used independently of evaluation."""

    def solve(self, matrix: Any, right_hand_side: Any) -> Any:
        """Return the solution of one square linear system."""


@dataclasses.dataclass(frozen=True)
class _DenseLinearSolver:
    numpy: Any

    def solve(self, matrix: Any, right_hand_side: Any) -> Any:
        return self.numpy.linalg.solve(matrix, right_hand_side)


@dataclasses.dataclass(frozen=True)
class _SparseLinearSolver:
    sparse: Any
    sparse_linalg: Any

    def solve(self, matrix: Any, right_hand_side: Any) -> Any:
        return self.sparse_linalg.spsolve(self.sparse.csc_array(matrix), right_hand_side)


@dataclasses.dataclass(frozen=True)
class SimulationLayout:
    """Semantic lookup information for numerical state and outputs."""

    state: StateLayout
    auxiliaries: dict[tuple[str, str], int]
    port_nets: dict[tuple[str, str], str]
    element_ports: dict[str, tuple[str, ...]]
    differential_states: tuple[int, ...]

    @property
    def state_size(self) -> int:
        """Number of scalar variables in the flattened DAE state."""
        return len(self.state.potentials) + len(self.state.currents)


class _SignalAccess:
    layout: SimulationLayout

    def _state_values(self) -> Any:
        raise NotImplementedError

    def _auxiliary_values(self) -> Any:
        raise NotImplementedError

    def potential(self, net: str) -> Any:
        """Return one potential or a result column for a named net."""
        if net == REFERENCE_NODE:
            values = self._state_values()
            return 0.0 if values.ndim == 1 else values[:, 0] * 0
        return self._state_values()[..., self.layout.state.potential_index(net)]

    def current(self, element: str, port: str) -> Any:
        """Return one port current or result column, reconstructing the reference port."""
        index = self.layout.state.currents.get((element, port))
        if index is not None:
            return self._state_values()[..., index]
        ports = self.layout.element_ports[element]
        if port != ports[0]:
            raise KeyError((element, port))
        result: Any = 0.0
        for other in ports[1:]:
            result = result - self._state_values()[..., self.layout.state.current_index(element, other)]
        return result

    def port_voltage(self, element: str, port: str) -> Any:
        """Return a port voltage relative to the model's reference port."""
        ports = self.layout.element_ports[element]
        return self.potential(self.layout.port_nets[element, port]) - self.potential(self.layout.port_nets[element, ports[0]])

    def auxiliary(self, element: str, name: str) -> Any:
        """Return one model-defined auxiliary value or result column."""
        return self._auxiliary_values()[..., self.layout.auxiliaries[element, name]]


@dataclasses.dataclass(frozen=True)
class OperatingPointResult(_SignalAccess):
    """Converged stationary state."""

    state: Any
    auxiliaries: Any
    layout: SimulationLayout
    time: float
    iterations: int
    residual_norm: float
    trust_radius: float
    trust_limited_steps: int
    backtracks: int
    agreement_ratio: float | None

    def _state_values(self) -> Any:
        return self.state

    def _auxiliary_values(self) -> Any:
        return self.auxiliaries


@dataclasses.dataclass(frozen=True)
class TransientResult(_SignalAccess):
    """Adaptive transient samples at their actual accepted times."""

    times: Any
    states: Any
    auxiliaries: Any
    layout: SimulationLayout
    accepted_steps: int
    rejected_steps: int

    @property
    def sample_count(self) -> int:
        """Number of recorded accepted samples, including the initial state."""
        return len(self.times)

    def _state_values(self) -> Any:
        return self.states

    def _auxiliary_values(self) -> Any:
        return self.auxiliaries


@dataclasses.dataclass(frozen=True)
class ACVoltageInput:
    """Unit small-signal voltage constraint on one non-reference net."""

    net: str


@dataclasses.dataclass(frozen=True)
class ACCurrentInput:
    """Unit small-signal current injection into one non-reference net."""

    net: str


type ACInput = ACVoltageInput | ACCurrentInput


@dataclasses.dataclass(frozen=True)
class ACResult(_SignalAccess):
    """Complex small-signal response over a frequency grid."""

    frequencies: Any
    states: Any
    auxiliaries: Any
    layout: SimulationLayout
    input: ACInput

    @property
    def sample_count(self) -> int:
        """Number of evaluated frequency points."""
        return len(self.frequencies)

    def _state_values(self) -> Any:
        return self.states

    def _auxiliary_values(self) -> Any:
        return self.auxiliaries


@dataclasses.dataclass
class SolverWorkspace:
    """Reusable mutable numerical storage for one non-concurrent analysis."""

    state: Any
    stage_derivatives: Any
    next_state: Any
    residual: Any
    jacobian: Any
    stationary_residual: Any
    stationary_jacobian: Any
    auxiliary_values: Any
    ac_state_jacobian: Any
    ac_derivative_jacobian: Any
    ac_auxiliary_state_jacobian: Any
    ac_auxiliary_derivative_jacobian: Any


class _TransientBuffer:
    def __init__(self, numpy: Any, state_size: int, auxiliary_count: int, capacity: int = 256) -> None:
        self.numpy = numpy
        self.count = 0
        self.capacity = capacity
        self.times = numpy.empty(capacity, dtype=numpy.float64)
        self.states = numpy.empty((capacity, state_size), dtype=numpy.float64)
        self.auxiliaries = numpy.empty((capacity, auxiliary_count), dtype=numpy.float64)

    def append(self, time: float, state: Any, auxiliaries: Any) -> None:
        if self.count == self.capacity:
            self._grow()
        self.times[self.count] = time
        self.states[self.count] = state
        self.auxiliaries[self.count] = auxiliaries
        self.count += 1

    def _grow(self) -> None:
        capacity = 2 * self.capacity
        times = self.numpy.empty(capacity, dtype=self.numpy.float64)
        states = self.numpy.empty((capacity, self.states.shape[1]), dtype=self.numpy.float64)
        auxiliaries = self.numpy.empty((capacity, self.auxiliaries.shape[1]), dtype=self.numpy.float64)
        times[: self.count] = self.times[: self.count]
        states[: self.count] = self.states[: self.count]
        auxiliaries[: self.count] = self.auxiliaries[: self.count]
        self.times, self.states, self.auxiliaries = times, states, auxiliaries
        self.capacity = capacity


@dataclasses.dataclass(frozen=True)
class LinearizedSystem:
    """Immutable small-signal linearization around one operating state."""

    compiled: PythonSolver
    state_jacobian: Any
    derivative_jacobian: Any
    auxiliary_state_jacobian: Any
    auxiliary_derivative_jacobian: Any
    time: float

    def sweep(self, input: ACInput, frequencies: Any) -> ACResult:  # noqa: A002 - domain term
        """Solve one unit excitation over the supplied positive frequencies."""
        np = self.compiled.numpy
        frequencies = np.asarray(frequencies, dtype=np.float64)
        if frequencies.ndim != 1 or not len(frequencies) or not np.all(np.isfinite(frequencies)) or not np.all(frequencies > 0):
            msg = "frequencies must be a non-empty one-dimensional array of positive finite values"
            raise ValueError(msg)
        try:
            net_index = self.compiled.layout.state.potential_index(input.net)
        except (KeyError, ValueError) as error:
            msg = f"unknown or reference AC input net: {input.net!r}"
            raise ValueError(msg) from error
        size = self.compiled.layout.state_size
        states = np.empty((len(frequencies), size), dtype=np.complex128)
        for sample, frequency in enumerate(frequencies):
            matrix = self.state_jacobian.astype(np.complex128) + 2j * math.pi * frequency * self.derivative_jacobian
            if isinstance(input, ACVoltageInput):
                augmented = np.zeros((size + 1, size + 1), dtype=np.complex128)
                augmented[:size, :size] = matrix
                augmented[net_index, size] = -1
                augmented[size, net_index] = 1
                rhs = np.zeros(size + 1, dtype=np.complex128)
                rhs[size] = 1
                states[sample] = self.compiled.linear_solver.solve(augmented, rhs)[:size]
            else:
                rhs = np.zeros(size, dtype=np.complex128)
                rhs[net_index] = 1
                states[sample] = self.compiled.linear_solver.solve(matrix, rhs)
        auxiliaries = states @ self.auxiliary_state_jacobian.T.astype(np.complex128)
        auxiliaries += (states * (2j * math.pi * frequencies[:, None])) @ self.auxiliary_derivative_jacobian.T
        return ACResult(frequencies.copy(), states, auxiliaries, self.compiled.layout, input)


@dataclasses.dataclass(frozen=True)
class PythonSolver:
    """Reusable compiled Python representation of one circuit."""

    system: EquationSystem
    layout: SimulationLayout
    kernels: PythonKernels
    numpy: Any
    linear_solver: LinearSolver

    @property
    def generated_source(self) -> str:
        """Return the native Python evaluator source for inspection."""
        return self.kernels.source

    def create_workspace(self) -> SolverWorkspace:
        """Allocate reusable scratch storage for one analysis at a time."""
        np = self.numpy
        size = self.layout.state_size
        auxiliary_count = len(self.layout.auxiliaries)
        return SolverWorkspace(
            state=np.zeros(size),
            stage_derivatives=np.zeros(2 * size),
            next_state=np.zeros(size),
            residual=np.zeros(2 * size),
            jacobian=np.zeros((2 * size, 2 * size)),
            stationary_residual=np.zeros(size),
            stationary_jacobian=np.zeros((size, size)),
            auxiliary_values=np.zeros(auxiliary_count),
            ac_state_jacobian=np.zeros((size, size)),
            ac_derivative_jacobian=np.zeros((size, size)),
            ac_auxiliary_state_jacobian=np.zeros((auxiliary_count, size)),
            ac_auxiliary_derivative_jacobian=np.zeros((auxiliary_count, size)),
        )

    def operating_point(
        self,
        time: float = 0.0,
        *,
        initial_state: Any | None = None,
        workspace: SolverWorkspace | None = None,
        max_iterations: int = 100,
        residual_tolerance: float = 1e-10,
        minimum_step_multiplier: float = 2**-20,
        initial_trust_radius: float = 1.0,
        maximum_trust_radius: float = 1e6,
        voltage_scale: float = 1.0,
        current_scale: float = 1.0,
    ) -> OperatingPointResult:
        """Solve the stationary DAE using residual-backtracked Newton steps."""
        np = self.numpy
        self._validate_operating_options(
            time,
            max_iterations,
            residual_tolerance,
            minimum_step_multiplier,
            initial_trust_radius,
            maximum_trust_radius,
            voltage_scale,
            current_scale,
        )
        workspace = workspace or self.create_workspace()
        state = workspace.state
        if initial_state is None:
            state.fill(0)
        else:
            initial = np.asarray(initial_state, dtype=np.float64)
            if initial.shape != state.shape:
                msg = f"initial_state must have shape {state.shape}"
                raise ValueError(msg)
            state[:] = initial
        norm = math.inf
        trust_radius = initial_trust_radius
        trust_limited_steps = 0
        backtracks = 0
        agreement_ratio = None
        scales = np.where(
            np.arange(state.size) < len(self.layout.state.potentials),
            voltage_scale,
            current_scale,
        )
        for iteration in range(max_iterations):
            self.kernels.stationary_evaluate(state, time, workspace.stationary_residual, workspace.stationary_jacobian)
            norm = float(np.max(np.abs(workspace.stationary_residual), initial=0))
            if not math.isfinite(norm):
                msg = f"operating-point residual is non-finite at t={time}"
                raise RuntimeError(msg)
            if norm <= residual_tolerance:
                return self._operating_result(
                    workspace,
                    time,
                    iteration,
                    norm,
                    trust_radius,
                    trust_limited_steps,
                    backtracks,
                    agreement_ratio,
                )
            try:
                correction = self.linear_solver.solve(workspace.stationary_jacobian, workspace.stationary_residual)
            except Exception as error:
                msg = f"operating-point Jacobian is singular or ill-conditioned at t={time}"
                raise RuntimeError(msg) from error
            previous = state.copy()
            correction_norm = float(np.linalg.norm(correction / scales))
            trust_multiplier = min(1.0, trust_radius / correction_norm) if correction_norm else 1.0
            trust_limited_steps += trust_multiplier < 1.0
            previous_norm = norm
            multiplier = trust_multiplier
            while multiplier >= trust_multiplier * minimum_step_multiplier:
                state[:] = previous - multiplier * correction
                self.kernels.stationary_evaluate(state, time, workspace.stationary_residual, workspace.stationary_jacobian)
                candidate = float(np.max(np.abs(workspace.stationary_residual), initial=0))
                if math.isfinite(candidate) and candidate < norm:
                    norm = candidate
                    break
                multiplier *= 0.5
                backtracks += 1
            else:
                state[:] = previous
                msg = f"operating-point backtracking failed at t={time} below relative step multiplier {minimum_step_multiplier}"
                raise RuntimeError(msg)
            predicted_reduction = multiplier * previous_norm
            agreement_ratio = (previous_norm - norm) / predicted_reduction if predicted_reduction > 0 else 0.0
            boundary_step = trust_multiplier < 1.0 and multiplier == trust_multiplier
            step_norm = multiplier * correction_norm
            if agreement_ratio < 0.25:
                trust_radius = max(np.finfo(np.float64).eps, 0.25 * step_norm)
            elif agreement_ratio > 0.75 and boundary_step:
                trust_radius = min(maximum_trust_radius, 2 * trust_radius)
            if norm <= residual_tolerance:
                return self._operating_result(
                    workspace,
                    time,
                    iteration + 1,
                    norm,
                    trust_radius,
                    trust_limited_steps,
                    backtracks,
                    agreement_ratio,
                )
        msg = f"operating-point Newton iteration did not converge at t={time} after {max_iterations} iterations (residual norm {norm})"
        raise RuntimeError(msg)

    def transient(
        self,
        *,
        start_time: float,
        end_time: float,
        minimum_step_size: float,
        maximum_step_size: float,
        initial_state: Any | None = None,
        workspace: SolverWorkspace | None = None,
        relative_tolerance: float = 1e-4,
        voltage_absolute_tolerance: float = 1e-7,
        current_absolute_tolerance: float = 1e-10,
        residual_tolerance: float = 1e-10,
        max_iterations: int = 20,
    ) -> TransientResult:
        """Run adaptive Radau-IIA integration without resampling outputs."""
        numeric_options = (
            start_time,
            end_time,
            minimum_step_size,
            maximum_step_size,
            relative_tolerance,
            voltage_absolute_tolerance,
            current_absolute_tolerance,
            residual_tolerance,
        )
        if not all(math.isfinite(value) for value in numeric_options):
            msg = "transient times, step sizes, and tolerances must be finite"
            raise ValueError(msg)
        if end_time < start_time or minimum_step_size <= 0 or maximum_step_size < minimum_step_size:
            msg = "invalid transient time or step-size interval"
            raise ValueError(msg)
        if min(relative_tolerance, voltage_absolute_tolerance, current_absolute_tolerance, residual_tolerance) <= 0:
            msg = "transient tolerances must be positive"
            raise ValueError(msg)
        if not isinstance(max_iterations, int) or max_iterations < 1:
            msg = "max_iterations must be a positive integer"
            raise ValueError(msg)
        workspace = workspace or self.create_workspace()
        if initial_state is None:
            initial_state = self.operating_point(start_time, workspace=workspace, residual_tolerance=residual_tolerance).state
        workspace.state[:] = initial_state
        workspace.stage_derivatives.fill(0)
        buffer = _TransientBuffer(self.numpy, self.layout.state_size, len(self.layout.auxiliaries))
        buffer.append(start_time, workspace.state, self._evaluate_auxiliaries(workspace, start_time))
        time = start_time
        step_size = math.sqrt(minimum_step_size * maximum_step_size)
        accepted = rejected = 0
        options = (max_iterations, residual_tolerance)
        while time < end_time:
            remaining = end_time - time
            trial = min(step_size, maximum_step_size, remaining)
            accepted_step, error_norm, suggested, failure = self._adaptive_step(
                workspace,
                time=time,
                step_size=trial,
                relative_tolerance=relative_tolerance,
                voltage_absolute_tolerance=voltage_absolute_tolerance,
                current_absolute_tolerance=current_absolute_tolerance,
                options=options,
            )
            if not accepted_step:
                rejected += 1
                if trial <= minimum_step_size * (1 + 16 * self.numpy.finfo(float).eps) or end_time - time < minimum_step_size:
                    reason = f": {failure}" if failure else f" (estimated error {error_norm})"
                    msg = f"adaptive integration failed at t={time}; minimum step size {minimum_step_size} is insufficient{reason}"
                    raise RuntimeError(msg)
                step_size = max(minimum_step_size, suggested)
                continue
            time += trial
            if end_time - time <= 32 * self.numpy.finfo(float).eps * max(1.0, abs(end_time)):
                time = end_time
            accepted += 1
            buffer.append(time, workspace.state, self._evaluate_auxiliaries(workspace, time))
            step_size = max(minimum_step_size, min(maximum_step_size, suggested))
        return TransientResult(
            buffer.times[: buffer.count],
            buffer.states[: buffer.count],
            buffer.auxiliaries[: buffer.count],
            self.layout,
            accepted,
            rejected,
        )

    def linearize(self, state: Any, time: float = 0.0, *, workspace: SolverWorkspace | None = None) -> LinearizedSystem:
        """Evaluate the small-signal DAE matrices around one supplied state."""
        if not math.isfinite(time):
            msg = "linearization time must be finite"
            raise ValueError(msg)
        state = self.numpy.asarray(state, dtype=self.numpy.float64)
        if state.shape != (self.layout.state_size,):
            msg = f"linearization state must have shape {(self.layout.state_size,)}"
            raise ValueError(msg)
        workspace = workspace or self.create_workspace()
        self.kernels.ac_linearize(state, time, workspace.ac_state_jacobian, workspace.ac_derivative_jacobian)
        if self.kernels.ac_auxiliary_linearize is not None:
            self.kernels.ac_auxiliary_linearize(
                state,
                time,
                workspace.ac_auxiliary_state_jacobian,
                workspace.ac_auxiliary_derivative_jacobian,
            )
        return LinearizedSystem(
            self,
            workspace.ac_state_jacobian.copy(),
            workspace.ac_derivative_jacobian.copy(),
            workspace.ac_auxiliary_state_jacobian.copy(),
            workspace.ac_auxiliary_derivative_jacobian.copy(),
            time,
        )

    def ac(
        self,
        input: ACInput,  # noqa: A002 - domain term
        frequencies: Any,
        *,
        time: float = 0.0,
        operating_state: Any | None = None,
        workspace: SolverWorkspace | None = None,
        residual_tolerance: float = 1e-10,
    ) -> ACResult:
        """Solve an operating point, linearize it, and run one AC sweep."""
        workspace = workspace or self.create_workspace()
        if operating_state is None:
            operating_state = self.operating_point(
                time,
                workspace=workspace,
                residual_tolerance=residual_tolerance,
            ).state
        return self.linearize(operating_state, time, workspace=workspace).sweep(input, frequencies)

    @staticmethod
    def _validate_operating_options(
        time: float,
        max_iterations: int,
        residual_tolerance: float,
        minimum_step_multiplier: float,
        initial_trust_radius: float,
        maximum_trust_radius: float,
        voltage_scale: float,
        current_scale: float,
    ) -> None:
        if not math.isfinite(time):
            msg = "operating-point time must be finite"
            raise ValueError(msg)
        if not isinstance(max_iterations, int) or max_iterations < 1:
            msg = "max_iterations must be a positive integer"
            raise ValueError(msg)
        if not math.isfinite(residual_tolerance) or residual_tolerance <= 0:
            msg = "residual_tolerance must be positive and finite"
            raise ValueError(msg)
        if not math.isfinite(minimum_step_multiplier) or not 0 < minimum_step_multiplier <= 1:
            msg = "minimum_step_multiplier must be finite and in (0, 1]"
            raise ValueError(msg)
        if not math.isfinite(initial_trust_radius) or initial_trust_radius <= 0:
            msg = "initial_trust_radius must be positive and finite"
            raise ValueError(msg)
        if not math.isfinite(maximum_trust_radius) or maximum_trust_radius < initial_trust_radius:
            msg = "maximum_trust_radius must be finite and not smaller than initial_trust_radius"
            raise ValueError(msg)
        if not math.isfinite(voltage_scale) or voltage_scale <= 0:
            msg = "voltage_scale must be positive and finite"
            raise ValueError(msg)
        if not math.isfinite(current_scale) or current_scale <= 0:
            msg = "current_scale must be positive and finite"
            raise ValueError(msg)

    def _operating_result(
        self,
        workspace: SolverWorkspace,
        time: float,
        iterations: int,
        norm: float,
        trust_radius: float,
        trust_limited_steps: int,
        backtracks: int,
        agreement_ratio: float | None,
    ) -> OperatingPointResult:
        workspace.stage_derivatives.fill(0)
        auxiliaries = self._evaluate_auxiliaries(workspace, time).copy()
        return OperatingPointResult(
            workspace.state.copy(),
            auxiliaries,
            self.layout,
            time,
            iterations,
            norm,
            trust_radius,
            trust_limited_steps,
            backtracks,
            agreement_ratio,
        )

    def _evaluate_auxiliaries(self, workspace: SolverWorkspace, time: float) -> Any:
        if self.kernels.evaluate_auxiliaries is not None:
            size = self.layout.state_size
            self.kernels.evaluate_auxiliaries(
                workspace.state,
                workspace.stage_derivatives[size:],
                time,
                workspace.auxiliary_values,
            )
        return workspace.auxiliary_values

    def _step(self, workspace: SolverWorkspace, time: float, step_size: float, options: tuple[int, float]) -> None:
        np = self.numpy
        size = self.layout.state_size
        max_iterations, residual_tolerance = options
        workspace.stage_derivatives[:size] = workspace.stage_derivatives[size:]
        for attempt in range(2):
            if attempt:
                workspace.stage_derivatives.fill(0)
            for _ in range(max_iterations):
                self.kernels.radau_evaluate(
                    workspace.stage_derivatives,
                    time,
                    step_size,
                    workspace.state,
                    workspace.residual,
                    workspace.jacobian,
                )
                norm = float(np.max(np.abs(workspace.residual), initial=0))
                if norm <= residual_tolerance:
                    self._update_next_state(workspace, step_size)
                    workspace.state[:] = workspace.next_state
                    return
                try:
                    correction = self.linear_solver.solve(workspace.jacobian, workspace.residual)
                except Exception as error:
                    failure = error
                    break
                previous = workspace.stage_derivatives.copy()
                multiplier = 1.0
                while multiplier >= 2**-20:
                    workspace.stage_derivatives[:] = previous - multiplier * correction
                    self.kernels.radau_evaluate(
                        workspace.stage_derivatives,
                        time,
                        step_size,
                        workspace.state,
                        workspace.residual,
                        workspace.jacobian,
                    )
                    candidate = float(np.max(np.abs(workspace.residual), initial=0))
                    if math.isfinite(candidate) and candidate < norm:
                        break
                    multiplier *= 0.5
                else:
                    failure = RuntimeError("Newton backtracking failed")
                    break
            else:
                msg = f"Newton iteration did not converge after {max_iterations} iterations"
                failure = RuntimeError(msg)
        msg = f"transient solve failed at t={time + step_size}: {failure}"
        raise RuntimeError(msg) from failure

    def _update_next_state(self, workspace: SolverWorkspace, step_size: float) -> None:
        size = self.layout.state_size
        workspace.next_state[:] = workspace.state + step_size * (3 / 4 * workspace.stage_derivatives[:size] + 1 / 4 * workspace.stage_derivatives[size:])

    def _adaptive_step(
        self,
        workspace: SolverWorkspace,
        *,
        time: float,
        step_size: float,
        relative_tolerance: float,
        voltage_absolute_tolerance: float,
        current_absolute_tolerance: float,
        options: tuple[int, float],
    ) -> tuple[bool, float, float, Exception | None]:
        np = self.numpy
        initial_state = workspace.state.copy()
        initial_stage = workspace.stage_derivatives.copy()
        try:
            self._step(workspace, time, step_size, options)
            full_state = workspace.state.copy()
            workspace.state[:] = initial_state
            workspace.stage_derivatives[:] = initial_stage
            half = step_size / 2
            self._step(workspace, time, half, options)
            self._step(workspace, time + half, half, options)
        except Exception as error:
            workspace.state[:] = initial_state
            workspace.stage_derivatives[:] = initial_stage
            return False, math.inf, step_size * 0.25, error
        differential = np.asarray(self.layout.differential_states, dtype=np.intp)
        absolute = np.where(differential < len(self.layout.state.potentials), voltage_absolute_tolerance, current_absolute_tolerance)
        scale = absolute + relative_tolerance * np.maximum(np.abs(initial_state[differential]), np.abs(workspace.state[differential]))
        error_norm = float(np.max(np.abs(workspace.state[differential] - full_state[differential]) / (7 * scale), initial=0))
        factor = 5.0 if error_norm == 0 else max(0.2, min(5.0, 0.9 * error_norm**-0.25))
        if error_norm > 1:
            workspace.state[:] = initial_state
            workspace.stage_derivatives[:] = initial_stage
            return False, error_norm, step_size * min(0.9, factor), None
        return True, error_norm, step_size * factor, None


def compile_python(
    circuit: Circuit,
    *,
    linear_solver: Literal["dense", "sparse"] | LinearSolver = "dense",
) -> PythonSolver:
    """Compile a circuit into a reusable optional-dependency Python solver."""
    try:
        numpy = importlib.import_module("numpy")
    except ImportError as error:
        msg = "the Python solver requires NumPy; install antispice[python]"
        raise PythonBackendUnavailable(msg) from error
    if linear_solver == "dense":
        strategy: LinearSolver = _DenseLinearSolver(numpy)
    elif linear_solver == "sparse":
        try:
            sparse = importlib.import_module("scipy.sparse")
            sparse_linalg = importlib.import_module("scipy.sparse.linalg")
        except ImportError as error:
            msg = "the sparse Python solver requires SciPy; install antispice[sparse]"
            raise PythonBackendUnavailable(msg) from error
        strategy = _SparseLinearSolver(sparse, sparse_linalg)
    elif isinstance(linear_solver, str):
        msg = f"unknown linear solver: {linear_solver!r}"
        raise ValueError(msg)
    else:
        strategy = linear_solver
    system = compile_circuit(circuit)
    auxiliaries = {key: index for index, key in enumerate(system.auxiliaries)}
    port_nets: dict[tuple[str, str], str] = {}
    element_ports: dict[str, tuple[str, ...]] = {}
    for reference, element in circuit.elements.items():
        ports = circuit.resolve_model(element).ports
        element_ports[reference] = ports
        port_nets.update(((reference, port), net) for port, net in zip(ports, element.nodes, strict=True))
    layout = SimulationLayout(system.layout, auxiliaries, port_nets, element_ports, differential_state_indices(system))
    return PythonSolver(system, layout, generate_python_kernels(system, numpy), numpy, strategy)
