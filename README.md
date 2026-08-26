# Antispice

Antispice is a circuit-model compiler and numerical simulation toolkit. It turns
a declarative circuit into a reusable solver for either native Python or a web
browser. The core package deliberately contains no schematic editor, graphical
symbols, or plotting code; those belong to applications such as
[antispice-web](https://github.com/ts109/antispice-web).

Antispice is intended for experimenting with circuit models and simulation
methods without requiring every model to conform to a large, fixed SPICE
vocabulary. Models are equations, parts bind model parameters, and elements
connect those definitions into a circuit.

## A first circuit

This example simulates the step response of an RC low-pass filter with the
native Python target:

```python
import antispice

circuit = antispice.Circuit(
    elements={
        "V1": antispice.Element(
            "voltage-source",
            ("0", "input"),
            {"voltage": "where(t > 0, 1, 0)"},
        ),
        "R1": antispice.Element(
            "resistor",
            ("input", "output"),
            {"resistance": 1_000},
        ),
        "C1": antispice.Element(
            "capacitor",
            ("0", "output"),
            {"capacitance": 1e-6},
        ),
    }
)

solver = antispice.compile_python(circuit)
result = solver.transient(
    start_time=0,
    end_time=5e-3,
    minimum_step_size=1e-8,
    maximum_step_size=1e-4,
)

times = result.times
output_potential = result.potential("output")
```

The solver first finds a stationary state at `start_time`, then performs an
adaptive Radau-IIA integration. The returned samples are the actual accepted
integration points; Antispice does not resample them onto an equidistant grid.

The Python target requires NumPy. SciPy is only required for sparse solves:

```sh
pip install "antispice[python]"
pip install "antispice[sparse]"
```

## The mental model

The main concepts form a short chain:

```text
Model -> Part -> Element -> Circuit -> Compiled solver -> Results
```

### Models

A `Model` is a constitutive mathematical description. It declares named ports,
parameter names, equations, and optional auxiliary expressions. It contains no
parameter values and has no graphical symbol.

The first model port is its **reference port**. Voltages at all other ports are
measured relative to it. For a port named `p`, equations may use:

- `U_p` and `I_p` for port voltage and current;
- `Udot_p` and `Idot_p` for their time derivatives;
- `t` for simulation time;
- the model's declared parameters and auxiliaries.

Port current is positive into the element. A model with `n` ports supplies
`n - 1` equations; circuit assembly adds Kirchhoff's current equations.

For example, a resistor model can be written as:

```python
resistor = antispice.Model(
    ports=("ref", "p"),
    parameters=("resistance",),
    equations=("I_p - U_p / resistance",),
    auxiliaries={"power": "U_p * I_p"},
)
```

Auxiliaries are named model-local expressions. They make equations easier to
read and remain available as observable simulation results, but do not add
unknowns to the system.

### Parts

A `Part` binds concrete values to all parameters of a model. For example, a
named transistor part may bind the parameters of an Ebers-Moll model. Parts do
not add or replace model equations, and a part cannot inherit from another
part.

An element using a part may still override individual parameter values.

The built-in library includes compact dynamic models alongside the simpler DC
models:

- `transformer` couples two galvanically isolated windings through
  `coupling_factor * sqrt(primary_inductance * secondary_inductance)`. A
  negative coupling factor reverses the winding polarity.
- `tapped-inductor` models two magnetically coupled series sections with a
  shared electrical tap. Its positive coupling direction is series-aiding.
- `diode-charge-storage` adds junction capacitance and diffusion-charge transit
  time. The built-in `1n4148` and `1n4007` bind representative fast and slow
  switching parameters.
- `bjt-charge-control` extends Ebers-Moll with base-emitter/base-collector
  capacitance and forward/reverse transit time.
- `fet-shichman-hodges-capacitive` adds gate-source, gate-drain, and
  drain-source capacitance to the level-1 channel model. Its cutoff and
  triode-to-saturation boundaries use differentiable square-root transitions.
- `opamp-slew-limited` has conventional positive- and negative-supply ports
  and models open-loop gain, input saturation headroom, output dropout, a
  dominant small-signal regime, and smooth slew limiting. Its approximate
  unity-gain angular bandwidth is `slew_rate / transition_voltage`; open-loop
  gain does not multiply the closed-loop pole, and the open-loop transfer is
  centered at the midpoint of the two supply rails. Each zero saturation
  or dropout parameter disables its corresponding limit; nonzero values enable
  differentiable soft clipping at that distance from the relevant supply rail.

### Elements and circuits

An `Element` places a model or part into a circuit. Its node tuple follows the
port order declared by the model. Thus these are different namespaces:

- a **port** belongs to a model and describes its interface;
- a **net** belongs to a circuit and connects element ports.

A `Circuit` contains element instances and may contain its own local library of
models and parts. The net named `"0"` is the fixed global reference potential.
This is distinct from a model's first port: that port may be connected to any
circuit net.

Antispice validates connections and parameter completeness, but intentionally
does not reject unconventional yet mathematically meaningful models or circuit
topologies.

## Definitions and parameter resolution

Models and parts share one definition namespace. An element may refer to:

- a definition in its circuit's local library;
- a definition from `antispice.BUILTIN_LIBRARY` by name;
- an inline `Model` or `Part` object.

Parameter values are resolved in this order:

1. A part supplies the model's parameter values.
2. The element's parameter mapping overrides values explicitly supplied for
   that instance.
3. Missing or unknown model parameters are validation errors.

The core package performs this resolution before compiling the equations.

## Compilation targets

Python and WebAssembly are separate, maintained targets built from the same
flattened differential-algebraic equation system.

### Native Python

```python
solver = antispice.compile_python(circuit)
```

The generated residual and Jacobian evaluators are native Python code operating
on NumPy arrays. Dense linear solves use `numpy.linalg.solve`; pass
`linear_solver="sparse"` to use `scipy.sparse.linalg.spsolve`, or provide an
object implementing the public `LinearSolver` protocol.

A `PythonSolver` is reusable and largely stateless. It supports:

- stationary operating-point solutions with Newton backtracking;
- adaptive Radau-IIA transient integration;
- small-signal AC linearization and frequency sweeps.

For repeated or concurrent analyses, allocate explicit scratch memory with
`solver.create_workspace()`. One workspace may be reused sequentially, but must
not be shared by concurrent analyses.

### WebAssembly

```python
artifact = antispice.compile_wasm(circuit)
```

The returned `WasmArtifact` contains:

- `module`: the WebAssembly binary;
- `javascript`: an ES module wrapping the solver and its memory;
- `layout`: public state-vector metadata for the host.

The wrapper performs numerical simulation in JavaScript using the generated
WASM evaluators and dependency-free dense LU solver. Applications may also
compile dedicated AC input cases into the wrapper. `antispice-web` is the
reference browser host for this target.

## Analyses and results

### Operating point

```python
operating_point = solver.operating_point(time=0)
```

This solves the original circuit equations with all state derivatives set to
zero. It is also the default initial condition for transient and AC analyses.

### Transient analysis

```python
transient = solver.transient(
    start_time=0,
    end_time=1e-3,
    minimum_step_size=1e-9,
    maximum_step_size=1e-5,
)
```

The geometric mean of the step-size bounds is used as the initial step. Step
acceptance is controlled adaptively, while every implicit step retains Newton
backtracking.

### AC analysis

```python
import numpy as np

frequencies = np.geomspace(10, 1e6, 200)
response = solver.ac(
    antispice.ACCurrentInput("input"),
    frequencies,
)
```

AC analysis linearizes the circuit around a stationary operating point and
solves one complex small-signal system per frequency. Inputs are dedicated
unit voltage constraints or current injections, not ordinary circuit sources.

### Accessing signals

Operating-point, transient, and AC result objects use the same semantic access
methods:

```python
result.potential("output")
result.port_voltage("Q1", "C")
result.current("Q1", "C")
result.auxiliary("Q1", "i_forward")
```

Depending on the analysis, these return a scalar or a NumPy array over time or
frequency. Raw state and auxiliary arrays remain available when direct numeric
access is preferable.

## Public API map

| Area | Main public members |
| --- | --- |
| Circuit model | `Model`, `Part`, `Element`, `Circuit` |
| Definitions | `BUILTIN_LIBRARY`, `REFERENCE_NODE` |
| Python target | `compile_python`, `PythonSolver`, `SolverWorkspace` |
| Results | `OperatingPointResult`, `TransientResult`, `ACResult` |
| AC inputs | `ACVoltageInput`, `ACCurrentInput` |
| WASM target | `compile_wasm`, `WasmArtifact`, `WasmLayout` |
| Extension points | `LinearSolver` |

Lower-level symbolic compiler and code-generation APIs remain available from
their respective `antispice.compiler`, `antispice.python_codegen`,
`antispice.wasm_target`, `antispice.wasm`, and `antispice.javascript` modules.
They are intended for backend development rather than ordinary simulation.

## Scope and license

Antispice provides circuit definitions, equation compilation, numerical
solvers, and numerical results. Schematic editing, model-to-symbol conventions,
and plotting are deliberately outside the core package.

The project requires Python 3.14 or newer and is distributed under the
[MIT License](LICENSE). Antispice is free software and may be used, modified,
and redistributed under those terms.
