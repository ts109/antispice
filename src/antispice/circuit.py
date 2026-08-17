"""Minimal, simulation-independent circuit data model.

Models and parameter-bound parts share one library namespace.  An element may
use either kind of definition, by name or inline.  Expressions remain strings
until a later compiler turns them into symbolic expressions.
"""

import dataclasses
import tomllib
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

type Expression = str | int | float
type NodeReference = str


@dataclasses.dataclass
class Model:
    """Constitutive equations and their parameter names.

    The first port is the reference port.  Every other port ``p`` has the
    canonical variables ``U_p``, ``I_p``, ``Udot_p``, and ``Idot_p``. Auxiliary
    expressions are local shorthand substituted into the equations; they do
    not add state variables. A model with ``n`` ports must provide exactly
    ``n - 1`` equations.
    """

    ports: tuple[str, ...]
    parameters: tuple[str, ...]
    equations: tuple[str, ...]
    auxiliaries: dict[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.ports) < 2:
            raise ValueError("a model must have at least two ports")
        if len(set(self.ports)) != len(self.ports):
            raise ValueError("model port names must be unique")
        if len(set(self.parameters)) != len(self.parameters):
            raise ValueError("model parameter names must be unique")
        if len(self.equations) != len(self.ports) - 1:
            raise ValueError(f"a model with {len(self.ports)} ports must have {len(self.ports) - 1} equations")


@dataclasses.dataclass
class Part:
    """A complete parameter binding for a model."""

    model: str | Model
    parameters: dict[str, Expression]


type Definition = Model | Part
type DefinitionReference = str | Definition


@dataclasses.dataclass
class Element:
    """A named or inline definition connected to circuit nodes."""

    use: DefinitionReference
    nodes: tuple[NodeReference, ...]
    parameters: dict[str, Expression] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Circuit:
    """A shared model/part library and its element instances."""

    library: dict[str, Definition] = dataclasses.field(default_factory=dict)
    elements: dict[str, Element] = dataclasses.field(default_factory=dict)

    def resolve_definition(self, reference: DefinitionReference) -> Definition:
        """Resolve a named or inline definition."""
        if not isinstance(reference, str):
            return reference

        if reference.startswith("builtin."):
            builtin_name = reference.removeprefix("builtin.")
            try:
                return BUILTIN_LIBRARY[builtin_name]
            except KeyError as error:
                raise KeyError(f"unknown built-in definition: {reference!r}") from error

        if reference in self.library:
            return self.library[reference]

        try:
            return BUILTIN_LIBRARY[reference]
        except KeyError as error:
            raise KeyError(f"unknown definition: {reference!r}") from error

    def resolve_model(self, element: Element | str) -> Model:
        """Return the ordinary, flat model used by an element."""
        element = self._resolve_element(element)
        definition = self.resolve_definition(element.use)

        if isinstance(definition, Model):
            return definition

        model = self.resolve_definition(definition.model)
        if isinstance(model, Part):
            raise ValueError("a part must reference a model, not another part")
        return model

    def resolve_parameters(self, element: Element | str) -> dict[str, Expression]:
        """Return the fully bound parameters used by an element."""
        element = self._resolve_element(element)
        definition = self.resolve_definition(element.use)
        model = self.resolve_model(element)

        if isinstance(definition, Part):
            parameters = definition.parameters | element.parameters
        else:
            parameters = dict(element.parameters)

        self._validate_parameters(model, parameters)
        return parameters

    def validate(self) -> None:
        """Validate library bindings, connections, and element parameters."""
        for name, definition in self.library.items():
            if isinstance(definition, Part):
                model = self.resolve_definition(definition.model)
                if isinstance(model, Part):
                    raise ValueError(f"part {name!r} must reference a model, not another part")
                self._validate_parameters(model, definition.parameters, f"part {name!r}")

        for name, element in self.elements.items():
            model = self.resolve_model(element)
            if len(element.nodes) != len(model.ports):
                raise ValueError(f"element {name!r} connects {len(element.nodes)} nodes, but its model has {len(model.ports)} ports")
            self.resolve_parameters(element)

    def _resolve_element(self, element: Element | str) -> Element:
        if not isinstance(element, str):
            return element
        try:
            return self.elements[element]
        except KeyError as error:
            raise KeyError(f"unknown element: {element!r}") from error

    @staticmethod
    def _validate_parameters(
        model: Model,
        parameters: dict[str, Expression],
        subject: str = "element",
    ) -> None:
        expected = set(model.parameters)
        actual = set(parameters)

        unknown = actual - expected
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"{subject} has unknown model parameters: {names}")

        missing = expected - actual
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{subject} is missing model parameters: {names}")


BUILTIN_LIBRARY_FILENAME = "builtin_library.toml"


def _decode_definition(name: str, value: Any) -> Definition:
    if not isinstance(value, dict):
        raise ValueError(f"library definition {name!r} must be a table")

    keys = set(value)
    if "model" in keys:
        expected = {"model", "parameters"}
        if keys != expected:
            raise ValueError(_shape_error(name, "part", keys, expected))
        if not isinstance(value["model"], str):
            raise ValueError(f"library part {name!r} must reference its model by name")
        if not isinstance(value["parameters"], dict):
            raise ValueError(f"library part {name!r} parameters must be a table")
        return Part(model=value["model"], parameters=dict(value["parameters"]))

    expected = {"ports", "parameters", "equations"}
    allowed = expected | {"auxiliaries"}
    if not keys <= allowed or not expected <= keys:
        raise ValueError(_shape_error(name, "model", keys, expected, allowed))
    if not isinstance(value["parameters"], list):
        raise ValueError(f"library model {name!r} parameters must be an array")
    if not isinstance(value.get("auxiliaries", {}), dict):
        raise ValueError(f"library model {name!r} auxiliaries must be a table")
    return Model(
        ports=tuple(value["ports"]),
        parameters=tuple(value["parameters"]),
        equations=tuple(value["equations"]),
        auxiliaries=dict(value.get("auxiliaries", {})),
    )


def _shape_error(
    name: str,
    kind: str,
    actual: set[str],
    required: set[str],
    allowed: set[str] | None = None,
) -> str:
    missing = required - actual
    unknown = actual - (allowed or required)
    details = []
    if missing:
        details.append(f"missing {', '.join(sorted(missing))}")
    if unknown:
        details.append(f"unknown {', '.join(sorted(unknown))}")
    return f"library {kind} {name!r} has invalid fields: {'; '.join(details)}"


def _load_library(root: Any, filename: str = BUILTIN_LIBRARY_FILENAME) -> dict[str, Definition]:
    return _load_library_file(root, PurePosixPath(filename), ())


def _load_library_file(
    root: Any,
    filename: PurePosixPath,
    stack: tuple[PurePosixPath, ...],
) -> dict[str, Definition]:
    if filename in stack:
        chain = " -> ".join(map(str, (*stack, filename)))
        raise ValueError(f"cyclic library include: {chain}")

    location = root.joinpath(*filename.parts)
    try:
        with location.open("rb") as source:
            encoded = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        chain = " -> ".join(map(str, (*stack, filename)))
        raise ValueError(f"cannot load library file {filename}: {error}; include chain: {chain}") from error

    version = encoded.pop("format", 1)
    if version != 1:
        raise ValueError(f"library file {filename} has unsupported format {version!r}")
    includes = encoded.pop("include", [])
    if not isinstance(includes, list) or not all(isinstance(include, str) for include in includes):
        raise ValueError(f"library file {filename} include must be an array of paths")

    library: dict[str, Definition] = {}
    for include in includes:
        relative = PurePosixPath(include)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"library file {filename} has invalid include path {include!r}")
        child = filename.parent.joinpath(relative)
        _merge_library(library, _load_library_file(root, child, (*stack, filename)), child)

    definitions = {name: _decode_definition(name, value) for name, value in encoded.items()}
    _merge_library(library, definitions, filename)
    return library


def _merge_library(target: dict[str, Definition], source: dict[str, Definition], filename: PurePosixPath) -> None:
    duplicate = target.keys() & source.keys()
    if duplicate:
        names = ", ".join(sorted(duplicate))
        raise ValueError(f"duplicate library definitions while including {filename}: {names}")
    target.update(source)


def _load_builtin_library() -> dict[str, Definition]:
    override = Path.cwd() / BUILTIN_LIBRARY_FILENAME
    if override.is_file():
        return _load_library(Path.cwd())
    return _load_library(resources.files(__package__))


BUILTIN_LIBRARY: dict[str, Definition] = _load_builtin_library()
