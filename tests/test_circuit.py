"""Tests for the public circuit data model."""

import json
import os
import tempfile
import unittest
from unittest import mock

import antispice
from antispice import circuit as circuit_module


class ModelTest(unittest.TestCase):
    def test_model_requires_one_equation_per_non_reference_port(self) -> None:
        with self.assertRaisesRegex(ValueError, "must have 2 equations"):
            antispice.Model(
                ports=("reference", "a", "b"),
                parameters=(),
                equations=("U_a",),
            )

    def test_model_names_must_be_unique(self) -> None:
        with self.assertRaisesRegex(ValueError, "port names must be unique"):
            antispice.Model(
                ports=("reference", "reference"),
                parameters=(),
                equations=("U_reference",),
            )

        with self.assertRaisesRegex(ValueError, "parameter names must be unique"):
            antispice.Model(
                ports=("reference", "p"),
                parameters=("value", "value"),
                equations=("U_p - value",),
            )


class CircuitResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.transistor = antispice.Model(
            ports=("source", "gate", "drain"),
            parameters=("threshold", "gain"),
            equations=(
                "U_gate - threshold",
                "I_drain - gain * U_drain",
            ),
        )

    def test_part_binds_parameters_and_element_can_override_them(self) -> None:
        circuit = antispice.Circuit(
            library={
                "mosfet": self.transistor,
                "2N7002": antispice.Part(
                    model="mosfet",
                    parameters={"threshold": 1.8, "gain": 0.12},
                ),
            },
            elements={
                "Q1": antispice.Element(
                    use="2N7002",
                    nodes=("0", "input", "output"),
                    parameters={"threshold": 2.0},
                )
            },
        )

        circuit.validate()

        self.assertIs(circuit.resolve_model("Q1"), self.transistor)
        self.assertEqual(
            circuit.resolve_parameters("Q1"),
            {"threshold": 2.0, "gain": 0.12},
        )

    def test_models_and_parts_share_one_namespace(self) -> None:
        local_resistor = antispice.Model(
            ports=("reference", "p"),
            parameters=("resistance",),
            equations=("U_p - 2 * resistance * I_p",),
        )
        circuit = antispice.Circuit(
            library={"resistor": local_resistor},
            elements={
                "local": antispice.Element("resistor", ("0", "a"), {"resistance": 10}),
                "builtin": antispice.Element("builtin.resistor", ("0", "b"), {"resistance": 10}),
            },
        )

        circuit.validate()

        self.assertIs(circuit.resolve_model("local"), local_resistor)
        self.assertIs(
            circuit.resolve_model("builtin"),
            antispice.BUILTIN_LIBRARY["resistor"],
        )

    def test_part_cannot_reference_another_part(self) -> None:
        circuit = antispice.Circuit(
            library={
                "mosfet": self.transistor,
                "base": antispice.Part("mosfet", {"threshold": 1.8, "gain": 0.12}),
                "derived": antispice.Part("base", {"threshold": 2.0, "gain": 0.1}),
            }
        )

        with self.assertRaisesRegex(ValueError, "not another part"):
            circuit.validate()

    def test_validation_rejects_incomplete_and_unknown_parameters(self) -> None:
        missing = antispice.Circuit(
            elements={
                "R1": antispice.Element(
                    use="resistor",
                    nodes=("0", "output"),
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "missing.*resistance"):
            missing.validate()

        unknown = antispice.Circuit(
            elements={
                "R1": antispice.Element(
                    use="resistor",
                    nodes=("0", "output"),
                    parameters={"resistance": 10, "temperature": 25},
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "unknown.*temperature"):
            unknown.validate()

    def test_validation_rejects_wrong_node_count(self) -> None:
        circuit = antispice.Circuit(
            elements={
                "R1": antispice.Element(
                    use="resistor",
                    nodes=("0", "a", "b"),
                    parameters={"resistance": 10},
                )
            }
        )

        with self.assertRaisesRegex(ValueError, "connects 3 nodes"):
            circuit.validate()


class BuiltinLibraryTest(unittest.TestCase):
    def test_packaged_library_is_loaded_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(circuit_module.Path, "cwd", return_value=circuit_module.Path(directory)):
            library = circuit_module._load_builtin_library()

        self.assertEqual(library, antispice.BUILTIN_LIBRARY)

    def test_library_in_current_directory_overrides_packaged_library(self) -> None:
        encoded = {
            "wire": {
                "type": "model",
                "ports": ["ref", "p"],
                "parameters": [],
                "equations": ["U_p"],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, circuit_module.BUILTIN_LIBRARY_FILENAME)
            with open(filename, "w", encoding="utf-8") as destination:
                json.dump(encoded, destination)
            with mock.patch.object(circuit_module.Path, "cwd", return_value=circuit_module.Path(directory)):
                library = circuit_module._load_builtin_library()

        self.assertEqual(library, {"wire": antispice.Model(("ref", "p"), (), ("U_p",))})


if __name__ == "__main__":
    unittest.main()
