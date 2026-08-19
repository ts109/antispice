"""Tests for the public circuit data model."""

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

    def test_auxiliaries_are_model_local_expressions(self) -> None:
        model = antispice.Model(
            ports=("ref", "p"),
            parameters=("gain",),
            equations=("I_p - scaled_voltage",),
            auxiliaries={"scaled_voltage": "gain * U_p"},
        )

        self.assertEqual(model.auxiliaries, {"scaled_voltage": "gain * U_p"})


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

    def test_transistor_models_use_signed_polarity(self) -> None:
        bjt = antispice.BUILTIN_LIBRARY["bjt-ebers-moll"]
        fet = antispice.BUILTIN_LIBRARY["fet-shichman-hodges"]

        self.assertIsInstance(bjt, antispice.Model)
        self.assertEqual(bjt.ports, ("E", "B", "C"))
        self.assertIn("polarity", bjt.parameters)
        self.assertEqual(set(bjt.auxiliaries), {"v_be", "v_bc", "i_forward", "i_reverse"})
        self.assertIsInstance(fet, antispice.Model)
        self.assertEqual(fet.ports, ("S", "G", "D"))
        self.assertIn("polarity", fet.parameters)
        self.assertEqual(
            set(fet.auxiliaries),
            {"v_gs", "v_ds", "overdrive", "gate_activation", "positive_overdrive", "effective_v_ds", "channel_current"},
        )

    def test_dynamic_semiconductor_and_opamp_models_are_available(self) -> None:
        diode = antispice.BUILTIN_LIBRARY["diode-charge-storage"]
        bjt = antispice.BUILTIN_LIBRARY["bjt-charge-control"]
        fet = antispice.BUILTIN_LIBRARY["fet-shichman-hodges-capacitive"]
        opamp = antispice.BUILTIN_LIBRARY["opamp-slew-limited"]

        self.assertIsInstance(diode, antispice.Model)
        self.assertIn("Udot_A", diode.equations[0])
        self.assertIn("forward_transit_time", bjt.parameters)
        self.assertTrue(any("Udot_B" in equation for equation in bjt.equations))
        self.assertIn("gate_source_capacitance", fet.parameters)
        self.assertIn("Udot_G", fet.equations[0])
        self.assertEqual(opamp.ports, ("negative_supply", "positive_supply", "noninverting", "inverting", "output"))
        self.assertNotIn("upper_supply_voltage", opamp.parameters)
        self.assertNotIn("lower_supply_voltage", opamp.parameters)
        self.assertIn("Udot_output", opamp.equations[-1])
        self.assertIn("normalized_error", opamp.auxiliaries)
        self.assertEqual(opamp.auxiliaries["normalized_error"], "(target_voltage - U_output) / (open_loop_gain * transition_voltage)")
        self.assertIn("upper_dropout_voltage ** 2", opamp.auxiliaries["upper_dropout_activation"])
        self.assertIn("lower_input_saturation_voltage ** 2", opamp.auxiliaries["lower_saturation_activation"])

    def test_builtin_models_have_no_discontinuous_where_expressions(self) -> None:
        for name, definition in antispice.BUILTIN_LIBRARY.items():
            if not isinstance(definition, antispice.Model):
                continue
            with self.subTest(name=name):
                expressions = (*definition.equations, *definition.auxiliaries.values())
                self.assertFalse(any("where(" in expression for expression in expressions))

    def test_named_parts_use_dynamic_semiconductor_models(self) -> None:
        expected = {
            "1n4148": "diode-charge-storage",
            "1n4007": "diode-charge-storage",
            "2n3904": "bjt-charge-control",
            "2n3906": "bjt-charge-control",
            "2n7000": "fet-shichman-hodges-capacitive",
            "bs170": "fet-shichman-hodges-capacitive",
        }
        for name, model in expected.items():
            with self.subTest(name=name):
                definition = antispice.BUILTIN_LIBRARY[name]
                self.assertIsInstance(definition, antispice.Part)
                self.assertEqual(definition.model, model)

    def test_packaged_parts_bind_complete_models(self) -> None:
        for name in ("1n4148", "1n4007", "2n3904", "2n3906", "2n7000", "bs170", "bc547", "bc548", "bc549"):
            with self.subTest(name=name):
                definition = antispice.BUILTIN_LIBRARY[name]
                self.assertIsInstance(definition, antispice.Part)
                circuit = antispice.Circuit(elements={"X1": antispice.Element(name, tuple("0" for _ in circuit_module.BUILTIN_LIBRARY[definition.model].ports))})
                circuit.validate()

    def test_library_in_current_directory_overrides_packaged_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            filename = circuit_module.Path(directory) / circuit_module.BUILTIN_LIBRARY_FILENAME
            filename.write_text('[wire]\nports = ["ref", "p"]\nparameters = []\nequations = ["U_p"]\n', encoding="utf-8")
            with mock.patch.object(circuit_module.Path, "cwd", return_value=circuit_module.Path(directory)):
                library = circuit_module._load_builtin_library()

        self.assertEqual(library, {"wire": antispice.Model(("ref", "p"), (), ("U_p",))})

    def test_includes_are_relative_and_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = circuit_module.Path(directory)
            (root / "models").mkdir()
            (root / "parts").mkdir()
            (root / "library.toml").write_text('include = ["models/basic.toml", "parts/common.toml"]\n', encoding="utf-8")
            (root / "models/basic.toml").write_text('[wire]\nports = ["ref", "p"]\nparameters = []\nequations = ["U_p"]\n', encoding="utf-8")
            (root / "parts/common.toml").write_text('include = ["../invalid.toml"]\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid include path"):
                circuit_module._load_library(root, "library.toml")

            (root / "parts/common.toml").write_text('[jumper]\nmodel = "wire"\n[jumper.parameters]\n', encoding="utf-8")
            library = circuit_module._load_library(root, "library.toml")

        self.assertEqual(set(library), {"wire", "jumper"})
        self.assertEqual(library["jumper"], antispice.Part("wire", {}))

    def test_include_cycles_and_duplicate_definitions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = circuit_module.Path(directory)
            (root / "a.toml").write_text('include = ["b.toml"]\n', encoding="utf-8")
            (root / "b.toml").write_text('include = ["a.toml"]\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cyclic.*a.toml -> b.toml -> a.toml"):
                circuit_module._load_library(root, "a.toml")

            definition = '[wire]\nports = ["ref", "p"]\nparameters = []\nequations = ["U_p"]\n'
            (root / "a.toml").write_text('include = ["b.toml", "c.toml"]\n', encoding="utf-8")
            (root / "b.toml").write_text(definition, encoding="utf-8")
            (root / "c.toml").write_text(definition, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate.*wire"):
                circuit_module._load_library(root, "a.toml")

    def test_definition_kind_is_inferred_strictly(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid fields.*unknown typo"):
            circuit_module._decode_definition(
                "broken",
                {"ports": ["ref", "p"], "parameters": [], "equations": ["U_p"], "typo": True},
            )


if __name__ == "__main__":
    unittest.main()
