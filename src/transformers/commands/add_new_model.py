# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

from ..utils import logging
from . import BaseTransformersCLICommand


try:
    from cookiecutter.main import cookiecutter

    _has_cookiecutter = True
except ImportError:
    _has_cookiecutter = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def add_new_model_command_factory(args: Namespace):
    return AddNewModelCommand(args.testing, args.testing_file, path=args.path)


class AddNewModelCommand(BaseTransformersCLICommand):
    def __init__(self, testing: bool, testing_file: str, path=None, *args):
        self._testing = testing
        self._testing_file = testing_file
        self._path = path

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        add_new_model_parser = parser.add_parser("add-new-model")
        add_new_model_parser.add_argument("--testing", action="store_true", help="If in testing mode.")
        add_new_model_parser.add_argument("--testing_file", type=str, help="Configuration file on which to run.")
        add_new_model_parser.add_argument(
            "--path", type=str, help="Path to cookiecutter. Should only be used for testing purposes."
        )
        add_new_model_parser.set_defaults(func=add_new_model_command_factory)

    def run(self):
        warnings.warn(
            f"The command `transformers-cli add-new-model` is deprecated and will be removed in v5 of Transformers. "
            f"It is not actively maintained anymore, so might give a result that won't pass all tests and quality "
            f"checks, you should use `transformers-cli add-new-model-like` instead."
        )
        if not _has_cookiecutter:
            raise ImportError(
                "Model creation dependencies are required to use the `add_new_model` command. Install them by running "
                "the following at the root of your `transformers` clone:\n\n\t$ pip install -e .[modelcreation]\n"
            )

        path_to_transformer_root = (
            Path(__file__).parent.parent.parent.parent if self._path is None else Path(self._path).parent.parent
        )
        path_to_cookiecutter = path_to_transformer_root / "templates" / "adding_a_new_model"
        self._execute_cookiecutter(path_to_cookiecutter)

        directory = [directory for directory in os.listdir() if directory.startswith("cookiecutter-template-")][0]

        with open(f"{directory}/configuration.json", "r") as configuration_file:
            configuration = json.load(configuration_file)

        lowercase_model_name = configuration["lowercase_modelname"]
        generate_tensorflow_pytorch_and_flax = configuration["generate_tensorflow_pytorch_and_flax"]

        self._remove_file(f"{directory}/configuration.json")
        output_pytorch, output_tensorflow, output_flax = self._get_output_options(generate_tensorflow_pytorch_and_flax)

        model_dir = f"{path_to_transformer_root}/src/transformers/models/{lowercase_model_name}"
        self._create_directories(model_dir, f"{path_to_transformer_root}/tests/models/{lowercase_model_name}")

        self._move_files(
            [(f"__init__.py", f"{model_dir}/__init__.py"), (f"configuration_{lowercase_model_name}.py", f"{model_dir}/configuration_{lowercase_model_name}.py")],
            [f"{directory}/__init__.py", f"{directory}/configuration_{lowercase_model_name}.py"],
        )

        if output_pytorch:
            self._move_files(
                [
                    (f"modeling_{lowercase_model_name}.py", f"{model_dir}/modeling_{lowercase_model_name}.py"),
                    (f"test_modeling_{lowercase_model_name}.py", f"{path_to_transformer_root}/tests/models/{lowercase_model_name}/test_modeling_{lowercase_model_name}.py"),
                ],
                [f"{directory}/modeling_{lowercase_model_name}.py", f"{directory}/test_modeling_{lowercase_model_name}.py"],
            )
        else:
            self._remove_files([f"{directory}/modeling_{lowercase_model_name}.py", f"{directory}/test_modeling_{lowercase_model_name}.py"])

        if output_tensorflow:
            self._move_files(
                [
                    (f"modeling_tf_{lowercase_model_name}.py", f"{model_dir}/modeling_tf_{lowercase_model_name}.py"),
                    (f"test_modeling_tf_{lowercase_model_name}.py", f"{path_to_transformer_root}/tests/models/{lowercase_model_name}/test_modeling_tf_{lowercase_model_name}.py"),
                ],
                [f"{directory}/modeling_tf_{lowercase_model_name}.py", f"{directory}/test_modeling_tf_{lowercase_model_name}.py"],
            )
        else:
            self._remove_files([f"{directory}/modeling_tf_{lowercase_model_name}.py", f"{directory}/test_modeling_tf_{lowercase_model_name}.py"])

        if output_flax:
            self._move_files(
                [
                    (f"modeling_flax_{lowercase_model_name}.py", f"{model_dir}/modeling_flax_{lowercase_model_name}.py"),
                    (f"test_modeling_flax_{lowercase_model_name}.py", f"{path_to_transformer_root}/tests/models/{lowercase_model_name}/test_modeling_flax_{lowercase_model_name}.py"),
                ],
                [f"{directory}/modeling_flax_{lowercase_model_name}.py", f"{directory}/test_modeling_flax_{lowercase_model_name}.py"],
            )
        else:
            self._remove_files([f"{directory}/modeling_flax_{lowercase_model_name}.py", f"{directory}/test_modeling_flax_{lowercase_model_name}.py"])

        self._move_files(
            [
                (f"{lowercase_model_name}.md", f"{path_to_transformer_root}/docs/source/en/model_doc/{lowercase_model_name}.md"),
                (f"tokenization_{lowercase_model_name}.py", f"{model_dir}/tokenization_{lowercase_model_name}.py"),
                (f"tokenization_fast_{lowercase_model_name}.py", f"{model_dir}/tokenization_{lowercase_model_name}_fast.py"),
            ],
            [
                f"{directory}/{lowercase_model_name}.md",
                f"{directory}/tokenization_{lowercase_model_name}.py",
                f"{directory}/tokenization_fast_{lowercase_model_name}.py",
            ],
        )

        self._replace_in_files(f"{directory}/to_replace_{lowercase_model_name}.py")

        os.rmdir(directory)

    def _execute_cookiecutter(self, path_to_cookiecutter):
        if not self._testing:
            cookiecutter(str(path_to_cookiecutter))
        else:
            with open(self._testing_file, "r") as configuration_file:
                testing_configuration = json.load(configuration_file)

            cookiecutter(
                str(path_to_cookiecutter if self._path is None else self._path),
                no_input=True,
                extra_context=testing_configuration,
            )

    def _get_output_options(self, generate_tensorflow_pytorch_and_flax):
        output_pytorch = "PyTorch" == generate_tensorflow_pytorch_and_flax
        output_tensorflow = "TensorFlow" == generate_tensorflow_pytorch_and_flax
        output_flax = "Flax" == generate_tensorflow_pytorch_and_flax
        return output_pytorch, output_tensorflow, output_flax

    def _create_directories(self, *dirs):
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "__init__.py"), "w"):
                pass

    def _remove_file(self, file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    def _remove_files(self, files):
        for file_path in files:
            self._remove_file(file_path)

    def _move_files(self, file_pairs, source_paths):
        for (src, dest), source_path in zip(file_pairs, source_paths):
            shutil.move(os.path.join(source_path, src), dest)

    def _replace_in_files(self, path_to_datafile):
        with open(path_to_datafile) as datafile:
            lines_to_copy = []
            skip_file = False
            skip_snippet = False
            for line in datafile:
                if "# To replace in: " in line and "##" not in line:
                    file_to_replace_in = line.split('"')[1]
                    skip_file = self._skip_units(line)
                elif "# Below: " in line and "##" not in line:
                    line_to_copy_below = line.split('"')[1]
                    skip_snippet = self._skip_units(line)
                elif "# End." in line and "##" not in line:
                    if not skip_file and not skip_snippet:
                        self._replace(file_to_replace_in, line_to_copy_below, lines_to_copy)

                    lines_to_copy = []
                elif "# Replace with" in line and "##" not in line:
                    lines_to_copy = []
                elif "##" not in line:
                    lines_to_copy.append(line)

        os.remove(path_to_datafile)

    def _skip_units(self, line):
        return (
            ("generating PyTorch" in line and not output_pytorch)
            or ("generating TensorFlow" in line and not output_tensorflow)
            or ("generating Flax" in line and not output_flax)
        )

    def _replace(self, original_file: str, line_to_copy_below: str, lines_to_copy: List[str]):
        # Create temp file
        with open(original_file, "r") as f:
            lines = f.readlines()
        with open(original_file, "w") as f:
            for line in lines:
                f.write(line)
                if line_to_copy_below in line:
                    for line_to_copy in lines_to_copy:
                        f.write(line_to_copy)


if __name__ == "__main__":
    # Example usage:
    args = Namespace(testing=False, testing_file="test_config.json", path=None)
    add_new_model_command_factory(args).run()

    print("san")
