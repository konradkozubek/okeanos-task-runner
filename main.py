import argparse
from typing import Tuple, List  # Union,
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import abc
import os
import concurrent.futures
from mpi4py.futures import MPIPoolExecutor
from sklearn.model_selection import ParameterGrid


# TODO Add logging
class TaskRunner(metaclass=abc.ABCMeta):
    program_argument_types: dict = {"name": str,
                                    "logarithmic": bool,
                                    "sequence": Tuple[int, int, int],  # Union[Tuple[int, int, int], None],
                                    # "text": Union[str, None]
                                    }
    program: str

    def __init__(self, args):
        self.program: Path = Path.cwd() / args.program
        self.arguments: pd.DataFrame = self.parse_program_arguments(args.arguments)
        self.output_directory: Path = self.create_output_directory(args.output)
        common_process_arguments: list
        task_arguments_index: int
        common_process_arguments, task_arguments_index = self.prepare_process_arguments(args.signature, args.input)
        self.common_process_arguments = common_process_arguments
        self.task_arguments_index = task_arguments_index
        self.output_subdirectories: bool = args.output_subdirectories

    def parse_program_arguments(self, program_args: str) -> pd.DataFrame:
        program_args_data_list: list = []
        for program_arg_data_str in program_args.split(":"):
            program_arg_data: list = program_arg_data_str.split(",")
            program_args_data_list.append({"name": program_arg_data[0],
                                           "logarithmic": len(program_arg_data) == 5 and program_arg_data[-1] == "log",
                                           "sequence": tuple(map(int, program_arg_data[1:4])),
                                               # if len(program_arg_data) >= 4 else None,
                                           # "text": program_arg_data[1] if len(program_arg_data) == 2 else None
                                           })
        program_args_data_dataframe: pd.DataFrame = pd.DataFrame(program_args_data_list,
                                                                 columns=self.program_argument_types)
        return program_args_data_dataframe

    def create_output_directory(self, output_path: str) -> Path:
        output_directory: Path = Path.cwd() / output_path
        if not output_directory.exists():
            output_directory.mkdir(parents=True)  # exist_ok=True
        return output_directory

    def prepare_process_arguments(self, call_signature: str, input_file_path: str) -> Tuple[list, int]:
        input_file = Path.cwd() / input_file_path
        common_process_arguments: list = [self.program.as_posix(), call_signature[2:], input_file.as_posix()]
        task_arguments_index: int = 1 if call_signature[0] == "a" else 3
        # self.arguments["text"]...
        return common_process_arguments, task_arguments_index

    @abc.abstractmethod
    def task_args_to_process_arguments(self, task_args: np.ndarray) -> List[str]:
        pass

    def run_task(self, task_args: np.ndarray) -> None:  # task_args_names: np.ndarray,
        task_signature: str = "_".join(map(str, task_args))
        if self.output_subdirectories:
            # Create task output directory
            task_output_directory: Path = self.output_directory / task_signature
            if not task_output_directory.exists():
                task_output_directory.mkdir()
        # Prepare task process arguments
        process_arguments: list = self.common_process_arguments[:]
        # task_process_arguments: list = list(np.vstack((task_args_names, task_args)).transpose().flatten())
        task_process_arguments: list = self.task_args_to_process_arguments(task_args)
        process_arguments[self.task_arguments_index:self.task_arguments_index] = task_process_arguments
        # Run task process
        if self.output_subdirectories:
            task_output_file_path: Path = task_output_directory / "o.txt"
        else:
            task_output_file_path: Path = self.output_directory / "o_{}.txt".format(task_signature)
        with task_output_file_path.open(mode="w+") as task_output_file:
            with subprocess.Popen(process_arguments,
                                  stdout=task_output_file,
                                  stderr=task_output_file,
                                  cwd=task_output_directory if self.output_subdirectories else self.output_directory) \
                    as task_process:
                # pid = task_process.pid
                task_process.wait()  # return_code =

    def run(self) -> None:
        slurm_job_num_nodes = os.getenv("SLURM_JOB_NUM_NODES")
        if slurm_job_num_nodes is not None:
            nodes_number = int(slurm_job_num_nodes) - 1
        # Maybe add here: pool_workers_number - to use below as nodes_number

        # def task_nodes_number(task_number: int, parallel_tasks_number: int) -> int:
        #     number = nodes_number // parallel_tasks_number
        #     if number < nodes_number - parallel_tasks_number * number:
        #         number += 1
        #     return number

        arguments_values_lists: dict = {}
        argument_values_list: np.ndarray
        for argument_data in self.arguments.itertuples():
            if not argument_data.logarithmic:
                argument_values_list = np.linspace(*argument_data.sequence)
            else:
                argument_values_list = np.logspace(*argument_data.sequence)
            arguments_values_lists[argument_data.name] = argument_values_list
        arguments_grid: ParameterGrid = ParameterGrid(arguments_values_lists)
        pending_tasks: int = len(arguments_grid)
        futures: list = []
        with MPIPoolExecutor() as executor:
            for arguments in arguments_grid:
                # task_args = np.array(list(arguments.values()))
                task_args = np.array([arguments[argument_name] for argument_name in self.arguments["name"]])
                futures.append(executor.submit(self.run_task, task_args))
            while pending_tasks > 0:
                future = next(concurrent.futures.as_completed(futures))
                print(future.result())
                pending_tasks -= 1


class RoundedRectangleRSATaskRunner(TaskRunner):

    def task_args_to_process_arguments(self, task_args: np.ndarray) -> List[str]:
        particle_attributes_parameters: dict = {}
        process_arguments: list = ["simulate"]
        # argument_index,  # list(range(len(task_args))),
        for argument_name, argument in zip(self.arguments["name"], task_args):
            if argument_name[0] == "-":
                process_arguments.append(argument_name + "=" + str(argument))
                # process_arguments.extend([argument_name, str(argument)])
            else:
                particle_attributes_parameters[argument_name] = argument
        particle_attributes = "{0} 4 xy {1} 0 {1} 1 0 1 0 0 4 0 1 2 3".format(particle_attributes_parameters["r"],
                                                                              particle_attributes_parameters["x"])
        process_arguments.append("-particleAttributes=" + particle_attributes)
        # process_arguments.extend(["-particleAttributes", particle_attributes])
        return process_arguments


def parse_arguments() -> argparse.Namespace:
    module_description: str = "Parallel task runner for ICM UW Okeanos cluster"
    arg_parser: argparse.ArgumentParser = argparse.ArgumentParser(description=module_description)
    arg_parser.add_argument("-p", "--program", help="executable of a program to be run")
    arg_parser.add_argument("-a", "--arguments", help="arguments of tasks; "
                                                      "illustrative arguments: \"-a,0,10,11:--bbb,1,100,11,log"  #:-c,"
                                                      # "const\""
                            )
    arg_parser.add_argument("-s", "--signature", help="program call signature; "
                                                      "illustrative signature: \"ai-f\" or \"ia--input\"")
    arg_parser.add_argument("-i", "--input", help="common input file")
    arg_parser.add_argument("-o", "--output", help="output directory")
    arg_parser.add_argument("-d", "--output-subdirectories", action="store_true", help="create output subdirectories "
                                                                                       "for tasks")
    args: argparse.Namespace = arg_parser.parse_args()
    return args


def main() -> None:
    args: argparse.Namespace = parse_arguments()
    task_runner = RoundedRectangleRSATaskRunner(args)
    task_runner.run()


if __name__ == "__main__":
    main()
