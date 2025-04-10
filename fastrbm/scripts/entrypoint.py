import os
import subprocess
import sys


def main_rcm():
    # Get the directory of the current script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Check if the first positional argument is provided
    if len(sys.argv) < 2:
        print("Error: No command provided. Use 'train', 'mesh' or 'to_rbm'.")
        sys.exit(1)

    # Assign the first positional argument to a variable
    COMMAND = sys.argv[1]

    # Map the command to the corresponding script
    match COMMAND:
        case "train":
            SCRIPT = "train_rcm.py"
        case "mesh":
            SCRIPT = "mesh.py"
        case "to_rbm":
            SCRIPT = "to_rbm.py"
        case _:
            print(
                f"Error: Invalid command '{COMMAND}'. Use 'train', 'mesh' or 'to_rbm'."
            )
            sys.exit(1)

    # Run the corresponding Python script with the remaining optional arguments
    script_path = os.path.join(SCRIPT_DIR, SCRIPT)
    proc = subprocess.call(
        [sys.executable, script_path] + sys.argv[2:],
    )
    print(proc)


def main_fastrbm():
    # Get the directory of the current script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Check if the first positional argument is provided
    if len(sys.argv) < 2:
        print("Error: No command provided. Use 'train', 'mesh' or 'to_rbm'.")
        sys.exit(1)

    # Assign the first positional argument to a variable
    COMMAND = sys.argv[1]

    # Map the command to the corresponding script
    match COMMAND:
        case "train":
            SCRIPT = "train.py"
        case _:
            print(
                f"Error: Invalid command '{COMMAND}'. Use 'train', 'mesh' or 'to_rbm'."
            )
            sys.exit(1)

    # Run the corresponding Python script with the remaining optional arguments
    script_path = os.path.join(SCRIPT_DIR, SCRIPT)
    proc = subprocess.call(
        [sys.executable, script_path] + sys.argv[2:],
    )
    print(proc)
