import csv
import os
import argparse
from collections import deque, defaultdict
from tabulate import tabulate  # Importing the tabulate library for aesthetic output

class NTM:
    def __init__(self, filename):
        """Initializes the NTM by loading the machine configuration from a CSV file."""
        self.filename = filename
        self.load_machine(filename)

    def load_machine(self, filename):
        """Loads the NTM/DTM configuration from a .csv file."""
        with open(filename, 'r') as file:
            reader = csv.reader(file)

            # Read the machine's metadata from the CSV header lines
            self.name = next(reader)[0]  # Name of the machine
            self.states = set(next(reader))  # List of state names (Q)
            self.input_symbols = set(next(reader))  # List of input symbols (Σ)
            self.tape_symbols = set(next(reader))  # List of tape symbols (Γ)
            self.start_state = next(reader)[0]  # Start state
            self.accept_state = next(reader)[0]  # Accept state
            self.reject_state = next(reader)[0]  # Reject state

            # Initialize a dictionary to store the machine's transitions
            self.transitions = defaultdict(list)

            # Read the transitions (state, input, next_state, write_symbol, direction)
            for row in reader:
                row = row[:5]  # Only process the first 5 values
                if len(row) != 5:
                    raise ValueError(
                        f"Invalid transition format in row: {row}. Expected 5 values."
                    )
                current_state, read_symbol, next_state, write_symbol, direction = row
                self.transitions[(current_state, read_symbol)].append(
                    (next_state, write_symbol, direction)  # Store each transition
                )

    def trace(self, input_string, max_depth=1000, max_transitions=1000):
        """Traces all possible paths of the NTM starting from the initial configuration."""
        tree = []  # This will store the entire computation tree
        explored_configs_at_depth = []  # List to store configurations explored at each depth level
        initial_config = ("", self.start_state, tuple(input_string))  # Convert input_string to a tuple
        queue = deque([(initial_config, [])])  # Queue to perform breadth-first search, with paths recorded
        explored_configs = set()  # To avoid reprocessing the same configuration
        depth = 0
        transitions_count = 0
        accept_path = None  # Path to the accepting configuration
        reject_depth = 0  # Depth at which the last rejection occurs
        total_new_configs = 0  # Total number of new configurations explored
        total_non_leaf_configs = 0  # Number of configurations that aren't leaves in the tree

        # Perform breadth-first search up to max_depth or max_transitions
        while queue and depth < max_depth and transitions_count < max_transitions:
            level_size = len(queue)  # Number of configurations to process at this level
            current_level = []  # To store the configurations at the current level
            for _ in range(level_size):
                config, path = queue.popleft()  # Pop the next configuration to process

                # config_tuple is now guaranteed to be hashable
                config_tuple = config

                if config_tuple in explored_configs:
                    continue  # Skip if we've already explored this configuration

                explored_configs.add(config_tuple)
                current_level.append(config)

                left, state, right = config  # Unpack the configuration
                if state == self.accept_state:
                    accept_path = path + [config]  # If we reach an accepting state, store the path
                    break  # No need to explore further
                if state == self.reject_state:
                    reject_depth = max(reject_depth, len(path))  # Track rejection depth
                    continue  # Skip this path as it leads to rejection

                # Look for transitions based on the current state and tape symbol under the head
                tape_head = right[0] if right else '_'
                transitions = self.transitions.get((state, tape_head), [])
                if transitions:
                    total_non_leaf_configs += 1  # Increment for non-leaf configurations (those with transitions)

                # Process all possible transitions for the current configuration
                for next_state, write_symbol, direction in transitions:
                    # Ensure all parts of the new configuration are tuples
                    new_left = left + tape_head if direction == 'L' else left
                    new_right = right[1:] if direction == 'R' and len(right) > 1 else right  # Update right tape
                    if direction == 'R':
                        new_right = (write_symbol,) + new_right if right else (write_symbol,)
                    new_config = (new_left, next_state, new_right)  # Create the new configuration
                    queue.append((new_config, path + [config]))  # Add the new configuration to the queue

                    total_new_configs += len(transitions)  # Count the new configurations generated
                    transitions_count += len(transitions)

            tree.append(current_level)  # Add the current level to the computation tree
            explored_configs_at_depth.append(current_level)  # Store the configurations at this depth
            depth += 1
            if accept_path:
                break  # Stop searching once we find an accepting path

        # Summarize the results from the tracing process
        return self.summarize_trace(input_string, tree, accept_path, depth, reject_depth, max_depth, transitions_count, total_new_configs, total_non_leaf_configs, max_transitions, explored_configs_at_depth)

    def summarize_trace(self, input_string, tree, accept_path, depth, reject_depth, max_depth, transitions_count, total_new_configs, total_non_leaf_configs, max_transitions, explored_configs_at_depth):
        """Summarizes the trace results and calculates average nondeterminism."""
        avg_nondeterminism = (
            total_new_configs / total_non_leaf_configs if total_non_leaf_configs > 0 else 0  # Compute average nondeterminism
        )
        # Prepare the result dictionary with trace summary details
        result = {
            "Machine": self.name,
            "Input": input_string,
            "Depth": depth,
            "Configurations": sum(len(level) for level in tree),  # Total number of configurations explored
            "Result": "Accepted" if accept_path else "Rejected",  # Outcome of the trace
            "Transitions": transitions_count,  # Number of transitions taken
            "Average Nondeterminism": avg_nondeterminism,  # Average nondeterminism
            "Configurations Explored": explored_configs_at_depth  # Add configurations explored to the result
        }
        if depth >= max_depth and not accept_path:
            result["Result"] = "Ran too long"  # If the trace exceeds max depth, mark as "Ran too long"
        if not accept_path and transitions_count >= max_transitions:
            result["Result"] = "Ran too many transitions"  # If the transition limit is exceeded

        return result


def run_and_log(machine_file, inputs, output_file, max_depth=1000, max_transitions=1000):
    """Runs the NTM for all inputs and logs the results to a CSV file."""
    ntm = NTM(machine_file)  # Create an NTM object with the given machine file
    results = []
    for input_str in inputs:
        # Trace the execution for each input string and append the result
        results.append(ntm.trace(input_str, max_depth, max_transitions))

    # Write the results to the output CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())  # Create CSV writer with appropriate headers
        writer.writeheader()  # Write the header row
        writer.writerows(results)  # Write the result rows

    # Create a formatted table as a string
    formatted_table = tabulate(results, headers="keys", tablefmt="grid")

    # Write the formatted table to the output CSV file
    with open(output_file, 'a', newline='') as file:
        file.write("\nFormatted Table:\n")
        file.write(formatted_table)

    print(f"Results logged to {output_file}")


def main():
    """Main function to handle command-line arguments and execute the script."""
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Trace NTM behavior and log results.")
    parser.add_argument(
        "machine_file",
        type=str,
        help="Path to the CSV file describing the NTM."  # Path to the machine description file
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output CSV file to store results."  # Path where the output will be saved
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        default=["aaa", "111000", "abc", ""],
        help="Space-separated list of input strings to test on the NTM. Default: ['aaa', '111000', 'abc', '']"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=1000,
        help="Maximum depth of the configuration tree to trace. Default is 1000."
    )
    parser.add_argument(
        "--max_transitions",
        type=int,
        default=1000,
        help="Maximum number of transitions to simulate. Default is 1000."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Run the NTM tracing and log results
    run_and_log(args.machine_file, args.inputs, args.output_file, args.max_depth, args.max_transitions)


if __name__ == "__main__":
    main()
