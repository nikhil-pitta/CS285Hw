import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

def run_training(discount=0, network_size=0, batch_size=0, learning_rate=0, rtg=0, na=0, gae_lambda=0, default = False): ### num_runs, exp_name, seed):
    # Define the command to run
    if default:
        command = "python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 --exp_name pendulum_default -rtg --use_baseline -na --batch_size 5000"
    else:
        command = f"python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 -n 100 --use_baseline"

        # Add optional arguments based on function arguments
        if discount:
            command += f" --discount {discount}"
        if network_size:
            command += f" --n_layers {network_size}"
        if batch_size:
            command += f" --batch_size {batch_size}"
        if learning_rate:
            command += f" -lr {learning_rate}"
        if rtg:
            command += " -rtg"
        if na:
            command += f" -na"
        if gae_lambda:
            command += f" --gae_lambda {gae_lambda}"
        # if seed:
        #     command += f" --seed {seed}"

            # expert name
        label_string = f"discount={discount}_layers={network_size}_batch_size={batch_size}_lr={learning_rate}_rtg={rtg}_na={na}_gae_lambda={gae_lambda}"
        command += f" --exp_name {label_string}"

    # Initialize a list to store returns from all runs
    all_returns = []

    # Run the command and capture output logs for each run
    for seed in range(1, 6):
        iter_num = f"RUNNING DEFAULT: {default}, SEED:{seed}, discount={discount}, layers={network_size}, batch_size={batch_size}, lr={learning_rate}, rtg={rtg}, na={na}, gae_lambda={gae_lambda}\n\n\n"
        print(iter_num)
        command += f" --seed {seed}"
        output = subprocess.check_output(command, shell=True, text=True)
        returns = []

        # Parse the output logs
        lines = output.split('\n')
        for line in lines:
            if "Eval_AverageReturn" in line:
                match = re.search(r"Eval_AverageReturn : (-?\d+\.\d+)", line)
                if match:
                    returns.append(float(match.group(1)))

        all_returns.append(returns)

    # Calculate the average return across all runs for each training step
    average_returns = np.mean(all_returns, axis=0)
    training_steps = np.arange(0, len(average_returns)) * 1000  # Assuming each step is 1000 iterations

    return training_steps, average_returns


# for default agent

# training_steps, average_returns = run_training(default=True)
# plt.plot(training_steps, average_returns, label=f"pendulum_default")


# Define lists of parameter values to iterate through
discounts = [0, 0.95, 0.98, 0.99, 1]
network_sizes = [3, 4]
batch_sizes = [2500, 5000, 10000]
learning_rates = [0.05, 0.01, 0.001]
rtg_values = [True, False]
na_values = [True, False]
gae_values = [0, 0.95, 0.98, 0.99, 1]


# Create a single plot for all parameter combinations
plt.figure(figsize=(10, 6))

# discount=0, network_size=0, batch_size=0, learning_rate=0, rtg=0, na=0, gae_lambda=0, default = False

for disc in discounts:
    for nl in network_sizes:
        for batch_size in batch_sizes:
            for lr in learning_rates:
                for rtg in rtg_values:
                    for na in na_values:
                        for gae_lambda in gae_values:
                            label_string = f"discount={disc}, layers={nl}, batch_size={batch_size}, lr={lr}, rtg={rtg}, na={na}, gae_lambda={gae_lambda}"
                            print(label_string)

                            training_steps, average_returns = run_training(discount=disc, network_size=nl, batch_size=batch_size, learning_rate=lr, 
                                                                           rtg=rtg, na=na, gae_lambda=gae_lambda, default=False)
                            # Plot the average return for the current parameter combination
                            plt.plot(training_steps, average_returns, label=label_string)

plt.xlabel("Training Steps")
plt.ylabel("Average Return")
plt.legend()
plt.title("Average Return vs. Training Steps (Average of 5 Runs)")
plt.grid(True)
plt.show()
