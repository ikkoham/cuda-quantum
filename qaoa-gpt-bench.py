import sys
import os
import timeit
import random
import numpy as np
import cudaq

# Add the directory containing qaoa_gpt_src to sys.path
sys.path.append(os.path.abspath("docs/sphinx/applications/python"))

from qaoa_gpt_src.generate_adapt_qaoa_data import generate_data_max_cut, generate_data_mis

if __name__ == '__main__':
    # out_dir is where user wants to save the output data
    # The output will be saved in a folder called adapt_results in the current directory.
    # The output will contain the results of the ADAPT-QAOA algorithm for various graphs.
    adapt_data = 'adapt_results'

    # Set seeds for global determinism
    SEED = 16
    random.seed(SEED)
    np.random.seed(SEED)
    cudaq.set_random_seed(SEED)


    print("Starting data generation with fixed seeds...")
    start_time = timeit.default_timer()
    generate_data_max_cut(output_dir= adapt_data, graphs_number=1, n_nodes=12, weighted=True,
                                              use_negative_weights=False,
                                              use_brute_force=True, use_simulated_annealing=True,
                                              use_one_exchange=True,
                                              op_pool='all_pool',
                                              init_gamma=[0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
                                              scaling_coef=1.0, norm_weights=False, norm_coef=1.0,
                                              trials_per_graph=1, optimizer='BFGS',
                                              approx_ratio=0.97, norm_threshold= 1e-4,
                                              energy_threshold = 1e-15, max_iter=15,
                                              p_init=0.3, p_final=0.9,
                                              seed_g=SEED, seed_weight=SEED, seed_adapt=SEED,
                                              verbose=True)
    end_time = timeit.default_timer()
    print(f"Total time for MaxCut data generation: {end_time - start_time} seconds")

    # --- MIS benchmark ---
    adapt_data_mis = 'adapt_results_mis'
    print("\nStarting MIS data generation with fixed seeds...")
    start_time_mis = timeit.default_timer()
    generate_data_mis(output_dir=adapt_data_mis, graphs_number=1, n_nodes=8,
                      penalty=2.0,
                      use_brute_force=True, use_greedy=True,
                      use_simulated_annealing=True,
                      op_pool='all_pool',
                      init_gamma=[0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
                      trials_per_graph=1, optimizer='BFGS',
                      approx_ratio=0.97, norm_threshold=1e-4,
                      energy_threshold=1e-15, max_iter=15,
                      p_init=0.3, p_final=0.9,
                      seed_g=SEED, seed_adapt=SEED,
                      verbose=True)
    end_time_mis = timeit.default_timer()
    print(f"Total time for MIS data generation: {end_time_mis - start_time_mis} seconds")
