# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:32:42 2024

@author: ade.fajemisin
"""

import pandas as pd
import functions
import optimization

def main():
    # Get complete network: F, D, S, P_locations, 
    city_network_filename = 'input/city_network_training_config.xlsx'
    city_network_df = pd.read_excel(city_network_filename, sheet_name='node_info')
    
    # Maximum possible size of network
    max_num_DSPs = len(city_network_df.loc[city_network_df['type'].str.contains('LM')])
    max_num_lockers = len(city_network_df.loc[city_network_df['type'].str.contains('locker')])
    max_locker_nodes_list = list(city_network_df.loc[city_network_df['type']=='locker']['node'])
    max_num_package_destinations = len(city_network_df.loc[city_network_df['type'].str.contains('package_destination')])
    max_num_nodes = len(city_network_df)
    
    # Last-miler nodes - READ FROM FILE
    last_milers_df = pd.read_excel(city_network_filename, sheet_name='last_milers')
    all_LM_nodes = list(last_milers_df['LM_node_id'])
    num_vehicles_per_DSP = list(last_milers_df['num_vehicles'])
    assert len(num_vehicles_per_DSP) == max_num_DSPs, 'Not all DSPs have vehicles. Recheck network config file.'
    
    # Satellite nodes
    # Total satellite capacities - READ FROM FILE
    satellite_hubs_df = city_network_df[city_network_df['type']=='locker']
    max_locker_capacities = satellite_hubs_df.set_index('node')['locker_capacity'].to_dict()
    all_locker_capacities = list(max_locker_capacities.values())
    
    
    # Electricity generation info - READ FROM FILE
    electricity_inputs = pd.read_excel(city_network_filename, sheet_name='electricity_generation_breakdwn')
    emission_factor = list(electricity_inputs['Emission Factor'])
    generation_percentage = list(electricity_inputs['Generation Percentage'])
    
    # Battery capacity of electric vehicle in kWh - READ FROM FILE
    bc_df = pd.read_excel(city_network_filename, sheet_name='battery_capacity')
    battery_capacity = bc_df['Capacity (kWh)'].values[0]
    
    time_periods_multiplier = [1] # Only 1 time period for now...
    start_time_periods = [min(city_network_df['earliest'])]
    end_time_periods = [max(city_network_df['latest'])]
    
    time_violation_df = pd.read_excel(city_network_filename, sheet_name='LM_time_violation_penalty')
    time_violation_penalty = time_violation_df['violation penalty'][0]
    
    distance_matrix, travel_time_matrix = functions.generate_distance_time_mat(city_network_df, time_periods_multiplier)
    leave_time_start, leave_time_end, arrive_time, earliest, latest, bigM_matrix = functions.generate_all_times_and_bigMs(city_network_df, start_time_periods,end_time_periods, travel_time_matrix)
    num_time_periods_matrix = functions.get_num_time_periods(distance_matrix)
    emissions_matrix_EV = functions.compute_emissions_EV(battery_capacity, emission_factor, generation_percentage, distance_matrix, travel_time_matrix, num_time_periods_matrix, max_num_nodes)

    num_points_df = pd.read_excel(city_network_filename, sheet_name='num_training_points')
    feasible_ys = []
    feasible_lamdas = []
    ps = []
    dests = []
    dspdeps = []
    dspnods = []
    dsparcs = []
    lockconf = []
    sellocknods=[]
    pckconf = []
    
    num_to_generate = num_points_df['Number of training points'][0]
    counter = 0    
    print('Training should be completed in about', (num_to_generate*7/2000), 'days')
    print('Generating', num_to_generate, 'training data points...')
    while counter < num_to_generate:
        locker_config = functions.generate_locker_config(max_num_lockers)
        lockconf.append(locker_config)
        selected_locker_nodes = [a*b for a,b in zip(locker_config, list(max_locker_capacities.keys()))]
        selected_locker_nodes = [s for s in selected_locker_nodes if s>0]
        sellocknods.append(selected_locker_nodes)
        total_available_capacity = sum([a*b for a,b in zip(locker_config, all_locker_capacities)])
        package_dest_config = functions.generate_package_config(max_num_package_destinations, int(max_num_package_destinations/10), total_available_capacity)
        pckconf.append(package_dest_config)  
        lm_config = functions.create_lm_config(max_num_DSPs)
    
        package_dest_ids, destinations, dsp_depots, dsp_d_nodes, dsp_d_arcs = functions.get_sub_problem_info(city_network_df, lm_config, locker_config, package_dest_config, selected_locker_nodes)
        ps.append(package_dest_ids)
        dests.append(destinations)
        dspdeps.append(dsp_depots)
        dspnods.append(dsp_d_nodes)
        dsparcs.append(dsp_d_arcs)
        
        # generate y, lamda and save
        y_sol = functions.generate_feasible_y(max_num_package_destinations, max_num_DSPs, package_dest_ids, all_LM_nodes, dsp_depots)
        feasible_ys.append(y_sol)
        lamda_sol = functions.generate_feasible_lamda(max_num_package_destinations, max_num_nodes, max_locker_capacities,selected_locker_nodes, package_dest_ids)
        feasible_lamdas.append(lamda_sol)
    
        # update counter
        counter += 1
    
    
    # Store results in dataframe
    # Create headers
    headers = []
    for p in range(max_num_package_destinations):
        for e in range(max_num_DSPs):
            headers.append('y_'+str(p)+'_'+str(e))
    for p in range(max_num_package_destinations):
        for s in max_locker_nodes_list:
            headers.append('lamda_'+str(p)+'_'+str(s))
    headers.append('LM_objective')
    headers.append('LM_best_bound')
    headers.append('LM_gap')
    headers.append('LM_sol_time')
    headers.append('LM_emissions')
    
    # write headers to file
    output_filename = 'output/city_network_training_data.csv'
    with open(output_filename, 'w') as file:
        file.write(','.join(headers)+'\n')
        
    # Solve LM follower problem
    mip_time_limit = 300 # seconds
    mip_emphasis = 1     # prioritize feasible solutions
    
    training_outputs = []
    for i in range(len(feasible_ys)):
        print('i=',i)
        d = dspdeps[i][0]
        obj, bb, gap, stime, sol_df = optimization.Last_Mile_Follower(feasible_ys[i][:, d], feasible_lamdas[i], dspnods[i][0], dsparcs[i][0],
                                                                      dspdeps[i][0], sellocknods[i], ps[i], dests[i], num_vehicles_per_DSP[d],
                                                                      distance_matrix, travel_time_matrix, num_time_periods_matrix, bigM_matrix,
                                                                      earliest, latest, leave_time_start, leave_time_end, emissions_matrix_EV, 
                                                                      time_violation_penalty, mip_time_limit, mip_emphasis)
        emissions = functions.last_miler_emissions_training(sol_df, emissions_matrix_EV)
        sol = [obj, bb, gap, stime, emissions]
        training_outputs.append(sol)
        
        # Write y_p_d to list
        fy_flam = []
        for row in feasible_ys[i]:
            fy_flam.extend(row)
        
        # Write lamda_p_s to list
        lams = feasible_lamdas[i][:,max_locker_nodes_list]
        for p in range(max_num_package_destinations):
            for s in range(len(max_locker_nodes_list)):
                fy_flam.append(lams[p,s])
                
        # Add solutions
        fy_flam.append(sol[0])
        fy_flam.append(sol[1])
        fy_flam.append(sol[2])
        fy_flam.append(sol[3])
        fy_flam.append(sol[4])
        
        # Write to file
        with open(output_filename, 'a') as file:
            file.write(','.join(str(v) for v in fy_flam)+'\n')
    
    
    # Read in training data file
    foll_training_data = pd.read_csv('output/city_network_training_data.csv')
    # Keep only rows with useful data
    foll_training_data = foll_training_data[foll_training_data['LM_gap'] != 'None']
    foll_training_data.reset_index(inplace=True, drop=True)
    # Train model and return trained model in a file
    print('Training predictive model...')    
    trained_model_svr, test_score_svr, total_train_time_svr = functions.train_regression_models_grid_search_svm(foll_training_data)
    learned_constraint = functions.constraint_extrapolation_SVM(trained_model_svr)    
    output_filename = 'output/learned_constraint.xlsx'
    learned_constraint.to_excel(output_filename, index=False)
    print('Model training complete!')
        

if __name__ == "__main__":
    main()
