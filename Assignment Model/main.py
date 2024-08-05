# -*- coding: utf-8 -*-
"""
Created on Mon Aug 05 08:32:42 2024

@author: ade.fajemisin
"""

import pandas as pd
import optfunctions

def main():
    # Read in learned constraint
    learned_constraint_filename = 'input/learned_constraint.xlsx'
    learned_constraint = pd.read_excel(learned_constraint_filename)
    
    # Read in complete network
    complete_network_filename = 'input/city_network_training_config.xlsx'
    complete_network = pd.read_excel(complete_network_filename)
    
    # Read in instance and confirm that size of instance is less than that of original network
    instance_filename = 'input/test_input_1.xlsx'
    instance_network = pd.read_excel(instance_filename, sheet_name='problem_instance_info')
    max_num_nodes = len(complete_network)
    num_original_LMs = len(complete_network[complete_network['type'].str.contains('depot', case=False)])
    num_original_microhubs = len(complete_network[complete_network['type']=='locker'])
    num_original_destinations = len(complete_network[complete_network['type']=='package_destination'])
    num_instance_LMs = len(instance_network[instance_network['type'].str.contains('depot', case=False)])
    num_instance_microhubs = len(instance_network[instance_network['type']=='locker'])
    num_instance_destinations = len(instance_network[instance_network['type']=='package_destination'])
    assert num_instance_LMs <= num_original_LMs, 'The instance network should be smaller or the same size as the complete network'
    assert num_instance_microhubs <= num_original_microhubs, 'The instance network should be smaller or the same size as the complete network'
    assert num_instance_destinations <= num_original_destinations, 'The instance network should be smaller or the same size as the complete network'
    inst_locker_dfs = instance_network[instance_network['type']=='locker']
    comp_net_locker_dfs = complete_network[complete_network['type']=='locker']
    closest_lockers_map_back = optfunctions.map_closest_lockers_back(comp_net_locker_dfs, inst_locker_dfs)
    
    
    # Create instance: do matching of DSPs to lockers and parcels to locations
    all_locker_nodes = list(complete_network[complete_network['type']=='locker']['node'])
    package_destination_ids, destinations, dsp_depots, dsp_d_nodes, dsp_d_arcs, lm_config, selected_locker_nodes, max_locker_capacities, lm_lookup, locker_lookup, destination_lookup = optfunctions.output_problem_instance(complete_network, instance_network)
    
    # Use CL to find optimal y and lamda assignments
    satellite_penalty = 5000
    y_sol_cl, lamda_sol_cl = optfunctions.solve_CL_leader(num_original_LMs, lm_config, max_num_nodes, num_original_destinations, 
                                                package_destination_ids, all_locker_nodes, selected_locker_nodes,
                                                    max_locker_capacities, learned_constraint, satellite_penalty)
    
    # Post-process (route) LM followers to get LM emissions
    # Number of LM vehicles
    lm_info_df = pd.read_excel(instance_filename, sheet_name='last_milers')
    total_foll_vehicles_all = list(lm_info_df['num_vehicles'])
    num_vehicles_for_followers = [total_foll_vehicles_all[i] for i in range(len(lm_config)) if lm_config[i] == 1]
    
    # Battery capacity of electric vehicle in kWh
    bc_df = pd.read_excel(instance_filename, sheet_name='battery_capacity')
    battery_capacity = bc_df['Capacity (kWh)'].values[0]
    
    # Electricity generation info 
    electricity_inputs = pd.read_excel(instance_filename, sheet_name='electricity_generation_breakdwn')
    emission_factor = list(electricity_inputs['Emission Factor'])
    generation_percentage = list(electricity_inputs['Generation Percentage'])
    
    # Time violation penalty
    time_violation_df = pd.read_excel(instance_filename, sheet_name='LM_time_violation_penalty')
    time_violation_penalty = time_violation_df['violation penalty'][0]
    time_periods_multiplier = [1]
    start_time_periods = [min(complete_network['earliest'])]
    end_time_periods = [max(complete_network['latest'])]
    distance_matrix, travel_time_matrix = optfunctions.generate_distance_time_mat(complete_network, time_periods_multiplier)
    leave_time_start, leave_time_end, arrive_time, earliest, latest, bigM_matrix = optfunctions.generate_all_times_and_bigMs(complete_network, start_time_periods,end_time_periods, travel_time_matrix)
    num_time_periods_matrix = optfunctions.get_num_time_periods(distance_matrix)
    emissions_matrix_EV = optfunctions.compute_emissions_EV(battery_capacity, emission_factor, generation_percentage, distance_matrix, travel_time_matrix, num_time_periods_matrix, max_num_nodes)
    
    
    # Solve follower problems given optimal y and lamda assignments to get emissions
    xm_Sol_DFs = []
    t_Sol_DFs = []
    for d in range(len(dsp_depots)):
        print('Solving for LM:', d)    
        total_assigned_d = y_sol_cl[:,d].sum(axis=0)
        if total_assigned_d < 15:
            # Solve normal vrp
            lm_sol_df = optfunctions.Last_Mile_Follower(y_sol_cl[:, d], lamda_sol_cl, dsp_d_nodes[d], dsp_d_arcs[d], dsp_depots[d],
                                       selected_locker_nodes, package_destination_ids, destinations, num_vehicles_for_followers[d],
                                       distance_matrix, travel_time_matrix, num_time_periods_matrix, bigM_matrix,
                                       earliest, latest, leave_time_start, leave_time_end, emissions_matrix_EV,
                                       time_violation_penalty, 300)
            # Extract t and xm dfs and append
            xm_sol = optfunctions.extract_xm(lm_sol_df, distance_matrix)
            xm_Sol_DFs.append(xm_sol)
            t_sol = optfunctions.extract_t(lm_sol_df)         
            t_Sol_DFs.append(t_sol)
            
        else:
            # Solve clustered
            xm_sol, t_sol = optfunctions.run_clustering_vrp(y_sol_cl,lamda_sol_cl, complete_network, d, dsp_d_nodes, dsp_d_arcs, dsp_depots,
                           selected_locker_nodes, package_destination_ids, destinations, num_vehicles_for_followers,
                           distance_matrix, travel_time_matrix, num_time_periods_matrix, bigM_matrix,
                           earliest, latest, leave_time_start, leave_time_end, emissions_matrix_EV,
                           time_violation_penalty)
            # save t and xm dfs
            xm_Sol_DFs.append(xm_sol)
            t_Sol_DFs.append(t_sol)
    
    # Write results to file
    res_filename = 'output/results_1.xlsx'
    optfunctions.write_instance_results_to_file(res_filename, y_sol_cl, lamda_sol_cl, xm_Sol_DFs, t_Sol_DFs,
                                       distance_matrix, emissions_matrix_EV, dsp_depots, selected_locker_nodes,
                                       package_destination_ids, destinations,destination_lookup, num_instance_LMs,
                                                instance_network, closest_lockers_map_back)
    

   
        

if __name__ == "__main__":
    main()
