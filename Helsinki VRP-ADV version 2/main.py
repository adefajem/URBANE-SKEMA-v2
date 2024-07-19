# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:04:44 2024

@author: ade.fajemisin
"""
import pandas as pd
pd.options.mode.chained_assignment = None
import functions

def main():
    
    # --- Inputs ---
    # Take in parcel demand data
    parcel_demand_filename = 'input/ParcelDemand.csv'
    all_nodes_df, destinations_counts_df = functions.process_parcel_demand_data(parcel_demand_filename)
    num_nodes = len(all_nodes_df)
    
    # Read in problem data
    problem_input_filename = 'input/test_input_1.xlsx'
    problem_input_data = pd.read_excel(problem_input_filename, sheet_name='problem_input')

    earliest = problem_input_data.set_index('postcode')['earliest'].to_dict()
    earliest = {k: v * 3600 for k, v in earliest.items()} # convert hours to seconds
    earliest_list = [earliest[postcode] for postcode in all_nodes_df['postcode']] # map list order to order in all_nodes_df

    latest = problem_input_data.set_index('postcode')['latest'].to_dict()
    latest = {k: v * 3600 for k, v in latest.items()} # convert hours to seconds
    latest_list = [latest[postcode] for postcode in all_nodes_df['postcode']] # map list order to order in all_nodes_df

    wait_times = problem_input_data.set_index('postcode')['wait time (seconds)'].to_dict() 
    wait_times_list = [wait_times[postcode] for postcode in all_nodes_df['postcode']] # map list order to order in all_nodes_df

    # The robot moves on the sidewalk at walking speed, which is about 1.42 metres per second
    robot_df = pd.read_excel(problem_input_filename, sheet_name='robot')
    robot_speed = robot_df['robot speed (metres per second)'][0] 

    # Electricity generation info
    electricity_inputs = pd.read_excel(problem_input_filename, sheet_name='electricity_generation_breakdwn')
    emission_factor = list(electricity_inputs['Emission Factor'])
    generation_percentage = list(electricity_inputs['Generation Percentage'])

    # Battery capacity of electric vehicle in kWh
    battery_capacity = robot_df['battery capacity (kWh)'].values[0]
    
    # Service time per parcel
    average_serve_time_per_parcel = robot_df['average service time per parcel (seconds)'].values[0]

    
    # --- Do routing ---
    distance_matrix, travel_time_matrix = functions.generate_distance_time_matrices(all_nodes_df, robot_speed)
    bigM_matrix = functions.generate_bigM_matrix(earliest_list, latest_list, wait_times_list, travel_time_matrix)

    try:
        routing_solution = functions.robot_routing(distance_matrix, travel_time_matrix, bigM_matrix, earliest_list, latest_list, wait_times_list, num_nodes)
    except:
            print('The problem is infeasible. Ajust inputs and try again.')
    
    
    # --- Output --- 
    # Get total emissions
    emissions_matrix_EV =  functions.get_emissions_matrix_EV(battery_capacity, emission_factor, generation_percentage, distance_matrix, travel_time_matrix, num_nodes)
    total_emissions = functions.calculate_routing_emissions(routing_solution, emissions_matrix_EV)
    emissions_df = pd.DataFrame([total_emissions], columns=['Total CO2 emissions (gCO2eq)'])

    # Get robot arrival times at destination nodes and number of parcels delivered
    result = functions.output_result(routing_solution, all_nodes_df, destinations_counts_df, wait_times_list, average_serve_time_per_parcel)
    
    # Write to file
    result_filename = 'output/results_1.xlsx'
    with pd.ExcelWriter(result_filename) as writer:  
        result.to_excel(writer, sheet_name='Delivery_results', index = False)        
        emissions_df.to_excel(writer, sheet_name='Total_emissions', index = False)


    
if __name__ == "__main__":
    main()
