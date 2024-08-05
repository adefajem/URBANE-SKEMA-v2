import pandas as pd
import numpy as np
import random
from geopy.distance import geodesic
pd.options.mode.chained_assignment = None
from time import perf_counter as pc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR

def generate_distance_time_mat(nodes_df, time_periods_multiplier):
    dist_mat = []
    time_mat = []

    avg_speed = 1250 #12.5 # m/s
    for i in range(len(nodes_df)):
        d_temp = []
        t_temp = []
        origin = (nodes_df.loc[i]["latitude"], nodes_df.loc[i]["longitude"])
        for j in range(len(nodes_df)):
            destination = (nodes_df.loc[j]["latitude"], nodes_df.loc[j]["longitude"])
            d_inner = []
            t_inner = []
            
            for k in time_periods_multiplier:
                dist = round(geodesic(origin, destination).meters * k, 2)
                time = round(dist/avg_speed, 2)
                
                d_inner.append(dist)
                t_inner.append(time)
                
            d_temp.append(d_inner)
            t_temp.append(t_inner)

        dist_mat.append(d_temp)
        time_mat.append(t_temp)        
    return dist_mat, time_mat

def get_num_time_periods(lst_of_lst_of_lst):
    lengths_result = []
    for inner_lst in lst_of_lst_of_lst:
        inner_lengths = [len(innermost_lst) for innermost_lst in inner_lst]
        lengths_result.append(inner_lengths)
    return lengths_result


def generate_all_times_and_bigMs(city_instance_df, start_time_periods,end_time_periods, travel_time_matrix):
    leave_time_start = []
    leave_time_end = []

    for i in range(len(city_instance_df)):
        leave_start_temp = []
        leave_end_temp = []
        for j in range(len(city_instance_df)):
            leave_start_inner = []
            leave_end_inner = []
            for m in start_time_periods:
                leave_start_inner.append(m)

            for m in end_time_periods:
                leave_end_inner.append(m)

            leave_start_temp.append(leave_start_inner)
            leave_end_temp.append(leave_end_inner)

        leave_time_start.append(leave_start_temp)
        leave_time_end.append(leave_end_temp)
    
    # arrive time = leave_time_start + travel_time
    arrive_time = []
    for i in range(len(leave_time_start)):
        arrive_temp = []

        for j in range(len(leave_time_start[i])):
            arrive_inner = []
            for m in range(len(leave_time_start[i][j])):
                arrive_inner.append(leave_time_start[i][j][m] + travel_time_matrix[i][j][m])

            arrive_temp.append(arrive_inner)

        arrive_time.append(arrive_temp)
    
    # big M for optimization
    bigM_matrix = []
    for i in range(len(arrive_time)):
        big_temp = []
        for j in range(len(arrive_time[i])):
            big_temp.append(max(arrive_time[i][j])+ city_instance_df.iloc[i]['latest'])

        bigM_matrix.append(big_temp)
        
    earliest = list(city_instance_df['earliest'])
    latest = list(city_instance_df['latest'])    
    return leave_time_start, leave_time_end, arrive_time, earliest, latest, bigM_matrix
    
def compute_distance(all_nodes_df, locker_num, deliv_loc):
    # 1. Get closest lockers to delivery locations
    # This is using basic distance. We can improve on this later...
    origin = (all_nodes_df.loc[locker_num]["latitude"], all_nodes_df.loc[locker_num]["longitude"])
    destination = (all_nodes_df.loc[deliv_loc]["latitude"], all_nodes_df.loc[deliv_loc]["longitude"])
    distance = round(geodesic(origin, destination).meters, 4)    
    return distance

def EVCO2(kwh_consumption, emission_factor, generation_percentage):
    energy_mix = sum([emission_factor[i] * generation_percentage[i] for i in range(len(emission_factor))])    
    # if you multiply gCO2eq/kWh with kWh you get gCO2eq
    emissions_gCO2 = kwh_consumption * energy_mix    
    return emissions_gCO2 # in gCO2eq
    
def compute_emissions_EV(battery_capacity, emission_factor, generation_percentage, distance_matrix, travel_time_matrix, num_time_periods_matrix, num_nodes):
    emission_ij = []
    
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):  
            emissions_m = []
            for m in range(num_time_periods_matrix[i][j]): 
                if travel_time_matrix[i][j][m]!=0:
                    kwh_consumption = battery_capacity * distance_matrix[i][j][m] / 100 # kilowatt-hours/100 km
                    emission = EVCO2(kwh_consumption, emission_factor, generation_percentage)
                                  
                    emissions_m.append(emission)
                else:
                    emissions_m.append(0)
            
            emission_ij.append(emissions_m)
                    
    # Put in matrix form
    E_ijkm = [emission_ij[i:i+num_nodes] for i in range(0, len(emission_ij), num_nodes)]                    
    return E_ijkm

def generate_package_config(max_num_destinations, max_train_destinations, total_available_capacity):
    if max_train_destinations > max_num_destinations:
        raise ValueError("Number of max_train_destinations must be less than max_num_destinations")
    # Generate package configuration
    # Select number of parcels - randomly btw  5 and max_train_destinations
    package_config = [0] * max_num_destinations
    indices = random.sample(range(max_num_destinations), random.randint(int(max_train_destinations/2), max_train_destinations))
    for ind in indices:
        package_config[ind] = 1
        
    # Make sure package_config is feasible
    while sum(package_config) > total_available_capacity:
        package_config = list(np.random.choice([0, 1], size=max_train_destinations))
        if all(elem == 0 for elem in package_config):
            index = random.randint(0, len(package_config)-1)
            package_config[index] = 1
    return package_config
    
def generate_locker_config(max_num_lockers):    
    # Generate locker configuration
    locker_config = list(np.random.choice([0, 1], size=max_num_lockers))
    if all(elem == 0 for elem in locker_config):
        index = random.randint(0, len(locker_config)-1)
        locker_config[index] = 1
    return locker_config

def generate_feasible_y(max_num_packages, max_num_DSPs, package_ids, all_LM_nodes, dsp_depots):
    rows = max_num_packages
    cols = max_num_DSPs
    
    # Initialize the array with zeros
    feasible_y_sol = np.zeros((rows, cols))
    # Set one value to one in each row randomly
    for i in package_ids:
        j = [all_LM_nodes.index(depot) for depot in dsp_depots][0] # use correct column index
        feasible_y_sol[i, j] = 1       
    return feasible_y_sol 

def generate_feasible_lamda(max_num_packages, max_num_nodes, max_locker_capacities, selected_locker_nodes, package_ids):      
    # Initialize the array with zeros
    feasible_lamda_sol = np.zeros((max_num_packages, max_num_nodes))
    # Set one value to one in each row randomly while maintaining column sum constraint
    for i in package_ids:
        # Generate random column index based on available locker nodes
        col_idx = random.choice(selected_locker_nodes)
        # Adjust column index if adding 1 would exceed locklist[col_idx]
        while feasible_lamda_sol[:, col_idx].sum() >= max_locker_capacities[col_idx]:
            col_idx = random.choice(selected_locker_nodes)
        feasible_lamda_sol[i, col_idx] = 1
    return feasible_lamda_sol

def get_destinations(df):
    destinations = {}
    for index, row in df.iterrows():
        if row['type'] == 'package_destination':
            destinations[int(row['destination_id'])] = int(row['node'])
    return destinations

def get_sub_problem_info(city_instance_df, lm_config, locker_config, package_config,
                        selected_locker_nodes):
    problem_config = lm_config + locker_config + package_config
    city_instance_df['problem_config'] = problem_config
    city_sub_instance = city_instance_df[city_instance_df['problem_config'] == 1]

    last_depots_df = city_sub_instance.loc[city_instance_df['type'].str.contains('LM')]
    last_depots_df.reset_index(inplace=True, drop=True)

    package_destination_ids = list(city_sub_instance.loc[city_sub_instance['type']=='package_destination']['destination_id'])
    package_destination_ids = [int(x) for x in package_destination_ids]    

    destinations = get_destinations(city_sub_instance)
        
    # DSP depots, nodes and arcs
    dsp_depots = []
    dsp_d_nodes = []
    # Assuming only one depot per DSP
    for d in range(len(last_depots_df)):
        dsp_depots.append(int(last_depots_df['node'][d]))
        dsp_d_nodes.append([int(last_depots_df['node'][d])] + selected_locker_nodes + list(destinations.values()))

    dsp_d_arcs = []
    for d in range(len(dsp_d_nodes)):
        arcs = [(i, j) for i in dsp_d_nodes[d] for j in dsp_d_nodes[d] if i!=j]
        dsp_d_arcs.append(arcs)        
    return package_destination_ids, destinations, dsp_depots, dsp_d_nodes, dsp_d_arcs

def create_lm_config(max_num_DSPs):
    lm_config = [0] * max_num_DSPs    
    dsp_index = random.randint(0, max_num_DSPs - 1)
    lm_config[dsp_index] = 1    
    return lm_config

def create_bounds(num_vehicles_per_DSP):
    veh_ranges = np.cumsum(num_vehicles_per_DSP)
    bounds = []
    bounds.append(range(veh_ranges[0]))
    for i in range(len(veh_ranges)):
        if i != 0:
            bounds.append(range(veh_ranges[i-1], veh_ranges[i]))
    return bounds

def last_miler_emissions_training(sol_df, emissions_matrix_EV):
    if sol_df is not None:
        x_rows = sol_df[sol_df['name'].str.startswith('x_')]
        x_rows.reset_index(inplace=True)
        # Remove possible duplicates
        if len(x_rows) > 0:
            for i in range(len(x_rows)):
                if x_rows.loc[i]['value'] < 0.1:
                    x_rows.drop([i], inplace=True)
        x_rows.reset_index(inplace=True)

        total_emissions_d = 0

        for i in range(x_rows.shape[0]):
            row = x_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            total_emissions_d += emissions_matrix_EV[row[0]][row[1]][0]
    else:
        total_emissions_d = None
    return total_emissions_d  

def train_regression_models_grid_search_svm(lm_data):
    param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.1, 0.2, 0.5, 1.0],
    'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
    'max_iter': [1000, 5000, 10000],
    'tol': [1e-4, 1e-3, 1e-2]
    }
    
    # --- Last Mile Agent ---
    lm_timer = pc()
    # Split into features and target
    target_LM = lm_data['LM_emissions']
    features_LM = lm_data.drop(['LM_objective','LM_best_bound','LM_gap','LM_sol_time','LM_emissions'], axis=1, inplace=False)
    # Create train-test split
    features_train_LM, features_test_LM, target_train_LM, target_test_LM = train_test_split(features_LM, target_LM, test_size=0.8, random_state=42)
    
    # Fit model for last mile agent
    svr_lm = LinearSVR()
    # Initialize GridSearchCV
    grid_search_lm = GridSearchCV(svr_lm, param_grid, cv=5, n_jobs=-1, verbose=2)    
    # Fit the model
    grid_search_lm.fit(features_train_LM, target_train_LM)
    # Get the best model
    svr_regr_lastm = grid_search_lm.best_estimator_
    test_score_l = svr_regr_lastm.score(features_test_LM, target_test_LM)  
    lm_total_train_time = pc()-lm_timer 
    return svr_regr_lastm, test_score_l, lm_total_train_time

def constraint_extrapolation_SVM(trained_model):
    '''
    :return: constraint: it has the following structure: Coeff*x+intercept >= 0
    '''
    columns = [feature for feature in trained_model.feature_names_in_]
    constraint = pd.DataFrame(data=[trained_model.coef_], columns=columns)
    constraint['intercept'] = trained_model.intercept_
    return constraint