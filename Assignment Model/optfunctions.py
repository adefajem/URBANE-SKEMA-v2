import numpy as np
import pandas as pd

import math
from docplex.mp.model import Model
from geopy.distance import geodesic
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def find_nearest_locations(instance_depots_df, comp_net_depots_df):
    # Initialize a list to keep track of the selected indices
    selected_indices = []

    # Function to compute the nearest depot
    def find_nearest(instance_location):
        min_distance = float('inf')
        nearest_index = -1
        instance_coords = (instance_location['latitude'], instance_location['longitude'])
        
        for idx, row in comp_net_depots_df.iterrows():
            if idx in selected_indices:
                continue
            comp_coords = (row['latitude'], row['longitude'])
            distance = geodesic(instance_coords, comp_coords).meters
            if distance < min_distance:
                min_distance = distance
                nearest_index = idx
        
        return nearest_index

    # Find the nearest depot for each instance depot
    instance_original_lookup = {}
    for idx, instance_row in instance_depots_df.iterrows():
        nearest_index = find_nearest(instance_row)
        instance_original_lookup[idx] = nearest_index
        selected_indices.append(nearest_index)
            
    # LM config or locker config
    config = [0]*len(comp_net_depots_df)
    for i in selected_indices:
        config[i] = 1
        
    return config, instance_original_lookup

def get_destinations(df):
    destinations = {}
    for index, row in df.iterrows():
        if row['type'] == 'package_destination':
            destinations[int(row['destination_id'])] = int(row['node'])
    return destinations

def get_problem_instance_info(complete_network, problem_config, selected_locker_nodes):    
    complete_network['problem_config'] = problem_config
    city_sub_instance = complete_network[complete_network['problem_config'] == 1]

    last_depots_df = city_sub_instance.loc[complete_network['type'].str.contains('LM')]
    last_depots_df.reset_index(inplace=True, drop=True)

    package_ids = list(city_sub_instance.loc[city_sub_instance['type']=='package_destination']['destination_id'])
    package_ids = [int(x) for x in package_ids]

    # Get package destinations
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

    return package_ids, destinations, dsp_depots, dsp_d_nodes, dsp_d_arcs

def output_problem_instance(complete_network, instance_network):
    # LM depot matching
    comp_net_depots_df = complete_network[complete_network['type'].str.contains('depot', case=False)]
    comp_net_depots_df.reset_index(inplace=True,drop=True)
    instance_depots_df = instance_network[instance_network['type'].str.contains('depot', case=False)]
    instance_depots_df.reset_index(inplace=True,drop=True)
    lm_config, lm_lookup = find_nearest_locations(instance_depots_df, comp_net_depots_df)

    # Microhub matching
    comp_net_microhubs_df = complete_network[complete_network['type']=='locker']
    comp_net_microhubs_df.reset_index(inplace=True,drop=True)
    instance_microhubs_df = instance_network[instance_network['type']=='locker']
    instance_microhubs_df.reset_index(inplace=True,drop=True)
    locker_config, locker_lookup_orig = find_nearest_locations(instance_microhubs_df, comp_net_microhubs_df)
    # Offset locker lookup
    locker_offset = complete_network[complete_network['type'].str.contains('locker', case=False)].iloc[0]['node']
    locker_lookup = {key: value+locker_offset for key, value in locker_lookup_orig.items()}

    # Destination matching
    comp_net_destinations_df = complete_network[complete_network['type']=='package_destination']
    comp_net_destinations_df.reset_index(inplace=True,drop=True)
    instance_destinations_df = instance_network[instance_network['type']=='package_destination']
    instance_destinations_df.reset_index(inplace=True,drop=True)
    package_config, destination_lookup = find_nearest_locations(instance_destinations_df, comp_net_destinations_df)

    # Create mask after matching
    problem_config = lm_config + locker_config + package_config

    # Instance microhub capacities
    # max_locker_capacities = comp_net_microhubs_df.set_index('node')['locker_capacity'].to_dict()
    max_locker_capacities = instance_microhubs_df.set_index('node')['locker_capacity'].to_dict()
    selected_locker_nodes = [value for value in locker_lookup.values()]
    selected_locker_capacities = {node: capacity for node, capacity in zip(selected_locker_nodes, max_locker_capacities.values())}


    # Create problem instance
    package_ids, destinations, dsp_depots, dsp_d_nodes, dsp_d_arcs = get_problem_instance_info(complete_network, problem_config, selected_locker_nodes)
    return package_ids, destinations, dsp_depots, dsp_d_nodes, dsp_d_arcs, lm_config, selected_locker_nodes, selected_locker_capacities, max_locker_capacities, lm_lookup, locker_lookup, destination_lookup

def map_closest_lockers_back(comp_net_locker_dfs, inst_locker_dfs):
    # Find the closest node in inst_locker_dfs for each node in comp_net_locker_dfs using Geodesic distance
    closest_lockers = {}
    for _, comp_row in comp_net_locker_dfs.iterrows():
        comp_node = comp_row['node']
        comp_lat = comp_row['latitude']
        comp_lon = comp_row['longitude']
        min_distance = float('inf')
        closest_node = None

        for _, inst_row in inst_locker_dfs.iterrows():
            inst_node = inst_row['node']
            inst_lat = inst_row['latitude']
            inst_lon = inst_row['longitude']        
            distance = geodesic((comp_lat, comp_lon), (inst_lat, inst_lon)).km        
            if distance < min_distance:
                min_distance = distance
                closest_node = inst_node
        closest_lockers[comp_node] = closest_node

    return closest_lockers


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

def solve_CL_leader(max_num_DSPs, lm_config, max_num_nodes, max_num_destinations, package_destination_ids, all_locker_nodes,
                    selected_locker_nodes, max_locker_capacities, selected_locker_capacities, learned_constraint, satellite_penalty):   
    model = Model(name = 'CL_Leader')
    model.parameters.timelimit = 900 # seconds
    model.parameters.mip.tolerances.mipgap = 1e-2
    model.parameters.emphasis.mip = 1 # prioritize feasible solutions
    model.parameters.threads = 4    
    
    # --- Optimization ---
    # Create a dict that maps z_f to y_p_d and lamda_p_s
    feature_names = list(learned_constraint.columns[:-1])
    z_feature_names = []
    for f in range(len(feature_names)):
        z_feature_names.append('z_'+str(f))
    z_mapping = {z_feature_names[i]: feature_names[i] for i in range(len(z_feature_names))}
    
    
    # ----- Sets -----
    all_D = range(max_num_DSPs)
    instance_D = range(sum(lm_config))
    all_P = list(range(max_num_destinations)) 
    delivery_P = package_destination_ids 
    not_considered = [value for value in all_P if value not in delivery_P]
    unused_lockers = list(set(all_locker_nodes) - set(selected_locker_nodes))
    
    # ----- Variables -----
    # ----- Leader Variables -----
    # y_pd = 1 if parcel p is offered to DSP d by the leader
    y = {(p, d): model.binary_var(name='y_%d_%d' % (p, d)) for p in all_P for d in all_D}
    # lamda_ps = 1 if parcel p is placed at satellite s
    lamda = {(p, s): model.binary_var(name='lamda_%d_%d' % (p, s)) for p in all_P for s in all_locker_nodes}    
    # z_f is based on value of y and lamda
    z = {f: model.binary_var(name='z_%d' % f) for f in range(len(z_feature_names))}
    # last mile emissions
    emissions_last = model.continuous_var(lb=0.0, name='last_emission')    

    # Number of satellites used
    # phi = {s: model.binary_var(name='phi_%d' % s) for s in all_locker_nodes}   
    phi = {s: model.binary_var(name='phi_%d' % s) for s in selected_locker_nodes}  
    # print('all_locker_nodes=',all_locker_nodes)
    
    
    # ----- Leader Objective Function: Minimize the emissions as well as the amount of satellites and cost -----
    model.minimize(emissions_last + model.sum(satellite_penalty*phi[s] for s in selected_locker_nodes)
                  )
    
    # ----- Constraints -----
    # ----- Filtering Constraints -----
    # Add constraints on y and lamda that fix the values of non-existents packages
    # to zero
    for p in not_considered:
        for d in all_D:
            model.add_constraint(model.get_var_by_name('y_%d_%d' % (p,d)) == 0)
        for s in all_locker_nodes:
            model.add_constraint(model.get_var_by_name('lamda_%d_%d' % (p,s)) == 0)

    # Add constraints on y and lamda that ensure that unused lockers remain unused
    for p in all_P:
        for s in unused_lockers:
            model.add_constraint(model.get_var_by_name('lamda_%d_%d' % (p,s)) == 0)

            
    # ----- Leader Constraints -----
    # Respect satellites' capacity constraint
    for s in selected_locker_nodes:
        model.add_constraint(model.sum(lamda[p, s] for p in delivery_P) <= selected_locker_capacities[s])
        
    
    # A parcel should only be assigned to one satellite
    for p in delivery_P:
        model.add_constraint(model.sum(lamda[p, s] for s in selected_locker_nodes) == 1)
    
    # A parcel should only be assigned to one DSP
    for p in delivery_P:
        model.add_constraint(model.sum(y[p, d] for d in all_D) == 1)
 
    # Add equity constraints: each DSP is guaranteed a minimum percentage of deliveries
    reserve = len(delivery_P)/10
    min_num_packages = int((len(delivery_P) - reserve)/len(instance_D))
    for d in instance_D:
        model.add_constraint(model.sum(y[p,d] for p in delivery_P) >= min_num_packages)
    
    # Add constraints on the number of vehicles and satellites used
    for p in all_P:
        # number of satellites used
        for s in all_locker_nodes:
            model.add_constraint(model.get_var_by_name('phi_%d' % s) >= model.get_var_by_name('lamda_%d_%d' % (p,s)), ctname='satPen')
    
    # ----- Learned Constraints -----
    # Add constraints that link z to y and lamda
    for fz in range(len(z_feature_names)):
        model.add_constraint(model.get_var_by_name(list(z_mapping.keys())[fz]) == model.get_var_by_name(list(z_mapping.values())[fz]))

    # Add learned constraints for last mile
    intercept = learned_constraint['intercept'].values[0]
    model.add_constraint(model.sum(z[f]*learned_constraint[feature_names[f]].values[0] for f in range(len(feature_names))) + intercept >= emissions_last,
                         ctname='learned_constr')
       

    # Solve the model
    solution = model.solve(log_output = False)   
    sol_df = None
    y_sol_cl = None
    lamda_sol_cl = None
    
    if model.solve_status.name == 'OPTIMAL_SOLUTION' or model.solve_status.name == 'FEASIBLE_SOLUTION':
        sol_df = solution.as_df() 

    # Extract y and lamda from sol_df
    y_sol_cl = extract_y(sol_df, max_num_destinations, sum(lm_config))
    lamda_sol_cl, lamda_df = extract_lamda(sol_df, max_num_destinations, max_num_nodes)
    
    return y_sol_cl, lamda_sol_cl, lamda_df

def Last_Mile_Follower(y, lamda, V2d, A2d, fol_depot, locker_nodes, packages, destinations, num_vehicles_for_follower,
                      distance_matrix, travel_time_matrix, num_time_periods_matrix, bigM_matrix,
                    earliest, latest, leave_time_start, leave_time_end, emissions_matrix_EV, time_violation_penalty, time_limit_seconds):
    model = Model(name = 'Last_Mile_Follower')
    model.parameters.timelimit = time_limit_seconds
    model.parameters.mip.tolerances.mipgap = 1e-2
    model.parameters.emphasis.mip = 1
    model.parameters.threads = 4
    
    # --- Sets ---
    
    P = packages    
    satellites = locker_nodes
        
    M = num_time_periods_matrix  
    K = range(num_vehicles_for_follower)

    # ----- Variables ----- 
    # mu_kp = 1 if package p is caried by vehicle k
    mu = {(k,p):model.binary_var(name='mu_%d_%d' % (k,p)) for k in K for p in P}
        
    # x_kij = 1 if arc (i,j) is traversed by vehicle k  
    x = {(k,i,j):model.binary_var(name='x_%d_%d_%d' % (k,i,j)) for k in K for i in V2d for j in V2d if i!=j}
    
    # x_kmij = 1 if arc (i,j) is traversed by vehicle k in time period m     
    x_m = {(i,j,k,m):model.binary_var(name='xm_%d_%d_%d_%d' % (i,j,k,m)) for i in V2d
                                     for j in V2d if j!=i
                                     for k in K
                                     for m in range(M[i][j]) }
    
    # Arrival time of vehicle k at node i - t_ki
    t = {(k,i):model.continuous_var(lb=0.0, name='t_%d_%d' % (k,i)) for k in K for i in V2d}  
        
    # Variables for earliest and latest times
    alpha_early = {(k,i): model.continuous_var(lb=0.0, name='alphaEarly_%d_%d' % (k,i)) for k in K for i in V2d}
    alpha_late = {(k,i): model.continuous_var(lb=0.0, name='alphaLate_%d_%d' % (k,i)) for k in K for i in V2d}
    
    
    # ----- Objective function: Minimize the emissions travel cost ----- 
    model.minimize(model.sum(emissions_matrix_EV[i][j][m] * x_m[i,j,k,m] for (i,j) in A2d for k in K for m in range(M[i][j]))
                   + time_violation_penalty * model.sum(alpha_early[k,i] for k in K for i in V2d)
                   + time_violation_penalty * model.sum(alpha_late[k,i] for k in K for i in V2d)
                  )

    
    # ----- Time Violation Constraints ----- 
    for k in K:
        for i in V2d:
            model.add_constraint(alpha_early[k,i] >= earliest[i]*model.sum(x[k,i,j] for j in V2d if i!=j) - t[k,i])
            model.add_constraint(alpha_late[k,i] >= t[k,i] - latest[i]*model.sum(x[k,i,j] for j in V2d if i!=j))
    
    # ----- Constraints -----     
    
    # If a DSP is assigned a package, it should be carried on one vehicle k
    for p in P:
        model.add_constraint(model.sum(mu[k,p] for k in K) == y[p], "assign")
        
    # If a vehicle k picks up a package p, it should go to the destination of p
    for p in P:
        for k in K:
            model.add_constraint(model.sum(x[k,j,destinations[p]] for j in V2d if j != destinations[p]) == mu[k,p])
    
    # A package p should be picked up from its satellite s
    for p in P:
        for k in K:
            model.add_constraint(model.sum(lamda[p,s]*x[k,s,j] for s in satellites for j in V2d if s!=j) >= mu[k,p])        
            
    # A vehicle shoud go from a depot to a satellite
    for k in K:
        model.add_constraint(model.sum(x[k,fol_depot,s] for s in satellites) <= model.sum(mu[k,p] for p in P))
    
    # A vehicle may go from a satellite to another node
    for k in K:
        for s in satellites:
            model.add_constraint(model.sum(x[k,s,j] for j in V2d if s!=j) <= 1)
    
    # Flow conservation at all nodes
    for k in K:
        for j in V2d:    
            model.add_constraint(model.sum(x[k,i,j] for i in V2d if i!=j) - model.sum(x[k,j,i] for i in V2d if i!=j) == 0)
 
    
    # Satellite Time constraints
    for p in P:
        for k in K:
            model.add_constraint(model.sum(lamda[p,s]*(t[k,s] + min(travel_time_matrix[s][destinations[p]])) for s in satellites) <= 
                                 t[k, destinations[p]] + (1-mu[k,p])*100)#bigN[p][k])
    
    # Arrival time
    for (i,j) in A2d:
        if j==fol_depot: continue
        for k in K:
            model.add_constraint(t[k,i] + model.sum(x_m[i,j,k,m] * travel_time_matrix[i][j][m] for m in range(M[i][j]) if i!=j) <= 
                                 t[k,j] + (1 - x[k,i,j]) * bigM_matrix[i][j])

    # Time period
    for k in K:
        for i in V2d:
            model.add_constraint(model.sum(x_m[i,j,k,m]*leave_time_start[i][j][m] for j in V2d for m in range(M[i][j]) if i!=j) <= t[k,i])
            model.add_constraint(t[k,i] <= model.sum(x_m[i,j,k,m]*leave_time_end[i][j][m] for j in V2d for m in range(M[i][j]) if i!=j)) 
    
    # Select only one time period to leave
    for (i,j) in A2d: 
        for k in K:
            model.add_constraint(model.sum(x_m[i,j,k,m] for m in range(M[i][j])) == x[k,i,j])
    
    # Solve
    solution = model.solve(log_output = False)   
    sol_df = None    
    
    if model.solve_status.name == 'OPTIMAL_SOLUTION' or model.solve_status.name == 'FEASIBLE_SOLUTION':
        sol_df = solution.as_df()
        
    return sol_df


def extract_y(sol_df, num_original_destinations, num_instance_LMs):
    y_df = None    
    if len(sol_df) > 0:
        y_rows = sol_df[sol_df['name'].str.startswith('y')]
        y_rows = y_rows.reset_index(drop=True)
        # Remove possible duplicates
        if len(y_rows) > 0:
            for i in range(len(y_rows)):
                if y_rows.loc[i]['value'] < 0.1:
                    y_rows.drop([i], inplace=True)
        y_rows = y_rows.reset_index(drop=True)
    
    y_df = pd.DataFrame(columns=['p','d'])
    for i in range(y_rows.shape[0]):
        row = y_rows['name'][i].split('_')
        row.pop(0);    
        row = [int(i) for i in row]
        y_df.loc[len(y_df)] = row
        
    # Initialize the array with zeros
    y_sol_cl = np.zeros((num_original_destinations, num_instance_LMs), dtype=int)
    for p, d in zip(y_df['p'], y_df['d']):
        y_sol_cl[p, d] = 1

    return y_sol_cl

def extract_lamda(sol_df, num_original_destinations, max_num_nodes):    
    lamda_df = None    
    if len(sol_df) > 0:
        lamda_rows = sol_df[sol_df['name'].str.startswith('lamda')]
        lamda_rows = lamda_rows.reset_index(drop=True)
        # Remove possible duplicates
        if len(lamda_rows) > 0:
            for i in range(len(lamda_rows)):
                if lamda_rows.loc[i]['value'] < 0.1:
                    lamda_rows.drop([i], inplace=True)
        lamda_rows = lamda_rows.reset_index(drop=True)

    lamda_df = pd.DataFrame(columns=['p','s'])
    for i in range(lamda_rows.shape[0]):
        row = lamda_rows['name'][i].split('_')
        row.pop(0);    
        row = [int(i) for i in row]
        lamda_df.loc[len(lamda_df)] = row

    # Initialize the array with zeros
    lamda_sol_cl = np.zeros((num_original_destinations, max_num_nodes), dtype=int)
    for p, s in zip(lamda_df['p'], lamda_df['s']):
        lamda_sol_cl[p, s] = 1

    return lamda_sol_cl, lamda_df

def extract_t(lastmiler_final_sol):
    t_df = None    
    if len(lastmiler_final_sol) > 0:
        t_rows = lastmiler_final_sol[lastmiler_final_sol['name'].str.startswith('t_')]
        t_rows = t_rows.reset_index(drop=True)            

        t_df = pd.DataFrame(columns=['k', 'j','time'])
        for i in range(t_rows.shape[0]):
            row = t_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            row.append(t_rows['value'][i])
            t_df.loc[len(t_df)] = row    
      
    return t_df

def extract_xm(sol_df, distance_matrix):
    xm_df = None    
    if len(sol_df) > 0:
        xm_rows = sol_df[sol_df['name'].str.startswith('xm')]
        xm_rows = xm_rows.reset_index(drop=True)
        # Remove possible duplicate
        if len(xm_rows) > 0:
            for i in range(len(xm_rows)):
                if xm_rows.loc[i]['value'] < 0.1:
                    xm_rows.drop([i], inplace=True)
        xm_rows.reset_index(inplace=True)
        
        xm_df = pd.DataFrame(columns=['i','j','k','m','c_ijm'])
        for i in range(xm_rows.shape[0]):
            row = xm_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            row.append(distance_matrix[row[0]][row[1]][row[3]])
            xm_df.loc[len(xm_df)] = row      
    return xm_df

def extract_xm_from_t(t_sol_df, distance_matrix):    
    xm_sol = t_sol_df.sort_values(by=['time'])
    xm_sol = xm_sol.reset_index(drop=True)
    i_list = xm_sol['j'].tolist()
    first = i_list[0]
    i_list.pop(0)
    i_list.append(first)
    xm_sol['i'] = i_list
    xm_sol = xm_sol[['k', 'j', 'i', 'time']]
    xm_sol.rename(columns={"k": "k", "j": "i", "i": "j", "time": "time"}, inplace=True)
    c_ijm = [] 
    for b in range(len(xm_sol)):
        c_ijm.append(distance_matrix[int(xm_sol['i'][b])][int(xm_sol['j'][b])][0])
    xm_sol['c_ijm'] = c_ijm
    xm_sol.drop(columns=['k'], inplace=True)    
    return xm_sol

def balance_clusters(df, kmeans, coordinates, max_size):
    cluster_sizes = df['cluster'].value_counts()
    large_clusters = cluster_sizes[cluster_sizes > max_size].index.tolist()
    
    for cluster in large_clusters:
        # Get indices of the points in the large cluster
        indices = df[df['cluster'] == cluster].index.tolist()
        
        # Calculate number of points to move
        num_to_move = cluster_sizes[cluster] - max_size
        
        # Get centroids of all clusters
        centroids = kmeans.cluster_centers_
        
        # Calculate distances from each point in the large cluster to the centroids of smaller clusters
        distances = np.min(cdist(coordinates[indices], centroids), axis=1)
        
        # Sort indices by their distance to other cluster centroids
        sorted_indices = sorted(zip(indices, distances), key=lambda x: x[1])
        
        # Move points to the nearest cluster
        for i in range(num_to_move):
            nearest_cluster = np.argmin(cdist([coordinates[sorted_indices[i][0]]], centroids))
            df.at[sorted_indices[i][0], 'cluster'] = nearest_cluster
            
    return df

def run_clustering_vrp(y_sol_cl,lamda_sol_cl, complete_network, d, dsp_d_nodes, dsp_d_arcs, dsp_depots,
                       selected_locker_nodes, package_destination_ids, destinations, num_vehicles_for_followers,
                       distance_matrix, travel_time_matrix, num_time_periods_matrix, bigM_matrix,
                       earliest, latest, leave_time_start, leave_time_end, emissions_matrix_EV,
                       time_violation_penalty):
    # --- Get assignments for LM d ---
    lm_d_assigned_df = complete_network.copy(deep=True)
    lm_d_assigned_df = lm_d_assigned_df[lm_d_assigned_df['destination_id'].notnull()]
    lm_d_assigned_df['mask'] = y_sol_cl[:,d].tolist()
    lm_d_assigned_df = lm_d_assigned_df[lm_d_assigned_df['mask']==1]
    lm_d_assigned_df = lm_d_assigned_df.reset_index(drop=True)
    
    # --- Cluster the addresses ---
    # Number of clusters needed
    num_clusters = math.ceil(len(lm_d_assigned_df) / 10)
    # Extract the coordinates for clustering
    coordinates = lm_d_assigned_df[['latitude', 'longitude']].values

    # Perform initial KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    lm_d_assigned_df['cluster'] = kmeans.fit_predict(coordinates)    
    # Balance the clusters
    lm_d_assigned_df = balance_clusters(lm_d_assigned_df, kmeans, coordinates, max_size=14)
    # Calculate centroids of each cluster
    centroids = kmeans.cluster_centers_
    
    # Find the closest destination_id to each centroid
    closest_destination_ids = {}
    for i, centroid in enumerate(centroids):
        # Use cdist to calculate the distances between the centroid and all coordinates
        distances = cdist([centroid], coordinates)
        closest_idx = np.argmin(distances)
        closest_destination_ids[f'Cluster {i}'] = lm_d_assigned_df.iloc[closest_idx]['destination_id']

        
    # Run VRP within clusters
    LM_Sol_TEMP_DFs = []
    for n in range(num_clusters):
        # Extract nodes for that cluster
        lm_d_cluster_nodes = lm_d_assigned_df[lm_d_assigned_df['cluster']==n]    
        subset_rows = list(lm_d_cluster_nodes['destination_id'])
        subset_rows = [int(i) for i in subset_rows]
        # Create a new array with all zeros
        new_y_sol_cl = np.zeros_like(y_sol_cl[:, d])
        # Set the elements at the specified indices to 1
        new_y_sol_cl[subset_rows] = 1
        # Run VRP
        sol_df_temp = Last_Mile_Follower(new_y_sol_cl, lamda_sol_cl, dsp_d_nodes[d], dsp_d_arcs[d], dsp_depots[d],
                                       selected_locker_nodes, package_destination_ids, destinations, num_vehicles_for_followers[d],
                                       distance_matrix, travel_time_matrix, num_time_periods_matrix, bigM_matrix,
                                       earliest, latest, leave_time_start, leave_time_end, emissions_matrix_EV,
                                       time_violation_penalty, 30)
        LM_Sol_TEMP_DFs.append(sol_df_temp)

    # Combine together
    all_cluster_t_dfs = []
    for i in range(len(LM_Sol_TEMP_DFs)):
        cluster_t_df = extract_t(LM_Sol_TEMP_DFs[i])
        all_cluster_t_dfs.append(cluster_t_df)

    merged_t_clust_df = pd.concat(all_cluster_t_dfs)
    t_sol_df = merged_t_clust_df.loc[merged_t_clust_df.groupby('j')['time'].idxmin()].drop_duplicates(subset=['j']).reset_index(drop=True)
    xm_sol_df = extract_xm_from_t(t_sol_df, distance_matrix)
    
    return xm_sol_df, t_sol_df


def get_times_at_destinations(t_Sol_DFs, y_sol_cl, packages, destinations, dsp_depots, locker_nodes,
                             destination_lookup, num_instance_LMs, instance_network):    
    all_arrive_times = pd.DataFrame(columns=['k','j','time'])
    irrelevant_rows = dsp_depots + locker_nodes
    
    for d in range(len(t_Sol_DFs)):    
        # If solution for DSP d exists:
        if len(t_Sol_DFs[d]) > 0:
            arrive_time_d = t_Sol_DFs[d]
            arrive_time_d_filtered = arrive_time_d[~arrive_time_d['j'].isin(irrelevant_rows)]
            
            all_arrive_times = pd.concat([all_arrive_times, arrive_time_d_filtered], axis=0)
            
    # Get times at which the nodes are visited in HH:MM
    converted_hours = []
    for i in range(len(all_arrive_times)):
        hours = int(all_arrive_times.iloc[i]['time'])
        minutes = int(all_arrive_times.iloc[i]['time']*60) % 60
        converted_hours.append("%02d:%02d" % (hours, minutes))
    
    # Create dataframe and return
    times_at_dest_df = pd.DataFrame()
    times_at_dest_df['Current Package ID'] = packages
    times_at_dest_df['Destination node'] = destinations
    times_at_dest_df['Arrival Time at Destination (HH:MM)'] = converted_hours
    # Initialize the reduced array with zeros
    y_sol_reduced = np.zeros((len(destination_lookup), num_instance_LMs), dtype=int)
    # Populate y_sol_reduced using the destination_lookup
    for i in range(len(destination_lookup)):
        y_sol_reduced[i] = y_sol_cl[destination_lookup[i]]
    times_at_dest_df['Carried by Last Miler'] = get_lastmiler_assgt(y_sol_reduced)
    
    # Put back original IDs
    original_ids = []
    value_to_key_lookup = {v: k for k, v in destination_lookup.items()}
    for i in range(len(times_at_dest_df)):
        key = value_to_key_lookup.get(times_at_dest_df['Current Package ID'][i], 'Key not found')
        original_ids.append(key)    
    times_at_dest_df['Package ID'] = original_ids
    times_at_dest_df.drop(columns=['Current Package ID','Destination node'], inplace=True)     
    times_at_dest_df = times_at_dest_df.sort_values(by=['Package ID'])
            
    # Put back original IDs
    filtered_df = instance_network[instance_network['destination_id'].notna()]
    # Create a dictionary from filtered_df where the destination_id is the key and the node is the value
    destination_to_node = dict(zip(filtered_df['destination_id'].astype(int), filtered_df['node']))
    # Add the 'original_dests' column to times_at_dest_df
    times_at_dest_df['Destination node'] = times_at_dest_df['Package ID'].map(destination_to_node)    
    times_at_dest_df = times_at_dest_df[['Package ID','Destination node','Arrival Time at Destination (HH:MM)','Carried by Last Miler']]
    times_at_dest_df = times_at_dest_df.reset_index(drop=True)
    
    return times_at_dest_df


def get_distance_and_emissions(xm_sol_dfs_final, emissions_matrix_EV):
    dist_res = []
    emm_res = []
    for d in range(len(xm_sol_dfs_final)):    
        # If solution exists:
        if len(xm_sol_dfs_final[d]) > 0:
            # Compute total distance by last-miler
            xm_sol_final_d = xm_sol_dfs_final[d]    
            dist_res.append(xm_sol_final_d['c_ijm'].sum()/1000)  

            # Compute total emissions by last-miler
            total_emissions_d = 0
            for i in range(len(xm_sol_final_d)):
                i_index = int(xm_sol_final_d.loc[i]['i'])
                j_index = int(xm_sol_final_d.loc[i]['j'])
                m_index = 0#int(xm_sol_final_d.loc[i]['m'])
                total_emissions_d += emissions_matrix_EV[i_index][j_index][m_index]
            emm_res.append(total_emissions_d/1000)

        else:
            dist_res.append(0) 
            emm_res.append(0)

    # Create columns based on number of last-milers
    cols = []
    for i in range(len(xm_sol_dfs_final)):
        cols.append('Last Miler ' + str(i))

    # create dataframe and write to file
    dist_emm_df = pd.DataFrame([dist_res, emm_res], columns = cols, index = (['Total distance (km)', 'Total CO2 emissions (gCO2eq)']))
    
    return dist_emm_df

def get_locker_assignments(lamda_df, selected_locker_nodes, max_locker_capacities, destination_lookup):
    # Add a new column 'Package ID' by replacing the values in 'p' using reversed_package_lookup
    reversed_package_lookup = {v: k for k, v in destination_lookup.items()}
    lamda_df['Package ID'] = lamda_df['p'].map(reversed_package_lookup)
   
    # Add the 'Locker node' column by mapping the 's' column using reverse_locker_map
    reverse_locker_map = {node: size for node, size in zip(selected_locker_nodes, max_locker_capacities.keys())}
    lamda_df['Locker node'] = lamda_df['s'].map(reverse_locker_map)

    # Drop columns 'p' and 's' from the dataframe
    lamda_df = lamda_df.drop(columns=['p', 's'])   
    lamda_df = lamda_df.sort_values(by=['Package ID'])         
    return lamda_df

def get_lastmiler_assgt(y_sol_final):
    y_sol_final = np.round(y_sol_final)
    indices = []
    for row in y_sol_final:
        index = next((i for i, x in enumerate(row) if x == 1), None)
        indices.append(index if index is not None else -1)
    return indices
 
def write_instance_results_to_file(res_filename, y_sol_cl, lamda_df, selected_locker_nodes, max_locker_capacities,
                                   xm_Sol_DFs, t_Sol_DFs, emissions_matrix_EV, dsp_depots,
                                   packages, destinations, destination_lookup, num_instance_LMs, instance_network):    
    # Get distance travelled and total last-mile emissions
    dist_emm_df = get_distance_and_emissions(xm_Sol_DFs, emissions_matrix_EV)
      
    # Get package assignments to Lockers
    locker_ass_df = get_locker_assignments(lamda_df, selected_locker_nodes, max_locker_capacities, destination_lookup)
    
    # Get package arrival times
    times_at_dest_df = get_times_at_destinations(t_Sol_DFs, y_sol_cl, packages, destinations,
                          dsp_depots, selected_locker_nodes, destination_lookup, num_instance_LMs, instance_network)
    
    # Write all to file
    with pd.ExcelWriter(res_filename) as writer:  
        dist_emm_df.to_excel(writer, sheet_name='Distance and Emissions')
        locker_ass_df.to_excel(writer, sheet_name='Assignments to Lockers', index=False)
        times_at_dest_df.to_excel(writer, sheet_name='Package Arrival Times', index=False)

    return