import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from geopy.distance import geodesic
from docplex.mp.model import Model


def convert_coordinates(x_3879, y_3879):
    # Define the CRS for EPSG:3879 and WGS84
    crs_3879 = CRS.from_epsg(3879)
    crs_4326 = CRS.from_epsg(4326)
    # Create a transformer object
    transformer = Transformer.from_crs(crs_3879, crs_4326, always_xy=True)
    # Perform the transformation
    lon, lat = transformer.transform(x_3879, y_3879)    
    return lat, lon


def process_parcel_demand_data(parcel_demand_filename):    
    parcel_demand_df = pd.read_csv(parcel_demand_filename)
    # Get postcodes and convert X and Y coordinates to latitudes and longitudes
    # Apply the conversion to each row
    parcel_demand_df[['O_latitude', 'O_longitude']] = parcel_demand_df.apply(
        lambda row: convert_coordinates(row['O_zone_X'], row['O_zone_Y']), axis=1, result_type='expand'
    )

    parcel_demand_df[['D_latitude', 'D_longitude']] = parcel_demand_df.apply(
        lambda row: convert_coordinates(row['D_zone_X'], row['D_zone_Y']), axis=1, result_type='expand'
    )

    locations_df = parcel_demand_df[['O_latitude', 'O_longitude', 'D_latitude', 'D_longitude','Postcode_Ozone', 'Postcode_Dzone']]

    # Group by D_latitude and D_longitude and count number of deliveries to that destination
    grouped_df = locations_df.groupby(['D_latitude', 'D_longitude']).size().reset_index(name='Number_of_parcels_to_deliver')

    # Merge the 'Number_deliveries' column back into the original dataframe
    locations_df_with_counts = pd.merge(locations_df, grouped_df, on=['D_latitude', 'D_longitude'])

    # Select unique rows based on D_zone_X and D_zone_Y
    destinations_counts_df = locations_df_with_counts.drop_duplicates(subset=['D_latitude', 'D_longitude'])
    destinations_counts_df.reset_index(inplace=True, drop=True)

    unique_depot_df = locations_df.drop_duplicates(subset=['O_latitude', 'O_longitude'])
    unique_depot_df = unique_depot_df[['O_latitude', 'O_longitude']]
    unique_depot_df.rename(columns={'O_latitude': 'latitude', 'O_longitude': 'longitude'}, inplace=True)
    unique_depot_df['postcode'] = destinations_counts_df['Postcode_Ozone'][0]
    unique_depot_df['type'] = 'depot'


    unique_destinations_df = pd.DataFrame()
    unique_destinations_df['latitude'] = destinations_counts_df['D_latitude']
    unique_destinations_df['longitude'] = destinations_counts_df['D_longitude']
    unique_destinations_df['postcode'] = destinations_counts_df['Postcode_Dzone']
    unique_destinations_df['type'] = 'destination'

    all_nodes_df = pd.concat([unique_depot_df, unique_destinations_df], ignore_index=True)
    return all_nodes_df, destinations_counts_df

def generate_distance_time_matrices(nodes_df, robot_speed):
    all_distances = []
    all_times = []
    avg_speed = robot_speed 
    for i in range(len(nodes_df)):
        origin = (nodes_df.loc[i]["latitude"], nodes_df.loc[i]["longitude"])
        for j in range(len(nodes_df)):
            destination = (nodes_df.loc[j]["latitude"], nodes_df.loc[j]["longitude"])            
            dist = round(geodesic(origin, destination).meters , 2)
            time = round(dist/avg_speed, 2)                
            all_distances.append(dist)
            all_times.append(time)
            
    # Create distance and time matrices
    num_nodes = len(nodes_df)
    distance_matrix = [all_distances[i:i + num_nodes] for i in range(0, len(all_distances), num_nodes)]
    travel_time_matrix = [all_times[i:i + num_nodes] for i in range(0, len(all_times), num_nodes)]                
    return distance_matrix, travel_time_matrix


def generate_bigM_matrix(earliest, latest, wait_time, time_matrix):
    bigM_matrix = []    
    for i in range(len(time_matrix)):
        temp = []
        for j in range(len(time_matrix[i])):
            temp.append(latest[i] + wait_time[i] + time_matrix[i][j])
        bigM_matrix.append(temp)       
    return bigM_matrix


def robot_routing(distance_matrix, time_matrix, bigM_matrix, earliest, latest, wait_times, num_nodes):
    model = Model(name = 'LMAD_routing')
    model.parameters.timelimit = 30
    # ----- Sets -----
    K = range(1) #range(num_robots)
    V = range(num_nodes)  
    A = [(i,j) for i in V for j in V if i!=j]
        
    # ----- Variables ----- 
    # x_kij = 1 if arc (i,j) is traversed by vehicle k  
    x = {(k,i,j):model.binary_var(name='x_%d_%d_%d' % (k,i,j)) for k in K for i in V for j in V if i!=j}
    
    # Arrival time of vehicle k at node i - s_ki
    s = {(k,i):model.continuous_var(lb=0.0, name='s_%d_%d' % (k,i)) for k in K for i in V}
    
    # ----- Objective function ----- 
    model.minimize(model.sum(distance_matrix[i][j] * x[k,i,j] for k in K for (i,j) in A))         

    # ----- Constraints -----
    # Leave depot
    for k in K:
        model.add_constraint(model.sum(x[k,0,j] for j in V if j!=0) == 1, ctname = "leave depot")    
   
    # Visit all delivery nodes
    for i in V:
         model.add_constraint(model.sum(x[k,i,j] for k in K for j in V if i!=j) == 1, ctname = "visit all nodes")    
    
    # Flow conservation
    for k in K:
        for j in V:    
            model.add_constraint(model.sum(x[k,i,j] for i in V if i!=j) - model.sum(x[k,j,i] for i in V if i!=j) == 0,
                                ctname = "flow conservation")        
    # Arrival times
    for (i,j) in A:
        if j==0: continue
        for k in K:
            model.add_constraint(s[k,i] + wait_times[i] + time_matrix[i][j] <= s[k,j] + (1 - x[k,i,j]) * bigM_matrix[i][j],
                                ctname = "arrival times")
         
    # Respect time windows
    for k in K:
        for i in V:
            model.add_constraint(earliest[i] <= s[k,i], ctname = "early window")
            model.add_constraint(s[k,i] <= latest[i], ctname = "late window")
         
    # Solve
    solution = model.solve(log_output = False)   
    print(model.solve_status.name)
    sol_df = None

    if model.solve_status.name == 'OPTIMAL_SOLUTION' or model.solve_status.name == 'FEASIBLE_SOLUTION':
        sol_df = solution.as_df()    
    return sol_df


def EVCO2(kwh_consumption, emission_factor, generation_percentage):
    energy_mix = sum([emission_factor[i] * generation_percentage[i] for i in range(len(emission_factor))])    
    # if you multiply gCO2eq/kWh with kWh you get gCO2eq
    emissions_gCO2 = kwh_consumption * energy_mix    
    return emissions_gCO2 # in gCO2eq


def get_emissions_matrix_EV(battery_capacity, emission_factor, generation_percentage, distance_matrix, travel_time_matrix, num_nodes):
    emission_ij = []    
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):              
            if travel_time_matrix[i][j]!=0:
                kwh_consumption = battery_capacity * distance_matrix[i][j] / 100000 # kilowatt-hours/100 km
                emission = EVCO2(kwh_consumption, emission_factor, generation_percentage)
                emission_ij.append(emission)
            else:
                emission_ij.append(0)                    
    # Put in matrix form
    E_ij = [emission_ij[i:i+num_nodes] for i in range(0, len(emission_ij), num_nodes)]                    
    return E_ij

def calculate_total_emissions_and_distance(sol_df, emissions_matrix_EV, distance_matrix):
    if sol_df is not None:
        x_rows = sol_df[sol_df['name'].str.startswith('x_')]
        x_rows.reset_index(inplace=True, drop=True)
        # Remove possible duplicates
        if len(x_rows) > 0:
            for i in range(len(x_rows)):
                if x_rows.loc[i]['value'] < 0.1:
                    x_rows.drop([i], inplace=True)
        x_rows.reset_index(inplace=True)
        total_emissions = 0
        total_distance = 0
        for i in range(x_rows.shape[0]):
            row = x_rows['name'][i].split('_')
            row.pop(0);    
            row = [int(i) for i in row]
            total_emissions += emissions_matrix_EV[row[1]][row[2]]
            total_distance += distance_matrix[row[1]][row[2]]
    else:
        total_emissions = None
        total_distance = None
    return total_emissions, total_distance  

def single_server_queue(arrival_rate, service_rate, num_customers, max_service_time):
    ''' M/M/1 single-server queue '''    
    # Initialize variables
    inter_arrival_times = np.random.exponential(1 / arrival_rate, num_customers)
    arrival_times = np.cumsum(inter_arrival_times)
    service_times = np.random.exponential(1 / service_rate, num_customers)
    
    # Simulate the queue
    departure_times = []
    current_time = 0
    served_customers = 0
    for i in range(num_customers):
        if len(departure_times) == 0 or arrival_times[i] >= departure_times[-1]:
            current_time = arrival_times[i]
        else:
            current_time = departure_times[-1]

        # Check if the service can be completed within the max service time
        if current_time + service_times[i] <= max_service_time:
            departure_times.append(current_time + service_times[i])
            served_customers += 1
        else:
            break
    
    # Calculate the percentage of customers served
    percentage_served = round((served_customers / num_customers) * 100, 2)
    
    # Calculate metrics for served customers
    if served_customers > 0:
        waiting_times = np.array(departure_times)[:served_customers] - np.array(arrival_times)[:served_customers]
        average_waiting_time = round(np.mean(waiting_times), 2)
        average_queue_length = round(np.mean([np.sum((arrival_times[:served_customers] <= t) & (t < np.array(departure_times)[:served_customers])) for t in arrival_times[:served_customers]]), 2)
    
    return served_customers, percentage_served, average_waiting_time, average_queue_length

def output_result(route_sol, all_nodes_df, destinations_counts_df, arrival_rate, wait_times, average_serve_time_per_parcel):    
    result = None    
    # Extract times
    t_rows = route_sol[route_sol['name'].str.startswith('s')]    
    t_rows = t_rows.sort_values("value")
    t_rows = t_rows.reset_index(drop=True)

    # Get times at which the nodes are visited in HH:MM
    t_df = pd.DataFrame(columns=['j','Arrival_time'])
    for i in range(t_rows.shape[0]):
        row = t_rows['name'][i].split('_')
        row.pop(0);    
        row = [int(i) for i in row]
        j = row[-1]
        hours = int(t_rows['value'][i] //3600)            
        minutes = int((t_rows['value'][i] % 3600) // 60)
        res = [j,"%02d:%02d" % (hours, minutes)]
        t_df.loc[len(t_df)] = res

    # Match node numbers to names
    name_to_num_match = dict(zip(all_nodes_df['postcode'], list(all_nodes_df.index)))
    names =[]
    for i in range(len(t_df)):
        for j in range(len(name_to_num_match)):
            if t_df.loc[i]["j"] == list(name_to_num_match.values())[j]:
                names.append(list(name_to_num_match.keys())[j])
    t_df["Delivery_point"] = names
    
    # Get arrival times at delivery points
    result = t_df[['Delivery_point','Arrival_time']].copy()           
    
    # Compute number of deliveries made based on problem instance input paramaters
    postcode_and_counts = destinations_counts_df[['Postcode_Dzone','Number_of_parcels_to_deliver']]
    postcode_and_counts = postcode_and_counts.rename(columns={'Postcode_Dzone': 'Delivery_point'})
    result = pd.merge(result, postcode_and_counts, on='Delivery_point')
    
    result['Number_of_parcels_delivered'] = ''
    result['Percentage_of_parcels_delivered_(%)'] = ''
    result['Average_waiting_time_(minutes)'] = ''
    result['Average_queue_length'] = ''

    for row in range(len(result)):
        current_node = result['Delivery_point'][row]
        num_customers = result['Number_of_parcels_to_deliver'][result['Delivery_point'] == current_node].values[0]
        max_service_time =  wait_times[current_node]/60 # in minutes
        served_customers, percentage_served, average_waiting_time, average_queue_length = single_server_queue(arrival_rate[current_node], average_serve_time_per_parcel, num_customers, max_service_time)

        result['Number_of_parcels_delivered'][row] = served_customers
        result['Percentage_of_parcels_delivered_(%)'][row] = percentage_served
        result['Average_waiting_time_(minutes)'][row] = average_waiting_time
        result['Average_queue_length'][row] = average_queue_length
 
    return result
