**Model Description:**

This model solves a Vehicle Routing Problem with Time Windows (VRPTW) in order to simulate the operation of the LMAD robot. The deliveries must occur within specified time windows while minimizing the total distance travelled. The routing is accomplished with the aid of the CPLEX solver. At each delivery point, the robot waits for a specified time for the parcel recipients. At each delivery point while the robot is waiting, we simulate the operation of a single-server queue in order to determine the percentage of customers served, the average waiting time for served customers, and the average queue length.

**Input Files:**

1. The Parcel Demand file. This file contains the origin and destination for each parcel to be delivered.
2. A problem instance file

**Input parameters:**

Each problem instance contains a file with the following information.

1. Delivery points’ postcodes (these should match the locations in the Parcel Demand file)
2. Earliest time at which a delivery point can be served
3. Latest time at which a delivery point can be served
4. The customer arrival rate (per minute) at each delivery point
5. The waiting time (also known as the service time) at each delivery point
6. The electricity generation breakdown of the city
7. The robot speed (currently set to walking speed (1.42 m/s) in all test instances)
8. The battery capacity of the robot
9. The average time it takes to deliver a single parcel

**Outputs:**

For the given problem instance, the output file contains:

1. The time at which the robot arrives at each delivery point
2. The number of parcels delivered at each delivery point
3. The percentage of parcels successfully delivered at each delivery point
4. The average waiting time at each delivery point
5. The average queue length at each delivery point
6. The total emissions (in gCO2eq) during the delivery operation
