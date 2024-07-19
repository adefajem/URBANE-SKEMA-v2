**Model Description:**

This model solves a Vehicle Routing Problem with Time Windows (VRPTW) in order to simulate the operation of the LMAD robot. The deliveries must occur within specified time windows while minimizing the total distance travelled. At each delivery point, the robot waits for a specified time for the parcel recipients. The total number of parcels delivered at each node is a function of the number of parcels to be delivered, the waiting time at each delivery point, and the time it takes to deliver a single parcel. The model uses the CPLEX solver.

**Input Files:**

1. The Parcel Demand file. This file contains the origin and destination for each parcel to be delivered.
2. A problem instance file

**Input parameters:**

Each problem instance contains a file with the following information.

1. Delivery points’ postcodes (these should match the locations in the Parcel Demand file)
2. Earliest time at which a delivery point can be served
3. Latest time at which a delivery point can be served
4. The waiting time at each delivery point
5. The electricity generation breakdown of the city
6. The robot speed (currently set to walking speed (1.42 m/s) in all test instances)
7. The battery capacity of the robot
8. The average time it takes to deliver a single parcel

**Outputs:**

For the given problem instance, the output file contains:

1. The time at which the robot arrives at each delivery point
2. The number of parcels delivered (based on the waiting time and the average delivery time)
3. The percentage of parcels successfully delivered
4. The total emissions (in gCO2eq) during the delivery operation
