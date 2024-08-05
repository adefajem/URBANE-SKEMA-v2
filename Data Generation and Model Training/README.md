**Model Description:**

This model is used to learn the emissions associated with certain routes in a network. Given a complete city network with a set of micro-hub and delivery locations, the model selects subnetworks, generates random assignments, runs the Pickup and Delivery Vehicle Routing Problem with Time Windows (PDPTW) to get a route, and then computes the emissions associated with that route. It does this thousands of times to generate a training set such that train a predictive model to learn the emissions associated with routes in the network.

**Input Files:**

1. The city_network_training_config file. This file contains:
    1. the locations of possible last-mile depots, micro-hubs and historical parcel destinations. origin and destination for each parcel to be delivered,
    2. information on number of possible last-milers and number of vehicles they have,
    3. time-violation penalty which controls the degree to which time window may be violated by the last-milers,
    4. the electricity generation breakdown for the city,
    5. the battery capacity if the electric delivery vehicles being used, and
    6. the number of training points to be generated.

**Outputs:**

The output file contains:

1. The city_network_training_data: which contains the data relating the assignment of parcels to last-milers and micro-hubs to the emissions associated with these assignments.
2. The learned_constraint file: which contains a constraint learning the relationship between the assignments and emissions.
