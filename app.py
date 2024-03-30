#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math


# In[2]:


import pandas as pd
import numpy as np

# Load aircraft dataset
aircraft_data = pd.read_csv('aircraftDataset.csv')

# Load airport dataset
airport_data = pd.read_csv('airportDataset.csv')

# Load weather dataset
weather_data = pd.read_csv('weatherDataset.csv')

# Remove null values from data
weather_df = weather_data.dropna()
aircraft_df = aircraft_data.dropna()
airport_df = airport_data.dropna()

# Display sample data for each dataset
print("\nWeather Data:")
print(weather_df.head())


print("\nAircraft Dataset:")
print(aircraft_df.head())


print("\nAirport Dataset:")
print(airport_df.head())





# Data Cleaning

# In[3]:


# Data Cleaning

# Check for missing values in each dataset
print("Aircraft Dataset - Missing Values:")
print(aircraft_df.isnull().sum())

print("\nAirport Dataset - Missing Values:")
print(airport_df.isnull().sum())

print("\nWeather Dataset - Missing Values:")
print(weather_df.isnull().sum())




# In[4]:


# Assuming the data is already loaded and cleaned

# Function to initialize a population for the Genetic Algorithm
def initialize_population(population_size, num_genes):
    return np.random.choice(aircraft_df['ICAO CODES'], size=(population_size, num_genes))


# Function to calculate the great-circle distance between two points given their coordinates
def haversine(coord1, coord2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    # Haversine formula for distance calculation
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    radius_earth = 6371  # Radius of Earth in kilometers
    distance = radius_earth * c

    return distance

# Function to get wind speed for a given route and date
def get_wind_speed(route, source, destination, date):
    # Filter weather data for the given route and date
    relevant_weather = weather_df[(weather_df['City'].isin([source, destination])) & (weather_df['Date'] == date)]

    # Calculate average wind speed for the selected cities
    average_wind_speed = relevant_weather['Wind Speed'].mean()

    return average_wind_speed

# Function to calculate the fuel consumption for a given route
def calculate_fuel_consumption(route, source, destination, date):
    # Fetch relevant data for the selected aircraft
    aircraft_data = aircraft_df[aircraft_df['ICAO CODES'].isin(route)]

    # Assuming a linear relationship between cruise speed and fuel consumption
    average_cruise_speed = aircraft_data['Cruise Speed'].mean()

    # Fetch latitude and longitude for source and destination
    source_coords = airport_df.loc[airport_df['ICAO Code'] == source, ['Latitude', 'Longitude']].values.flatten()
    destination_coords = airport_df.loc[airport_df['ICAO Code'] == destination, ['Latitude', 'Longitude']].values.flatten()

    # Calculate the great-circle distance between source and destination
    distance = haversine(source_coords, destination_coords)

    # Fetch wind speed for the given route and date
    wind_speed = get_wind_speed(route, source, destination, date)

    # Assume a simple linear relationship between wind speed and fuel consumption
    # Adjust the coefficients based on your specific knowledge or dataset
    fuel_consumption = distance / average_cruise_speed + 0.1 * wind_speed

    return fuel_consumption

# Function to calculate the fitness for a given route, considering fuel consumption and wind speed
def calculate_fitness(route, source, destination, date):
    # Calculate fuel consumption for the route
    fuel_consumption = calculate_fuel_consumption(route, source, destination, date)

    # Fetch wind speed for the given route and date
    wind_speed = get_wind_speed(route, source, destination, date)

    # In this case, fitness is the inverse of fuel consumption weighted by wind speed
    # We use the inverse of wind speed to favor routes with low wind speed
    fitness = 1 / (fuel_consumption * (1 / (wind_speed + 1)))

    return 3

# Function for tournament selection
def tournament_selection(population, fitness_values, tournament_size):
    selected_indices = []
    for i in range(len(population)):
        competitors = np.random.choice(len(population), size=tournament_size, replace=False)
        winner_index = competitors[np.argmax(fitness_values[competitors])]
        selected_indices.append(winner_index)
    return selected_indices

# Function for crossover operation (single-point crossover)
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Function for mutation operation
def mutate(individual, mutation_rate):
    mutated_individual = individual.copy()

    # Randomly select genes to mutate based on mutation_rate
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            mutated_individual[i] = np.random.choice(aircraft_df['ICAO CODES'])

    return mutated_individual

# Function to perform Genetic Algorithm
def genetic_algorithm(population_size, num_genes, generations, mutation_rate, source, destination, date):
    # Initialize population
    population = initialize_population(population_size, num_genes)

    for generation in range(generations):
        # Calculate fitness for each individual
        fitness_values = np.array([calculate_fitness(route, source, destination, date) for route in population])

        # Select individuals for crossover using tournament selection
        selected_indices = tournament_selection(population, fitness_values, tournament_size=5)

        # Perform crossover
        for i in range(0, len(selected_indices), 2):
            parent1 = population[selected_indices[i]]
            parent2 = population[selected_indices[i + 1]]
            child1, child2 = crossover(parent1, parent2)
            population[selected_indices[i]] = child1
            population[selected_indices[i + 1]] = child2

        # Perform mutation
        for i in range(len(population)):
            if np.random.rand() < mutation_rate:
                population[i] = mutate(population[i], mutation_rate)

    # Return the best route based on the final population
    best_route_index = np.argmax(fitness_values)
    best_route = population[best_route_index]
    # Remove duplicates in the route
    best_route = list(dict.fromkeys(best_route))
    # Remove source and destination from the route
    if source in best_route:
        best_route.remove(source)

    if destination in best_route:
        best_route.remove(destination)
    # Remove duplicates from the route
    best_route = list(dict.fromkeys(best_route))
    # Add source and destination to the beginning and end of the route
    best_route = [source] + best_route + [destination]
    return best_route

# Example usage
population_size = 100
num_genes = 10
generations = 5
mutation_rate = 0.1

# Get user input
# source = input("Enter the source airport (ICAO Code): ")
# destination = input("Enter the destination airport (ICAO Code): ")
# date = input("Enter the date of the flight (YYYY-MM-DD): ")

# print("Best Route:", best_route)


from flask import Flask, render_template, request,jsonify


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    
    
    source = request.form.get('source')
    destination = request.form.get('destination')
    date = request.form.get('date')
    best_route = genetic_algorithm(population_size, num_genes, generations, mutation_rate, source, destination, date)
    print(best_route)
    return jsonify({'best_route': best_route})
#     return render_template('index.html', best_route=best_route)

if __name__ == '__main__':
    app.run(debug=True)

