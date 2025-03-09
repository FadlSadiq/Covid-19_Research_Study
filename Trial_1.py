import numpy as np
import matplotlib.pyplot as plt

class CBFModel:
    def __init__(self, population, initial_infected, population_by_age, contact_matrix, age_groups=10, hemisphere='north'):
        # Model parameters
        self.population = population #int
        self.population_by_age = population_by_age
        self.hemisphere = hemisphere
        
        # Compartments for each age group
        self.S = np.zeros(age_groups)  # Susceptible
        self.S_B = np.zeros(age_groups)  # Behavioral Susceptible
        self.E = np.zeros(age_groups)  # Exposed
        self.I = np.zeros(age_groups)  # Infected
        self.R = np.zeros(age_groups)  # Recovered
        
        # Initialize population distribution
        #self.S = np.random.multinomial(population, [1/age_groups]*age_groups) # later check
        
        # Initialize some initial infected
        for i in range(age_groups):
            self.I[i] = min(initial_infected // age_groups, self.S[i])
            self.S[i] -= self.I[i]
        
        # Model parameters
        self.beta = 0.3  # Base transmission rate
        self.beta_B = 0.1  # Behavioral change rate
        self.mu_B = 0.5  # Behavior relaxation rate
        self.gamma = 0.1  # Sensitivity to reported deaths
        self.r = 0.5  # Efficacy of behavioral changes
        self.epsilon = 1/5.2  # Rate of progression from Exposed to Infected
        self.mu = 1/7  # Recovery rate
    
        # Normalize contact matrix
        self.contact_matrix = self.normalize_contact_matrix(
            contact_matrix, 
            population_by_age
        )

        # Initialize compartments
        for i in range(age_groups):
            # Distribute initial population
            self.S[i] = population_by_age[i]
            
            # Distribute initial infected
            infected_in_group = min(
                initial_infected * (population_by_age[i] / self.population), 
                self.S[i]
            )
            self.I[i] = infected_in_group
            self.S[i] -= infected_in_group
    
    def normalize_contact_matrix(self, contact_matrix, population_by_age):
        # Create a copy to avoid modifying the original matrix
        normalized_matrix = contact_matrix.copy()
    
        # Compute normalized contact rates
        for i in range(len(population_by_age)):
            for j in range(len(population_by_age)):
                # Normalize by population sizes
                normalized_matrix[i, j] *= (
                population_by_age[j] / population_by_age[i]
                )
    
        return normalized_matrix

    def Mult1(self, m, p1, p2):
        """Multinomial draw for first event"""
        p3 = max(0, 1 - p1 - p2)
        #print(p1, p2, p3)
        return np.random.multinomial(m, [p1, p2, p3])[0]
    
    def Mult2(self, m, p1, p2):
        """Multinomial draw for second event"""
        p3 = max(0, 1 - p1 - p2)
        return np.random.multinomial(m, [p1, p2, p3])[1]
    
    def calculate_seasonal_modulation(self, t, hemisphere='north'):
        # Seasonality Parameter
        s_min = 0.6  # Can be calibrated, paper suggests uniform distribution between 0.6-1.0
        s_max = 1.0
        alpha_min = 1 #range 0.5-1
        alpha_max = 0.5 #range 0.05 - 0.7

        # Time of maximum seasonality
        if hemisphere == 'north':
            t_max = 15  # January 15th
        else:
            t_max = 195  # July 15th (6 months later)
    
        # Seasonal modulation calculation
        seasonal_term = 0.5 * ((1 - s_min/s_max) * np.sin(2 * np.pi / 365 * (t - t_max) + np.pi/2) + 1 + alpha_min/alpha_max)
    
        return seasonal_term

    def calculate_force_of_infection_cbf(self, t):
        """Calculate force of infection with seasonal modulation"""
        # Simplified seasonal modulation
        s_t = self.calculate_seasonal_modulation(t)
        
        force_of_infection_cbf = np.zeros(len(self.population_by_age))
        
        for k in range(len(self.population_by_age)):
            for k_prime in range(len(self.population_by_age)):
                # Reduced transmission for behavioral susceptible
                contact_rate = self.contact_matrix[k, k_prime]
                force_of_infection_cbf[k] = (
                    self.r * self.beta * s_t * contact_rate * 
                    (self.I[k_prime] / self.population_by_age[k_prime]) 
                    
                )
        
        return force_of_infection_cbf
    
    def calculate_force_of_infection(self, t):
        """Calculate force of infection with seasonal modulation"""
        # Simplified seasonal modulation
        s_t = self.calculate_seasonal_modulation(t)
        
        force_of_infection = np.zeros(len(self.population_by_age))
        
        for k in range(len(self.population_by_age)):
            for k_prime in range(len(self.population_by_age)):
                # Reduced transmission for behavioral susceptible
                contact_rate = self.contact_matrix[k, k_prime]
                force_of_infection[k] = (
                    self.beta * s_t * contact_rate * 
                    (self.I[k_prime] / self.population_by_age[k_prime])
                )
        
        return force_of_infection

    def simulate(self, days):
        """Simulate epidemic progression"""
        # Store results
        results = {
            'S': [self.S.copy()],
            'S_B': [self.S_B.copy()],
            'E': [self.E.copy()],
            'I': [self.I.copy()],
            'R': [self.R.copy()]
        }
        
        for t in range(days):
            # Calculate force of infection
            force_of_infection = self.calculate_force_of_infection(t)
            force_of_infection_cbf = self.calculate_force_of_infection_cbf(t)

            # Temporary storage for transitions
            new_S = self.S.copy()
            new_S_B = self.S_B.copy()
            new_E = self.E.copy()
            new_I = self.I.copy()
            new_R = self.R.copy()
            
            for k in range(len(self.population_by_age)):
                # Transition S → S_B (adopt protective behavior)
                # Adoption Rate
                behavioral_change_rate = self.beta_B * (1 - np.exp(-self.gamma * np.sum(self.I)))
                #susceptible_exposed= self.Mult1(int(self.S[k]), force_of_infection[k], 1-force_of_infection[k])
                
                # Transition S_B → S (return to normal)
                # Relaxation Rate
                relaxation_rate = self.mu_B * (np.sum(self.S) + np.sum(self.R)) / self.population
                
                # Changes using multinomial
                print(force_of_infection[k])
                susceptible_exposed= self.Mult1(int(self.S[k]), 
                                                  force_of_infection[k], 
                                                  behavioral_change_rate)
                

                behavior_exposed = self.Mult1(int(self.S_B[k]), 
                                                  force_of_infection_cbf[k], 
                                                  relaxation_rate)
                
                Susceptible_behavior = self.Mult2(int(self.S[k]), 
                                                  force_of_infection[k], 
                                                  behavioral_change_rate)
                behavior_relaxation = self.Mult2(int(self.S_B[k]), 
                                                  force_of_infection_cbf[k], 
                                                  relaxation_rate)

                # Susceptible to behavior/Exposed
                new_S[k] = new_S[k] - susceptible_exposed - Susceptible_behavior + behavior_relaxation #add one more
                new_S_B[k] = new_S[k] + Susceptible_behavior - behavior_exposed - behavior_relaxation
                
                # Exposed to Infected
                exposed_to_infected = np.random.binomial(int(self.E[k]), self.epsilon)
                new_E[k] = new_E[k] + susceptible_exposed + behavior_exposed - exposed_to_infected
                
                # Infected to Recovered
                infected_to_recovered = np.random.binomial(int(self.I[k]), self.mu)
                new_I[k] = new_I[k] + exposed_to_infected - infected_to_recovered
                new_R[k] = new_R[k] + infected_to_recovered
            
            # Update compartments
            self.S = new_S
            self.S_B = new_S_B
            self.E = new_E
            self.I = new_I
            self.R = new_R
            
            # Store results
            results['S'].append(self.S.copy())
            results['S_B'].append(self.S_B.copy())
            results['E'].append(self.E.copy())
            results['I'].append(self.I.copy())
            results['R'].append(self.R.copy())
        
        return results
    
    def visualize_simulation(self, results, days):
        """Visualize simulation results"""
        plt.figure(figsize=(15, 10))
        
        # Total population for each compartment
        S_total = np.sum(results['S'], axis=1)
        S_B_total = np.sum(results['S_B'], axis=1)
        E_total = np.sum(results['E'], axis=1)
        I_total = np.sum(results['I'], axis=1)
        R_total = np.sum(results['R'], axis=1)
        
        # Plot results
        plt.plot(range(days+1), S_total, label='Susceptible', color='blue')
        plt.plot(range(days+1), S_B_total, label='Behavioral Susceptible', color='green')
        plt.plot(range(days+1), E_total, label='Exposed', color='yellow')
        plt.plot(range(days+1), I_total, label='Infected', color='red')
        plt.plot(range(days+1), R_total, label='Recovered', color='purple')
        
        plt.title('CBF Model Simulation')
        plt.xlabel('Days')
        plt.ylabel('Number of Individuals')
        plt.legend()
        plt.grid(True)
        plt.show()

# Run simulation
np.random.seed(42)
population_by_age = [
    100,  # 0-9
    120,  # 10-19
    150,  # 20-24
    200,  # 25-29
    250,  # 30-39
    220,  # 40-49
    180,  # 50-59
    150,  # 60-69
    100,  # 70-79
    100    # 80+
]
contact_matrix = np.array([
       [ 2,  7,  5,  5,  2,  7,  4,  1,  5,  3],
       [ 7,  8,  6,  2,  4,  6,  5,  3,  6,  5],
       [ 5,  6,  1,  9,  9,  3,  4,  2,  3,  4],
       [ 5,  2,  9,  2,  7,  5,  4,  3,  8,  7],
       [ 2,  4,  9,  7,  6,  6,  8,  5,  4,  5],
       [ 7,  6,  3,  5,  6,  9,  6,  3,  8,  4],
       [ 4,  5,  4,  4,  8,  6,  1,  5,  4,  5],
       [ 1,  3,  2,  3,  5,  3,  5,  1,  3,  7],
       [ 5,  6,  3,  8,  4,  8,  4,  3, 10,  2],
       [ 3,  5,  4,  7,  5,  4,  5,  7,  2,  8],
])
days = 100

model = CBFModel(
    population=sum(population_by_age),
    initial_infected=500,
    population_by_age=population_by_age,
    contact_matrix=contact_matrix
)
results = model.simulate(days)
model.visualize_simulation(results, days)