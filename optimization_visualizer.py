import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from matplotlib.widgets import Button
import time

class OptimizationVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)
        
        # Create buttons
        self.ax_button1 = plt.axes([0.2, 0.05, 0.2, 0.075])
        self.ax_button2 = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.ax_button3 = plt.axes([0.6, 0.05, 0.2, 0.075])
        
        self.button1 = Button(self.ax_button1, 'Hill Climbing')
        self.button2 = Button(self.ax_button2, 'Knapsack')
        self.button3 = Button(self.ax_button3, 'TSP')
        
        self.button1.on_clicked(self.run_hill_climbing)
        self.button2.on_clicked(self.run_knapsack)
        self.button3.on_clicked(self.run_tsp)
        
        self.current_algorithm = None
        self.animation = None

    def rosenbrock(self, x, y):
        a = 1
        b = 100
        return (a - x)**2 + b * (x**2 - y)**2

    def visualize_hill_climbing(self):
        # Create grid for visualization
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-1, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.rosenbrock(X, Y)
        
        # Plot the surface
        self.ax.clear()
        self.ax.contourf(X, Y, Z, levels=20, cmap='viridis')
        self.ax.set_title('Hill Climbing with Random Restarts')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        
        # Initialize starting point
        current_x, current_y = 0, 0
        best_x, best_y = current_x, current_y
        points = [(current_x, current_y)]
        
        def update(frame):
            nonlocal current_x, current_y, best_x, best_y
            
            # Generate candidate solution
            candidate_x = current_x + (random.uniform(-1, 1) * 0.1)
            candidate_y = current_y + (random.uniform(-1, 1) * 0.1)
            
            if self.rosenbrock(candidate_x, candidate_y) < self.rosenbrock(best_x, best_y):
                best_x, best_y = candidate_x, candidate_y
                points.append((best_x, best_y))
            
            current_x, current_y = candidate_x, candidate_y
            
            # Update plot
            self.ax.clear()
            self.ax.contourf(X, Y, Z, levels=20, cmap='viridis')
            self.ax.plot([p[0] for p in points], [p[1] for p in points], 'r-')
            self.ax.scatter([p[0] for p in points], [p[1] for p in points], c='red')
            self.ax.set_title(f'Iteration: {frame}')
            
            if frame % 50 == 0:  # Random restart
                current_x, current_y = random.uniform(-2, 2), random.uniform(-1, 3)
                points.append((current_x, current_y))
        
        self.animation = FuncAnimation(self.fig, update, frames=200, interval=100)
        plt.show()

    def visualize_knapsack(self):
        items = [
            {'value': 60, 'weight': 10},
            {'value': 100, 'weight': 20},
            {'value': 120, 'weight': 30}
        ]
        max_weight = 50
        
        def f(v):
            return sum(items[i]['value'] * v[i] for i in range(len(items)))
        
        def h(v):
            total_weight = sum(items[i]['weight'] * v[i] for i in range(len(items)))
            return max(total_weight - max_weight, 0)
        
        def F(v, c=1.0):
            return f(v) - c * h(v)
        
        current_solution = [0] * len(items)
        best_solution = current_solution.copy()
        history = [F(current_solution)]
        
        def update(frame):
            nonlocal current_solution, best_solution
            
            candidate = current_solution.copy()
            i = random.randint(0, len(items) - 1)
            candidate[i] = 1 - candidate[i]
            
            if F(candidate) > F(best_solution):
                best_solution = candidate.copy()
            
            current_solution = candidate
            history.append(F(current_solution))
            
            self.ax.clear()
            self.ax.plot(history, 'b-')
            self.ax.set_title(f'Knapsack Optimization\nCurrent Value: {f(current_solution)}')
            self.ax.set_xlabel('Iteration')
            self.ax.set_ylabel('Objective Value')
            
            if frame % 50 == 0:  # Random restart
                current_solution = [random.choice([0, 1]) for _ in range(len(items))]
        
        self.animation = FuncAnimation(self.fig, update, frames=200, interval=100)
        plt.show()

    def visualize_tsp(self):
        # Example cities (coordinates)
        cities = np.array([
            [0, 0],
            [1, 2],
            [2, 1],
            [3, 3]
        ])
        
        def total_distance(tour):
            return sum(np.linalg.norm(cities[tour[i]] - cities[tour[i-1]]) 
                      for i in range(len(tour)))
        
        current_tour = list(range(len(cities)))
        best_tour = current_tour.copy()
        T = 1000
        alpha = 0.95
        
        def update(frame):
            nonlocal current_tour, best_tour, T
            
            candidate = current_tour.copy()
            i, j = random.sample(range(len(cities)), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            
            delta = total_distance(candidate) - total_distance(current_tour)
            if delta < 0 or random.random() < np.exp(-delta/T):
                current_tour = candidate
                if total_distance(candidate) < total_distance(best_tour):
                    best_tour = candidate.copy()
            
            T *= alpha
            
            self.ax.clear()
            self.ax.plot(cities[:, 0], cities[:, 1], 'ro')
            self.ax.plot(cities[current_tour + [current_tour[0]], 0],
                        cities[current_tour + [current_tour[0]], 1], 'b-')
            self.ax.set_title(f'TSP Optimization\nTemperature: {T:.2f}')
            
        self.animation = FuncAnimation(self.fig, update, frames=200, interval=100)
        plt.show()

    def run_hill_climbing(self, event):
        if self.animation:
            self.animation.event_source.stop()
        self.visualize_hill_climbing()

    def run_knapsack(self, event):
        if self.animation:
            self.animation.event_source.stop()
        self.visualize_knapsack()

    def run_tsp(self, event):
        if self.animation:
            self.animation.event_source.stop()
        self.visualize_tsp()

if __name__ == "__main__":
    visualizer = OptimizationVisualizer()
    plt.show() 