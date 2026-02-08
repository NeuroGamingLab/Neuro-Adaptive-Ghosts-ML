"""
Evolutionary Ghost Behaviors Module

Uses genetic algorithms to evolve ghost hunting strategies:
- Genome encodes behavior traits (aggression, coordination, prediction, etc.)
- Fitness based on catching Pacman and survival
- Selection, crossover, mutation for evolution
- Ghosts improve over generations!

Each ghost has a unique genome that affects:
- Chase aggression (direct vs cautious)
- Prediction ability (anticipate Pacman moves)
- Coordination tendency (pack hunting vs solo)
- Ambush preference (cut-off vs chase)
- Risk tolerance (chase into power pellet zones)
"""

import numpy as np
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
import random


@dataclass
class GhostGenome:
    """
    Genetic representation of ghost behavior.
    Each trait is a float [0, 1] that influences decisions.
    """
    # Core behavior traits
    aggression: float = 0.5        # 0=cautious, 1=aggressive chase
    prediction: float = 0.5        # 0=reactive, 1=predictive targeting
    coordination: float = 0.5      # 0=solo hunter, 1=pack hunter
    ambush: float = 0.5            # 0=direct chase, 1=cut-off/ambush
    risk_tolerance: float = 0.5    # 0=avoid danger, 1=ignore threats
    
    # Movement traits
    speed_preference: float = 0.5  # 0=slow/methodical, 1=fast/erratic
    persistence: float = 0.5       # 0=easily distracted, 1=focused pursuit
    randomness: float = 0.2        # Random decision factor
    
    # Advanced traits
    learning_rate: float = 0.5     # How fast ghost adapts to player
    memory: float = 0.5            # Remember player patterns
    territorial: float = 0.5       # Guard area vs roam freely
    
    # Fitness tracking (not inherited)
    fitness: float = field(default=0.0, repr=False)
    catches: int = field(default=0, repr=False)
    deaths: int = field(default=0, repr=False)
    generation: int = field(default=0, repr=False)
    
    def to_array(self) -> np.ndarray:
        """Convert traits to numpy array for operations."""
        return np.array([
            self.aggression, self.prediction, self.coordination,
            self.ambush, self.risk_tolerance, self.speed_preference,
            self.persistence, self.randomness, self.learning_rate,
            self.memory, self.territorial
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, generation: int = 0) -> 'GhostGenome':
        """Create genome from numpy array."""
        return cls(
            aggression=float(np.clip(arr[0], 0, 1)),
            prediction=float(np.clip(arr[1], 0, 1)),
            coordination=float(np.clip(arr[2], 0, 1)),
            ambush=float(np.clip(arr[3], 0, 1)),
            risk_tolerance=float(np.clip(arr[4], 0, 1)),
            speed_preference=float(np.clip(arr[5], 0, 1)),
            persistence=float(np.clip(arr[6], 0, 1)),
            randomness=float(np.clip(arr[7], 0, 1)),
            learning_rate=float(np.clip(arr[8], 0, 1)),
            memory=float(np.clip(arr[9], 0, 1)),
            territorial=float(np.clip(arr[10], 0, 1)),
            generation=generation
        )
    
    @classmethod
    def random(cls, generation: int = 0) -> 'GhostGenome':
        """Create a random genome."""
        return cls(
            aggression=random.random(),
            prediction=random.random(),
            coordination=random.random(),
            ambush=random.random(),
            risk_tolerance=random.random(),
            speed_preference=random.random(),
            persistence=random.random(),
            randomness=random.uniform(0.1, 0.4),
            learning_rate=random.random(),
            memory=random.random(),
            territorial=random.random(),
            generation=generation
        )
    
    def get_behavior_description(self) -> str:
        """Get human-readable description of this ghost's personality."""
        traits = []
        
        if self.aggression > 0.7:
            traits.append("Aggressive")
        elif self.aggression < 0.3:
            traits.append("Cautious")
        
        if self.prediction > 0.7:
            traits.append("Predictive")
        
        if self.coordination > 0.7:
            traits.append("Pack Hunter")
        elif self.coordination < 0.3:
            traits.append("Lone Wolf")
        
        if self.ambush > 0.7:
            traits.append("Ambusher")
        elif self.ambush < 0.3:
            traits.append("Chaser")
        
        if self.risk_tolerance > 0.7:
            traits.append("Reckless")
        elif self.risk_tolerance < 0.3:
            traits.append("Careful")
        
        return ", ".join(traits) if traits else "Balanced"


class GhostEvolution:
    """
    Manages the evolutionary process for ghost behaviors.
    
    Runs a genetic algorithm to evolve better ghost hunters:
    1. Initialize population with random genomes
    2. Evaluate fitness during gameplay
    3. Select best performers
    4. Create next generation via crossover + mutation
    5. Repeat!
    """
    
    def __init__(self, population_size: int = 20, elite_count: int = 4,
                 mutation_rate: float = 0.15, mutation_strength: float = 0.2,
                 save_path: str = None):
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.save_path = save_path or "ghost_evolution.json"
        
        self.population: List[GhostGenome] = []
        self.generation = 0
        self.best_fitness_ever = 0.0
        self.best_genome_ever: Optional[GhostGenome] = None
        self.generation_history: List[Dict] = []
        
        # Active ghosts (4 ghosts in game)
        self.active_genomes: List[GhostGenome] = []
        
        # Load or initialize
        if os.path.exists(self.save_path):
            self.load()
        else:
            self._initialize_population()
        
        print(f"Ghost Evolution initialized - Gen {self.generation}, Pop: {len(self.population)}")
    
    def _initialize_population(self):
        """Create initial random population."""
        self.population = [GhostGenome.random(0) for _ in range(self.population_size)]
        self.generation = 0
        
        # Select initial active ghosts
        self._select_active_ghosts()
    
    def _select_active_ghosts(self):
        """Select 4 ghosts for the current game."""
        if len(self.population) < 4:
            # Not enough, create more
            while len(self.population) < 4:
                self.population.append(GhostGenome.random(self.generation))
        
        # Select diverse set: best + random
        sorted_pop = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        
        # Take top 2 and 2 random from rest
        self.active_genomes = sorted_pop[:2]
        remaining = sorted_pop[2:]
        if remaining:
            self.active_genomes.extend(random.sample(remaining, min(2, len(remaining))))
        
        # Fill if needed
        while len(self.active_genomes) < 4:
            self.active_genomes.append(GhostGenome.random(self.generation))
    
    def get_active_genomes(self) -> List[GhostGenome]:
        """Get the 4 genomes for active ghosts."""
        if len(self.active_genomes) < 4:
            self._select_active_ghosts()
        return self.active_genomes[:4]
    
    def record_catch(self, ghost_idx: int):
        """Record that a ghost caught Pacman."""
        if 0 <= ghost_idx < len(self.active_genomes):
            self.active_genomes[ghost_idx].catches += 1
            self.active_genomes[ghost_idx].fitness += 100
    
    def record_death(self, ghost_idx: int):
        """Record that a ghost was eaten."""
        if 0 <= ghost_idx < len(self.active_genomes):
            self.active_genomes[ghost_idx].deaths += 1
            self.active_genomes[ghost_idx].fitness -= 20
    
    def record_chase_time(self, ghost_idx: int, time_near_pacman: float):
        """Record time spent near Pacman (good hunting behavior)."""
        if 0 <= ghost_idx < len(self.active_genomes):
            self.active_genomes[ghost_idx].fitness += time_near_pacman * 0.1
    
    def record_coordination_bonus(self, ghost_indices: List[int]):
        """Bonus for coordinated attacks."""
        for idx in ghost_indices:
            if 0 <= idx < len(self.active_genomes):
                self.active_genomes[idx].fitness += 25
    
    def end_game(self, pacman_won: bool, final_score: int):
        """
        Called when a game ends. Updates fitness and potentially evolves.
        """
        # Penalty if Pacman won
        if pacman_won:
            for genome in self.active_genomes:
                genome.fitness -= 50
        else:
            # Bonus for winning
            for genome in self.active_genomes:
                genome.fitness += 30
        
        # Update population with active genomes' fitness
        for active in self.active_genomes:
            for pop_genome in self.population:
                if self._genomes_equal(active, pop_genome):
                    pop_genome.fitness = active.fitness
                    pop_genome.catches = active.catches
                    pop_genome.deaths = active.deaths
                    break
        
        # Check if we should evolve
        games_played = sum(g.catches + g.deaths for g in self.population)
        if games_played > 0 and games_played % 10 == 0:
            self.evolve()
        
        # Select new active ghosts for next game
        self._select_active_ghosts()
    
    def _genomes_equal(self, g1: GhostGenome, g2: GhostGenome) -> bool:
        """Check if two genomes have same traits."""
        return np.allclose(g1.to_array(), g2.to_array(), atol=0.001)
    
    def evolve(self):
        """Run one generation of evolution."""
        print(f"\n=== EVOLVING GENERATION {self.generation} -> {self.generation + 1} ===")
        
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        
        # Track best
        best = sorted_pop[0]
        avg_fitness = np.mean([g.fitness for g in self.population])
        
        print(f"Best fitness: {best.fitness:.1f}")
        print(f"Avg fitness: {avg_fitness:.1f}")
        print(f"Best traits: {best.get_behavior_description()}")
        
        # Record history
        self.generation_history.append({
            'generation': self.generation,
            'best_fitness': best.fitness,
            'avg_fitness': avg_fitness,
            'best_traits': best.get_behavior_description()
        })
        
        # Update best ever
        if best.fitness > self.best_fitness_ever:
            self.best_fitness_ever = best.fitness
            self.best_genome_ever = GhostGenome.from_array(
                best.to_array(), best.generation
            )
        
        # === SELECTION ===
        # Elitism: keep best performers
        new_population = []
        for elite in sorted_pop[:self.elite_count]:
            new_genome = GhostGenome.from_array(elite.to_array(), self.generation + 1)
            new_population.append(new_genome)
        
        # === CROSSOVER ===
        # Tournament selection + crossover for rest
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select(sorted_pop)
            parent2 = self._tournament_select(sorted_pop)
            
            # Crossover
            child = self._crossover(parent1, parent2)
            child.generation = self.generation + 1
            
            # Mutation
            child = self._mutate(child)
            
            new_population.append(child)
        
        # Update population
        self.population = new_population
        self.generation += 1
        
        # Reset fitness for new generation
        for genome in self.population:
            genome.fitness = 0.0
            genome.catches = 0
            genome.deaths = 0
        
        # Save progress
        self.save()
        
        print(f"New generation {self.generation} created!\n")
    
    def _tournament_select(self, sorted_pop: List[GhostGenome], 
                           tournament_size: int = 3) -> GhostGenome:
        """Select parent via tournament selection."""
        contestants = random.sample(sorted_pop, min(tournament_size, len(sorted_pop)))
        return max(contestants, key=lambda g: g.fitness)
    
    def _crossover(self, parent1: GhostGenome, parent2: GhostGenome) -> GhostGenome:
        """Create child via crossover of two parents."""
        arr1 = parent1.to_array()
        arr2 = parent2.to_array()
        
        # Uniform crossover with blend
        child_arr = np.zeros_like(arr1)
        for i in range(len(arr1)):
            if random.random() < 0.5:
                # Blend crossover
                alpha = random.uniform(0.3, 0.7)
                child_arr[i] = alpha * arr1[i] + (1 - alpha) * arr2[i]
            else:
                # Pick from one parent
                child_arr[i] = arr1[i] if random.random() < 0.5 else arr2[i]
        
        return GhostGenome.from_array(child_arr)
    
    def _mutate(self, genome: GhostGenome) -> GhostGenome:
        """Apply mutation to genome."""
        arr = genome.to_array()
        
        for i in range(len(arr)):
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                mutation = random.gauss(0, self.mutation_strength)
                arr[i] = np.clip(arr[i] + mutation, 0, 1)
        
        return GhostGenome.from_array(arr, genome.generation)
    
    def get_behavior_modifier(self, ghost_idx: int, behavior: str) -> float:
        """
        Get behavior modifier for a ghost based on its genome.
        Used by game to modify ghost behavior.
        """
        if ghost_idx >= len(self.active_genomes):
            return 0.5  # Default
        
        genome = self.active_genomes[ghost_idx]
        
        modifiers = {
            'aggression': genome.aggression,
            'prediction': genome.prediction,
            'coordination': genome.coordination,
            'ambush': genome.ambush,
            'risk': genome.risk_tolerance,
            'speed': genome.speed_preference,
            'persistence': genome.persistence,
            'randomness': genome.randomness,
            'learning': genome.learning_rate,
            'memory': genome.memory,
            'territorial': genome.territorial,
        }
        
        return modifiers.get(behavior, 0.5)
    
    def get_ghost_personality(self, ghost_idx: int) -> str:
        """Get personality description for display."""
        if ghost_idx >= len(self.active_genomes):
            return "Unknown"
        return self.active_genomes[ghost_idx].get_behavior_description()
    
    def get_stats(self) -> str:
        """Get evolution statistics for display."""
        if not self.population:
            return "No population"
        
        avg_fitness = np.mean([g.fitness for g in self.population])
        best = max(self.population, key=lambda g: g.fitness)
        
        return f"Gen {self.generation} | Best: {best.fitness:.0f} | Avg: {avg_fitness:.0f}"
    
    def save(self):
        """Save evolution state to file."""
        data = {
            'generation': self.generation,
            'best_fitness_ever': self.best_fitness_ever,
            'population': [asdict(g) for g in self.population],
            'history': self.generation_history[-100:],  # Keep last 100
        }
        
        if self.best_genome_ever:
            data['best_genome_ever'] = asdict(self.best_genome_ever)
        
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load evolution state from file."""
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            
            self.generation = data.get('generation', 0)
            self.best_fitness_ever = data.get('best_fitness_ever', 0)
            self.generation_history = data.get('history', [])
            
            # Reconstruct population
            self.population = []
            for g_data in data.get('population', []):
                genome = GhostGenome(**{k: v for k, v in g_data.items() 
                                        if k in GhostGenome.__dataclass_fields__})
                self.population.append(genome)
            
            # Load best ever
            if 'best_genome_ever' in data:
                self.best_genome_ever = GhostGenome(**data['best_genome_ever'])
            
            self._select_active_ghosts()
            print(f"Loaded evolution state: Gen {self.generation}")
            
        except Exception as e:
            print(f"Failed to load evolution: {e}")
            self._initialize_population()
    
    def force_evolve(self):
        """Force evolution now (for testing)."""
        self.evolve()
        self._select_active_ghosts()


class EvolvingGhostBehavior:
    """
    Applies evolved genome to ghost decision-making.
    Wraps around existing ghost logic to modify behavior based on traits.
    """
    
    def __init__(self, genome: GhostGenome):
        self.genome = genome
        self.player_pattern_memory = []
        self.last_pacman_positions = []
    
    def should_chase_directly(self, distance_to_pacman: float) -> bool:
        """Decide if ghost should chase directly based on aggression."""
        # High aggression = more likely to chase
        threshold = 1.0 - self.genome.aggression
        return random.random() > threshold * 0.5
    
    def should_ambush(self) -> bool:
        """Decide if ghost should try to cut off Pacman."""
        return random.random() < self.genome.ambush
    
    def should_coordinate(self) -> bool:
        """Decide if ghost should coordinate with pack."""
        return random.random() < self.genome.coordination
    
    def get_prediction_offset(self, pacman_dir: Tuple[int, int]) -> Tuple[int, int]:
        """Get predicted position offset based on prediction trait."""
        if self.genome.prediction < 0.3:
            return (0, 0)  # No prediction
        
        # Predict further ahead with higher prediction
        look_ahead = int(self.genome.prediction * 6)
        return (pacman_dir[0] * look_ahead, pacman_dir[1] * look_ahead)
    
    def should_take_risk(self, near_power_pellet: bool) -> bool:
        """Decide if ghost should take risk near danger."""
        if not near_power_pellet:
            return True
        return random.random() < self.genome.risk_tolerance
    
    def get_random_factor(self) -> float:
        """Get randomness to add to decisions."""
        return self.genome.randomness
    
    def update_player_memory(self, pacman_x: int, pacman_y: int):
        """Track player movement patterns."""
        self.last_pacman_positions.append((pacman_x, pacman_y))
        
        # Keep last N positions based on memory trait
        max_memory = int(self.genome.memory * 20) + 5
        if len(self.last_pacman_positions) > max_memory:
            self.last_pacman_positions.pop(0)
    
    def predict_player_direction(self) -> Tuple[int, int]:
        """Predict player's likely direction based on memory."""
        if len(self.last_pacman_positions) < 3:
            return (0, 0)
        
        # Calculate average direction from recent moves
        recent = self.last_pacman_positions[-5:]
        if len(recent) < 2:
            return (0, 0)
        
        dx = sum(recent[i+1][0] - recent[i][0] for i in range(len(recent)-1))
        dy = sum(recent[i+1][1] - recent[i][1] for i in range(len(recent)-1))
        
        # Normalize
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
        
        return (dx, dy)


# Convenience function
def create_evolution_system(save_path: str = None) -> GhostEvolution:
    """Create or load ghost evolution system."""
    return GhostEvolution(save_path=save_path)
