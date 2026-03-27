"""
Genetic Algorithm for optimizing Fuzzy Controller parameters.

Chromosome = flattened MF parameters (a, b, c) for all fuzzy sets
  - 4 queue sets × 3 params = 12
  - 4 wait sets × 3 params = 12
  - 5 green sets × 3 params = 15
  Total = 39 genes per chromosome

GA operators:
  - Selection: Tournament (k=3)
  - Crossover: BLX-alpha (blend)
  - Mutation: Gaussian perturbation
  - Elitism: top 10% survive unchanged

Fitness = average reward over N simulation episodes using the chromosome's fuzzy params.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from .fuzzy_controller import FuzzyController
from .simulator import TrafficSimulator


@dataclass
class Individual:
    genes: list[float]
    fitness: float = float("-inf")


@dataclass
class GAConfig:
    population_size: int = 50
    generations: int = 30
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    mutation_sigma: float = 2.0      # std dev for Gaussian mutation
    tournament_k: int = 3
    elitism_ratio: float = 0.1
    blx_alpha: float = 0.5          # BLX-alpha crossover parameter
    eval_episodes: int = 5           # simulation episodes per fitness eval
    eval_cycles: int = 50            # cycles per episode
    seed: int | None = None


class GeneticOptimizer:
    """
    GA optimizer for fuzzy MF parameters.

    Usage:
        ga = GeneticOptimizer(config)
        best = ga.run(progress_callback)
        fuzzy_controller.from_chromosome(best.genes)
    """

    def __init__(self, config: GAConfig | None = None) -> None:
        self.config = config or GAConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._population: list[Individual] = []
        self._best: Individual | None = None
        self._history: list[dict] = []
        self._running = False
        self._generation = 0

    @property
    def best(self) -> Individual | None:
        return self._best

    @property
    def history(self) -> list[dict]:
        return self._history

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False

    # ── Core GA ──────────────────────────────────────────────────────────────

    def run(
        self,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> Individual:
        """
        Run the GA optimization.

        progress_callback(generation, total_generations, best_fitness)
        Returns the best individual found.
        """
        cfg = self.config
        self._running = True
        self._history.clear()

        # Initialize population from default + random perturbations
        self._init_population()

        # Evaluate initial population
        self._evaluate_all()

        for gen in range(cfg.generations):
            if not self._running:
                break

            self._generation = gen + 1

            # Selection + Crossover + Mutation
            new_pop = self._evolve()

            # Elitism: keep top individuals
            n_elite = max(1, int(cfg.population_size * cfg.elitism_ratio))
            sorted_pop = sorted(self._population, key=lambda x: x.fitness, reverse=True)
            elites = [Individual(genes=list(ind.genes), fitness=ind.fitness) for ind in sorted_pop[:n_elite]]

            self._population = elites + new_pop[:cfg.population_size - n_elite]

            # Evaluate new individuals (elites keep their fitness)
            for ind in self._population[n_elite:]:
                ind.fitness = self._evaluate(ind.genes)

            # Track best
            gen_best = max(self._population, key=lambda x: x.fitness)
            if self._best is None or gen_best.fitness > self._best.fitness:
                self._best = Individual(genes=list(gen_best.genes), fitness=gen_best.fitness)

            avg_fit = np.mean([ind.fitness for ind in self._population])
            self._history.append({
                "generation": self._generation,
                "best_fitness": gen_best.fitness,
                "avg_fitness": float(avg_fit),
                "global_best": self._best.fitness,
            })

            if progress_callback:
                progress_callback(self._generation, cfg.generations, self._best.fitness)

        self._running = False
        return self._best or Individual(genes=FuzzyController().to_chromosome())

    def _init_population(self) -> None:
        """Create initial population: default params + random variations."""
        cfg = self.config
        default_genes = FuzzyController().to_chromosome()
        n_genes = len(default_genes)

        self._population = [Individual(genes=list(default_genes))]  # seed with default

        for _ in range(cfg.population_size - 1):
            noise = self._rng.normal(0, cfg.mutation_sigma * 2, n_genes)
            genes = [max(0, g + n) for g, n in zip(default_genes, noise)]
            # Enforce a <= b <= c within each triplet
            genes = self._enforce_ordering(genes)
            self._population.append(Individual(genes=genes))

    def _evaluate_all(self) -> None:
        for ind in self._population:
            ind.fitness = self._evaluate(ind.genes)

        best = max(self._population, key=lambda x: x.fitness)
        self._best = Individual(genes=list(best.genes), fitness=best.fitness)

    def _evaluate(self, genes: list[float]) -> float:
        """Evaluate fitness by running simulation episodes with these fuzzy params."""
        ctrl = FuzzyController()
        ctrl.from_chromosome(genes)
        sim = TrafficSimulator()

        total_reward = 0.0
        for ep in range(self.config.eval_episodes):
            def policy(obs: dict) -> float:
                return ctrl.decide(
                    queue_length=obs["total_queue"],
                    waiting_time=obs["avg_wait"],
                )

            result = sim.run_episode(
                policy_fn=policy,
                max_cycles=self.config.eval_cycles,
                seed=ep * 1000 + (self.config.seed or 0),
            )
            total_reward += result["avg_reward"]

        return total_reward / self.config.eval_episodes

    def _evolve(self) -> list[Individual]:
        """Create next generation via selection, crossover, mutation."""
        cfg = self.config
        offspring: list[Individual] = []

        while len(offspring) < cfg.population_size:
            # Tournament selection
            p1 = self._tournament_select()
            p2 = self._tournament_select()

            # Crossover
            if self._rng.random() < cfg.crossover_rate:
                c1_genes, c2_genes = self._blx_crossover(p1.genes, p2.genes)
            else:
                c1_genes, c2_genes = list(p1.genes), list(p2.genes)

            # Mutation
            c1_genes = self._mutate(c1_genes)
            c2_genes = self._mutate(c2_genes)

            offspring.append(Individual(genes=c1_genes))
            offspring.append(Individual(genes=c2_genes))

        return offspring

    def _tournament_select(self) -> Individual:
        """Tournament selection with k competitors."""
        candidates = self._rng.choice(
            len(self._population),
            size=min(self.config.tournament_k, len(self._population)),
            replace=False,
        )
        best = max(candidates, key=lambda i: self._population[i].fitness)
        return self._population[best]

    def _blx_crossover(
        self, p1: list[float], p2: list[float]
    ) -> tuple[list[float], list[float]]:
        """BLX-alpha crossover: offspring genes sampled within expanded range."""
        alpha = self.config.blx_alpha
        c1, c2 = [], []
        for g1, g2 in zip(p1, p2):
            lo, hi = min(g1, g2), max(g1, g2)
            span = hi - lo
            new_lo = lo - alpha * span
            new_hi = hi + alpha * span
            c1.append(max(0, self._rng.uniform(new_lo, new_hi)))
            c2.append(max(0, self._rng.uniform(new_lo, new_hi)))
        return self._enforce_ordering(c1), self._enforce_ordering(c2)

    def _mutate(self, genes: list[float]) -> list[float]:
        """Gaussian mutation on random genes."""
        result = list(genes)
        for i in range(len(result)):
            if self._rng.random() < self.config.mutation_rate:
                result[i] = max(0, result[i] + self._rng.normal(0, self.config.mutation_sigma))
        return self._enforce_ordering(result)

    @staticmethod
    def _enforce_ordering(genes: list[float]) -> list[float]:
        """Ensure a <= b <= c for every (a, b, c) triplet."""
        result = list(genes)
        for i in range(0, len(result) - 2, 3):
            a, b, c = result[i], result[i + 1], result[i + 2]
            result[i] = min(a, b, c)
            result[i + 1] = sorted([a, b, c])[1]
            result[i + 2] = max(a, b, c)
        return result
