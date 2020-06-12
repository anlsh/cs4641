import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
// import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    private static final int N = 50;
    static final int MAX_ITER = 200000;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {

        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();
        }
        /* TWEAK PARAMETERS HERE TO CHANGE PROBLEM */
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        /* END PROBLEM TWEAKABLE PARAMETERS */

        final String outfile = args[1];
        final int log_int = Integer.valueOf(args[2]);
        final boolean run_rhc = Boolean.valueOf(args[3]);
        final boolean run_sa = Boolean.valueOf(args[4]);
        final boolean run_ga = Boolean.valueOf(args[5]);
        final boolean run_mimic = Boolean.valueOf(args[6]);

        FixedIterationTrainerMod fit;
        double start_time;
        double end_time;
        double elapsed_seconds;
        String algo_name;

        // for rhc, sa, and ga we use a permutation based encoding
        Distribution odd = new DiscretePermutationDistribution(N);

        /* RANDOMIZED HILL CLIMBING CODE */
        if (run_rhc) {

            // RANDOMIZED HILL CLIMBING PARAMETERS
            NeighborFunction neighbor_func = new SwapNeighbor();
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, neighbor_func);
            // END RANDOMIZED HILL CLIMBING PARAMETERS

            algo_name = "Randomized Hill Climbing";
            start_time = System.currentTimeMillis();
            RandomizedHillClimbing problem = new RandomizedHillClimbing(hcp);
            fit = new FixedIterationTrainerMod(problem, MAX_ITER);
            end_time = System.currentTimeMillis();

            String run_description = "* " + algo_name + " (max_iter: " + MAX_ITER + ")";
            System.out.println("Starting " + run_description);
            fit.train(run_description, outfile, log_int);

            elapsed_seconds = (end_time - start_time) / 1000;
            System.out.println(algo_name + " Optimal Value: " + ef.value(problem.getOptimal())
                    +  " in " + elapsed_seconds + "s");
        }

        if (run_sa) {
            // SIMULATED ANNEALING PARAMETERS
            NeighborFunction neighbor_func = new SwapNeighbor();
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, neighbor_func);
            // SIMULATED ANNEALING PARAMETERS

            algo_name = "Simulated Annealing";
            double start_temp = 1E12;
            double cooling_rate = .95;
            start_time = System.currentTimeMillis();
            SimulatedAnnealing problem = new SimulatedAnnealing(start_temp, cooling_rate, hcp);
            fit = new FixedIterationTrainerMod(problem, MAX_ITER);
            end_time = System.currentTimeMillis();

            String run_description = "* " + algo_name
                    + " (max_iter: " + MAX_ITER
                    + ", StartTemp: " + start_temp
                    + ", CoolingRate: " + cooling_rate
                    + ")";
            System.out.println("Starting " + run_description);
            fit.train(run_description, outfile, log_int);

            elapsed_seconds = (end_time - start_time) / 1000;
            System.out.println(algo_name + " Optimal Value: " + ef.value(problem.getOptimal())
                    +  " in " + elapsed_seconds + "s");
        }

        if (run_ga) {
            algo_name = "Genetic Algorithm";

            /* GENETIC ALGORITHM PARAMETERS */
            int population_size = 250;
            int to_mate = 150;
            int to_mutate = 20;
            CrossoverFunction crossover_func = new TravelingSalesmanCrossOver(ef);
            MutationFunction mutation_func = new SwapMutation();
            // END GENETIC ALGORITHM PARAMETERS

            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mutation_func, crossover_func);

            start_time = System.currentTimeMillis();
            StandardGeneticAlgorithm problem = new StandardGeneticAlgorithm(population_size, to_mate, to_mutate, gap);
            fit = new FixedIterationTrainerMod(problem, MAX_ITER);
            end_time = System.currentTimeMillis();

            String run_description = "* " + algo_name
                    + " (max_iter: " + MAX_ITER
                    + ", PopSize: " + population_size
                    + ", ToMate: " + to_mate
                    + ", ToMutate: " + to_mutate
                    + ")";
            System.out.println("Starting " + run_description);
            fit.train(run_description, outfile, log_int);

            elapsed_seconds = (end_time - start_time) / 1000;
            System.out.println(algo_name + " Optimal Value: " + ef.value(problem.getOptimal())
                    +  " in " + elapsed_seconds + "s");
        }

        // TODO Why the fuck does it swap stuff up?
        // for mimic we use a sort encoding
        ef = new TravelingSalesmanSortEvaluationFunction(points);
        int[] ranges = new int[N];
        Arrays.fill(ranges, N);
        odd = new  DiscreteUniformDistribution(ranges);

        if (run_mimic) {
            algo_name = "MIMIC";
            int num_samples = 200;
            int tokeep = 100;

            double discrete_dep = .1;

            start_time = System.currentTimeMillis();
            Distribution df = new DiscreteDependencyTree(discrete_dep, ranges);
            MIMIC problem = new MIMIC(num_samples, tokeep,
                    new GenericProbabilisticOptimizationProblem(ef, odd, df));
            fit = new FixedIterationTrainerMod(problem, MAX_ITER);
            end_time = System.currentTimeMillis();

            String run_description = "* " + algo_name
                    + " (max_iter: " + MAX_ITER
                    + "), (DiscreteDep: " + discrete_dep
                    + "), (NumSamples: " + num_samples
                    + "), (ToKeep: " + tokeep
                    + ")";
            System.out.println("Starting " + run_description);
            fit.train(run_description, outfile, log_int);

            elapsed_seconds = (end_time - start_time) / 1000;
            System.out.println(algo_name + " Optimal Value: " + ef.value(problem.getOptimal())
                    +  " in " + elapsed_seconds + "s");
        }

    }
}
