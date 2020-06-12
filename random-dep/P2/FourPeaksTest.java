import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
// import FixedIterationTrainerMod;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The number of total points */
    private static final int N = 200;
    /** The t value */
    private static final int T = N / 5;
    static final int MAX_ITER = 200000;
    
    public static void main(String[] args) {

        /* TWEAK THESE PARAMETERS TO CHANGE THE PROBLEM */
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        /* END PROBLEM TWEAKABLE PARAMETERS */

        final String outfile = args[1];
        final int log_int = Integer.valueOf(args[2]);
        final boolean run_rhc = Boolean.valueOf(args[3]);
        final boolean run_sa = Boolean.valueOf(args[4]);
        final boolean run_ga = Boolean.valueOf(args[5]);
        final boolean run_mimic = Boolean.valueOf(args[6]);

        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        Distribution odd = new DiscreteUniformDistribution(ranges);

        FixedIterationTrainerMod fit;
        double start_time;
        double end_time;
        double elapsed_seconds;
        String algo_name;

        /* RANDOMIZED HILL CLIMBING CODE */
        if (run_rhc) {

            // PARAMETERS
            NeighborFunction neighbor_func = new DiscreteChangeOneNeighbor(ranges);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, neighbor_func);
            // END PARAMETERS

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

            // PARAMETERS
            NeighborFunction neighbor_func = new DiscreteChangeOneNeighbor(ranges);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, neighbor_func);
            // END PARAMETERS

            algo_name = "Simulated Annealing";
            double start_temp = 1E11;
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

            // PARAMETERS
            MutationFunction mutation_func = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction crossover_func = new SingleCrossOver();
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mutation_func, crossover_func);
            // END PARAMETERS

            algo_name = "Genetic Algorithm";
            int population_size = 200;
            int to_mate = 100;
            int to_mutate = 10;

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

        if (run_mimic) {
            algo_name = "MIMIC";
            int num_samples = 200;
            int tokeep = 20;
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
