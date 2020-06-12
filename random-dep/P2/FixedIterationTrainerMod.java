// package shared;
import shared.Trainer;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

/**
 * A fixed iteration trainer
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FixedIterationTrainerMod {

    /**
     * The inner trainer
     */
    private Trainer trainer;

    /**
     * The number of iterations to train
     */
    private int iterations;

    /**
     * Make a new fixed iterations trainer
     * @param t the trainer
     * @param iter the number of iterations
     */
    public FixedIterationTrainerMod(Trainer t, int iter) {
        trainer = t;
        iterations = iter;
    }

    /**
     * @see shared.Trainer#train()
     */
    public double train(String run_name, String outfilename, int print_interval) {
        double starttime = System.nanoTime();
        double sum = 0;

        try (BufferedWriter out = new BufferedWriter(
                new FileWriter(outfilename, true))) {
            out.write(run_name + "\n");
            for (int i = 0; i < iterations; i++) {
                double fitness = trainer.train();
                String outstring = "Iteration: " + String.valueOf(i)
                            + ", Wall-Clock: " + String.valueOf((System.nanoTime() - starttime) / 60)
                            + ", Fitness: " + String.valueOf(fitness)
                            + "\n";

                out.write(outstring);
                out.close();

                if (print_interval > 0 && i % print_interval == 0) {
                    System.out.println(outstring);
                }
                sum += fitness;
            }
        }
        catch (IOException e) {
            System.out.println("exception occoured" + e);
        }
        return sum / iterations;
    }


}
