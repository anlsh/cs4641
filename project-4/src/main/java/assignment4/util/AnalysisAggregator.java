package assignment4.util;

import burlap.oomdp.core.values.DoubleArrayValue;

import java.util.ArrayList;
import java.util.List;

public final class AnalysisAggregator {
	private static List<Double> numIterations = new ArrayList<Double>();
	private static List<Double> stepsToFinishValueIteration = new ArrayList<Double>();
	private static List<Double> stepsToFinishPolicyIteration = new ArrayList<Double>();
	private static List<Double> stepsToFinishQLearning = new ArrayList<Double>();

	private static List<Double> millisecondsToFinishValueIteration = new ArrayList<Double>();
	private static List<Double> millisecondsToFinishPolicyIteration = new ArrayList<Double>();
	private static List<Double> millisecondsToFinishQLearning = new ArrayList<Double>();

	private static List<Double> rewardsForValueIteration = new ArrayList<Double>();
	private static List<Double> rewardsForPolicyIteration = new ArrayList<Double>();
	private static List<Double> rewardsForQLearning = new ArrayList<Double>();

	public static void addNumberOfIterations(Double numIterations1){
		numIterations.add(numIterations1);
	}
	public static void addStepsToFinishValueIteration(Double stepsToFinishValueIteration1){
		stepsToFinishValueIteration.add(stepsToFinishValueIteration1);
	}
	public static void addStepsToFinishPolicyIteration(Double stepsToFinishPolicyIteration1){
		stepsToFinishPolicyIteration.add(stepsToFinishPolicyIteration1);
	}
	public static void addStepsToFinishQLearning(Double stepsToFinishQLearning1){
		stepsToFinishQLearning.add(stepsToFinishQLearning1);
	}
	public static void printValueIterationResults(){
		printList(stepsToFinishValueIteration);
    System.out.println(",");
	}
	public static void printPolicyIterationResults(){
		printList(stepsToFinishPolicyIteration);
    System.out.println(",");
	}
	public static void printQLearningResults(){
		printList(stepsToFinishQLearning);
    System.out.println(",");
	}


	public static void addMillisecondsToFinishValueIteration(Double millisecondsToFinishValueIteration1){
		millisecondsToFinishValueIteration.add(millisecondsToFinishValueIteration1);
	}
	public static void addMillisecondsToFinishPolicyIteration(Double millisecondsToFinishPolicyIteration1){
		millisecondsToFinishPolicyIteration.add(millisecondsToFinishPolicyIteration1);
	}
	public static void addMillisecondsToFinishQLearning(Double millisecondsToFinishQLearning1){
		millisecondsToFinishQLearning.add(millisecondsToFinishQLearning1);
	}
	public static void addValueIterationReward(double reward) {
		rewardsForValueIteration.add(reward);
	}
	public static void addPolicyIterationReward(double reward) {
		rewardsForPolicyIteration.add(reward);
	}
	public static void addQLearningReward(double reward) {
		rewardsForQLearning.add(reward);
	}
	public static void printValueIterationTimeResults(){
		printList(millisecondsToFinishValueIteration);
    System.out.println(",");
    System.out.println("");
	}
	public static void printPolicyIterationTimeResults(){
		printList(millisecondsToFinishPolicyIteration);
    System.out.println(",");
    System.out.println("");
	}

	public static void printQLearningTimeResults(){
		printList(millisecondsToFinishQLearning);
    System.out.println(",");
    System.out.println("");
	}

	public static void printValueIterationRewards(){
		printDoubleList(rewardsForValueIteration);
    System.out.println(",");
    System.out.println("");
	}

	public static void printPolicyIterationRewards(){
		printDoubleList(rewardsForPolicyIteration);
    System.out.println(",");
    System.out.println("");
	}

	public static void printQLearningRewards(){
		printDoubleList(rewardsForQLearning);
    System.out.println(",");
    System.out.println("");
	}

	public static void printNumIterations(){
		printList(numIterations);
    System.out.println(",");
    System.out.println("");
	}
	private static void printList(List<Double> valueList){
		int counter = 0;
    System.out.print("[");
		for(double value : valueList){
			System.out.print(String.valueOf(value));
			if(counter != valueList.size()-1){
				System.out.print(",");
			}
			counter++;
		}
    System.out.print("]");
		System.out.println();
	}
	private static void printDoubleList(List<Double> valueList){
		int counter = 0;
    System.out.print("[");
		for(double value : valueList){
			System.out.print(String.valueOf(value));
			if(counter != valueList.size()-1){
				System.out.print(",");
			}
			counter++;
		}
    System.out.print("]");
		System.out.println();
	}
	public static void printAggregateAnalysis(){
		System.out.println("//Aggregate Analysis//\n");
		// System.out.println("The data below shows the number of steps/actions the agent required to reach \n"
		// 		+ "the terminal state given the number of iterations the algorithm was run.");
    System.out.println("Episode Length vs # Iterations");
    System.out.println("-------------------------------------------");

    System.out.println("~[");
		printNumIterations();
		printValueIterationResults();
		printPolicyIterationResults();
		printQLearningResults();
    System.out.println("]~");
		System.out.println();
		// System.out.println("The data below shows the number of milliseconds the algorithm required to generate \n"
		// 		+ "the optimal policy given the number of iterations the algorithm was run.");

    System.out.println("# of milliseconds to converge vs iteration");
    System.out.println("-------------------------------------------");

    System.out.println("~[");
		printNumIterations();
		printValueIterationTimeResults();
		printPolicyIterationTimeResults();
		printQLearningTimeResults();
    System.out.println("]~");

		// System.out.println("\nThe data below shows the total reward gained for \n"
		// 		+ "the optimal policy given the number of iterations the algorithm was run.");
    System.out.println("Cumulative Reward vs # Iterations");
    System.out.println("-------------------------------------------");

    System.out.println("~[");
		printNumIterations();
		printValueIterationRewards();
		printPolicyIterationRewards();
		printQLearningRewards();
    System.out.println("]~");
	}
}
