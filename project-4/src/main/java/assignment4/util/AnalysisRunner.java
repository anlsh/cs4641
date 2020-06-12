package assignment4.util;

import assignment4.BasicGridWorld;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.statehashing.HashableStateFactory;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;

import java.util.List;

public class AnalysisRunner {

	final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

	private int MAX_ITERATIONS;
	private int NUM_INTERVALS;

    final private int NUM_TO_AVERAGE = 1000;

	public AnalysisRunner(int MAX_ITERATIONS, int NUM_INTERVALS){
		this.MAX_ITERATIONS = MAX_ITERATIONS;
		this.NUM_INTERVALS = NUM_INTERVALS;

		double increment = (int)(MAX_ITERATIONS/NUM_INTERVALS);
		for(double numIterations = increment;
        numIterations<=MAX_ITERATIONS;
        numIterations+=increment ){
			AnalysisAggregator.addNumberOfIterations(numIterations);

		}

	}
	public void runValueIteration(BasicGridWorld gen, Domain domain,
			State initialState, RewardFunction rf, TerminalFunction tf, boolean showPolicyMap) {
		System.out.println("//Value Iteration Analysis//");
		ValueIteration vi = null;
		Policy p = null;
		EpisodeAnalysis ea = null;
		int increment = MAX_ITERATIONS/NUM_INTERVALS;
		for(int numIterations = increment;numIterations<=MAX_ITERATIONS;numIterations+=increment ){
			long startTime = System.nanoTime();
			vi = new ValueIteration(
					domain,
					rf,
					tf,
					0.99,
					hashingFactory,
					-1, numIterations); //Added a very high delta number in order to guarantee that value iteration occurs the max number of iterations
										   //for comparison with the other algorithms.

			// run planning from our initial state
			p = vi.planFromState(initialState);
			AnalysisAggregator.addMillisecondsToFinishValueIteration((double)((int) (System.nanoTime()-startTime)/1000000));

			// evaluate the policy with one roll out visualize the trajectory
      double reward = 0;
      double steps = 0;
      for (int i = 0; i < NUM_TO_AVERAGE; ++i) {
          ea = p.evaluateBehavior(initialState, rf, tf);
          reward += calcRewardInEpisode(ea);
          steps += ea.numTimeSteps();
      }
      reward /= NUM_TO_AVERAGE;
      steps /= NUM_TO_AVERAGE;

			AnalysisAggregator.addValueIterationReward(reward);
			AnalysisAggregator.addStepsToFinishValueIteration(steps);
		}

//		Visualizer v = gen.getVisualizer();
//		new EpisodeSequenceVisualizer(v, domain, Arrays.asList(ea));
		AnalysisAggregator.printValueIterationResults();
		MapPrinter.printPolicyMap(vi.getAllStates(), p, gen.getMap());
		System.out.println("\n\n");
		if(showPolicyMap){
			simpleValueFunctionVis((ValueFunction)vi, p, initialState, domain, hashingFactory, "Value Iteration");
		}
	}

	public void runPolicyIteration(BasicGridWorld gen, Domain domain,
			State initialState, RewardFunction rf, TerminalFunction tf, boolean showPolicyMap) {
		System.out.println("//Policy Iteration Analysis//");
		PolicyIteration pi = null;
		Policy p = null;
		EpisodeAnalysis ea = null;
		int increment = MAX_ITERATIONS/NUM_INTERVALS;
		for(int numIterations = increment;
        numIterations<=MAX_ITERATIONS;
        numIterations+=increment ){

			long startTime = System.nanoTime();
			pi = new PolicyIteration(
					domain,
					rf,
					tf,
					0.99,
					hashingFactory,
					-1, 1, numIterations);

			// run planning from our initial state
			p = pi.planFromState(initialState);
			AnalysisAggregator.addMillisecondsToFinishPolicyIteration((double)((int) (System.nanoTime()-startTime)/1000000));

			// evaluate the policy with one roll out visualize the trajectory
      double reward = 0;
      double steps = 0;
      for (int i = 0; i < NUM_TO_AVERAGE; ++i) {
          ea = p.evaluateBehavior(initialState, rf, tf);
          reward += calcRewardInEpisode(ea);
          steps += ea.numTimeSteps();
      }
      reward /= NUM_TO_AVERAGE;
      steps /= NUM_TO_AVERAGE;
			AnalysisAggregator.addPolicyIterationReward(reward);
			AnalysisAggregator.addStepsToFinishPolicyIteration(steps);
		}

//		Visualizer v = gen.getVisualizer();
//		new EpisodeSequenceVisualizer(v, domain, Arrays.asList(ea));
		AnalysisAggregator.printPolicyIterationResults();

		MapPrinter.printPolicyMap(getAllStates(domain,rf,tf,initialState), p, gen.getMap());
		System.out.println("\n\n");

		//visualize the value function and policy.
		if(showPolicyMap){
			simpleValueFunctionVis(pi, p, initialState, domain, hashingFactory, "Policy Iteration");
		}
	}

	public void simpleValueFunctionVis(ValueFunction valueFunction, Policy p,
			State initialState, Domain domain, HashableStateFactory hashingFactory, String title){

		List<State> allStates = StateReachability.getReachableStates(initialState,
				(SADomain)domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
				allStates, valueFunction, p);
		gui.setTitle(title);
		gui.initGUI();

	}

	public void runQLearning(BasicGridWorld gen, Domain domain,
			State initialState, RewardFunction rf, TerminalFunction tf,
			SimulatedEnvironment env, boolean showPolicyMap) {
		System.out.println("//Q Learning Analysis//");

		QLearning agent = null;
		Policy p = null;
		EpisodeAnalysis ea = null;
		int increment = MAX_ITERATIONS/NUM_INTERVALS;
		for(int numIterations = increment;numIterations<=MAX_ITERATIONS;numIterations+=increment ){
			long startTime = System.nanoTime();

			agent = new QLearning(
				domain,
				0.99,
				hashingFactory,
				0.99, 0.99);

			for (int i = 0; i < numIterations; i++) {
				ea = agent.runLearningEpisode(env);
				env.resetEnvironment();
			}
			AnalysisAggregator.addMillisecondsToFinishQLearning((double)((int) (System.nanoTime()-startTime)/1000000));


			agent.initializeForPlanning(rf, tf, 1);
			p = agent.planFromState(initialState);
      double reward = 0;
      double steps = 0;
      for (int i = 0; i < NUM_TO_AVERAGE; ++i) {
          ea = p.evaluateBehavior(initialState, rf, tf);
          reward += calcRewardInEpisode(ea);
          steps += ea.numTimeSteps();
      }
      reward /= NUM_TO_AVERAGE;
      steps /= NUM_TO_AVERAGE;
			AnalysisAggregator.addQLearningReward(reward);
			AnalysisAggregator.addStepsToFinishQLearning(steps);

      // ##############
      // # My section #
      // ##############

			// run planning from our initial state
			// p = agent.planFromState(initialState);
			// AnalysisAggregator.addMillisecondsToFinishPolicyIteration((int) (System.nanoTime()-startTime)/1000000);

			// evaluate the policy with one roll out visualize the trajectory
      // double reward = 0;
      // double steps = 0;
      // for (int i = 0; i < NUM_TO_AVERAGE; ++i) {
      //     ea = p.evaluateBehavior(initialState, rf, tf);
      //     reward += calcRewardInEpisode(ea);
      //     steps += ea.numTimeSteps();
      // }
      // reward /= NUM_TO_AVERAGE;
      // steps /= NUM_TO_AVERAGE;
			// AnalysisAggregator.addPolicyIterationReward(reward);
			// AnalysisAggregator.addStepsToFinishPolicyIteration(steps);

		}
		AnalysisAggregator.printQLearningResults();
		MapPrinter.printPolicyMap(getAllStates(domain,rf,tf,initialState), p, gen.getMap());
		System.out.println("\n\n");

		//visualize the value function and policy.
		if(showPolicyMap){
			simpleValueFunctionVis((ValueFunction)agent, p, initialState, domain, hashingFactory, "Q-Learning");
		}

	}

	private static List<State> getAllStates(Domain domain,
			 RewardFunction rf, TerminalFunction tf,State initialState){
		ValueIteration vi = new ValueIteration(
				domain,
				rf,
				tf,
				0.99,
				new SimpleHashableStateFactory(),
				.5, 100);
		vi.planFromState(initialState);

		return vi.getAllStates();
	}

	public double calcRewardInEpisode(EpisodeAnalysis ea) {
		double myRewards = 0;

		//sum all rewards
		for (int i = 0; i<ea.rewardSequence.size(); i++) {
			myRewards += ea.rewardSequence.get(i);
		}
		return myRewards;
	}

}
