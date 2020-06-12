package assignment4.util;

import assignment4.BasicGridWorld;
import burlap.oomdp.core.objects.ObjectInstance;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;

public class ComplexRewardFunction implements RewardFunction {

    int[][] pos_spec;
    int default_reward;

    public ComplexRewardFunction(int default_reward, int[][] pos_spec) {
        /*
          RewardSpec should be an int[n][3] with the first thing being
          the X, the second thing being the y, and the third being the
          reward for that location
         */
        this.default_reward = default_reward;
        this.pos_spec = pos_spec;
    }

    @Override
    public double reward(State s, GroundedAction a, State sprime) {

        // get location of agent in next state
        ObjectInstance agent = sprime.getFirstObjectOfClass(BasicGridWorld.CLASSAGENT);
        int ax = agent.getIntValForAttribute(BasicGridWorld.ATTX);
        int ay = agent.getIntValForAttribute(BasicGridWorld.ATTY);

        // are they at goal location?
        for (int[] ps : this.pos_spec) {
            if (ax == ps[0] && ay == ps[1]) {
                return ((double) ps[2]);
            }
        }

        return ((double) this.default_reward);
    }

}
