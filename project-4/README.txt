# Machine Learning - Assignment 4
Anish Moorthy's code/results/analysis for CS4641 Project 4 (Markov Decision Processes)

I wrote very little actual code for this project, having forked most of it from https://github.com/juanjose49/omscs-cs7641-machine-learning-assignment-4

## Downloading & compiling my code
To download my code, simply perform a

```
git clone https://github.com/xanlsh/cs4641-project4.git
```

We use the maven build system to compile. `cd` into `cs4641-project4` and run the following commands

```
mvn install

mvn clean compile assembly:single
```

## Running the Grid World: Low Difficulty Analysis:

From the project root, simply run the following command

```
java -cp target/cs7641-assignment-4-1.0-jar-with-dependencies.jar assignment4.EasyGridWorldLauncher
```

## Running the Grid World: High Difficulty Analysis:
From the project root, simply run the following command

```
java -cp target/cs7641-assignment-4-1.0-jar-with-dependencies.jar assignment4.HardGridWorldLauncher
```
