import java.util.ArrayList;

public class NNAI {
    private ScoredNeuralNetwork[] nns;
    private int numGamesPerGen;
    private int numNNs;
    private double chanceToMutate;
    private double mutationAmount;
    private double chanceToExplore;//as opposed to optimize - uses NN or just random
    private double explorationDecayRate;
    private ScoringAlgorithm scorer;
    private int currGen;
    private DumbNN dumbNet;
    private int numInputs;
    private int numOutputs;

    public NNAI(int numGamesPerGen, int numNNs, double chanceToMutate, double mutationAmount, double chanceToExplore, double explorationDecayRate, ScoringAlgorithm scorer, int... layers) {
        nns = new ScoredNeuralNetwork[numNNs];
        for (int i = 0; i < numNNs; i += 1) {
            nns[i] = new ScoredNeuralNetwork(scorer, layers);
        }
        this.numGamesPerGen = numGamesPerGen;
        this.numNNs = numNNs;
        this.chanceToMutate = chanceToMutate;
        this.mutationAmount = mutationAmount;
        this.chanceToExplore = chanceToExplore;
        this.explorationDecayRate = explorationDecayRate;
        this.scorer = scorer;
        currGen = 0;
        numInputs = layers[0];
        numOutputs = layers[layers.length - 1];
        dumbNet = new DumbNN(numOutputs);
    }

    public ScoredNeuralNetwork[] getNNs() {
        return nns;
    }
    public void setNNs(ScoredNeuralNetwork[] nns) {
        this.nns = nns;
    }
    public int getNumGamesPerGen() {
        return numGamesPerGen;
    }
    public void setNumGamesPerGen(int numGamesPerGen) {
        this.numGamesPerGen = numGamesPerGen;
    }
    public int getNumNNs() {
        return numNNs;
    }
    public void setNumNNs(int numNNs) {
        this.numNNs = numNNs;
    }
    public double getChanceToMutate() {
        return chanceToMutate;
    }
    public void setChanceToMutate(double chanceToMutate) {
        this.chanceToMutate = chanceToMutate;
    }
    public double getChanceToExplore() {
        return chanceToExplore;
    }
    public void setChanceToExplore(double chanceToExplore) {
        this.chanceToExplore = chanceToExplore;
    }
    public double getExplorationDecayRate() {
        return explorationDecayRate;
    }
    public void setExplorationDecayRate(double explorationDecayRate) {
        this.explorationDecayRate = explorationDecayRate;
    }
    public ScoringAlgorithm getScorer() {
        return scorer;
    }
    public void setScorer(ScoringAlgorithm scorer) {
        this.scorer = scorer;
    }
    public int getCurrGen() {
        return currGen;
    }
    public void setCurrGen(int currGen) {
        this.currGen = currGen;
    }
    public DumbNN getDumbNet() {
        return dumbNet;
    }
    public void setDumbNet(DumbNN dumbNet) {
        this.dumbNet = dumbNet;
    }
    public int getNumInputs() {
        return numInputs;
    }
    public void setNumInputs(int numInputs) {
        this.numInputs = numInputs;
    }
    public int getNumOutputs() {
        return numOutputs;
    }
    public void setNumOutputs(int numOutputs) {
        this.numOutputs = numOutputs;
    }

    public void testGeneration(Game g) {
        for (ScoredNeuralNetwork scoredNet : nns) {
            for (int i = 0; i < numGamesPerGen; i += 1) {
                if ((currGen == 0) || (Math.random() < chanceToExplore)) {
                    scoredNet.saveResults(g.play(dumbNet));
                }
                else {
                    scoredNet.saveResults(g.play(scoredNet.getNN()));
                }
            }
        }
        double sumScores = 0;
        int numGames = 0;
        for (ScoredNeuralNetwork scoredNet : nns) {
            sumScores = 0;
            numGames = 0;
            for (GameResults result : scoredNet.getResults()) {
                numGames += 1;
                sumScores += scoredNet.getScoringAlgorithm().scoreGame(result);
            }
            scoredNet.setAvgScore(sumScores/numGames);
        }
        sortNets();
    }

    public void evolveGeneration() {
        ScoredNeuralNetwork[] newNetArray = new ScoredNeuralNetwork[nns.length];
        for (int currChild = 0; currChild < newNetArray.length; currChild += 1) {

            int lowestNNIndex = (int) (numNNs * .9) - 1;
            int highestNNIndex = numNNs - 1;
            int rangeNNs = highestNNIndex - lowestNNIndex;
            int indexOne = (int) (Math.random() * rangeNNs) + lowestNNIndex;
            int indexTwo = (int) (Math.random() * rangeNNs) + lowestNNIndex;
            if (!nns[indexOne].getNN().isComparable(nns[indexTwo].getNN())) {
                throw new IllegalStateException("Neural Networks " + indexOne +
                        " and " + indexTwo + " are not comparable");
            }
            ArrayList<Integer> nodes = new ArrayList<Integer>();
            nodes.add(nns[indexOne].getNN().getNumInputNodes());
            for (int i = 0; i < nns[indexOne].getNN().getNumLayers(); i += 1) {
                nodes.add(nns[indexOne].getNN().getInnerLayers()[i].getNumNodes());
            }
            int[] constructorNodes = new int[nodes.size()];
            for (int i = 0; i < constructorNodes.length; i += 1) {
                constructorNodes[i] = nodes.remove(0);
            }

            NeuralNetwork newNet = new NeuralNetwork(constructorNodes);
            for (int i = 0; i < nns[indexOne].getNN().getNumLayers(); i += 1) {
                double[][] weightsOne = nns[indexOne].getNN().getInnerLayers()[i].getWeights();
                double[][] weightsTwo = nns[indexTwo].getNN().getInnerLayers()[i].getWeights();
                double[] biasOne = nns[indexOne].getNN().getInnerLayers()[i].getBias();
                double[] biasTwo = nns[indexTwo].getNN().getInnerLayers()[i].getBias();
                double[][] newWeights = new double[weightsOne.length][weightsOne[0].length];
                double[] newBias = new double[biasOne.length];
                for (int j = 0; j < weightsOne.length; j += 1) {
                    for (int k = 0; k < weightsOne[i].length; k += 1) {
                        if (Math.random() > .5) {
                            newWeights[j][k] = weightsOne[j][k];
                        } else {
                            newWeights[j][k] = weightsTwo[j][k];
                        }
                        if (Math.random() <= chanceToMutate) {
                            if (Math.random() >= .5) {
                                newWeights[j][k] += mutationAmount;
                            } else {
                                newWeights[j][k] -= mutationAmount;
                            }
                        }
                    }
                }
                for (int j = 0; j < biasOne.length; j += 1) {
                    if (Math.random() > .5) {
                        newBias[j] = biasOne[j];
                    } else {
                        newBias[j] = biasTwo[j];
                    }
                    if (Math.random() <= chanceToMutate) {
                        if (Math.random() >= .5) {
                            newBias[j] += mutationAmount;
                        } else {
                            newBias[j] -= mutationAmount;
                        }
                    }
                }
                newNet.getInnerLayers()[i].setWeights(newWeights);
                newNet.getInnerLayers()[i].setBias(newBias);
            }
            ScoredNeuralNetwork newScoredNet = new ScoredNeuralNetwork(scorer);
            newScoredNet.setNN(newNet);
            newNetArray[currChild] = newScoredNet;
        }
        nns = newNetArray;
        chanceToExplore = chanceToExplore * (1 - explorationDecayRate);
        currGen += 1;
    }

    public void sortNets() {
        innerSortNets(0, nns.length - 1);
    }
    private void innerSortNets(int start, int end) {
        double firstScore = nns[start].getAvgScore();
        int midIndex = (start + end) / 2;
        double midScore = nns[midIndex].getAvgScore();
        double lastScore = nns[end].getAvgScore();
        ScoredNeuralNetwork tempNet;
        if (firstScore > midScore) {
            if (midScore > lastScore) {
                //lastScore < midScore < firstScore
                tempNet = nns[start];
                nns[start] = nns[end];
                nns[end] = tempNet;
            }
            else {
                if (firstScore > lastScore) {
                    //midScore < lastScore < firstScore
                    tempNet = nns[start];
                    nns[start] = nns[midIndex];
                    nns[midIndex] = nns[end];
                    nns[end] = tempNet;
                }
                else {
                    //midScore < firstScore < lastScore
                    tempNet = nns[start];
                    nns[start] = nns[midIndex];
                    nns[midIndex] = tempNet;
                }
            }
        }
        else {
            //firstScore < midScore
            if (lastScore > midScore) {
                //firstScore < midScore < lastScore
                //no rearranging necessary
            }
            else {
                if (firstScore > lastScore) {
                    //lastScore < firstScore < midScore
                    tempNet = nns[end];
                    nns[end] = nns[midIndex];
                    nns[midIndex] = nns[start];
                    nns[start] = tempNet;
                }
                else {
                    //firstScore < lastScore < midScore
                    tempNet = nns[midIndex];
                    nns[midIndex] = nns[end];
                    nns[end] = tempNet;
                }
            }
        }

        //now, nns[midIndex] is the middle of three

        if ((end - start) > 3) {
            //next, we partition based on middle value
            double partitionScore = nns[midIndex].getAvgScore();
            tempNet = null;
            int i = start;
            int j = end;
            while (i < j) {
                while (nns[i].getAvgScore() < partitionScore) {
                    i += 1;
                }
                while (nns[j].getAvgScore() > partitionScore) {
                    j -= 1;
                }
                if (i < j) {
                    tempNet = nns[i];
                    nns[i] = nns[j];
                    nns[j] = tempNet;
                    i += 1;
                    j -= 1;
                }
            }
            innerSortNets(start,j);
            innerSortNets(i,end);
        }
    }
}
