public class InnerLayer {
    public enum MappingConfig {
        RELU,
        SIGMOID;
    }
    private double[] bias;
    private double[][] weights;
    private int numNodes;
    private MappingConfig mappingConfig;
    public InnerLayer(int prevLayer, int thisLayer) {
        numNodes = thisLayer;
        bias = new double[numNodes];
        weights = new double[numNodes][prevLayer];
        mappingConfig = MappingConfig.RELU;
        setRandomWeights();
        setRandomBiases();
    }
    public InnerLayer(int prevLayer, int thisLayer, MappingConfig map) {
        numNodes = thisLayer;
        bias = new double[numNodes];
        weights = new double[numNodes][prevLayer];
        mappingConfig = map;
        setRandomWeights();
        setRandomBiases();
    }

    public double[] getBias() {
        return bias;
    }
    public void setBias(double[] bias) {
        if (bias.length == numNodes) {
            this.bias = bias;
        }
        else {
            throw new IllegalArgumentException();
        }
    }

    public double[][] getWeights() {
        return weights;
    }
    public void setWeights(double[][] weights) {
        if ((weights.length == numNodes)) {
            this.weights = weights;
        }
        else {
            throw new IllegalArgumentException();
        }
    }

    public int getNumNodes() {
        return numNodes;
    }
    public void setNumNodes(int numNodes) {
        this.numNodes = numNodes;
    }

    public MappingConfig getMappingConfig() {
        return mappingConfig;
    }
    public void setMappingConfig(MappingConfig mappingConfig) {
        this.mappingConfig = mappingConfig;
    }

    public void setRandomWeights() {
        for (int i = 0; i < weights.length; i += 1) {
            for (int j = 0; j < weights[i].length; j += 1) {
                weights[i][j] = (Math.random() * 2) - 1;
            }
        }
    }
    public void setRandomBiases() {
        for (int i = 0; i < bias.length; i += 1) {
            bias[i] = (Math.random() *  2) - 1;
        }
    }

    public double[] activate(double[] inputs) {
        if (inputs.length != weights[0].length) {
            return null;
        }
        double[] results = new double[numNodes];
        double sum = 0;
        for (int i = 0; i < numNodes; i += 1) {
            sum = 0;
            for (int j = 0; j < weights[i].length; j += 1) {
                sum += inputs[j] * weights[i][j];
            }
            sum += bias[i];
            results[i] = sum;
        }
        results = mapInputs(results, mappingConfig);
        return results;
    }
    public static double[] mapInputs(final double[] nums, MappingConfig map) {
        double[] mapped = new double[nums.length];
        if (map == MappingConfig.RELU) {
            double maxPositive = 0;
            for (int i = 0; i < mapped.length; i += 1) {
                if (nums[i] > maxPositive) {
                    maxPositive = nums[i];
                }
            }
            if (maxPositive > 0) {
                for (int i = 0; i < nums.length; i += 1) {
                    if (nums[i] > 0) {
                        mapped[i] = nums[i]/maxPositive;
                    }
                    else {
                        mapped[i] = 0;
                    }
                }
            }
            else {
                for (int i = 0; i < nums.length; i += 1) {
                    mapped[i] = 0;
                }
            }
        }
        else {
            double maxAbsVal = 0;
            for (int i = 0; i < nums.length; i += 1) {
                if (Math.abs(nums[i]) > maxAbsVal) {
                    maxAbsVal = Math.abs(nums[i]);
                }
            }
            for (int i = 0; i < nums.length; i += 1) {
                mapped[i] = (nums[i]/maxAbsVal) + .5;
            }
        }
        return mapped;
    }
}
