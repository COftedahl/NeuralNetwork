public class NeuralNetwork {
    private InnerLayer[] innerLayers;
    private int numLayers;
    private int numInputNodes;
    public NeuralNetwork(int ... layers) {
        numLayers = layers.length;
        innerLayers = new InnerLayer[numLayers - 1];
        for (int i = 0; i < innerLayers.length; i += 1) {
            innerLayers[i] = new InnerLayer(layers[i],layers[i + 1]);
        }
        numInputNodes = layers[0];
    }
    public NeuralNetwork(InnerLayer.MappingConfig mappingConfig, int... layers) {
        numLayers = layers.length;
        innerLayers = new InnerLayer[numLayers - 1];
        for (int i = 0; i < innerLayers.length; i += 1) {
            innerLayers[i] = new InnerLayer(layers[i],layers[i + 1], mappingConfig);
        }
        numInputNodes = layers[0];
    }

    public InnerLayer[] getInnerLayers() {
        return innerLayers;
    }
    public void setInnerLayers(InnerLayer[] innerLayers) {
        this.innerLayers = innerLayers;
    }
    public int getNumLayers() {
        return numLayers;
    }
    public void setNumLayers(int numLayers) {
        this.numLayers = numLayers;
    }
    public int getNumInputNodes() {
        return numInputNodes;
    }
    public void setNumInputNodes(int numInputNodes) {
        this.numInputNodes = numInputNodes;
    }

    public boolean isComparable(NeuralNetwork otherNet) {
        boolean comparable = true;
        if ((numLayers != otherNet.numLayers) ||
            (numInputNodes != otherNet.numInputNodes) ||
            (innerLayers.length != otherNet.innerLayers.length))
        {
            comparable = false;
        }
        else {
            for (int i = 0; i < innerLayers.length; i += 1) {
                if ((innerLayers[i].getWeights().length !=
                        otherNet.innerLayers[i].getWeights().length) ||
                    (innerLayers[i].getWeights()[0].length !=
                        otherNet.innerLayers[i].getWeights()[0].length) ||
                    (innerLayers[i].getNumNodes() !=
                        otherNet.innerLayers[i].getNumNodes()))
                {
                    comparable = false;
                }
            }
        }
        return comparable;
    }

    public double[] compute(double[] inputs) {
        return compute(inputs, 0, 1);
    }

    public double[] compute(double[] inputs, double minOut, double maxOut) {
        if (inputs.length != numInputNodes) {
            throw new IllegalArgumentException();
        }
        double[] output = inputs;
        for (int i = 0; i < innerLayers.length; i += 1) {
            output = innerLayers[i].activate(output);
        }
        final double rangeOut = maxOut - minOut;
        for (int i = 0; i < output.length; i += 1) {
            output[i] = (output[i] * rangeOut) + minOut;
        }
        return output;
    }
}
