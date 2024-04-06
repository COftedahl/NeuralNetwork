public class DumbNN extends NeuralNetwork{
    private int numOuts;
    public DumbNN(int outs) {}
    public double[] compute() {
        return compute(0, 1);
    }
    public double[] compute(double minOut, double maxOut) {
        double[] results = new double[numOuts];
        for (double res : results) {
            res = ((Math.random() * (maxOut - minOut)) + minOut);
        }
        return results;
    }
}
