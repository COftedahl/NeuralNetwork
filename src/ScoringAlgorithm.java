public abstract class ScoringAlgorithm {
    private double maxScore;
    private double minScore;

    public double getMaxScore() {
        return maxScore;
    }
    public void setMaxScore(double maxScore) {
        this.maxScore = maxScore;
    }
    public double getMinScore() {
        return minScore;
    }
    public void setMinScore(double minScore) {
        this.minScore = minScore;
    }
    public ScoringAlgorithm(int min, int max) {
        minScore = min;
        maxScore = max;
    }

    public abstract double scoreGame(GameResults results);
}
