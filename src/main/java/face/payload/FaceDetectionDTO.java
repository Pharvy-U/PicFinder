package face.payload;

import java.util.Arrays;

public class FaceDetectionDTO {

    private float[][][] boxes;

    private float[][] scores;

    private float[] numDetections;

    public FaceDetectionDTO() {
    }

    public FaceDetectionDTO(float[][][] boxes, float[][] scores, float[] numDetections) {
        this.boxes = boxes;
        this.scores = scores;
        this.numDetections = numDetections;
    }

    public float[][][] getBoxes() {
        return boxes;
    }

    public void setBoxes(float[][][] boxes) {
        this.boxes = boxes;
    }

    public float[][] getScores() {
        return scores;
    }

    public void setScores(float[][] scores) {
        this.scores = scores;
    }

    public float[] getNumDetections() {
        return numDetections;
    }

    public void setNumDetections(float[] numDetections) {
        this.numDetections = numDetections;
    }

    @Override
    public String toString() {
        return "FaceDetectDTO{" +
                "numDetections=" + Arrays.toString(numDetections) +
                '}' + ": Note that confidence scores have not been put into consideration here";
    }
}
