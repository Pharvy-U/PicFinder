package face.util;

import face.payload.ImageDTO;
import org.tensorflow.Tensor;
import org.tensorflow.op.core.TPUReplicatedInput;
import org.tensorflow.types.UInt8;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class ImageHandler {

    public static ImageDTO readImage(String imagePath) {
        BufferedImage image = null;
        BufferedImage imageRGB = null;
        Graphics2D g = null;
        try {
            image = ImageIO.read(new File(imagePath));
            imageRGB = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_RGB);
            g = imageRGB.createGraphics();
            g.drawImage(image, 0, 0, null);

        } catch (IOException e) {
            e.printStackTrace();
            return new ImageDTO(imageRGB, g);
        }
        return new ImageDTO(imageRGB, g);
    }

    public static ImageDTO cropImage(BufferedImage image, float x0,float y0, float x1, float y1) throws IOException {
        int width = (int) (image.getWidth() * (x1-x0));
        int height = (int) (image.getHeight() * (y1-y0));
        int x_pad = (int) (image.getWidth() * x0);
        int y_pad = (int) (image.getHeight() * y0);
        BufferedImage cropped = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for(int y=0; y< cropped.getHeight(); y++) {
            for(int x=0; x< cropped.getWidth(); x++) {
                cropped.setRGB(x,y,image.getRGB(x_pad+x, y_pad+y));
            }
        }

        Graphics2D g = cropped.createGraphics();
        return new ImageDTO(cropped, g);
    }
    // resize function
    public static BufferedImage resize (BufferedImage img, int newHeight, int newWidth) {
//        System.out.println("  Scaling image.");
        double hScaleFactor = (double) newHeight/img.getHeight();
        double wScaleFactor = (double) newWidth/img.getWidth();
        BufferedImage scaledImg = new BufferedImage(newWidth, newHeight, img.getType());
        AffineTransform at = new AffineTransform();
        at.scale(wScaleFactor, hScaleFactor);
        AffineTransformOp scaleOp = new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
        BufferedImage tt = scaleOp.filter(img, scaledImg);
//        display(tt);
        return tt;

    }

    public static void display(BufferedImage image) {
//        System.out.println("... Displaying Image...");
        image = resize(image, 1000, 1000);
        JFrame frame = new JFrame();
        JLabel label = new JLabel();
        frame.setSize(image.getWidth(), image.getHeight());
        label.setIcon(new ImageIcon(image));
        frame.getContentPane().add(label, BorderLayout.CENTER);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.pack();
        frame.setVisible(true);
    }

    public static Tensor<UInt8> convertImageToTensor(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();

        // Convert BufferedImage to ByteBuffer
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(3 * width * height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                byteBuffer.put((byte) ((rgb >> 16) & 0xFF));
                byteBuffer.put((byte) ((rgb >> 8) & 0xFF));
                byteBuffer.put((byte) (rgb & 0xFF));
            }
        }
        byteBuffer.rewind();
        return Tensor.create(UInt8.class, new long[]{1, height, width, 3}, byteBuffer);
    }

    public static Tensor<Float> convertImageToTensorFloat(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();

        // Convert BufferedImage to ByteBuffer
        FloatBuffer floatBuffer = FloatBuffer.allocate(3 * width * height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                floatBuffer.put(((rgb >> 16) & 0xFF));
                floatBuffer.put(((rgb >> 8) & 0xFF));
                floatBuffer.put((rgb & 0xFF));
            }
        }
        floatBuffer.rewind();

        // Create a Tensor from the ByteBuffer
        return Tensor.create(new long[]{1, height, width, 3}, floatBuffer);
    }

    public static Tensor<Float> convertImageToTensorFloatNorm(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();

        // Convert BufferedImage to ByteBuffer
        FloatBuffer floatBuffer = FloatBuffer.allocate(3 * width * height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                floatBuffer.put(((rgb >> 16) & 0xFF) / 255.0f);
                floatBuffer.put(((rgb >> 8) & 0xFF) / 255.0f);
                floatBuffer.put((rgb & 0xFF) / 255.0f);
            }
        }
        floatBuffer.rewind();

        // Create a Tensor from the ByteBuffer
        return Tensor.create(new long[]{1, height, width, 3}, floatBuffer);
    }

    /////

    // Method to slice a 4D array
    public static float[][][][] sliceArray(float[][][][] array, int startChannel, int endChannel) {
        // Get dimensions of the input array
        int batchSize = array.length;
        int height = array[0].length;
        int width = array[0][0].length;
        int numChannels = array[0][0][0].length;

        // Calculate the number of channels after slicing
        if(endChannel == 0 || endChannel > numChannels) {
            endChannel = numChannels;
        }
        int newNumChannels = endChannel - startChannel;

        // Initialize the new sliced array
        float[][][][] slicedArray = new float[batchSize][height][width][newNumChannels];

        // Copy the data from the original array to the new sliced array
        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    if (endChannel >= startChannel && numChannels >= endChannel)
                        System.arraycopy(array[b][h][w], startChannel, slicedArray[b][h][w], 0, newNumChannels);
                }
            }
        }

        return slicedArray;
    }

    public static float[][] sliceArray2D(float[][] array, int startPos, int endPos) {
        // Get dimensions of the input array
        int width = array.length;
        int numChannels = array[0].length;
        int newNumChannels = endPos - startPos;

        // Calculate the number of channels after slicing
        if(endPos == 0 || endPos > numChannels) {
            newNumChannels = numChannels - startPos;
        }

        // Initialize the new sliced array
        float[][] slicedArray = new float[width][newNumChannels];

        // Copy the data from the original array to the new sliced array
                for (int w = 0; w < width; w++) {
                    if (endPos >= startPos && numChannels >= endPos)
                        System.arraycopy(array[w], startPos, slicedArray[w], 0, newNumChannels);
                }

        return slicedArray;
    }

    public static float[] slice2DArray1D(float[][] array, int index) {
        // Get dimensions of the input array
        int width = array.length;

        // Initialize the new sliced array
        float[] slicedArray = new float[width];

        // Copy the data from the original array to the new sliced array
        for (int w = 0; w < width; w++) {
            slicedArray[w] = array[w][index];
        }

        return slicedArray;
    }

//    public static float[][][][] anchorsPlane(int height, int width, int stride, float[][] baseAnchors) {
//        int A = baseAnchors.length;
//        int baseAnchorSize = baseAnchors[0].length;

//        // Initialize the result array with dimensions [height, width, A, baseAnchorSize]
//        float[][][][] allAnchors = new float[height][width][A][baseAnchorSize];
//
//        // Create the c_0_2 and c_1_3 arrays
//        float[][][] c0_2 = new float[height][width][A];
//        float[][][] c1_3 = new float[height][width][A];
//
//        // Fill c_0_2 with repeated columns and c_1_3 with repeated rows
//        for (int h = 0; h < height; h++) {
//            for (int w = 0; w < width; w++) {
//                for (int a = 0; a < A; a++) {
//                    c0_2[h][w][a] = w;
//                    c1_3[h][w][a] = h;
//                }
//            }
//        }
//
//        // Calculate the allAnchors array
//        for (int h = 0; h < height; h++) {
//            for (int w = 0; w < width; w++) {
//                for (int a = 0; a < A; a++) {
//                    for (int b = 0; b < baseAnchorSize; b++) {
//                        allAnchors[h][w][a][b] = (c0_2[h][w][a] * stride + baseAnchors[a][b]);
//                        allAnchors[h][w][a][b + baseAnchorSize] = (c1_3[h][w][a] * stride + baseAnchors[a][b]);
//                    }
//                }
//            }
//        }
//
//        return allAnchors;
//    }

    // Method to generate anchor planes
    public static float[][][][] anchorsPlane(int height, int width, int stride, float[][] baseAnchors) {
        int A = baseAnchors.length;  // Number of base anchors

        // Generate a 2D array (height x width) of coordinates for x and y using stride
        float[][][] c0_2 = new float[height][width][A];  // For x-coordinates (c_0_2)
        float[][][] c1_3 = new float[height][width][A];  // For y-coordinates (c_1_3)

        // Fill c0_2 (x-coordinates)
        for (int w = 0; w < width; w++) {
            for (int h = 0; h < height; h++) {
                for (int a = 0; a < A; a++) {
                    c0_2[h][w][a] = w * stride;  // Stride applied to the x-axis
                }
            }
        }

        // Fill c1_3 (y-coordinates)
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int a = 0; a < A; a++) {
                    c1_3[h][w][a] = h * stride;  // Stride applied to the y-axis
                }
            }
        }

        // Create the all_anchors array: height x width x A x 4
        float[][][][] allAnchors = new float[height][width][A][4];

        // Fill all_anchors by combining c0_2 and c1_3, and adding base_anchors
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int a = 0; a < A; a++) {
                    allAnchors[h][w][a][0] = c0_2[h][w][a] + baseAnchors[a][0];  // x1
                    allAnchors[h][w][a][1] = c1_3[h][w][a] + baseAnchors[a][1];  // y1
                    allAnchors[h][w][a][2] = c0_2[h][w][a] + baseAnchors[a][2];  // x2
                    allAnchors[h][w][a][3] = c1_3[h][w][a] + baseAnchors[a][3];  // y2
                }
            }
        }

        return allAnchors;
    }

    public static float[] reshape1D(float[][][][] original) {
        int totalElements = original.length * original[0].length * original[0][0].length * original[0][0][0].length;

        float[] flattenedArray = new float[totalElements];
        int index = 0;

        for (float[][][] threeD : original) {
            for (float[][] twoD : threeD) {
                for (float[] oneD : twoD) {
                    for (float value : oneD) {
                        flattenedArray[index++] = value;
                    }
                }
            }
        }
        return flattenedArray;
    }

    // Method to reshape a 4D float array into a 2D float array
    public static float[][] reshape(float[][][][] original, int newRows, int newCols) {
        // Calculate the total number of elements in the original 4D array
        int totalElements = 0;

        for (float[][][] threeD : original) {
            for (float[][] twoD : threeD) {
                for (float[] oneD : twoD) {
                    totalElements += oneD.length;
                }
            }
        }

        if (newRows == -1) {
            newRows = totalElements / newCols;
        }

        if (newCols == -1) {
            newCols = totalElements / newRows;
        }

        if (totalElements != newRows * newCols) {
            throw new IllegalArgumentException("The new dimensions must match the total number of elements.");
        }

        float[] flattenedArray = new float[totalElements];
        int index = 0;

        for (float[][][] threeD : original) {
            for (float[][] twoD : threeD) {
                for (float[] oneD : twoD) {
                    for (float value : oneD) {
                        flattenedArray[index++] = value;
                    }
                }
            }
        }

        float[][] reshapedArray = new float[newRows][newCols];

        for (int i = 0; i < flattenedArray.length; i++) {
            reshapedArray[i / newCols][i % newCols] = flattenedArray[i];
        }

        return reshapedArray;
    }

    // Method to calculate predicted bounding boxes using float arrays
    public static float[][] bboxPred(float[][] boxes, float[][] boxDeltas) {

//        System.out.println("\n\n\n" + boxes.length + "\n" + boxDeltas.length);

        // Handle the case where there are no boxes
        if (boxes.length == 0) {
            return new float[0][boxDeltas.length];
        }

        // Initialize variables
        float[][] predBoxes = new float[boxes.length][boxDeltas.length];

        // Extract widths, heights, and centers of the boxes
        float[] widths = new float[boxes.length];
        float[] heights = new float[boxes.length];
        float[] ctr_x = new float[boxes.length];
        float[] ctr_y = new float[boxes.length];

        for (int i = 0; i < boxes.length; i++) {
            widths[i] = boxes[i][2] - boxes[i][0] + 1.0f;
            heights[i] = boxes[i][3] - boxes[i][1] + 1.0f;
            ctr_x[i] = boxes[i][0] + 0.5f * (widths[i] - 1.0f);
            ctr_y[i] = boxes[i][1] + 0.5f * (heights[i] - 1.0f);
        }

        // Extract deltas from boxDeltas array

        float[] dx = slice2DArray1D(boxDeltas, 0);
        float[] dy = slice2DArray1D(boxDeltas, 1);
        float[] dw = slice2DArray1D(boxDeltas, 2);
        float[] dh = slice2DArray1D(boxDeltas, 3);

        // Calculate the new centers and dimensions
        for (int i = 0; i < boxes.length; i++) {
            float pred_ctr_x = dx[i] * widths[i] + ctr_x[i];
            float pred_ctr_y = dy[i] * heights[i] + ctr_y[i];
            float pred_w = (float) Math.exp(dw[i]) * widths[i];
            float pred_h = (float) Math.exp(dh[i]) * heights[i];

            // Assign new coordinates to predBoxes
            // x1
            predBoxes[i][0] = pred_ctr_x - 0.5f * (pred_w - 1.0f);
            // y1
            predBoxes[i][1] = pred_ctr_y - 0.5f * (pred_h - 1.0f);
            // x2
            predBoxes[i][2] = pred_ctr_x + 0.5f * (pred_w - 1.0f);
            // y2
            predBoxes[i][3] = pred_ctr_y + 0.5f * (pred_h - 1.0f);
        }

        // If boxDeltas has more than 4 values, copy the remaining deltas
        if (boxDeltas.length > 4 * boxes.length) {
            for (int i = 0; i < boxes.length; i++) {
                if (boxDeltas.length / boxes.length - 4 >= 0)
                    System.arraycopy(boxDeltas, i * (boxDeltas.length / boxes.length) + 4, predBoxes[i], 4, boxDeltas.length / boxes.length - 4);
            }
        }
        return predBoxes;
    }

    // Helper method to slice arrays
    private static float[] sliceArray(float[] array, int start, int end) {
//        System.out.println("\n\n" + array.length + " " + start + " " + end);
        float[] slice = new float[end - start];
        System.arraycopy(array, start, slice, 0, end - start);
        return slice;
    }

    // Method to clip the boxes to the image boundaries
    public static float[][] clipBoxes(float[][] boxes, int imHeight, int imWidth) {
        int numBoxes = boxes.length;

        for (int i = 0; i < numBoxes; i++) {
            boxes[i][0] = Math.max(Math.min(boxes[i][0], imWidth - 1), 0); // x1
        }

        for (int i = 0; i < numBoxes; i++) {
            boxes[i][1] = Math.max(Math.min(boxes[i][1], imHeight - 1), 0); // y1
        }

        for (int i = 0; i < numBoxes; i++) {
            boxes[i][2] = Math.max(Math.min(boxes[i][2], imWidth - 1), 0); // x2
        }

        for (int i = 0; i < numBoxes; i++) {
            boxes[i][3] = Math.max(Math.min(boxes[i][3], imHeight - 1), 0); // y2
        }

        return boxes;
    }

    // Method to find indices where array values are greater than or equal to threshold
    public static int[] findIndices(float[] array, double threshold) {
//        System.out.println(Arrays.toString(array));
        List<Integer> indicesList = new ArrayList<>();

        for (int i = 0; i < array.length; i++) {
            if (array[i] >= threshold) {
//                System.out.println("Order Output" + array[i]);
                indicesList.add(i);
            }
        }

//        return indicesList;
        return indicesList.stream().mapToInt(Integer::intValue).toArray();
    }

    public static float[] filterByConf1D(float[] array, int[] indices) {
        float[] filteredScores = new float[indices.length];

        for (int i = 0; i < indices.length; i++) {
            filteredScores[i] = array[indices[i]];
        }

        return filteredScores;
    }

    // Method to filter a 2D float array based on indices
    public static float[][] filterByConf2D(float[][] array, int[] indices) {
        float[][] filteredScores = new float[indices.length][array[0].length];

        for (int i = 0; i < indices.length; i++) {
            filteredScores[i] = array[indices[i]];
        }

        return filteredScores;
    }

    // Stack multiple 1D float arrays into a 2D array (vstack equivalent)
    public static float[] vstack(List<float[]> scoresList) {
//        int totalRows = scoresList.size();
        int cols = 0;
        int i = 0;

        for (float[] scores : scoresList) {
            cols += scores.length;
        }

        // Create the stacked array
        float[] stackedScores = new float[cols];

        for (float[] scores : scoresList) {
            for (float score : scores) {
                stackedScores[i++] = score;
            }
        }

//        System.out.println("The stack is " + stackedScores.length + " " + stackedScores[0].length);

        return stackedScores;
    }

    // Stack multiple 2D float arrays vertically (vstack equivalent)
    public static float[][] vstack2D(List<float[][]> scoresList) {
        int totalRows = 0;
//        System.out.println("\nTesting theory\n" + Objects.isNull(scoresList.get(0)));
        int cols = 0;

        // Calculate the total number of rows
        for (float[][] floats : scoresList) {
            totalRows += floats.length;
            if (floats.length > cols) {
                cols = floats[0].length;
            }
        }

        // Create the stacked array
        float[][] stackedScores = new float[totalRows][cols];
        int currentRow = 0;

        for (float[][] scores : scoresList) {
            for (float[] row : scores) {
                stackedScores[currentRow++] = row;
            }
        }

        return stackedScores;
    }

    // Method to horizontally stack two 2D float arrays
    public static float[][] hstack2D(float[][] proposals, float[][] scores) {
        int rows = proposals.length;
        int proposalsCols = proposals[0].length;
        int scoresCols = scores[0].length;

        // Ensure both arrays have the same number of rows
        if (rows != scores.length) {
            throw new IllegalArgumentException("The number of rows in proposals and scores must be the same.");
        }

        // Create a new array to hold the stacked arrays
        float[][] preDet = new float[rows][proposalsCols + scoresCols];

        // Copy proposals data into the new array
        for (int i = 0; i < rows; i++) {
            System.arraycopy(proposals[i], 0, preDet[i], 0, proposalsCols);
            System.arraycopy(scores[i], 0, preDet[i], proposalsCols, scoresCols);
        }

        return preDet;
    }

    public static float[][] hstack(float[][] proposals, float[] scores) {
        int rows = proposals.length;
        int proposalsCols = proposals[0].length;
//        int scoresCols = scores.length;

        // Ensure both arrays have the same number of rows
        if (rows != scores.length) {
            throw new IllegalArgumentException("The number of rows in proposals and scores must be the same.");
        }

        // Create a new array to hold the stacked arrays
        float[][] preDet = new float[rows][proposalsCols + 1];

        // Copy proposals data into the new array
        for (int i = 0; i < rows; i++) {
            System.arraycopy(proposals[i], 0, preDet[i], 0, proposalsCols);
            preDet[i][proposalsCols] = scores[i];
//            System.arraycopy(scores, 0, preDet[i], proposalsCols, scoresCols);
        }

        return preDet;
    }

    // Flatten a 2D array into a 1D array
    public static float[] ravel(float[][] scores) {
        int rows = scores.length;
        int cols = scores[0].length;
        float[] ravelled = new float[rows * cols];
        int index = 0;

        System.out.println("\n\n Let's assess\n" + rows + cols);

        for (float[] row : scores) {
            for (float value : row) {
                System.out.println(index);
                ravelled[index++] = value;
            }
        }

        return ravelled;
    }

    // Get the descending order of the indices after sorting
    public static int[] argsortDescending(float[] array) {
        Integer[] indices = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            indices[i] = i;
        }

        // Sort the indices based on the values in array (in descending order)
        Arrays.sort(indices, (i1, i2) -> Float.compare(array[i2], array[i1]));

        // Convert to int[]
        return Arrays.stream(indices).mapToInt(Integer::intValue).toArray();
    }

    // Method to perform Non-Maximum Suppression (NMS)
    public static int[] cpuNms(float[][] dets, double threshold) {
        int ndets = dets.length;

        // Extracting x1, y1, x2, y2, and scores from the dets array
        float[] x1 = new float[ndets];
        float[] y1 = new float[ndets];
        float[] x2 = new float[ndets];
        float[] y2 = new float[ndets];
        float[] scores = new float[ndets];

        for (int i = 0; i < ndets; i++) {
            x1[i] = dets[i][0];
            y1[i] = dets[i][1];
            x2[i] = dets[i][2];
            y2[i] = dets[i][3];
            scores[i] = dets[i][4];
        }

        // Compute the areas of the bounding boxes
        float[] areas = new float[ndets];
        for (int i = 0; i < ndets; i++) {
            areas[i] = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1);
        }

        // Sort the indices of the scores in descending order
        Integer[] order = new Integer[ndets];
        for (int i = 0; i < ndets; i++) {
            order[i] = i;
        }

        // Sort based on scores (in descending order)
        Arrays.sort(order, (i, j) -> Float.compare(scores[j], scores[i]));

        int[] suppressed = new int[ndets];  // Array to keep track of suppressed boxes
        ArrayList<Integer> keep = new ArrayList<>();  // List of kept boxes

        // Non-Maximum Suppression algorithm
        for (int _i = 0; _i < ndets; _i++) {
            int i = order[_i];
            if (suppressed[i] == 1) {
                continue;
            }
            keep.add(i);

            float ix1 = x1[i];
            float iy1 = y1[i];
            float ix2 = x2[i];
            float iy2 = y2[i];
            float iarea = areas[i];

            for (int _j = _i + 1; _j < ndets; _j++) {
                int j = order[_j];
                if (suppressed[j] == 1) {
                    continue;
                }

                // Compute overlap (IoU)
                float xx1 = Math.max(ix1, x1[j]);
                float yy1 = Math.max(iy1, y1[j]);
                float xx2 = Math.min(ix2, x2[j]);
                float yy2 = Math.min(iy2, y2[j]);

                float w = Math.max(0.0f, xx2 - xx1 + 1);
                float h = Math.max(0.0f, yy2 - yy1 + 1);
                float inter = w * h;

                float ovr = inter / (iarea + areas[j] - inter);

                // Suppress box if IoU is above the threshold
                if (ovr >= threshold) {
                    suppressed[j] = 1;
                }
            }
        }

        return keep.stream().mapToInt(Integer::intValue).toArray();
    }

}
