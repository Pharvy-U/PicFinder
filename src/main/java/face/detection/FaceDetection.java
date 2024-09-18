package face.detection;

import face.payload.FaceDetectionDTO;
import face.payload.ImageDTO;
import face.util.ImageHandler;
import org.tensorflow.*;
import org.tensorflow.types.UInt8;

import java.awt.*;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.ResourceBundle;


public class FaceDetection {

    private static final ResourceBundle rb = ResourceBundle.getBundle("application");

    private static final double confidenceThreshold = Double.parseDouble(rb.getString("detection-confidence-threshold"));

    private static final double minSimilarityScore = Double.parseDouble(rb.getString("minimum-similarity-score"));

    private static final double nms_threshold = 0.4;

    private static final double decay = 0.5;

    private static final float[][] anchor32 = {{-248, -248, 263, 263}, {-120, -120, 135, 135}};
    private static final float[][] anchor16 = {{-56, -56, 71, 71}, {-24, -24, 39, 39}};
    private static final float[][] anchor8 = {{-8, -8, 23, 23}, {0, 0, 15, 15}};

    static final Session s = SavedModelBundle.load(rb.getString("detection-model"), "serve").session();

    static final Session s_rec = SavedModelBundle.load(rb.getString("recognition-model"), "serve").session();

    public static List<String> getFaces(String folderPath, String userImagePath, boolean displayImages) throws Exception {

        List<String> photoMatches = new ArrayList<>();

        File folder = new File(folderPath);
        File userImage = new File(userImagePath);

        if(isImage(userImage)) {
            ImageDTO userImageDto = ImageHandler.readImage(userImage.getPath());
            FaceDetectionDTO facesToMatch = performInference(userImageDto, displayImages);

            if(facesToMatch.getNumDetections() == 0) {
                throw new Exception("no faces found in input image. Please input a valid image");
            }

            if(folder.isDirectory()) {
                File[] files = folder.listFiles();

                assert files != null;
                System.out.println("Length of files in folder: " + files.length);
                int count = 1;
                for (File file : files) {
                    System.out.println("Iteration " + count++);
                    if (isImage(file)){
                        ImageDTO systemImageDto = ImageHandler.readImage(file.getPath());
                        FaceDetectionDTO facesToMatchWith = performInference(systemImageDto, displayImages);

                        int numFacesDetected = (int) facesToMatchWith.getNumDetections();
                        System.out.println("Number of faces detected: " + numFacesDetected);
                        boolean breakOuter = false;

                        for (int j = 0; j < numFacesDetected; j++) {
                            if (breakOuter) {
                                break;
                            }
                            ImageDTO croppedSystemImage = ImageHandler.cropImage(systemImageDto.getBufferedImage(),
                                    facesToMatchWith.getBoxes()[j][1],
                                    facesToMatchWith.getBoxes()[j][0],
                                    facesToMatchWith.getBoxes()[j][3],
                                    facesToMatchWith.getBoxes()[j][2]);

                            // save faces for external tests
//                            File save = new File("C:\\Users\\FAVOUR-UKASOANYA\\Desktop\\Projects\\Photo_Cluster\\test\\cropped\\" + ImageHandler.counter() + ".jpg");
//                            ImageIO.write(croppedSystemImage.getBufferedImage(), "jpg", save);

                            for (int i = 0; i < (int) facesToMatch.getNumDetections(); i++) {
                                ImageDTO croppedUserImage = ImageHandler.cropImage(userImageDto.getBufferedImage(),
                                        facesToMatch.getBoxes()[i][1],
                                        facesToMatch.getBoxes()[i][0],
                                        facesToMatch.getBoxes()[i][3],
                                        facesToMatch.getBoxes()[i][2]);

                                double similarity = compareFaceTensors(croppedUserImage, croppedSystemImage);
                                System.out.printf("The similarity score between images [%s] and [%s] is: [%f]\n", userImage.getName(), file.getName(), similarity);
                                if (similarity >= minSimilarityScore) {
                                    photoMatches.add(file.getAbsolutePath());
                                    breakOuter = true;
                                    break;
                                }
                            }

                        }
                    }
                }
            }
        }

        return photoMatches;
    }

    public static int getNumFaces(String imagePath, boolean displayImage) throws Exception {
        File image = new File(imagePath);

        if (isImage(image)) {
            ImageDTO userImageDto = ImageHandler.readImage(image.getPath());
            FaceDetectionDTO foundFaces = performInference(userImageDto, displayImage);

            return (int) foundFaces.getNumDetections();
        }
        return 0;
    }

    private static FaceDetectionDTO performInference(ImageDTO imageDto, boolean displayImages) {
//        System.out.println("IT\nER\n1" + " " + imageDto.getBufferedImage().getHeight() + " " + imageDto.getBufferedImage().getWidth());
        FaceDetectionDTO faceDetectionDTO = new FaceDetectionDTO();
        Tensor<Float> imageTensor = null;

        // handle image resize better
        if (imageDto.getBufferedImage().getHeight() * imageDto.getBufferedImage().getWidth() > 1000000) {
            if (imageDto.getBufferedImage().getHeight() > imageDto.getBufferedImage().getWidth()) {
                imageTensor = ImageHandler.convertImageToTensorFloat(ImageHandler.resize(imageDto.getBufferedImage(), 1080, 720));
            } else if (imageDto.getBufferedImage().getHeight() < imageDto.getBufferedImage().getWidth()) {
                imageTensor = ImageHandler.convertImageToTensorFloat(ImageHandler.resize(imageDto.getBufferedImage(), 720, 1080));
            } else {
                imageTensor = ImageHandler.convertImageToTensorFloat(ImageHandler.resize(imageDto.getBufferedImage(), 1000, 1000));
            }
        } else {
            imageTensor = ImageHandler.convertImageToTensorFloat(imageDto.getBufferedImage());
        }

        System.out.println(Arrays.toString(imageTensor.shape()));

        List<Tensor<?>> detections = s.runner()
                .feed("serving_default_data:0", imageTensor) //input_tensors
                .fetch("StatefulPartitionedCall:0")
                .fetch("StatefulPartitionedCall:1")
                .fetch("StatefulPartitionedCall:2")
                .fetch("StatefulPartitionedCall:3")
                .fetch("StatefulPartitionedCall:4")
                .fetch("StatefulPartitionedCall:5")
                .fetch("StatefulPartitionedCall:6")
                .fetch("StatefulPartitionedCall:7")
                .fetch("StatefulPartitionedCall:8")
                .run();

//        int numMaxClasses = 300;
//        float[][][] boxes = new float[1][numMaxClasses][4];
//        float[][] scores = new float[1][numMaxClasses];
//        float[] num_detections = new float[1];


        float[][][][] scores_1 = new float[1][(int) detections.get(6).shape()[1]][(int) detections.get(6).shape()[2]][4];
        float[][][][] boxes_1 = new float[1][(int) detections.get(1).shape()[1]][(int) detections.get(1).shape()[2]][8];
        float[][][][] lm_1 = new float[1][(int) detections.get(4).shape()[1]][(int) detections.get(4).shape()[2]][20];
        float[][][][] scores_2 = new float[1][(int) detections.get(7).shape()[1]][(int) detections.get(7).shape()[2]][4];
        float[][][][] boxes_2 = new float[1][(int) detections.get(0).shape()[1]][(int) detections.get(0).shape()[2]][8];
        float[][][][] lm_2 = new float[1][(int) detections.get(3).shape()[1]][(int) detections.get(3).shape()[2]][20];
        float[][][][] scores_3 = new float[1][(int) detections.get(8).shape()[1]][(int) detections.get(8).shape()[2]][4];
        float[][][][] boxes_3 = new float[1][(int) detections.get(2).shape()[1]][(int) detections.get(2).shape()[2]][8];
        float[][][][] lm_3 = new float[1][(int) detections.get(5).shape()[1]][(int) detections.get(5).shape()[2]][20];

//        float[][][][] scores_1 = new float[1][(int) detections.get(7).shape()[1]][(int) detections.get(7).shape()[2]][4];
//        float[][][][] boxes_1 = new float[1][(int) detections.get(1).shape()[1]][(int) detections.get(1).shape()[2]][8];
//        float[][][][] lm_1 = new float[1][(int) detections.get(4).shape()[1]][(int) detections.get(4).shape()[2]][20];
//        float[][][][] scores_2 = new float[1][(int) detections.get(6).shape()[1]][(int) detections.get(6).shape()[2]][4];
//        float[][][][] boxes_2 = new float[1][(int) detections.get(0).shape()[1]][(int) detections.get(0).shape()[2]][8];
//        float[][][][] lm_2 = new float[1][(int) detections.get(3).shape()[1]][(int) detections.get(3).shape()[2]][20];
//        float[][][][] scores_3 = new float[1][(int) detections.get(8).shape()[1]][(int) detections.get(8).shape()[2]][4];
//        float[][][][] boxes_3 = new float[1][(int) detections.get(2).shape()[1]][(int) detections.get(2).shape()[2]][8];
//        float[][][][] lm_3 = new float[1][(int) detections.get(5).shape()[1]][(int) detections.get(5).shape()[2]][20];

        for(int i=0; i<9; i++) {
            System.out.printf("Shape of Output %s is: " + Arrays.toString(detections.get(i).shape()) + "\n", i);
        }

        detections.get(0).copyTo(boxes_2);
        detections.get(1).copyTo(boxes_1);
        detections.get(2).copyTo(boxes_3);
        detections.get(3).copyTo(lm_2);
        detections.get(4).copyTo(lm_1);
        detections.get(5).copyTo(lm_3);
        detections.get(6).copyTo(scores_1);
        detections.get(7).copyTo(scores_2);
        detections.get(8).copyTo(scores_3);

//        detections.get(0).copyTo(boxes_2);
//        detections.get(6).copyTo(scores_2);
//        detections.get(3).copyTo(lm_2);
//        detections.get(2).copyTo(boxes_3);
//        detections.get(8).copyTo(scores_3);
//        detections.get(5).copyTo(lm_3);
//        detections.get(7).copyTo(scores_1);
//        detections.get(1).copyTo(boxes_1);
//        detections.get(4).copyTo(lm_1);

//        System.out.println(Arrays.deepToString(scores_1[0][0]) + "\n");
//        System.out.println(Arrays.deepToString(scores_2[0][0]) + "\n");
//        System.out.println(Arrays.deepToString(scores_3[0][0]) + "\n");
//        System.out.println(Arrays.deepToString(boxes_1[0][0]) + "\n");
//        System.out.println(Arrays.deepToString(boxes_2[0][0]) + "\n");
//        System.out.println(Arrays.deepToString(boxes_3[0][0]) + "\n");
//        System.out.println(Arrays.deepToString(lm_1[0][0]) + "\n");
//        System.out.println(Arrays.deepToString(lm_2[0][0]) + "\n");
//        System.out.println(Arrays.deepToString(lm_3[0][0]) + "\n");

//        System.out.println(scores_1[0][0].length + "\n");
//        System.out.println(scores_2[0][0].length + "\n");
//        System.out.println(scores_3[0][0].length + "\n");

        List<float[]> scoresList = new ArrayList<>();
        List<float[][]> proposalsList = new ArrayList<>();

        System.out.println(proposalsList);

        computeBoundingBoxes(scores_1, boxes_1, 32, anchor32, scoresList, proposalsList, imageDto);
        computeBoundingBoxes(scores_2, boxes_2, 16, anchor16, scoresList, proposalsList, imageDto);
        computeBoundingBoxes(scores_3, boxes_3, 8, anchor8, scoresList, proposalsList, imageDto);

//        System.out.println("Final score list before stacking" + Arrays.toString(proposalsList.get(1)));

        float[][] proposals = ImageHandler.vstack2D(proposalsList);

        if(proposals.length == 0) {
            return faceDetectionDTO;
        }

        float[] scores = ImageHandler.vstack(scoresList);
        int[] order = ImageHandler.argsortDescending(scores);

        proposals = ImageHandler.filterByConf2D(proposals, order);
        scores = ImageHandler.filterByConf1D(scores, order);

        float[][] preDet = ImageHandler.hstack(ImageHandler.sliceArray2D(proposals, 0, 4), scores);

        int[] keep = ImageHandler.cpuNms(preDet, nms_threshold);

        float[][] det = ImageHandler.filterByConf2D(preDet, keep);

        System.out.println("\n\nPrinting out the dets\n" + det.length);
        System.out.println(Arrays.deepToString(ImageHandler.sliceArray2D(det, 0, 4)));
        System.out.println("Keeps\n" + Arrays.toString(ImageHandler.slice2DArray1D(det, 4)));
//        System.out.println(Arrays.deepToString(det));


        faceDetectionDTO.setBoxes(ImageHandler.sliceArray2D(det, 0, 4));
        faceDetectionDTO.setScores(ImageHandler.slice2DArray1D(det, 4));
        faceDetectionDTO.setNumDetections(det.length);
//
//        int sliceIndex = -1;
//        for (int i = 0; i < scores[0].length; i++) {
//            if (scores[0][i] >= confidenceThreshold) {
//                sliceIndex = i;
//            } else {
//                break;
//            }
//        }
//
//        faceDetectionDTO.getNumDetections()[0] = sliceIndex + 1;
//
//        if(displayImages) {
//            for (int i = 0; i <= sliceIndex; i++) {
//
//                int ymin = Math.round(boxes[0][i][0] * imageDto.getBufferedImage().getHeight());
//                int xmin = Math.round(boxes[0][i][1] * imageDto.getBufferedImage().getWidth());
//                int ymax = Math.round(boxes[0][i][2] * imageDto.getBufferedImage().getHeight());
//                int xmax = Math.round(boxes[0][i][3] * imageDto.getBufferedImage().getWidth());
//
//                imageDto.getGraphics2D().setColor(Color.GREEN);
//                imageDto.getGraphics2D().setStroke(new BasicStroke(4));
//                imageDto.getGraphics2D().drawRect(xmin, ymin, xmax - xmin, ymax - ymin);
//                imageDto.getGraphics2D().setColor(Color.RED);
//                imageDto.getGraphics2D().setFont(new Font("Arial", Font.PLAIN, 50));
//                imageDto.getGraphics2D().drawString(String.valueOf(scores[0][i]), xmin, ymin);
//                ImageHandler.display(imageDto.getBufferedImage());

//            ImageDTO cropped = ImageHandler.cropImage(imageDto.getBufferedImage(), boxes[0][i][1], boxes[0][i][0], boxes[0][i][3], boxes[0][i][2]);
//            ImageHandler.display(cropped.getBufferedImage());
//            }
//        }

        return faceDetectionDTO;
    }

    private static double compareFaceTensors(ImageDTO systemImage, ImageDTO userImage) {
        Tensor<Float> systemImageTensor = ImageHandler.convertImageToTensorFloatNorm(ImageHandler.resize(systemImage.getBufferedImage(), 160, 160));
        Tensor<Float> userImageTensor = ImageHandler.convertImageToTensorFloatNorm(ImageHandler.resize(userImage.getBufferedImage(), 160, 160));

        double score = 0;

        List<Tensor<?>> userEmbed = s_rec.runner()
                .feed("serving_default_input_1:0", userImageTensor)
                .fetch("StatefulPartitionedCall:0")
                .run();

        List<Tensor<?>> systemEmbed = s_rec.runner()
                .feed("serving_default_input_1:0", systemImageTensor)
                .fetch("StatefulPartitionedCall:0")
                .run();

        float[][] userImageEmbed = new float[1][512];
        float[][] systemImageEmbed = new float[1][512];

        userEmbed.get(0).copyTo(userImageEmbed);
        systemEmbed.get(0).copyTo(systemImageEmbed);

        double dot = 0;
        double mag_1 = 0;
        double mag_2 = 0;

        for(int i=0; i<256; i++) {
            dot = dot + (userImageEmbed[0][i] * systemImageEmbed[0][i]);
            mag_1 = mag_1 + (userImageEmbed[0][i] * userImageEmbed[0][i]);
            mag_2 = mag_2 + (systemImageEmbed[0][i] * systemImageEmbed[0][i]);
        }
        score = dot / (Math.sqrt(mag_1) * Math.sqrt(mag_2));

        return score;
    }

    private static void computeBoundingBoxes(float[][][][] scores, float[][][][] boxes,
                                             int stride, float[][] anchor, List<float[]> scoresList,
                                             List<float[][]> proposalsList, ImageDTO imageDto) {
//        System.out.println("scores_init " + Arrays.deepToString(scores));
        scores = ImageHandler.sliceArray(scores, 2, 0);
        int height = boxes[0].length;
        int width = boxes[0][0].length;
        int bboxPredLen = boxes[0][0][0].length / 2;
        int K = height * width;

        float[][][][] anchors = ImageHandler.anchorsPlane(height, width, stride, anchor);
        float[][] anchorsR = ImageHandler.reshape(anchors, K*2, 4);
        float[] scores_1R = ImageHandler.reshape1D(scores);
        float[][] boxes_1R = ImageHandler.reshape(boxes, -1, bboxPredLen);

        float[][] proposals = ImageHandler.bboxPred(anchorsR,boxes_1R);
        float[][] proposalsC = ImageHandler.clipBoxes(proposals, imageDto.getBufferedImage().getHeight(), imageDto.getBufferedImage().getWidth());

        if(stride == 4 && decay < 1) {
            for (int i = 0; i < scores_1R.length; i++) {
                scores_1R[i] *= decay;
            }
        }

//        for (float v : scores_1R) {
//            if (v > confidenceThreshold)
//                System.out.println(v);
//        }

        int[] order = ImageHandler.findIndices(scores_1R, confidenceThreshold); //now the problem is here
        float[] scoreFiltered = ImageHandler.filterByConf1D(scores_1R, order);
        float[][] proposalsFiltered = ImageHandler.filterByConf2D(proposalsC, order);

        scoresList.add(scoreFiltered);
        proposalsList.add(proposalsFiltered);
//        System.out.println("\nscores" + Arrays.deepToString(scores));
//        System.out.println("\nscores_IR" + Arrays.toString(scores_1R));
//        System.out.println("\norder" + Arrays.toString(order));
//        System.out.println("\nscores_filtered" + Arrays.toString(scoreFiltered));
//        System.out.println("\nproposals_filtered" + Arrays.deepToString(proposalsFiltered));
//        System.out.println("\nboxes_IR" + Arrays.toString(boxes_1R));
    }

    private static boolean isImage(File file) {
        return file.isFile() && file.getName().contains(".jpg") ||
                file.getName().contains(".jpeg") ||
                file.getName().contains(".png") ||
                file.getName().contains(".JPG") ||
                file.getName().contains(".JPEG") ||
                file.getName().contains(".PNG");
    }

}
