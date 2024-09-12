package face.detection;

import face.payload.FaceDetectionDTO;
import face.payload.ImageDTO;
import face.util.ImageHandler;
import org.tensorflow.*;
import org.tensorflow.types.UInt8;

import java.awt.*;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.ResourceBundle;


public class FaceDetection {

    private static final ResourceBundle rb = ResourceBundle.getBundle("application");

    private static final double confidenceThreshold = Double.parseDouble(rb.getString("detection-confidence-threshold"));

    private static final double minSimilarityScore = Double.parseDouble(rb.getString("minimum-similarity-score"));

    static final Session s = SavedModelBundle.load(rb.getString("detection-model"), "serve").session();

    static final Session s_rec = SavedModelBundle.load(rb.getString("recognition-model"), "serve").session();

    public static List<String> getFaces(String folderPath, String userImagePath, boolean displayImages) throws Exception {

        List<String> photoMatches = new ArrayList<>();

        File folder = new File(folderPath);
        File userImage = new File(userImagePath);

        if(isImage(userImage)) {
            ImageDTO userImageDto = ImageHandler.readImage(userImage.getPath());
            FaceDetectionDTO facesToMatch = performInference(userImageDto, displayImages);

            if(facesToMatch.getNumDetections()[0] == 0) {
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

                        int numFacesDetected = (int) facesToMatchWith.getNumDetections()[0];
                        System.out.println("Number of faces detected: " + numFacesDetected);
                        boolean breakOuter = false;

                        for (int j = 0; j < numFacesDetected; j++) {
                            if (breakOuter) {
                                break;
                            }
                            ImageDTO croppedSystemImage = ImageHandler.cropImage(systemImageDto.getBufferedImage(),
                                    facesToMatchWith.getBoxes()[0][j][1],
                                    facesToMatchWith.getBoxes()[0][j][0],
                                    facesToMatchWith.getBoxes()[0][j][3],
                                    facesToMatchWith.getBoxes()[0][j][2]);

                            // save faces for external tests
//                            File save = new File("C:\\Users\\FAVOUR-UKASOANYA\\Desktop\\Projects\\Photo_Cluster\\test\\cropped\\" + ImageHandler.counter() + ".jpg");
//                            ImageIO.write(croppedSystemImage.getBufferedImage(), "jpg", save);

                            for (int i = 0; i < (int) facesToMatch.getNumDetections()[0]; i++) {
                                ImageDTO croppedUserImage = ImageHandler.cropImage(userImageDto.getBufferedImage(),
                                        facesToMatch.getBoxes()[0][i][1],
                                        facesToMatch.getBoxes()[0][i][0],
                                        facesToMatch.getBoxes()[0][i][3],
                                        facesToMatch.getBoxes()[0][i][2]);

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

            return (int) foundFaces.getNumDetections()[0];
        }
        return 0;
    }

    private static FaceDetectionDTO performInference(ImageDTO imageDto, boolean displayImages) {
        FaceDetectionDTO faceDetectionDTO = new FaceDetectionDTO();
        Tensor<UInt8> imageTensor = ImageHandler.convertImageToTensor(imageDto.getBufferedImage());

        List<Tensor<?>> detections = s.runner()
                .feed("serving_default_input_tensor:0", imageTensor) //input_tensors
                .fetch("StatefulPartitionedCall:1") //detection_boxes
                .fetch("StatefulPartitionedCall:4") //detection_scores
                .fetch("StatefulPartitionedCall:5") //num_detections
                .run();

        int numMaxClasses = 300;
        float[][][] boxes = new float[1][numMaxClasses][4];
        float[][] scores = new float[1][numMaxClasses];
        float[] num_detections = new float[1];

        detections.get(0).copyTo(boxes);
        detections.get(1).copyTo(scores);
        detections.get(2).copyTo(num_detections);

        faceDetectionDTO.setBoxes(boxes);
        faceDetectionDTO.setScores(scores);
        faceDetectionDTO.setNumDetections(num_detections);

        int sliceIndex = -1;
        for (int i = 0; i < scores[0].length; i++) {
            if (scores[0][i] >= confidenceThreshold) {
                sliceIndex = i;
            } else {
                break;
            }
        }

        faceDetectionDTO.getNumDetections()[0] = sliceIndex + 1;

        if(displayImages) {
            for (int i = 0; i <= sliceIndex; i++) {

                int ymin = Math.round(boxes[0][i][0] * imageDto.getBufferedImage().getHeight());
                int xmin = Math.round(boxes[0][i][1] * imageDto.getBufferedImage().getWidth());
                int ymax = Math.round(boxes[0][i][2] * imageDto.getBufferedImage().getHeight());
                int xmax = Math.round(boxes[0][i][3] * imageDto.getBufferedImage().getWidth());

                imageDto.getGraphics2D().setColor(Color.GREEN);
                imageDto.getGraphics2D().setStroke(new BasicStroke(4));
                imageDto.getGraphics2D().drawRect(xmin, ymin, xmax - xmin, ymax - ymin);
                imageDto.getGraphics2D().setColor(Color.RED);
                imageDto.getGraphics2D().setFont(new Font("Arial", Font.PLAIN, 50));
                imageDto.getGraphics2D().drawString(String.valueOf(scores[0][i]), xmin, ymin);
                ImageHandler.display(imageDto.getBufferedImage());

//            ImageDTO cropped = ImageHandler.cropImage(imageDto.getBufferedImage(), boxes[0][i][1], boxes[0][i][0], boxes[0][i][3], boxes[0][i][2]);
//            ImageHandler.display(cropped.getBufferedImage());
            }
        }

        return faceDetectionDTO;
    }

    private static double compareFaceTensors(ImageDTO systemImage, ImageDTO userImage) {
        Tensor<Float> systemImageTensor = ImageHandler.convertImageToTensorFloat(ImageHandler.resize(systemImage.getBufferedImage(), 160, 160));
        Tensor<Float> userImageTensor = ImageHandler.convertImageToTensorFloat(ImageHandler.resize(userImage.getBufferedImage(), 160, 160));

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

    private static boolean isImage(File file) {
        return file.isFile() && file.getName().contains(".jpg") ||
                file.getName().contains(".jpeg") ||
                file.getName().contains(".png") ||
                file.getName().contains(".JPG") ||
                file.getName().contains(".JPEG") ||
                file.getName().contains(".PNG");
    }

}
