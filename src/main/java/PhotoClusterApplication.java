import face.detection.FaceDetection;

import java.util.List;

public class PhotoClusterApplication {

    public static void main(String[]args) throws Exception {
        long heapsize = Runtime.getRuntime().totalMemory();
        System.out.println("heapsize is :: " + heapsize);

        List<String> matches = FaceDetection.getFaces("C:\\Users\\FAVOUR-UKASOANYA\\Desktop\\Projects\\Photo_Cluster\\test\\debug",
                "C:\\Users\\FAVOUR-UKASOANYA\\Desktop\\Projects\\Photo_Cluster\\test\\debug\\1.jpg", true); // C:\Users\FAVOUR-UKASOANYA\Desktop\Personal FIles\4Z0A2440.jpg

//        int num = FaceDetection.getNumFaces("C:\\Users\\FAVOUR-UKASOANYA\\Desktop\\Projects\\Photo_Cluster\\test\\test-image\\omo.jpg", true);
//        System.out.println("The number os faces found in given image: " + num);

        System.out.println("\n\n\n***** COMPLETE *****");
        System.out.println("Matches found " + matches);
//        System.exit(0);
    }

}
