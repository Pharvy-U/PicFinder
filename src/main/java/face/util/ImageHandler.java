package face.util;

import face.payload.ImageDTO;
import org.tensorflow.Tensor;
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
                floatBuffer.put(((rgb >> 16) & 0xFF) / 255.0f);
                floatBuffer.put(((rgb >> 8) & 0xFF) / 255.0f);
                floatBuffer.put((rgb & 0xFF) / 255.0f);
            }
        }
        floatBuffer.rewind();

        // Create a Tensor from the ByteBuffer
        return Tensor.create(new long[]{1, height, width, 3}, floatBuffer);
    }
}
