package face.payload;

import java.awt.*;
import java.awt.image.BufferedImage;

public class ImageDTO {

    private BufferedImage bufferedImage;

    private Graphics2D graphics2D;

    public ImageDTO(BufferedImage bufferedImage, Graphics2D graphics2D) {
        this.bufferedImage = bufferedImage;
        this.graphics2D = graphics2D;
    }

    public BufferedImage getBufferedImage() {
        return bufferedImage;
    }

    public void setBufferedImage(BufferedImage bufferedImage) {
        this.bufferedImage = bufferedImage;
    }

    public Graphics2D getGraphics2D() {
        return graphics2D;
    }

    public void setGraphics2D(Graphics2D graphics2D) {
        this.graphics2D = graphics2D;
    }

    @Override
    public String toString() {
        return "ImageDTO{" +
                "bufferedImage=" + bufferedImage +
                ", graphics2D=" + graphics2D +
                '}';
    }
}
