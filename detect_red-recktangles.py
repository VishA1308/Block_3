import cv2
import numpy
import argparse

def detect_red_rectangles(image_path: str, output_path='output.jpg'):
    """
    Обнаруживает красные прямоугольники в заданном изображении.

    Args:
        image_path: Путь к входному файлу изображения.
        output_path: Путь для сохранения выходного изображения.

    Returns:
        Количество обнаруженных красных прямоугольников.
    """

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Ошибка в загрузке файла: {}".format(image_path))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = numpy.array([0, 50, 50])
    upper_red1 = numpy.array([10, 255, 255])
    lower_red2 = numpy.array([160, 50, 50])
    upper_red2 = numpy.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = numpy.ones((5, 5), numpy.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image = image.copy()
    k = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        eps = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)

        if len(approx) == 4:
            if cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h

                if 0.5 <= aspect_ratio <= 2.0:
                    cv2.drawContours(output_image, [approx], -1, (0, 255, 0), 3)
                    k += 1
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(output_image, str(k), (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imwrite(output_path, output_image)
    cv2.imwrite('red_mask.jpg', red_mask)

    return k

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str)
    parser.add_argument("--output_path", type=str, default="output.jpg")
    args = parser.parse_args()

    try:
        num_rectangles = detect_red_rectangles(args.image_path, args.output_path)
        print("Количество обнаруженных красных прямоугольников: {}".format(num_rectangles))
    except Exception as e:
        print("Ошибка: {}".format(e))