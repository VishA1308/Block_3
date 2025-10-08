import os
import cv2
import numpy as np
from skimage import data
#CV-1-05:
from utils.coins_contour_detection import counting_contours
#CV-1-24:
from utils.proga1 import ShapeDetector


def classify_contours(contours, circularity_threshold=0.8):
    """
    Classify contours into circles, rectangles, and others.

    Args:
        contours (list[np.ndarray]): List of contours found in the image.
        circularity_threshold (float, optional): Threshold for circularity 
            (4π * Area / Perimeter²). Higher values → more likely a circle. 
            Default is 0.8.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
            - circles: contours classified as circles.
            - rects: contours classified as rectangles or squares.
            - others: all remaining contours.

    Raises:
        TypeError: If contours is not a list or contains invalid elements.
        ValueError: If no valid contours are provided.
    """
    if not isinstance(contours, (list, tuple)):
        raise TypeError("Contours must be provided as a list or tuple")
    if not contours:
        raise ValueError("Empty contour list provided")

    sd = ShapeDetector()
    circles, rects, others = [], [], []

    for cnt in contours:
        if not isinstance(cnt, np.ndarray):
            continue  # skip invalid elements

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0 or area <= 0:
            others.append(cnt)
            continue

        # Circularity formula: closer to 1 means more circular
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity > circularity_threshold:
            circles.append(cnt)
        else:
            shape = sd.detect(cnt)
            if shape in ("прямоугольник", "квадрат"):
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                box_area = cv2.contourArea(box)

                # Filter out noisy/irregular rectangles
                if box_area > 0 and area / box_area > 0.95:
                    rects.append(cnt)
                else:
                    others.append(cnt)
            else:
                others.append(cnt)

    return circles, rects, others


def draw_contours(image, contours, color=(0, 255, 0), thickness=2):
    """
    Draws given contours on an image.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        contours (list[np.ndarray]): Contours to draw.
        color (tuple[int, int, int], optional): BGR color for drawing. Default is green.
        thickness (int, optional): Contour line thickness. Default is 2.

    Returns:
        np.ndarray: Image with contours drawn.

    Raises:
        ValueError: If image is invalid or contours are empty.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid input image for drawing contours")

    output_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    for cnt in contours:
        cv2.drawContours(output_img, [cnt], -1, color, thickness)

    return output_img


def process_image(image_path=None, canny1=100, canny2=255, thresh_min=100, thresh_max=255, area_min=500, area_max=20000):
    """
    Loads an image (custom or default), preprocesses it and extracts contours.

    Args:
        image_path (str, optional): Path to user-provided image. If None, uses skimage.data.coins().
        canny1, canny2 (int): Lower and upper Canny thresholds.
        thresh_min, thresh_max (int): Binary thresholding limits.
        area_min, area_max (int): Minimum and maximum contour area for filtering.

    Returns:
        tuple[np.ndarray, list[np.ndarray]]: 
            - Grayscale image used for contour detection
            - List of detected contours

    Raises:
        FileNotFoundError: If provided image_path does not exist.
        ValueError: If no contours are found.
    """
    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to load image. Check file format or path.")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours = counting_contours(gray, canny1, canny2, thresh_min, thresh_max, area_min, area_max)
        print("Processing user image...")

    else:
        # Default test image from skimage
        image = data.coins()
        gray = cv2.equalizeHist(image)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        contours = counting_contours(gray, 50, 150, 127, 255, 200, 5000)
        print("Processing default coins image...")

    if not contours:
        raise ValueError("No contours detected in image.")
    return image, contours


def main():

    print("Print the image path (Enter for 'coins'):")
    image_path = input().strip() or None

    try:
        # Process image and extract contours
        image, contours = process_image(image_path)

        print(f"Countours found: {len(contours)}")

        # Classify contours
        circles, rects, others = classify_contours(contours, circularity_threshold=0.82)
        print(f"Circles: {len(circles)}, Rectangles: {len(rects)}, Others: {len(others)}")

        # Draw contours
        result_img = draw_contours(image, circles, (0, 255, 0))  # Green = circles
        result_img = draw_contours(result_img, rects, (255, 0, 0))  # Blue = rectangles

        # Save results
        output_path = "data/result.png"
        cv2.imwrite(output_path, result_img)
        print(f"Result saved: {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except TypeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpexted error: {e}")


if __name__ == "__main__":
    main()
