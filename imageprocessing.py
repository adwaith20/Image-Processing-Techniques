import cv2
import numpy as np

def split_image(image):
    height, width, _ = image.shape
    half_height = height // 2
    half_width = width // 2

    # Split the image into four quadrants
    top_left = image[:half_height, :half_width]
    top_right = image[:half_height, half_width:]
    bottom_left = image[half_height:, :half_width]
    bottom_right = image[half_height:, half_width:]

    return top_left, top_right, bottom_left, bottom_right

def display_images(images, window_names):
    for img, name in zip(images, window_names):
        cv2.imshow(name, img)
        cv2.resizeWindow(name, 800, 800)  # Set the window size

    print("Press any key to terminate.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_low_level_features(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Original Image", image_gray)

    # Apply Sobel filter to extract edges
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("Edges (Sobel Filter)", sobel_edges)

    # Apply Laplacian filter to extract edges
    laplacian_edges = cv2.Laplacian(image_gray, cv2.CV_64F)
    laplacian_edges = cv2.normalize(laplacian_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("Edges (Laplacian Filter)", laplacian_edges)

    # Apply Gaussian blur to extract textures
    gaussian_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    cv2.imshow("Gaussian Blur", gaussian_blur)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def contour_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Image with Contours", contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def histogram_equalization(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(image_gray)
    cv2.imshow("Histogram Equalization", equalized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def canny_edge_detection(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image_gray, 100, 200)
    cv2.imshow("Canny Edge Detection", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def thresholding(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresholding", binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def blurring(image):
    blur = cv2.blur(image, (5, 5))
    gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
    median_blur = cv2.medianBlur(image, 5)

    display_images([blur, gaussian_blur, median_blur], ["Average Blurring", "Gaussian Blurring", "Median Blurring"])

def sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    cv2.imshow("Sharpening", sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def prewitt_edge_detection(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prewitt_x = cv2.filter2D(image_gray, -1, np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]))
    prewitt_y = cv2.filter2D(image_gray, -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
    prewitt_edges = cv2.magnitude(prewitt_x, prewitt_y)
    prewitt_edges = cv2.normalize(prewitt_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("Edges (Prewitt Filter)", prewitt_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def harris_corner_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow('Harris Corner Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dilation_and_erosion(image):
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(image, kernel, iterations=1)
    erosion = cv2.erode(image, kernel, iterations=1)
    display_images([dilation, erosion], ["Dilation", "Erosion"])

def main():
    while True:
        print("\nSelect an option:")
        print("1. Split image into quadrants")
        print("2. Extract and display low-level features (edges, textures)")
        print("3. Contour detection")
        print("4. Histogram Equalization")
        print("5. Canny Edge Detection")
        print("6. Thresholding")
        print("7. Blurring")
        print("8. Sharpening")
        print("9. Prewitt Edge Detection")
        print("10. Harris Corner Detection")
        print("11. Dilation and Erosion")
        print("12. Exit")

        choice = input("Enter your choice (1-12): ")

        if choice in {str(i) for i in range(1, 12)}:
            image_path = input("Enter the path to your image: ")
            image = cv2.imread(image_path)
            if image is None:
                print("Failed to load the image.")
                continue

        if choice == '1':
            top_left, top_right, bottom_left, bottom_right = split_image(image)
            display_images([top_left, top_right, bottom_left, bottom_right], ["Top Left", "Top Right", "Bottom Left", "Bottom Right"])

        elif choice == '2':
            extract_low_level_features(image)

        elif choice == '3':
            contour_image(image)

        elif choice == '4':
            histogram_equalization(image)

        elif choice == '5':
            canny_edge_detection(image)

        elif choice == '6':
            thresholding(image)

        elif choice == '7':
            blurring(image)

        elif choice == '8':
            sharpening(image)

        elif choice == '9':
            prewitt_edge_detection(image)

        elif choice == '10':
            harris_corner_detection(image)

        elif choice == '11':
            dilation_and_erosion(image)

        elif choice == '12':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 12.")

if __name__ == "_main_":
    main()