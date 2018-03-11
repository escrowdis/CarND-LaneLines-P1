# **Finding Lane Lines on the Road**

This is the lane line detection project by using single color camera.

[img_rgb]: ./imgs/img_rgb.png "RGB color image"
[img_roi]: ./imgs/img_roi.png "ROI in image"
[img_hsl_h]: ./imgs/img_hsl_h.png "Hue in HSL"
[img_hsl_s]: ./imgs/img_hsl_s.png "Saturation in HSL"
[img_hsl_l]: ./imgs/img_hsl_l.png "Lightness in HSL"
[img_hsv_h]: ./imgs/img_hsv_h.png "Hue in HSV"
[img_hsv_s]: ./imgs/img_hsv_s.png "Saturation in HSV"
[img_hsv_v]: ./imgs/img_hsv_v.png "Value in HSV"
[img_yellow]: ./imgs/img_yellow.png "Detection of Yellow Lane Lines"
[img_white]: ./imgs/img_white.png "Detection of White Lane Lines"
[img_lanes]: ./imgs/img_lanes.png "Detection of Lane Lines"
[img_canny]: ./imgs/img_canny.png "Canny Edge Method"
[img_hough]: ./imgs/img_hough.png "Hough Line Detection"
[img_left_right_lanes]: ./imgs/img_left_right_lanes.png "Both Sides Lanes"
[img_result]: ./imgs/img_result.png "Result"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 4 steps:
1. Cropped the ROI (region of interest) in image and converted to HSL color space
2. Filtered out lane lines by color
3. Detected lane lines using Canny edge and Hough line methods
4. Distinguished left-side and right-side lane lines and averaged it to draw

First, the images were cropped with a tuned quadrilateral to focus on related lane lines in front of.

![][img_rgb]
![][img_roi]

```python
rows = img.shape[0]
cols = img.shape[1]
roi_y_min = rows * 0.58
roi_x_min = cols * 0.45
roi_x_max = cols * 0.55

vertices = np.array([[(0, rows),
                      (roi_x_min, roi_y_min),
                      (roi_x_max, roi_y_min),
                      (cols, rows)]], dtype=np.int32)
img_roi = region_of_interest(img_rgb, vertices)
```

And converted it to HSL (Hue, Saturation and Lightness) color space. I've tested to threshold out lane lines in RGB, HSV and HSL color spaces and I thought both of HSV and HSL color spaces would perform better than RGB one under different situations, such as shaded area, low light region...etc.

#### Images in Different Color Space
|     HSL      |     HSV      |
|--------------|--------------|
|![][img_hsl_h]|![][img_hsv_h]|
|![][img_hsl_s]|![][img_hsv_s]|
|![][img_hsl_l]|![][img_hsv_v]|

Second, the common lane lines, color in white and yellow, were filtered out by tuned thresholding. In the image, the lane line in yellow have strong responses in saturation, and the one in white has high lightness. So the thresholding was implemented shown as below.

![][img_yellow]
![][img_white]
![][img_lanes]

```python
# yellow lane
y_lower = np.array([10, 0, 0])
y_upper = np.array([50, 255, 255])
img_y = cv2.inRange(img_hsl, y_lower, y_upper)

# white lane
w_lower = np.array([0, 200, 0])
w_upper = np.array([255, 255, 255])
img_w = cv2.inRange(img_hsl, w_lower, w_upper)

img_result = cv2.bitwise_or(img_y, img_w)
```

Third, Canny edge and Hough line methods were used to extract the lane lines. The thresholds of Canny edge could be both 255 due to the input image was the binary image with detected lane lines. After that, the parameters of Hough line detection was adjusted to fit current situation.

![][img_canny]
![][img_hough]

```python
# Canny edge
thresh_canny_low = 50
thresh_canny_high = 150
img_canny = canny(img_th, thresh_canny_low, thresh_canny_high)

# Hough line detection
rho = 1
theta = np.pi/180
threshold = 20
min_line_length = 20
max_line_gap = 20
lines, img_hough = hough_lines(img_canny, rho, theta, threshold,
                               min_line_length, max_line_gap)
```


Forth, the lines have to be merged into one to represent the lane lines for each side. The lines were filtered and classified into left-side and right-side by slopes. The line fitting was implemented to find a line with minimum deviation. Finally, the lines were drawn in the ROI by finding the upper and lower point inside the ROI.

![][img_left_right_lanes]

```python
# Distinguish left- and right-side lane line
slope_lower = 0.4
slope_upper = 2.0
lane_left = []
lane_right = []
for line in lines:
    for x1, y1, x2, y2 in line:
        slope = (y2 - y1) / (x2 - x1)

    # Distinguish by checking slope
        if slope_lower <= slope <= slope_upper:
            lane_right.append((x1, y1))
            lane_right.append((x2, y2))
        elif -slope_upper <= slope <= -slope_lower:
            lane_left.append((x1, y1))
            lane_left.append((x2, y2))

# Line fitting
def lineFitting(x, y):
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    k = sum((x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y)) /
        sum((x_i - mean_x)**2 for x_i in x)
    m = mean_y - k * mean_x
    return k, m

merged_line_img = np.zeros((rows, cols, 3), dtype=np.uint8)
thick_line = 10
color_line = [255, 0, 0]

pts_left = np.array(lane_left)
if pts_left.size:
    slope_l, intercept_l = lineFitting(pts_left[:, 0], pts_left[:, 1])
    pt_left_y = [roi_y_min, rows]
    line_left_draw = np.array([[(pt_left_y[0] - intercept_l) / slope_l,
                                 pt_left_y[0],
                                (pt_left_y[1] - intercept_l) / slope_l,
                                 pt_left_y[1]]]).astype(int)
    draw_line(merged_line_img, line_left_draw, color_line, thick_line)

pts_right = np.array(lane_right)
if pts_right.size:
    slope_r, intercept_r = lineFitting(pts_right[:, 0], pts_right[:, 1])
    pt_right_y = [roi_y_min, rows]
    line_right_draw = np.array([[(pt_right_y[0] - intercept_r) / slope_r,
                                  pt_right_y[0],
                                 (pt_right_y[1] - intercept_r) / slope_r,
                                  pt_right_y[1]]]).astype(int)
    draw_line(merged_line_img, line_right_draw, color_line, thick_line)

return weighted_img(merged_line_img, img)
```

![][img_result]

### 2. Identify potential shortcomings with your current pipeline

This implementation still has lots of corner cases to be test. There are some potential shortcoming in this implementation listed as below:

- The strong constraint is that the driving lane lines should be inside ROI, so this method only works on straight lanes like driving on highway.

- If the lightness is inconsistent or the region was shaded, it might affect the line detection performance.

- The lane lines were drawn by finding the upper and lower bound of ROI, so the line drawn on image might not fit well if there is a curve lane in front.

### 3. Suggest possible improvements to your pipeline

I think using machine learning probably a good idea to detect lane lines because I am rookie in this learning-based methods. Or the image could be transform image to top-view to get further information.
