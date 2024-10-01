# Matlab Code

The code is divided into two parts:
1. Selecting the image for processing the projection.
2. Processing the projection of the selected image.

The dataset used is available [here](https://figshare.com/articles/dataset/low_resolution_images_dataset/9393062)

## Selected Images script
This script is separate into three part
1. Load the Video (obviously) _this part is not detailed_
2. Create the background model of each camera
3. Create the foreground of person and save the images


### Create the background model

The background model is computed by averaging all images when the scene has no one present. (l. 24 to 34). 

### Create the foreground of person and save the images.

For each pixel in a new frame, a correlation coefficient $\rho(p)$ is estimated. It represents the correlation between the pixel of the captured image and the corresponding pixel of the background model within the sliding window around the concerned pixel:
$$\rho(p)^t = \frac{(\sum_{p' \in \omega(p)} I(p')^t * I_{bg}(p)^t)^2}{(\sum_{p' \in \omega(p)} I(p')^t)^2 * (\sum_{p' \in \omega(p)}I_{bg}(p')^t)^2}$$ 

where $\omega(p)$ is a sliding square window centered at $p$ and $\rho(p)^t$ is the correlation coefficient between captured image pixel $I(p′)^t$ and background image pixel $I_{bg}(p′)^t$ over the pixels in $\omega(p)$. n this step, the pixel can be classified as background or foreground following:
$$FG(p) = \begin{cases} I(p), & \text{if} s = s + 1 \text{and} \varphi(p) < \text{min} \\ 0, & \text{otherwise} 
\end{cases}$$

where $\rho_{min}$ is the correlation threshold fixed between 0 and 1, and $s$ is the number of pixels constructing this foreground.

In this context, the image $FG$ is not utilized; instead, we rely solely on the variable $s$ to determine whether the image is selected or not.
 
In lines 50-51, we define the minimum and maximum thresholds for $s$; each threshold can represent a different value depending on the camera.

```matlab
thresh_min = [400,400,200,200];
thresh_max = [750,750,750,750];
```

## Projection
...