# Matlab Code

The code is divided into two parts:
1. Selecting the image for processing the projection.
2. Processing the projection of the selected image.

The dataset used is available here [[1](https://figshare.com/articles/dataset/low_resolution_images_dataset/9393062)]

## Low Processing action: Selected Images
`matlab -nodesktop -r "LowProcessing; quit"`
This script is separate into three part
1. Load the Video (obviously) _this part is not detailed_
2. Create the background model of each camera
3. Create the foreground of person and save the images

The dataset is stored in the `Camera` variable, which is of type `Cell`. Its array size is `Nb_Camera*Nb_Person`. To access the video of person 3 from camera 2, you would use `Camera{2, 3}`.

### Create the background model

The background model is computed by averaging all images when the scene has no one present. (l. 24 to 34). 

For each camera, the number of images to be averaged is different and is defined by the `interest` variable (line 22).

```matlab
interest = [6,17,6,13];
```

### Create the foreground of person and save the images.

For each pixel in a new frame, a correlation coefficient $\rho(p)$ is estimated. It represents the correlation between the pixel of the captured image and the corresponding pixel of the background model within the sliding window around the concerned pixel:

$$\rho(p)^t = \frac{(\sum_{p' \in \omega(p)} I(p')^t * I_{bg}(p)^t)^2}{(\sum_{p' \in \omega(p)} I(p')^t)^2 * (\sum_{p' \in \omega(p)}I_{bg}(p')^t)^2}$$ 

where $\omega(p)$ is a sliding square window centered at $p$ and $\rho(p)^t$ is the correlation coefficient between captured image pixel $I(p′)^t$ and background image pixel $I_{bg}(p′)^t$ over the pixels in $\omega(p)$. n this step, the pixel can be classified as background or foreground following: 

$$FG(p) = \begin{cases} I(p), & \text{if} s = s + 1 \text{and} \varphi(p) < \text{min} \\ 
0, & \text{otherwise} \end{cases}$$

where $\rho_{min}$ is the correlation threshold fixed between 0 and 1, and $s$ is the number of pixels constructing this foreground.

In this context, the image $FG$ is not utilized; instead, we rely solely on the variable $s$ to determine whether the image is selected or not.
 
In lines 50-51, we define the minimum and maximum thresholds for $s$; each threshold can represent a different value depending on the camera.

```matlab
thresh_min = [400,400,200,200];
thresh_max = [750,750,750,750];
```

## Medium Processing action: Projection of Latent Space
`matlab -nodesktop -r "MediumProcessing; quit"`
The medium processing is defined through three distinct actions:
1. Transform data sources
2. Compute the Latent Space
3. Project the data

### Data transformation
To maximize the representation of the latent space, we do not use the raw images directly. Instead, we calculate the mean of each event and subtract this average from each image within that event. This normalization process helps to emphasize the variations in the data, ensuring that the latent space effectively captures the underlying features and patterns.

$$\Phi_i = \Gamma_i - \Psi$$

With $\Phi_i$ representing the normalized event of $\Gamma_i$ by $\Psi$.

This section of the code is located between lines 28 and 44.
$\Psi$: 
```matlab
mean_target = cell(NB_CAMERA,NB_PERSON);
for i = 1:NB_CAMERA
    for j = 1:NB_PERSON
        mean_target{i,j} = transpose(mean(double(transpose(camera{i,j}))));
    end
end
```

$\Phi$:
```matlab
zero_mean = cell(NB_CAMERA,NB_PERSON);
for i = 1:NB_CAMERA
    for j = 1:NB_PERSON
        for k = 1:size(camera{i,j},2)
            zero_mean{i,j}(:,k) = camera{i,j}(:,k) - mean_target{i,j};
        end
    end
end
```

### Compute the Latent Space
We utilized a linear transformation to compute the latent space, applying Principal Component Analysis (PCA) as outlined by the Turk-Pentland method. This approach allows us to effectively reduce the dimensionality of the data while preserving its essential features. By projecting the original data into a lower-dimensional space, we can better identify patterns and relationships within the dataset.

The method is implemented in the tool `Turk_Pentland`



The two previous methods are described by Turk and Pentland in [[2](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=139758)] [[3](https://watermark.silverchair.com/jocn.1991.3.1.71.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA10wggNZBgkqhkiG9w0BBwagggNKMIIDRgIBADCCAz8GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMC1LGwXyLrXOK9hIcAgEQgIIDEF-1Hr_9nGfns_R4WUXkgnLhK9supUaOmthC5mHJwPkMqeB1vQOjeEMlsglM66IoNgwMoDoeUGkJYYxzgbgExVmh4dTzxFe8Yp2XRJdSSX0ulORzaHDTbDm9nUaMEtmvxR0oGlhKfC-2vMWAQTwz63kIABYPqyDisIPMJGo5EFCwR6fywz85WO2Pc_Ib3ZVMjdra7qQMQx23Stv0efe6RMzOs8mPB5ggZ9SZ0FDvXIO-WHvYLXcq62Z84sLYV0ExWRyr8tc6wDu0EIzRIDz9OXE9ni6QLh_6hyWTU-_dUS1WmVSIXUZzMzlmLQpIwJdahNh_f1rueYZl1DcGzh8BTkQ16ULDnB1flPqCtEegHXm5JcUE5qWi6Jagdz2HkWSV1qoooK6MGuyrf-BP9Efdj6-lkD_5m_a-J52K-DWNlMEI8LG8vmXj6OwxRW_Ug2bulrQCcLpYotdMg4egsmFqqMxf_vi2NrrdA27vCjCaG0F9XYWV_NIf6qGearBX3ebQTw7Ew66OzO-NWPGp_H_32HNJpGW_B3bwxAOGAjOZ9a3_uGwEdTH3hAInBr6W8Xl1XBv5AMjh-69vCwAU_9_Jz9H3DN65biYrjpIi-qZcDPLcwzGxEvxRoUKuM-x1SrFFSP2C3bRMAo6-cxi8jfiStBP6DGwJhIoMpfnqgalg1tAmDoP9YnOLiXWkh-lsrpxgQ0g11TPLzIEJu10H-VAFo7IKOVUpadqNkDiA2Nr2LPZzfHHoaUshqwq3LuTqAxCAKasT4zpNElJiG6wrPOzrnZ7I1h5QXt9RX4LRPKB3BdrbnOQuvbNm6kVBnVoIcayZmbxQxOZZt6bspbGM1KogyQCiTga1TmO76_visV9-sxXgTfveG3_q1UqzkCS-gEULZS6XOItMyk3P4Ys42Qs9lwp3K0jzy_HgF8EJW-2Xebnn8UtV9um96rBVfYkxomL7UbbhFidSsmxEmi7IFZrcu8zA2UMzPalnCc-BWHJqlKVT2r2qMdqRUrEOfMwjH91lBDtqsZs_3MEIbqWnmDV0k7U)].





## References
[[1](https://figshare.com/articles/dataset/low_resolution_images_dataset/9393062)] Lobna Ben Khelifa, François Berry and Jean-Charles Quiton, Low Resolution Images Dataset.

[[2](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=139758)] Matthew Turk and Alex Pentland, Face recognition using eigenfaces, 1991




[[3](https://watermark.silverchair.com/jocn.1991.3.1.71.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA10wggNZBgkqhkiG9w0BBwagggNKMIIDRgIBADCCAz8GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMC1LGwXyLrXOK9hIcAgEQgIIDEF-1Hr_9nGfns_R4WUXkgnLhK9supUaOmthC5mHJwPkMqeB1vQOjeEMlsglM66IoNgwMoDoeUGkJYYxzgbgExVmh4dTzxFe8Yp2XRJdSSX0ulORzaHDTbDm9nUaMEtmvxR0oGlhKfC-2vMWAQTwz63kIABYPqyDisIPMJGo5EFCwR6fywz85WO2Pc_Ib3ZVMjdra7qQMQx23Stv0efe6RMzOs8mPB5ggZ9SZ0FDvXIO-WHvYLXcq62Z84sLYV0ExWRyr8tc6wDu0EIzRIDz9OXE9ni6QLh_6hyWTU-_dUS1WmVSIXUZzMzlmLQpIwJdahNh_f1rueYZl1DcGzh8BTkQ16ULDnB1flPqCtEegHXm5JcUE5qWi6Jagdz2HkWSV1qoooK6MGuyrf-BP9Efdj6-lkD_5m_a-J52K-DWNlMEI8LG8vmXj6OwxRW_Ug2bulrQCcLpYotdMg4egsmFqqMxf_vi2NrrdA27vCjCaG0F9XYWV_NIf6qGearBX3ebQTw7Ew66OzO-NWPGp_H_32HNJpGW_B3bwxAOGAjOZ9a3_uGwEdTH3hAInBr6W8Xl1XBv5AMjh-69vCwAU_9_Jz9H3DN65biYrjpIi-qZcDPLcwzGxEvxRoUKuM-x1SrFFSP2C3bRMAo6-cxi8jfiStBP6DGwJhIoMpfnqgalg1tAmDoP9YnOLiXWkh-lsrpxgQ0g11TPLzIEJu10H-VAFo7IKOVUpadqNkDiA2Nr2LPZzfHHoaUshqwq3LuTqAxCAKasT4zpNElJiG6wrPOzrnZ7I1h5QXt9RX4LRPKB3BdrbnOQuvbNm6kVBnVoIcayZmbxQxOZZt6bspbGM1KogyQCiTga1TmO76_visV9-sxXgTfveG3_q1UqzkCS-gEULZS6XOItMyk3P4Ys42Qs9lwp3K0jzy_HgF8EJW-2Xebnn8UtV9um96rBVfYkxomL7UbbhFidSsmxEmi7IFZrcu8zA2UMzPalnCc-BWHJqlKVT2r2qMdqRUrEOfMwjH91lBDtqsZs_3MEIbqWnmDV0k7U)] Matthew Turk and Alex Pentland, Eigenfaces for Recognition, 1991
