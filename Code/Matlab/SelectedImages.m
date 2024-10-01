clear all; close all; clc;


DATASET_PATH = ... % Path to the dataset
%DATASET_PATH = '/home/ropouillard/Documents/Thesis_Working/DataSet/Lobna_Dataset/9393062/'
Nb_Camera = 4;
Nb_Person = 9;

% Load the video
Camera = cell(Nb_Camera,Nb_Person);
for i = 1:Nb_Camera
    for j = 1:Nb_Person
        pathfile = [DATASET_PATH, int2str(i), '-person', int2str(j), '.avi'];
        if exist(pathfile, 'file')
            Camera{i,j} = VideoReader(pathfile);
        else 
            Camera{i,j} = [];
        end
    end
end

% Create the background model
background_model = cell(Nb_Camera,1);
interest = [6,17,6,13];

for i = 1:Nb_Camera
    background_model{i} = zeros(Camera{i,1}.Height, Camera{i,1}.Width);
    if ~isempty(Camera{i,1})
        for k = 1:interest(i)
            frame = double(im2gray(readFrame(Camera{i,1})));
            background_model{i} = background_model{i} + frame;
        end
        background_model{i} = mat2gray(background_model{i} / interest(i));
        Camera{i,1}.CurrentTime = 0.0;
    end
end

% Display the background model
figure("Name", "Background Model");
for i = 1:Nb_Camera
    subplot(2,2,i);
    imshow(background_model{i});
end

% Create the foreground model and save the images into the folder
outpout_folder = './dataSet/ImageSelected/';
if ~exist(outpout_folder, 'dir')
    mkdir(outpout_folder);
end

rho_min = 0.992;
thresh_min = [400,400,200,200];
thresh_max = [750,750,750,750];
for i = 1:Nb_Camera
    camera_folder = [outpout_folder, '/video', int2str(i)];
    if ~exist(camera_folder, 'dir')
        mkdir(camera_folder);
    end
    for j = 1:Nb_Person
        cnt = 0;
        if ~isempty(Camera{i,j})
            person_folder = [camera_folder, '/p', int2str(j)];
            if ~exist(person_folder, 'dir')
                mkdir(person_folder);
            end
            Camera{i,j}.CurrentTime = 0.0;
            while hasFrame(Camera{i,j})
                frame = double(im2gray(readFrame(Camera{i,j})));
                frame = mat2gray(frame);
                mask = zeros(size(frame));
                pixel = 0;
                % Compute the ad-hoc foreground
                for h = 1:28
                    for w = 1:28
                        sum1 = 0; sum2 = 0; sum3 = 0;
                        for t = 1:3
                            for l = 1:3
                                sum1 = background_model{i}(h+t-1,w+l-1) * frame(h+t-1,w+l-1) + sum1;
                                sum2 = background_model{i}(h+t-1,w+l-1) * background_model{i}(h+t-1,w+l-1) + sum2;
                                sum3 = frame(h+t-1,w+l-1) * frame(h+t-1,w+l-1) + sum3;
                            end
                        end

                        rho = sum1*sum1 / (sum2 * sum3);
                        if rho < rho_min
                            pixel = pixel + 1;
                            mask(h+1,w+1) = 1;
                        end
                    end
                end
                if pixel > thresh_min(i) && pixel < thresh_max(i)
                    cnt = cnt + 1;
                    pathfile = [person_folder, '/', int2str(j), '-p',sprintf('%02d',cnt), '.png'];
                    imwrite(frame, pathfile);

                    % Save mask
                    pathfile = [person_folder, '/', int2str(j), '-p',sprintf('%02d',cnt), '-mask.png'];
                    imwrite(mask, pathfile);
                end
            end
        end
        fprintf('%d - %d - %d\n', i, j, cnt);
    end
end
