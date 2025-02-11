clear all; close all; clc;


%DATASET_PATH = ... % Path to the dataset
DATASET_PATH = '/home/ropouillard/Documents/Thesis_Working/DataSet/Lobna_Dataset/9393062/';
%DATASET_PATH = '/home/ropouillard/Documents/Thesis_Working/DataSet/Lobna_Dataset/9393062-Rearange/';
Nb_Camera = 4;
Nb_Person = 9;
SIZE_IMAGE = 30; % We consider the square image

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
background_model = cell(Nb_Camera,Nb_Person);
interest = [6,17,6,13];
%ref      = [1,1,2,8];
ref      = [1,1,1,1];

for i = 1:Nb_Camera
    for j = 1:Nb_Person
        background_model{i,j} = zeros(Camera{i,ref(i)}.Height, Camera{i,ref(i)}.Width);
        if ~isempty(Camera{i,ref(i)})
            for k = 1:interest(i)
                frame = double(im2gray(readFrame(Camera{i,ref(i)})));
                background_model{i,j} = background_model{i,j} + frame;
            end
            background_model{i,j} = mat2gray(background_model{i,j} / interest(i));
            Camera{i,1}.CurrentTime = 0.0;
        end
    end
end

% Display the background model
figure("Name", "Background Model");
for i = 1:Nb_Camera
    subplot(2,2,i);
    imshow(background_model{i,j});
end

% Create the foreground model and save the images into the folder
rho_min = 0.992; omega   = 3;
alpha = 0.01;
half_omega = round(omega / 2);
outpout_folder = './dataSet/ImageSelectedwithD'; %['./dataSet/Example/', int2str(omega), '_', num2str(rho_min), '/'];
if ~exist(outpout_folder, 'dir')
    mkdir(outpout_folder);
end


thresh_min = [350,400,200,400];
thresh_max = [600,650,350,750];


D = cell(Nb_Camera, Nb_Person);
for i = 1:Nb_Camera
    camera_folder = [outpout_folder, '/video', int2str(i)];
    if ~exist(camera_folder, 'dir')
        mkdir(camera_folder);
    end
    for j = 1:Nb_Person
        cnt = 0; D{i,j} = [];
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
                %for h = 1:SIZE_IMAGE - round(omega/2)
                %    for w = 1:SIZE_IMAGE - round(omega/2)
                %        sum1 = 0; sum2 = 0; sum3 = 0;
                %        for t = 1:omega
                %            for l = 1:omega
                %                sum1 = background_model{i,j}(h+t-1,w+l-1) * frame(h+t-1,w+l-1)                + sum1;
                %                sum2 = background_model{i,j}(h+t-1,w+l-1) * background_model{i,j}(h+t-1,w+l-1)  + sum2;
                %                sum3 = frame(h+t-1,w+l-1)               * frame(h+t-1,w+l-1)                + sum3;
                %            end
                %        end
%
                %        rho = sum1*sum1 / (sum2 * sum3);
                %        if rho < rho_min
                %            pixel = pixel + 1;
                %            mask(h+1,w+1) = 1;
                %        end
                %    end
                %end
                for h = 1:SIZE_IMAGE - omega + 1
                    for w = 1:SIZE_IMAGE - omega + 1
                        % Extraction de la sous-matrice
                        bg_patch = background_model{i,j}(h:h+omega-1, w:w+omega-1);
                        frame_patch = frame(h:h+omega-1, w:w+omega-1);
                
                        % Calcul des sommes
                        sum1 = sum(bg_patch(:) .* frame_patch(:));
                        sum2 = sum(bg_patch(:) .^ 2);
                        sum3 = sum(frame_patch(:) .^ 2);
                
                        % Calcul de rho
                        rho = (sum1^2) / (sum2 * sum3);
                
                        % Mise à jour du masque
                        if rho < rho_min
                            pixel = pixel + 1;
                            mask(h+half_omega, w+half_omega) = 1;
                        end
                    end
                end
                if pixel > thresh_min(i) && pixel < thresh_max(i)
                    cnt = cnt + 1;
                    pathfile = [person_folder, '/', int2str(j), '-p',sprintf('%02d',cnt), '.png'];
                    imwrite(frame, pathfile);
                    % Normalize pixel between thresh min and max
                    
                    D{i,j} = [D{i,j}, pixel];
                    % Save mask
                    %pathfile = [person_folder, '/', int2str(j), '-p',sprintf('%02d',cnt), '-mask.png'];
                    %imwrite(mask, pathfile);
                end
                background_model{i,j} = ((1-alpha) * background_model{i,j}) + (alpha * frame);
            end
        end
        fprintf('%d - %d - %d\n', i, j, cnt);
    end
end


save([outpout_folder,'/D.mat'], 'D');