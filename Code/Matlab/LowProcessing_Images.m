clear all; close all; clc;

%Disable PNG Library Warning
warning('off');


%DATASET_PATH = ... % Path to the dataset
DATASET_PATH = '/home/ropouillard/Documents/Thesis_Working/Computation_Autonomous_Low_Consumption/ViewerBLE/data/SortAcquisition/Go/';
%DATASET_PATH = '/home/ropouillard/Documents/Thesis_Working/Simulateur/HNCS/Generate_Dataset/Dataset49/Acquire/';

%DATASET_PATH = '/home/ropouillard/Documents/Thesis_Working/Computation_Autonomous_Low_Consumption/ViewerBLE/data/SortAcquisition/Y/Scenario_1/';

SIZE_IMAGE = 32; % We consider the square image

%Load the Images from Cameras
%folderCamera = dir(fullfile(DATASET_PATH, "Camera*"));
folderCamera = dir(fullfile(DATASET_PATH, "ESP*"));
%folderCamera = dir(fullfile(DATASET_PATH, "Camera*"));
folderCamera = folderCamera([folderCamera.isdir]);

disp('Found Folder:');
for i = 1:length(folderCamera)
    disp(folderCamera(i).name);
end
buff_cameras = cell(length(folderCamera), 1);


for i = 1:length(folderCamera)
    folderPerson = fullfile(DATASET_PATH, folderCamera(i).name);
    subFolderPerson = dir(folderPerson); 
    subFolderPerson = subFolderPerson([subFolderPerson.isdir]); 

    buff_cameras{i} = struct();

    for j = 1:length(subFolderPerson)
        if ~strcmp(subFolderPerson(j).name, '.') && ~strcmp(subFolderPerson(j).name, '..')
            disp(['- ', subFolderPerson(j).name]);

            personFolder = subFolderPerson(j).name;
            buff_cameras{i}.(personFolder) = {};  
            
            subFolderPersonPath = fullfile(folderPerson, personFolder);
            pngFiles = dir(fullfile(subFolderPersonPath, '*.png'));
            
            if ~isempty(pngFiles)
                for k = 1:length(pngFiles)
                    img = imread(fullfile(subFolderPersonPath, pngFiles(k).name));
                    buff_cameras{i}.(personFolder){end+1} = img; 
                end
            end
        end
    end
end

Nb_Camera = length(buff_cameras);
Nb_Person = length(fieldnames(buff_cameras{1}));
Camera = cell(Nb_Camera, Nb_Person);
for i = 1:Nb_Camera
    peopleFields = fieldnames(buff_cameras{i});
    
    for j = 1:Nb_Person
        personName = peopleFields{j}; 
        images = buff_cameras{i}.(personName); 
        for k = 1:size(images,2)
            Camera{i,j}(k,:)= images(:,k);  
        end
    end
end


%% Create the background model
background_model = cell(Nb_Camera,Nb_Person);
interest = [20,20,20,13];
%interest = [2,2,2,2];
ref      = [1,1,2,8];
%ref      = [1,1,1,1];

for i = 1:Nb_Camera
    for j = 1:Nb_Person
        background_model{i,j} = zeros(SIZE_IMAGE,SIZE_IMAGE);
        if ~isempty(Camera{i,j})
            for k = 1:interest(i)
                %frame = double(im2gray(readFrame(Camera{i,ref(i)})));
                frame = double(im2gray(Camera{i,j}{ref(i),1}));
                background_model{i,j} = background_model{i,j} + frame;
            end
            background_model{i,j} = mat2gray(background_model{i,j} / interest(i));
        end
    end
end

%% Display the background model
figure("Name", "Background Model");
for i = 1:Nb_Camera
    for j = 1:Nb_Person 
        subplot(Nb_Camera,Nb_Person,j + (i-1) * Nb_Person);
        imshow(background_model{i,j});
    end
end

%% Create the foreground model and save the images into the folder
rho_min = 0.99;
omega   = 3; alpha = 0.01;
half_omega = round(omega / 2);
%output_folder = ['./SimulatorDataset/Example/', int2str(omega), '_', num2str(rho_min), '/'];
%output_folder = ['./Scenario1/'];
output_folder = ['./NewDataset/'];
%output_folder = ['./SimulatorDataset/']
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end


%% Blender 
%thresh_min = [150,150,100,60];
%thresh_max = [200,200,200,200];
%% Romain
%thresh_min = [200,300,200,200]; % New Dataset
%thresh_min = [200,100,180,200]; % Scenario 1
%thresh_max = [500,600,900,750];
%% Lobna
thresh_min = [200,200,200];
thresh_max = [500,500,500];

for i = 1:Nb_Camera
    camera_folder = [output_folder, '/video', int2str(i)];
    %if ~exist(camera_folder, 'dir')
    %    mkdir(camera_folder);
    %end
    for j = 1:Nb_Person
        cnt = 0;
        if ~isempty(Camera{i,j})
            person_folder = [camera_folder, '/images/p', int2str(j)];
            mask_folder   = [camera_folder, '/mask/p', int2str(j)];
            dataset_folder = [camera_folder, '/p', int2str(j)];
            %if ~exist(person_folder, 'dir')
            %    mkdir(person_folder);
            %end
            %if ~exist(mask_folder, 'dir')
            %    mkdir(mask_folder)
            %end
            if ~exist(dataset_folder, 'dir')
                mkdir(dataset_folder);
            end
            mean_thresh = 0;
            for k = 1:size(Camera{i,j},1)
                frame = double(im2gray(Camera{i,j}{k,1}));
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
                if k < 6 
                    mean_thresh = mean_thresh + pixel;
                end
                if pixel > thresh_min(i) && pixel < thresh_max(i)
                    cnt = cnt + 1;
                    pathfile = [person_folder, '/', int2str(j), '-p',sprintf('%02d',cnt), '.png'];
                    %imwrite(frame, pathfile);

                    % Save mask
                    pathfile = [mask_folder, '/', int2str(j), '-p',sprintf('%02d',cnt), '-mask',sprintf('%03d', pixel),'.png'];
                    %imwrite(mask, pathfile);

                    pathfile = [dataset_folder, '/', int2str(j), '-p',sprintf('%02d',cnt), '.png'];
                    imwrite(frame, pathfile);
                end
                background_model{i,j} = ((1-alpha) * background_model{i,j}) + (alpha * frame);
            end
            mean_thresh = mean_thresh / 5;
            disp(['Person ', int2str(j), ' camera ',int2str(i),' | mean_segmentation = ', int2str(mean_thresh)])
        end
        fprintf('%d - %d - %d\n', i, j, cnt);
    end
end


figure("Name", "Background Model Ending");
for i = 1:Nb_Camera
    for j = 1:Nb_Person 
        subplot(Nb_Camera,Nb_Person,j + (i-1) * Nb_Person);
        imshow(background_model{i,j});
    end
end