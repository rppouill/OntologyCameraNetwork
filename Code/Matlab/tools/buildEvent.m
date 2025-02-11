function dataSet = buildEvent(varargin)
    %if nargin < 2
    %    NB_VECTOR = 5;
    %end
    camera_param    = varargin{1};
    NB_VECTOR       = varargin{2};
    D               = varargin{3};

    [NB_CAMERA, NB_PERSON] = size(camera_param);
    if size(camera_param{1,1},3) > 1
        [H,W,~] = size(camera_param{1,1});
        camera = cellfun(@(x) reshape(x, H*W, []), camera_param, 'UniformOutput', false);
    else
        H = 30; W = 30;
        camera = camera_param;
    end
    
    if nargin > 3
        if strcmp(varargin{4}, 'gradient')
            for i = 1:NB_CAMERA
                for j = 1:NB_PERSON
                    X = reshape(camera_param{i,j}, H, W, []);
                    [F1, F2, F3] = gradient(X);
                    grad = sqrt(F1.^2 + F2.^2 + F3.^2);
                    camera{i,j} = reshape(grad, [], size(grad,3));
                end
            end
        end
    end

    % Mean Image % 
    mean_target = cell(NB_CAMERA,NB_PERSON);
    for i = 1:NB_CAMERA
        for j = 1:NB_PERSON
            for k = 1:size(camera{i,j},2)
                camera{i,j}(:,k) = camera{i,j}(:,k) * D{i,j}(k); 
            end

            mean_target{i,j} = transpose(mean(transpose(camera{i,j})));
        end
    end

    % Zero Mean %
    zero_mean = cell(NB_CAMERA,NB_PERSON);
    for i = 1:NB_CAMERA
        for j = 1:NB_PERSON
            for k = 1:size(camera{i,j},2)
                zero_mean{i,j}(:,k) = camera{i,j}(:,k) - mean_target{i,j};
            end
        end
    end

    % PCA %
    dataSet = cell(NB_CAMERA, NB_PERSON);
    for i = 1:NB_CAMERA
        for j = 1:NB_PERSON
            [~,dataSet{i,j}] = Turk_Pentland(zero_mean{i,j}, NB_VECTOR);
        end
    end


end