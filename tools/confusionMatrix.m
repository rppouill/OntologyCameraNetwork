function accuracy = confusionMatrix(varargin)
    eigVec = varargin{1};
    M      = varargin{2};
    poly = false;
    if nargin >= 3 && strcmp(varargin{3}, 'poly')
        poly = true;
    end
    [NB_CAMERA, NB_PERSON]  = size(eigVec);
     N_VECTOR_EXAMPLE       = size(eigVec{1,1}, 2);

    if nargin < 3
        bias = zeros(size(eigVec{1,1},1),1);
    end

    accuracy = zeros(NB_CAMERA, NB_CAMERA);
    for cameraSource = 1:NB_CAMERA
        for cameraTarget = 1:NB_CAMERA
            %if cameraSource == cameraTarget
            %    continue;
            %end

            for person = 1:NB_PERSON
                % 1/ Init Camera
                P_Source = eigVec{cameraSource,person}(:,1:N_VECTOR_EXAMPLE);

                if poly
                    P_Target = polyval(M{cameraSource,cameraTarget,N_VECTOR_EXAMPLE}, P_Source);
                else
                    P_Target = M{cameraSource,cameraTarget,N_VECTOR_EXAMPLE} * P_Source;
                end
                
                P_Bad    = cell(NB_CAMERA - 2, 1);
                for badCamera = 1:NB_CAMERA
                    if badCamera == cameraSource || badCamera == cameraTarget
                        continue;
                    end
                    if poly
                        P_Bad{badCamera} = polyval(M{cameraSource,badCamera,N_VECTOR_EXAMPLE}, P_Source);
                    else
                        P_Bad{badCamera} = M{cameraSource,badCamera,N_VECTOR_EXAMPLE} * P_Source;
                    end
                end

                %2/ Good Camera
                P_Good = eigVec{cameraTarget,person}(:,1:N_VECTOR_EXAMPLE);
                rightDist = norm(P_Good - P_Target);

                %3/ Bad Camera
                wrongDist = zeros(NB_CAMERA - 2,NB_PERSON,1);
                cnt_pass = 0;
                for badCamera = 1:NB_CAMERA
                    if badCamera == cameraSource || badCamera == cameraTarget
                        cnt_pass = cnt_pass + 1;
                        continue;
                    end
                    for i = 1:NB_PERSON
                        P_Wrong = eigVec{badCamera,i}(:,1:N_VECTOR_EXAMPLE);
                        wrongDist(badCamera - cnt_pass,i) = norm(P_Wrong - P_Bad{badCamera});
                    end
                end
                [minValue, minIdx]  = min(wrongDist);
                [minValue, ~]       = min(minValue);
                class  = (minValue < rightDist) * minIdx(1) + (minValue >= rightDist) * cameraTarget;
                
                accuracy(cameraSource,cameraTarget) = accuracy(cameraSource,cameraTarget) + (class == cameraTarget);
            end
        end
    end
    
    accuracy = accuracy / NB_PERSON;
end