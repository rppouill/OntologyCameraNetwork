function accuracy = confusionMatrix(varargin)
    eigVec = varargin{1};
    M      = varargin{2};
    poly = false;
    if nargin >= 3 && strcmp(varargin{3}, 'poly')
        poly = true;
        mu = varargin{4};
    end
    [NB_CAMERA, NB_PERSON]  = size(eigVec);
    [NB_CAMERA, NB_PERSON, ~]  = size(M);
     DIM = length(size(eigVec{1,1}));
     N_VECTOR_EXAMPLE       = size(eigVec{1,1}, DIM);


    accuracy = zeros(NB_CAMERA, NB_CAMERA);
    for cameraSource = 1:NB_CAMERA
        for cameraTarget = 1:NB_CAMERA
            cnt = 0;
            for person = 1:NB_PERSON
                if isempty(eigVec{cameraSource,person}) || isempty(eigVec{cameraTarget,person})
                    continue;
                end
                cnt = cnt + 1;
                % 1/ Init Camera
                P_Source = eigVec{cameraSource,person}(:,1:N_VECTOR_EXAMPLE);

                if poly
                    P_Target = polyval(M{cameraSource,cameraTarget,N_VECTOR_EXAMPLE}, P_Source, [], mu{cameraSource,cameraTarget});
                else
                    P_Target = M{cameraSource,cameraTarget,N_VECTOR_EXAMPLE} * P_Source;
                end
                
                %P_Bad    = cell(NB_CAMERA - 2, 1);
                %for badCamera = 1:NB_CAMERA
                %    if badCamera == cameraSource || badCamera == cameraTarget
                %        continue;
                %    end
                %    if poly
                %        P_Bad{badCamera} = polyval(M{cameraSource,badCamera,N_VECTOR_EXAMPLE}, P_Source, [], mu{cameraSource,badCamera});
                %    else
                %        P_Bad{badCamera} = M{cameraSource,badCamera,N_VECTOR_EXAMPLE} * P_Source;
                %    end
                %end

                %2/ Good Camera
                P_Good = eigVec{cameraTarget,person}(:,1:N_VECTOR_EXAMPLE);
                rightDist = mse(P_Target, P_Good);

                %3/ Bad Camera
                wrongDist = zeros(NB_CAMERA,NB_PERSON,1) * inf;
                for badCamera = 1:NB_CAMERA
                    if (badCamera == cameraSource || badCamera == cameraTarget)
                        continue;
                    end
                    for i = 1:NB_PERSON
                        if i == person || isempty(eigVec{badCamera,i})
                            continue;
                        end
                        if poly
                            P_Bad = polyval(M{badCamera,cameraTarget,N_VECTOR_EXAMPLE}, eigVec{badCamera, i}, [], mu{badCamera,cameraTarget});
                        else
                            P_Bad = M{badCamera,cameraTarget,N_VECTOR_EXAMPLE} * eigVec{badCamera, i};
                        end
                        P_Wrong = eigVec{badCamera,i};
                        wrongDist(badCamera,i) = mse(P_Bad, P_Wrong);
                    end
                end
                [minValue, minIdx]  = min(wrongDist);
                [minValue, ~]       = min(minValue);
                class  = (minValue < rightDist) * minIdx(1) + (minValue >= rightDist) * cameraTarget;
                accuracy(cameraSource,cameraTarget) = accuracy(cameraSource,cameraTarget) + (class == cameraTarget);
            end
            accuracy(cameraSource,cameraTarget) = accuracy(cameraSource,cameraTarget) / cnt;
        end
    end
end