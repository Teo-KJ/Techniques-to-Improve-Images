%% 3.1a View image
% import image
macritchie = imread('maccropped.jpg');
macritchie = rgb2gray(macritchie); % convert to grayscale
imshow(macritchie)

%% 3.1b Create Sobel masks
% generate both vertical and horizontal 3-by-3 sobel masks
mask_Vertical = [-1, 0, 1;
                 -2, 0, 2;
                 -1, 0, 1];
mask_Horizontal = [-1, -1, -1;
                   0, 0, 0;
                   1, 1, 1];

% convolute the sobel masks on the image, with vertical, horizontal and
% both vertical and horizontal masks
macritchie_sobel_Vertical = conv2(macritchie, mask_Vertical);
macritchie_sobel_Horizontal = conv2(macritchie, mask_Horizontal);
macritchie_sobel_Both = conv2(macritchie_sobel_Vertical, mask_Horizontal);

figure, imshow(uint8(macritchie_sobel_Vertical))
figure, imshow(uint8(macritchie_sobel_Horizontal))
figure, imshow(uint8(macritchie_sobel_Both))

%% 3.1c Squaring vertical and horizontal edges
% generate the combined edge image
% values are squared because there are negative pixels
combined_edge = macritchie_sobel_Vertical.^2 + macritchie_sobel_Horizontal.^2;

figure, imshow(uint8(combined_edge))
figure, imshow(uint8(sqrt(combined_edge))) % root the image

%% 3.1d Thresholding
% generate image of edges with different threshold values
% Et = combined_edge > 100;
% figure, imshow(Et)
figure, imshow(combined_edge > 100)
figure, imshow(combined_edge > 1000)
figure, imshow(combined_edge > 10000)
figure, imshow(combined_edge > 50000)
figure, imshow(combined_edge > 100000)

%% 3.1e Canny Edge Detection
tl=0.04; th=0.1; sigma=1.0;
E = edge(macritchie, 'canny', [tl th], sigma); % apply Canny Edge detection
figure, imshow(E)

%% 3.1e (i) Trying out with different sigma values
s = 5;

% loop to generate image after applying Canny Edge at different sigma
% values of [1,5]
for sigma = 1:s
    cannyDiffSigma = edge(macritchie, 'canny', [tl th], sigma);
    figure, imshow(cannyDiffSigma)
end

%% 3.1e (ii) Changing the values of tl
valuesList = {0.01, 0.04, 0.08};

% loop to generate image after applying Canny Edge at different tl
% values in the valuesList
for tl = 1:length(valuesList)
    sigma = 1.0;
    cannyDiffTL = edge(macritchie, 'canny', [valuesList{tl} th], sigma);
    figure, imshow(cannyDiffTL)
end

%% 3.2a Canny Edge with sigma=1.0
% import image
P_mac = imread('maccropped.jpg');
P_mac = rgb2gray(P_mac);

% apply Canny Edge on the image
tl=0.04; th=0.1; sigma=1.0;
macritchie_Canny = edge(P_mac, 'canny', [tl th], sigma);
figure, imshow(macritchie_Canny)

%% 3.2b Find Hough Transform image
[H, xp] = radon(macritchie_Canny);
% figure, imshow(H)
% figure, imshow(uint8(H))
figure, imagesc(uint8(H)) % to visualise the waveforms better
colormap(gca, hot), colorbar;

%% 3.2c Finding maximum pixel intensity
maxH = max(H(:));
[radius, theta] = find(H == maxH);
radius = xp(radius); % obtain the radial coordinate for the maximum intensity

% converting from polar to cartesian
% % (theta, radius) = (104, -76)

%% 3.2d Derive the equations
% Derive values for A and B
[A,B] = pol2cart(theta * pi/180, radius);
B = -B;
% A = 18.39, B = 73.74

% Find center coordinate of the image
[numOfRows, numOfCols] = size(P_mac);
x_center = numOfCols / 2;
y_center = numOfRows / 2;
% x_center, y_center = 179, 145

% Obtain the C value from Ax + By = C
C = A*(A+x_center) + B*(B+y_center);
% C = 1.976e+04

%% 3.2e Compute y values
xl = 0; xr = numOfCols - 1;

% Equation to find y value: y = (C - Ax)/B
yl = (C - A*xl)/B;
yr = (C - A*xr)/B;
% yl = 267.96, yr = 178.95

%% 3.2f Display original image with the line
figure, imshow(P_mac);
path = line([xl xr], [yl yr]);
path.Color = "red";
% change sigma values

%% 3.2f Improving the line
tl=0.04; th=0.1; sigma=2.0;
macritchie_Canny2 = edge(P_mac, 'canny', [tl th], sigma);

[H, xp] = radon(macritchie_Canny2);

maxH = max(H(:));
[radius, theta] = find(H == maxH);
radius = xp(radius);
theta = theta - 1;

% Derive values for A and B
[A,B] = pol2cart(theta * pi/180, radius);
B = -B;

% Find center coordinate of the image
[numOfRows, numOfCols] = size(P_mac);
x_center = numOfCols / 2;
y_center = numOfRows / 2;

% Obtain the C value from Ax + By = C
C = A*(A+x_center) + B*(B+y_center);

xl = 0; xr = numOfCols - 1;

% Equation to find y value: y = (C - Ax)/B
yl = (C - A*xl)/B;
yr = (C - A*xr)/B;

figure, imshow(P_mac);
path = line([xl xr], [yl yr]);
path.Color = "yellow";

%% 3.3b Import the images
P_left = imread('corridorl.jpg');
P_right = imread('corridorr.jpg');

% Convert to grayscale
P_left = rgb2gray(P_left);
P_right = rgb2gray(P_right);

figure, imshow(P_left)
figure, imshow(P_right)

%% 3.3c Run Disparity Map function on both images
D = disparityMap(P_left, P_right, 11, 11);
figure, imshow(D,[-15 15]);

%% 3.3d Trying out Disparity May function on triclops image
P_triclops_Left = imread('triclopsi2l.jpg');
P_triclops_Right = imread('triclopsi2r.jpg');

% Convert to grayscale
P_triclops_Left = rgb2gray(P_triclops_Left);
P_triclops_Right = rgb2gray(P_triclops_Right);

D_triclops = disparityMap(P_triclops_Left, P_triclops_Right, 11, 11);
figure, imshow(D_triclops,[-15 15]);
% colormapeditor

%% Essential functions

% 3.3a
function [result] = disparityMap(pLeft, pRight, rowDim, colDim)
    % Convert image matrix to double
    pLeft = im2double(pLeft);
    pRight = im2double(pRight);
    
    % Obtain the dimension of left image
    [height, width] = size(pLeft); 
 
    % Get the disparity range and the template dimensions
    rowLength = floor(rowDim/2);
    colLength = floor(colDim/2);
    dispRange = 15;
    
    % Initialise matrix of 0s
    result = zeros(size(pLeft));

    % Outer loop - each pixel along the matrix column
    for i = 1:height
        % variables to keep the range in check
        minRow = max(1, i - rowLength);
        maxRow = min(height, i + rowLength);

        % loop for each pixel along the matrix row
        for j = 1:width
            % variables to keep the range in check
            minCol = max(1, j - colLength);
            maxCol = min(width, j + colLength);
            
            % variables to keep the horizontal search range in check
            minDisp = max(-dispRange, 1 - minCol); 
            maxDisp = min(dispRange, width - maxCol);
            
            % Obtain the template from the right image
            dispTemplate = pRight(minRow:maxRow, minCol:maxCol);

            % Initialise the variables for SSD comparison
            minSSD = inf;
            leastDifference = 0;
            
            % Inner loop - to do the searching in the search range
            for k = minDisp:maxDisp
                % Get the difference between left and right images
                newMinCol = minCol + k;
                newMaxCol = maxCol + k;
                block = pLeft(minRow:maxRow, newMinCol:newMaxCol);
                
                % Perform SSD
                squaredDifference = (dispTemplate - block).^2;
                ssd = sum(squaredDifference(:));
                
                % Get the lowest SSD
                if ssd < minSSD
                    minSSD = ssd;
                    leastDifference = k - minDisp + 1;
                end
            end
            
            % Return the SSD result
            result(i, j) = leastDifference + minDisp - 1;
        end
    end
end