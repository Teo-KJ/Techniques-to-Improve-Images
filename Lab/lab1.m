%% 2.1a View image
Pc = imread('mrttrainbland.jpg');
whos Pc; % check if image is in RGB

%% 2.1b Convert to Grayscale
P = rgb2gray(Pc);
imshow(P); % show image

%% 2.1c Check minimum and maximum intensities
a = min(P(:));
b = max(P(:));

%% Experimenting with the image - For Fun

% define the conditions
minTrue = P <= a;
maxTrue = P >= b;
middle = P > a & P < b;

multiplicant = 255/(b-a); % multiplicant for contrast scaling

P1 = P; % make a copy

% change pixel values based on earlier defined conditions
P1(minTrue) = P1(minTrue) - a;
P1(middle) = P1(middle) .* multiplicant .* (P1(middle) - a);
P1(maxTrue) = P1(maxTrue) + b;

min(P1(:)), max(P1(:))

figure, imshow(P1) % generate image

%% 2.1d,e Contrast stretching
P2 = imsubtract(P, 13);
P2 = immultiply(P2, 255/(204-13));
figure, imshow(P2)

min(P2(:)), max(P2(:))

%% 2.2a Display histogram
figure, imhist(P, 10) % display image P using histogram with 10 bins
figure, imhist(P, 256) % display image P using histogram with 256 bins

%% 2.2b, c Histogram Equalization processing
P3a = histeq(P, 255);
figure, imshow(P3a) % generate image with equalisation of 255 bins
figure, imshow(histeq(P, 10)) % generate image with equalisation of 10 bins

figure, imhist(P3a) % display histogram with 255 bins equalisation
figure, imhist(histeq(P, 10)) % display histogram with 10 bins equalisation

P3 = histeq(P3a, 255);
figure, imshow(P3)
figure, imhist(P3)

%% 2.3a Generate Gaussian filters
% σ=1, 5x5 neighbourhood elements
a1 = fspecial('gaussian',[5,5],1);
figure, mesh(a1)
% σ=2, 5x5 neighbourhood elements
a2 = fspecial('gaussian',[5,5],2);
figure, mesh(a2)

%% 2.3b View image
P_ntugn = imread('ntugn.jpg');
figure, imshow(P_ntugn)

%% 2.3c Apply filter to remove noise
% generate matrix from h function output
hMat = [h(2,2,2), h(1,2,2), h(0,2,2), h(1,2,2), h(2,2,2);
        h(2,1,2), h(1,1,2), h(0,1,2), h(1,1,2), h(2,1,2);
        h(2,0,2), h(1,0,2), h(0,0,2), h(1,0,2), h(2,0,2);
        h(2,1,2), h(1,1,2), h(0,1,2), h(1,1,2), h(2,1,2);
        h(2,2,2), h(1,2,2), h(0,2,2), h(1,2,2), h(2,2,2)];
% normalise the matrix such that sum of elements = 1
hMat = hMat / sum(sum(hMat));

% apply filter through convolution
P_ntugn_Imp = conv2(P_ntugn, hMat);

% Compare the difference
figure, imshow(P_ntugn)
figure, imshow(uint8(P_ntugn_Imp))

%% 2.3d View image
P_ntusp = imread('ntusp.jpg');
figure, imshow(P_ntusp)

%% 2.3e Attempt to remove speckle noise
% apply filter through convolution
P_ntusp_Imp = conv2(P_ntusp, hMat);
figure, imshow(uint8(P_ntusp_Imp))

%% 2.4 Median Filterning - ntugn.jpg (image with Gaussian noise)
% apply median filter
P_ntugn_MF_3 = medfilt2(P_ntugn, [3,3]);
P_ntugn_MF_5 = medfilt2(P_ntugn, [5,5]);

% Compare the difference
figure, imshow(P_ntugn)
figure, imshow(uint8(P_ntugn_MF_3))
figure, imshow(uint8(P_ntugn_MF_5))

%% 2.4 Median Filterning - ntusp.jpg (image with speckle noise)
% apply median filter
P_ntusp_MF_3 = medfilt2(P_ntusp, [3,3]);
P_ntusp_MF_5 = medfilt2(P_ntusp, [5,5]);

% Compare the difference
figure, imshow(P_ntusp)
figure, imshow(uint8(P_ntusp_MF_3))
figure, imshow(uint8(P_ntusp_MF_5))

%% 2.5a View image
P_PCK = imread('pckint.jpg');
figure, imshow(P_PCK)

%% 2.5b Apply Fourier Transform
% Fourier transform of image matrix
F_PCK = fft2(P_PCK);
% find power spectrum, S
S_PCK = abs(F_PCK).^2;
% visualise power spectrum
imagesc(fftshift(S_PCK.^0.1));
colormap('default');

%% 2.5c Redisplay without fftshift
% visualise power spectrum
imagesc(S_PCK.^0.1);
colormap('default');
% identified coordinates are (241,9) and (17,249)

%% 2.5d Re-compute with 5x5
% get transformed image matrix and define the identified coordinates
F_PCK_5 = fft2(P_PCK);
x1 = 241; y1 = 9;
x2 = 17 ; y2 = 249;

% set peaks and neighbour pixels to 0
F_PCK_5(x1-2:x1+2, y1-2:y1+2) = 0;
F_PCK_5(x2-2:x2+2, y2-2:y2+2) = 0;

% check power spectrum to identify if peaks are set to 0
S_PCK_5 = abs(F_PCK_5);
imagesc(S_PCK_5.^0.1);
colormap('default');

%% 2.5e Obtain back the image
% apply inverse transform to obtain back image
F_PCK_inv = ifft2(F_PCK_5);
figure, imshow(uint8(F_PCK_inv));

%% 2.5f Experimenting om the primate diagram
P_primate = imread('primatecaged.jpg');
P_primate = rgb2gray(P_primate); % convert to grayscale
figure, imshow(P_primate)

%% 2.5f Find power spectrum and identify peaks
% apply Fourier transform and find power spectrum
F_primate = fft2(P_primate);
S_primate = abs(F_primate).^2;
figure, imagesc(fftshift(S_primate.^0.1));
figure, imagesc(S_primate.^0.1);
colormap('default');
% identified coordinates are (252,11), (248,22), (5,247) and (10,236)

%% 2.5f set preaks to 0 with 5x5 neighbour pixels
% define peaks coordinates
x1 = 252; y1 = 11;
x2 = 248; y2 = 22;
x3 = 5; y3 = 247;
x4 = 10; y4 = 236;

% set peaks and neighbour pixels to 0
F_primate(x1-2:x1+2, y1-2:y1+2) = 0;
F_primate(x2-2:x2+2, y2-2:y2+2) = 0;
F_primate(x3-2:x3+2, y3-2:y3+2) = 0;
F_primate(x4-2:x4+2, y4-2:y4+2) = 0;

% check power spectrum again
S_primate_imp = abs(F_primate);
figure, imagesc(S_primate_imp.^0.1);

%% 2.5f inverse transform to obtain original image
% inverse Fourier transform to get back image
F_primate_inv = ifft2(F_primate);
figure, imshow(uint8(F_primate_inv));

%% 2.6a, b View image and Identify the coordinates
P_book = imread('book.jpg');
P_book = rgb2gray(P_book); % convert to grayscale
figure, imshow(P_book)

% Original corner coordinates in the image
% top left
x1 = 142; y1 = 28; 
% top right
x2 = 308; y2 = 48;
% bottom left
x3 = 3; y3 = 159;
% bottom right
x4 = 257; y4 = 216;

% set matrix of x and y coordinates
x = [x1; x2; x3; x4]; % top left; top right; bottom left, bottom right
y = [y1; y2; y3; y4]; % top left; top right; bottom left, bottom right

% desired corner coordinates in the image
X_corners = [0; 210; 0; 210]; % top left; top right; bottom left, bottom right
Y_corners = [0; 0; 297; 297]; % top left; top right; bottom left, bottom right

%% 2.6c Identify matrices for projective transformation
% matrix v
v = [X_corners(1); Y_corners(1); 
     X_corners(2); Y_corners(2);
     X_corners(3); Y_corners(3);
     X_corners(4); Y_corners(4)];

% matrix A
A = [
    [x(1) y(1) 1    0    0    0  -X_corners(1)*x(1)  -X_corners(1)*y(1)];
    [0    0    0    x(1) y(1) 1  -Y_corners(1)*x(1)  -Y_corners(1)*y(1)];
    [x(2) y(2) 1    0    0    0  -X_corners(2)*x(2)  -X_corners(2)*y(2)];
    [0    0    0    x(2) y(2) 1  -Y_corners(2)*x(2)  -Y_corners(2)*y(2)];
    [x(3) y(3) 1    0    0    0  -X_corners(3)*x(3)  -X_corners(3)*y(3)];
    [0    0    0    x(3) y(3) 1  -Y_corners(3)*x(3)  -Y_corners(3)*y(3)];
    [x(4) y(4) 1    0    0    0  -X_corners(4)*x(4)  -X_corners(4)*y(4)];
    [0    0    0    x(4) y(4) 1  -Y_corners(4)*x(4)  -Y_corners(4)*y(4)];
    ];
 
u = A \ v;
U = reshape([u;1], 3, 3)';

w = U*[x'; y'; ones(1,4)];
w = w ./ (ones(3,1) * w(3,:));

%% 2.6d warp the image
T = maketform('projective', U');
P_book_transform = imtransform(P_book, T, 'XData', [0 210], 'YData', [0 297]);
figure, imshow(uint8(P_book_transform)); % generate image

%% Essential functions

% 2.3a
function result = h(x, y, sig)
    result = (1/2*pi*(sig^2))*exp(-(x^2+y^2)/(2*sig^2));
end