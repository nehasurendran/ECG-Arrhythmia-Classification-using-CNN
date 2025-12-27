folderPath = 'C:\Users\nehas\OneDrive\Desktop\training2017';
cd(folderPath)
fileList = dir('*.mat');

dataCell = cell(length(fileList), 1);
for i = 1:length(fileList)
    fileName = fileList(i).name;
    data = load(fileName);
    dataCell{i} = data;
end
% Define the window size for the moving average filter
windowSize = 5;  % Adjust this value based on the characteristics of your data and the level of noise.

% Initialize a cell array to store the denoised data
denoisedDataCell = cell(length(dataCell), 1);

for i = 1:length(dataCell)
    % Access the data from the i-th file
    data = dataCell{i};
    
    % Check if the data is numeric and can be filtered
    if isnumeric(data)
        % Apply the moving average filter to smooth out the noise
        denoisedData = movmean(data, windowSize);
        
        % Store the denoised data in the cell array
        denoisedDataCell{i} = denoisedData;
    else
        % If the data is not numeric, handle the appropriate preprocessing for your specific data type.
        % You can add additional code here to handle other data types.
        denoisedDataCell{i} = data;
    end
end

% After the loop, denoisedDataCell will contain the denoised data for each dataset.
% You can use these denoised datasets for further analysis or processing.
% Define the cutoff frequency for the high-pass filter
cutoffFrequency = 0.1;  % Adjust this value based on the characteristics of your data and the baseline wander frequency.

% Initialize a cell array to store the data with baseline wander removed
baselineRemovedDataCell = cell(length(denoisedDataCell), 1);

for i = 1:length(denoisedDataCell)
    % Access the data from the i-th file
    data = denoisedDataCell{i};
    
    % Check if the data is numeric and can be filtered
    if isnumeric(data)
        % Design a high-pass Butterworth filter
        [b, a] = butter(2, cutoffFrequency, 'high');
        
        % Apply the high-pass filter to remove baseline wander
        baselineRemovedData = filtfilt(b, a, data);
        
        % Store the data with baseline wander removed in the cell array
        baselineRemovedDataCell{i} = baselineRemovedData;
    else
        % If the data is not numeric, handle the appropriate preprocessing for your specific data type.
        % You can add additional code here to handle other data types.
        baselineRemovedDataCell{i} = data;
    end
end

% After the loop, baselineRemovedDataCell will contain the data with baseline wander removed for each dataset.
% You can use these processed datasets for further analysis or processing.
% Initialize a cell array to store the normalized data
normalizedDataCell = cell(length(baselineRemovedDataCell), 1);

for i = 1:length(baselineRemovedDataCell)
    % Access the baseline-removed data from the i-th file
    data = baselineRemovedDataCell{i};
    
    % Check if the data is numeric and can be normalized
    if isnumeric(data)
        % Find the maximum and minimum values in the data
        maxVal = max(data);
        minVal = min(data);
        
        % Normalize the data to the range [-1, 1]
        normalizedData = -1 + 2 * (data - minVal) / (maxVal - minVal);
        
        % Store the normalized data in the cell array
        normalizedDataCell{i} = normalizedData;
    else
        % If the data is not numeric, handle the appropriate preprocessing for your specific data type.
        % You can add additional code here to handle other data types.
        normalizedDataCell{i} = data;
    end
end




% Assuming you already have the `normalizedDataCell` containing the normalized data

targetLength = 9000; % The desired length for resampling

% Initialize a new cell array to store resampled data
resampledDataCell = cell(size(normalizedDataCell));

% Loop through each dataset in normalizedDataCell
for i = 1:length(normalizedDataCell)
    data = normalizedDataCell{i};
    
    if isstruct(data) && isfield(data, 'val') && isnumeric(data.val)
        % If the data is a struct with a 'val' field containing numeric data
        % Resample the 'val' field data to have 9000 samples using interpolation
        resampledDataCell{i} = resample(data.val, targetLength, length(data.val));
    else
        % If the data is not a struct or does not have a 'val' field, directly store the data
        resampledDataCell{i} = data;
    end
end





numSamplesArray1 = zeros(length(resampledDataCell), 1);
for i = 1:length(resampledDataCell)
    numSamplesArray1(i) = numel(resampledDataCell{i})
end

% Assuming you have the `numSamplesArray1` from the previous code snippet
samplingFrequency = 300; % Sampling frequency in Hz

% Calculate the time duration for each dataset in seconds
timeInSeconds = numSamplesArray1 / samplingFrequency;

% Plot the histogram with time on the x-axis
histogram(timeInSeconds)

% Optionally, you can add labels and title to the plot for better visualization
xlabel('Time (seconds)')
ylabel('Frequency')
title('Histogram of Resampled Data (seconds)')

% Parameters for spectrogram computation
windowLength = 256;  % Length of the window for computing the short-time Fourier transform (STFT)
overlap = 128;       % Number of overlapping samples between consecutive windows


% Parameters for spectrogram computation
windowLength = 256;  % Length of the window for computing the short-time Fourier transform (STFT)
overlap = 128;       % Number of overlapping samples between consecutive windows

% Initialize an empty cell array to store the spectrograms
spectrogramsCell = cell(size(resampledDataCell));

% Compute spectrogram for each resampled ECG signal
for i = 1:length(resampledDataCell)
    % Get the resampled ECG signal
    ecgSignal = resampledDataCell{i};
    
    % Debug: Display the class and size of ecgSignal
    disp(['ecgSignal class: ', class(ecgSignal)]);
    disp(['ecgSignal size: ', num2str(size(ecgSignal))]);
    
    % Assuming ecgSignal contains the numeric ECG data directly
    if isnumeric(ecgSignal)  % Check if ecgSignal is numeric
        % Compute the spectrogram using the spectrogram function
        [S, f, t] = spectrogram(ecgSignal, windowLength, overlap, [], 300);  % Assuming 300 Hz sampling frequency
        
        % Store the spectrogram in the cell array
        spectrogramsCell{i} = S;
    else
        % Handle non-numeric data
        spectrogramsCell{i} = [];  % or any appropriate handling
    end
end

% ... (rest of your code)



% Initialize an array to store the number of samples for each dataset
numSamplesArray = zeros(length(spectrogramsCell), 1);

% Calculate the number of samples for each dataset
for i = 1:length(spectrogramsCell)
    numSamplesArray(i) = size(spectrogramsCell{i}, 2); % Assuming the second dimension is the number of samples
end

% Plot the histogram
histogram(numSamplesArray)
xlabel('Number of Samples')
ylabel('Frequency')
title('Histogram of Number of Samples')

% Load your spectrograms and corresponding labels
  % Load the 'spectrogramsCell' variable containing spectrogram data
         % Load the 'labels' variable containing corresponding labels
 labelsTable = readtable('C:\Users\nehas\OneDrive\Desktop\training2017\labels.csv');


filenames = labelsTable.Filename;
labels = labelsTable.Label;
% Load your labels from CSV


% Debug: Display unique labels and class of labels
disp(unique(labels));
disp(class(labels));


% Ensure labels are categorical
labels = categorical(labels);

% Ensure labels match the length of spectrogramsCell
if length(labels) ~= length(spectrogramsCell)
    error('Number of labels does not match the number of spectrograms.');
end

% Step 1: Split Data
splitRatio = 0.8; % 80% training, 20% testing
numSamples = length(spectrogramsCell);
numTrainSamples = round(splitRatio * numSamples);

% Shuffle the data and labels to ensure random distribution
shuffledIndices = randperm(numSamples);
trainIndices = shuffledIndices(1:numTrainSamples);
testIndices = shuffledIndices(numTrainSamples+1:end);

% Split the data and labels into training and testing sets
trainData = spectrogramsCell(trainIndices);
trainLabels = labels(trainIndices);
testData = spectrogramsCell(testIndices);
testLabels = labels(testIndices);



% Convert categorical labels to numeric using grp2idx
numericLabels = grp2idx(labels);

% Calculate the number of samples in each class
uniqueLabels = unique(numericLabels);
labelCounts = histc(numericLabels, uniqueLabels);

% Display the class counts
disp('Class Counts:');
disp([uniqueLabels, labelCounts]);


% Create a bar chart
bar(uniqueLabels, labelCounts)


% Assuming you have 'spectrogramsCell' containing spectrogram data and 'labels' containing corresponding labels.

% Assuming you have 'spectrogramsCell' containing spectrogram data and 'labels' containing corresponding labels.

% Assuming you have 'spectrogramsCell' containing spectrogram data and 'labels' containing corresponding labels.

% Assuming you have 'spectrogramsCell' containing spectrogram data and 'labels' containing corresponding labels.

% Assuming you have 'spectrogramsCell' containing spectrogram data and 'labels' containing corresponding labels.
% Step 1: Split Data
splitRatio = 0.8; % 80% training, 20% testing
numSamples = length(spectrogramsCell);
numTrainSamples = round(splitRatio * numSamples);

if numTrainSamples <= length(labels)
    shuffledIndices = randperm(numSamples);
    trainIndices = shuffledIndices(1:numTrainSamples);
    testIndices = shuffledIndices(numTrainSamples+1:end);
else
    error('Number of training samples exceeds the total number of samples.');
end

% Split the data and labels into training and testing sets
trainData = spectrogramsCell(trainIndices);
trainLabels = labels(trainIndices);
testData = spectrogramsCell(testIndices);
testLabels = labels(testIndices);

% Step 2: Convert cell array of spectrograms to a numeric array
% Assuming the spectrograms have consistent dimensions (rows x columns)
spectrogram_height = size(trainData{1}, 1);
spectrogram_width = size(trainData{1}, 2);
numChannels = 1; % Assuming the spectrograms are grayscale
numTrainData = numel(trainData);
numTestData = numel(testData);


% Initialize numeric arrays to store the training and testing data
trainDataNumeric = zeros(spectrogram_height, spectrogram_width, numChannels, numTrainData);
testDataNumeric = zeros(spectrogram_height, spectrogram_width, numChannels, numTestData);

% Convert the cell array to numeric array for training data
for i = 1:numTrainData
    curr_spectrogram = trainData{i};
    trainDataNumeric(:, :, 1, i) = abs(curr_spectrogram); % Use magnitude of complex spectrogram
end

% Convert the cell array to numeric array for testing data
for i = 1:numTestData
    curr_spectrogram = testData{i};
    testDataNumeric(:, :, 1, i) = abs(curr_spectrogram); % Use magnitude of complex spectrogram
end

% Step 3: Model Selection - Create the CNN model
numClasses = numel(categories(labels));

% Assuming you have loaded your labels from a CSV file and converted them to categorical
labelsTable = readtable('C:\Users\nehas\OneDrive\Desktop\dataset\labels_sample.csv');
labels = categorical(labelsTable.Label);
labels=grp2idx(labels)

% Calculate the number of samples in each class
uniqueLabels = unique(labels);
labelCounts = histc(labels, uniqueLabels);

% Create a bar chart
bar(uniqueLabels, labelCounts)

% Add labels and title
xlabel('Class Labels')
ylabel('Number of Samples')
title('Number of Samples in Each Class')





layers = [
    imageInputLayer([spectrogram_height, spectrogram_width, numChannels], 'SplitComplexInputs', true) % Set SplitComplexInputs to true
    convolution2dLayer(3, 32, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 64, 'Padding', 'same')
    reluLayer()
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(numClasses)
    softmaxLayer()
    classificationLayer()
];

% Step 4: Model Training
miniBatchSize = 16;
numEpochs = 1;
initialLearnRate = 0.001;

% Specify the training options
options = trainingOptions('adam', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', numEpochs, ...
    'ValidationData', {testDataNumeric, testLabels}, ...
    'InitialLearnRate', initialLearnRate, ...
    'ValidationFrequency', floor(numTrainSamples / miniBatchSize), ...
    'Plots', 'training-progress');

% Train the CNN model
cnnModel = trainNetwork(trainDataNumeric, trainLabels, layers, options);

% Step 5: Model Evaluation
% Make predictions using the trained model on the test data
predictedLabels = classify(cnnModel, testDataNumeric);

% Calculate accuracy (or other evaluation metrics) for the CNN model
accuracy = sum(predictedLabels == testLabels) / numel(testLabels);

% Display the accuracy
disp(['Test accuracy: ', num2str(accuracy)]);
% Calculate the confusion matrix
confusionMatrix = confusionmat(testLabels, predictedLabels);
plotconfusion(testLabels,predictedLabels)


