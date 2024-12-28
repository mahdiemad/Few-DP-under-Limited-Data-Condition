%% t-SNE 
% Load Excel file
data = readtable('few1_tsne_results.xlsx');

% Extract x, y, and labels
x = data.x;
y = data.y;
labels = data.label;

% Define the number of unique labels
unique_labels = unique(labels);
num_classes = length(unique_labels);
labels_ = ['Ex', 'IW', 'Inrush', 'T2T', 'Winding'];

% Create a colormap
colors = lines(num_classes);

% Plot t-SNE results
figure;
hold on;
for i = 1:num_classes
    % Find indices for the current class
    idx = labels == unique_labels(i);
    
    % Scatter plot for the current class
    scatter(x(idx), y(idx), 50, colors(i, :), 'filled', 'DisplayName', ['Class ' num2str(unique_labels(i))]);
    
    % Add label annotations
    x_mean = mean(x(idx));
    y_mean = mean(y(idx));
    text(x_mean, y_mean, num2str(unique_labels(i)), 'FontSize', 16, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end
hold off;

% Add title and legend
title(['t-SNE visualization for ${CNN}_{1D}$'], 'Interpreter', 'latex','FontSize',20, 'FontName', 'Times New Roman');
legend('Ex Fault', 'IW Fault', 'Inrush Fault', 'T2T Fault', 'Winding Fault');

set(gca, 'FontSize',15, 'FontName', 'Times New Roman');

%% Confusion matrix

% Load data
confusion_data = xlsread('few2_confusion_results', 1);
classLabels = {'Ex Fault', 'IW Fault', 'Inrush Fault', 'T2T Fault', 'Winding Fault'};

% Confusion Matrix
CM = confusionmat(confusion_data(:, 1), confusion_data(:, 2));

% Heatmap with Labels
figure();
h = heatmap(classLabels, classLabels, CM, ...
    "Title", "few2 Confusion Matrix ",'XLabel', 'Predicted Class', 'YLabel', 'True Class');

% Remove color bar from heatmap
h.ColorbarVisible = 'off';

% % Add custom LaTeX title
% annotation('textbox', [0.2, 0.9, 0.6, 0.1], ...
%     'String', '${CNN}_{1D}$ Confusion Matrix', ...
%     'Interpreter', 'latex', ...
%     'FontSize', 14, ...
%     'FontName', 'Times New Roman', ...
%     'HorizontalAlignment', 'right', ...
%     'EdgeColor', 'none');
set(gca, 'FontSize',12.5, 'FontName', 'Times New Roman');


