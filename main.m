close all; clear; clc;

%% Reading and organizing the input data

% Load data into a table then remove unnecessary data then sort rows by price
data_raw = readtable("train.csv", VariableNamingRule="preserve");
data_raw(:, ["Id", "MSSubClass"]) = []; 
data = sortrows(data_raw, width(data_raw));

%get numeric variables and replace NaNs with zeros
data_num = data(:, vartype("numeric")); 
data_num{:, :}(isnan(data_num.Variables)) = 0;

%get discrete variables
data_dis = table('Size', size(data_num), ...
    'VariableTypes', repmat({'double'}, [1, width(data_num)]), ...
    'VariableNames', data_num.Properties.VariableNames);

%identify discrete variables
for i = 1:width(data_num)
    if (height(unique(data_num(:, i))) < 20)
        data_dis(:, i) = data_num(:, i);
    end
end

data_dis(:, sum(data_dis.Variables) == 0) = [];

%remove discrete vars from numeric data
data_num(:, data_dis.Properties.VariableNames) = [];

%get categorical variables
data_str = data(:, vartype("cell"));                               

%% Statistical Analysis

%separate target variable
price = data_num.SalePrice; data_num(:, "SalePrice") = [];

%get histogram data on a selected variable
current_var = price; step = range(current_var)/20;
figure
h = histogram(current_var, round(range(current_var)/step));
y = h.Values; x = (h.BinEdges(2:end) + h.BinEdges(1:end-1))/2;

%define symbolic model for histogram fit
s = sym('s', [4, 1]); z = sym('z');

%define function for fit
F = s(1)*z.^s(2).*exp(s(3)*z + s(4)); 

%normalize data and define function handle for efficiency
x_s = x/max(x); y_s = y/max(y);
f = matlabFunction(subs(F, z, x_s), 'vars', {s});

%optimize using LM method
[k, e] = solvelm(@(s) mean((y_s - f(s)).^2), ones(4, 1), 1e2, 1e-6, 1e-3);  

%rescale fitted function (beacuase of normalization)
f = matlabFunction(subs(max(y)*F, [s; z], [k; z/max(x)]), 'vars', {z});

X = linspace(0, max(x), 1e3);
hold on
plot(X, f(X), 'r') 
hold off
title(['Normalized MSE = ', num2str(min(e)), ', r^2 = ', num2str(1 - sum((y - f(x)).^2)/sum((mean(y) - f(x)).^2))])
xlabel('Price')
ylabel('Number of Houses')
grid

%plot each numeric variable vs price
figure
title("Numeric Variables Vs. Price")
for i = 1:width(data_num)
    subplot(ceil(sqrt(width(data_num))) + 1, floor(sqrt(width(data_num))), i)
    plot(data_num{:, i}, price(:), '.')
    grid
    xlabel(data_num.Properties.VariableNames(i))
end

%% Principal Component Analysis

%compute mean and std for standardization
mu = mean(data_num); sigma = std(data_num);
data_num_std = (data_num - mu)./sigma;                             

%eigen decomposition of covariance matrix
[V, D] = eig(cov(data_num_std.Variables));
D = diag(D);
[D, ind] = sort(D, 'descend');
V = V(:, ind);

%percentage of variance explained and minimum threshold for retained variance
explained = D / sum(D) * 100; min_exp = 80;
num_var = (find(cumsum(explained) >= min_exp, 1)+ 1);

% Plot PCA spectrum and cumulative variance
figure
sgtitle('Principle Component Analysis Spectrum')
subplot(1, 2, 1)
bar(explained)                                                    % Bar plot of explained variance
grid
xlabel('Component'); ylabel('% Variance Explained')
axis tight

subplot(1, 2, 2)
hold on
bar(cumsum(explained))                                            % Cumulative variance plot
plot([0, width(data_num) + 1/2], min_exp*[1, 1], 'k--')            % Threshold line
text(width(data_num)/8, 1.05*min_exp, ...
    ['Retention Threshold = ', num2str(min_exp), '%'])            % Annotation
hold off
grid
xlabel('Number of Summed Components'); ylabel('% Variance Explained by X-Axis')
axis tight

%% Artificial Neural Network Training

%select training sample indices
train_ind = round(0.1*height(data_num)):round(0.6*height(data_num));

%project data to PCA space and reduce dimensions
train_data_num = data_num{train_ind, :}*V;
train_data = train_data_num(:, 1:num_var);
train_price = price(train_ind);

%feedforward neural net and its settings
ANN = feedforwardnet([30, 30]);
ANN.trainFcn = 'trainlm';
ANN.layers{1:end-1}.transferFcn = 'tansig';
ANN.layers{end}.transferFcn = 'purelin';
ANN.performFcn = 'sse';
ANN.trainParam.epochs = 1e3;
ANN.trainParam.max_fail = 50;
ANN.trainParam.goal = 1e-9;
ANN.trainParam.min_grad = 1e-12;
ANN.trainParam.mu_max = 1e12;
[ANN, perf] = train(ANN, train_data.', train_price.');

%ANN output
in = train_data.';
out = ANN( in ).';
e = out - train_price;

figure
plot(e, '.')
xlabel('House ID'); ylabel('Price Error');
title(['Mean of Price Error = ', num2str(mean(e))])
axis tight
grid

%% Saving the figures
% figs = findall(0, 'Type', 'figure'); figs(2) = [];
% for i = 1:length(figs)
%     figure(figs(i));
%     set(figs(i), 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
%     saveas(figs(i), ['figure_' num2str(figs(i).Number) '.svg']);
% end