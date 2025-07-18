%% Reading, Cleaning and Organizing the Input Data

close all; clear; clc;

% Load data into a table then remove unnecessary data then sort rows by price
data_raw = readtable("train.csv", VariableNamingRule="preserve");
data = sortrows(data_raw, width(data_raw));
data(:, ["Id", "MSSubClass", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "LowQualFinSF"]) = [];

%get numeric variables and replace NaNs with zeros
data_num = data(:, vartype("numeric"));
data_num{:, :}(isnan(data_num.Variables)) = 0;

%remove sparse variables and low variance variables
ind = [];
for i = 1:width(data_num)
    if length(data_num{(0 ==(data_num{:, i})), i}) > 0.5*height(data_num)
        ind = horzcat(i);
    end
end
data_num(:, ind) = [];

vr = std(data_num.Variables)./range(data_num.Variables) * 100;
ind = (vr < 10);
data_num(:, ind) = [];
%get discrete variables
data_dis = table('Size', size(data_num), 'VariableTypes', repmat({'double'}, [1, width(data_num)]),  'VariableNames', data_num.Properties.VariableNames);

for i = 1:width(data_num)
    if height(unique(data_num(:, i))) < 20
        data_dis(:, i) = data_num(:, i);
    end
end

data_dis(:, sum(data_dis.Variables) == 0) = [];

%remove discrete vars from numeric data
data_num(:, data_dis.Properties.VariableNames) = [];

%get categorical variables
data_str = data(:, vartype("cell"));

%separate target variable
price = data_num.SalePrice; data_num(:, "SalePrice") = [];

%% Exploritory Training Data Analysis

%get histogram data on a selected variable
current_var = price; step = range(current_var)/20;

%plot each numeric variable vs price
figure
sgtitle("Numeric Variables Plotted Against Price & Their Linear Correlation \it{r} ")
for i = 1:width(data_num)
    subplot(ceil(sqrt(width(data_num))), floor(sqrt(width(data_num))), i)
    r = round(corrcoef(price, data_num{:, i}), 2);
    plot(data_num{:, i}, price, '.')
    title(['r = ', num2str(r(2))])
    grid
    xlabel(data_num.Properties.VariableNames(i))
end

figure
sgtitle("Discrete Variables Plotted Against Price & Their Linear Correlation \it{r} ")
for i = 1:width(data_dis)
    subplot(ceil(sqrt(width(data_dis))), floor(sqrt(width(data_dis))), i)
    r = round(corrcoef(price, data_dis{:, i}), 2);
    plot(data_dis{:, i}, price, '.')
    title(['r = ', num2str(r(2))])
    grid
    xlabel(data_dis.Properties.VariableNames(i))
end

%% Principal Component Analysis for Continious Numeric Data

%compute mean and std for standardization
mu = mean(data_num); sigma = std(data_num);
data_num_std = (data_num - mu)./sigma;

%eigen decomposition of covariance matrix
[V, D] = eig(cov(data_num_std.Variables));
D = diag(D);
[D, ind] = sort(D, 'descend');
V = V(:, ind);

%percentage of variance explained and minimum threshold for retained variance
explained = D / sum(D) * 100; min_exp = 95;
num_var = (find(cumsum(explained) >= min_exp, 1)+ 1);

% Plot PCA spectrum and cumulative variance
figure
sgtitle('Principle Component Analysis Spectrum')
subplot(1, 2, 1)
bar(explained)
grid
xlabel('Component'); ylabel('% Variance Explained')
axis tight

subplot(1, 2, 2)
hold on
bar(cumsum(explained))
plot([0, width(data_num) + 1/2], min_exp*[1, 1], 'k--')
text(width(data_num)/8, 1.05*min_exp, ['Retention Threshold = ', num2str(min_exp), '%'])
hold off
grid
xlabel('Number of Summed Components'); ylabel('% Variance Explained by X-Axis')
axis tight

%% Regression Tree

%select training sample indices
train_ind = 1:height(data_num);

%project data to PCA space and reduce dimensions
train_data_num = data_num{train_ind, :}*V;
train_data = [train_data_num(:, 1:num_var), data_dis{train_ind, :}];
train_price = price(train_ind);

%define regression tree and train gradient boosting ensemble
tree = templateTree('MinLeafSize', 3);
GBM = fitensemble(train_data, train_price, 'LSBoost', 5e2, tree, 'LearnRate', 1e-1);

data_test_raw = readtable("test.csv", VariableNamingRule="preserve");
test_data_dis = data_test_raw(:, data_dis.Properties.VariableNames);
test_data_num = data_test_raw{:, data_num.Properties.VariableNames}*V;
test_data = [test_data_num(:, 1:num_var), test_data_dis{:, :}];


out = predict(GBM, test_data);
data_test_raw(:, end + 1) = table(out);
data_test_raw.Properties.VariableNames(end) = {'SalePrice'};

perf = [std(out)/std(price) , mean(out)/mean(price)];

%% Statistical Analysis of Sale Price in the Test Data

%get histogram data on a selected variable
figure
variables = {out, price}; titles = {"testing data", "training data"};
for i = [1, 2]
    current_var = variables{i}; step = range(current_var)/20;
    
    subplot(2, 1, i)
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
    legend({titles{i}, '$f = s_1 x^{s_2} e^{s_3 x + s_4}$'}, 'Interpreter', 'latex', 'FontSize', 14);
    xlabel('X, Price')
    ylabel('Y, Number of Houses')
    grid
end
%% Saving the figures
% figs = findall(0, 'Type', 'figure'); figs(2) = [];
% for i = 1:length(figs)
%     figure(figs(i));
%     set(figs(i), 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);
%     saveas(figs(i), ['figure_' num2str(figs(i).Number) '.svg']);
% end
