function H = hesse(f, v, h)

n = numel(v); %number of input variables of the function f
H = nan(n, n); %the hessian matrix
S = cell(1, n); %the differential vectors that help nudge the input in a certain to see a change in the output

% calcultaion of the nudge vectors
for i = 1:n
    S{i} = zeros(size(v));
    S{i}(i) = h;
end

for i = 1:n
    for j = 1:n
        H(i, j) = ( (f(v + S{i} + S{j}) - f(v - S{i} + S{j})) - (f(v + S{i} - S{j}) - f(v - S{i} - S{j})) )/(4*h^2);
    end
end

end