function J = jacobi(f, v, h)

n = numel(v); %number of input variables of the function f
m = numel(f(v)); %number of output variables of the function f
J = nan(m, n); %the jacobian matrix
S = cell(1, n); %the differential vectors that help nudge the input in a certain to see a change in the output

% calcultaion of the nudge vectors
for i = 1:n
    S{i} = zeros(size(v));
    S{i}(i) = h;
end

%the jacobian matrix calculation on a culumn by column basis
for i = 1:n
    J(:, i) = ( (f(v + S{i}) - f(v - S{i})) )/(2*h);
end

end