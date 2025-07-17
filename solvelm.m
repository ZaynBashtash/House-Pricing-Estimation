function [s, e] = solvelm(f, s, n_max, e_max, h)

r = 1e-2; d = 2;

e = inf(length(f(s)), n_max); e(1) = f(s);

for i = 2:(n_max + 1)

    J = jacobi(f, s, h);
    H = hesse(f, s, h);
    ds = - ( H + r*eye(size(H)) )\transpose(J);

    n = 0;
    while and(n < n_max/10, f(s + ds) >= e(i - 1))
        r = r*d;
        ds = - ( H + r*eye(size(H)) )\transpose(J);
        n = n + 1;
    end

    e(i) = f(s + ds);

    if or(or(e(i) <= e_max, e(i) > e(i - 1)), sum(isnan(ds)))
        break
    else
        s = s + ds;
        r = r/d;
    end

end

e = e(~(e == inf));
e = e(~isnan(e));
end