function [result] = kl_div(p,q)
  result = sum(p.*log(p) - p.*log(q),2);
  return;
end