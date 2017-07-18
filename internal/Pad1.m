function Zp = Pad1(Z)
% Repeated padding (1px wide) to handle boundary condition

Zp = {};
for c = 1:size(Z,3)
  Zp{c} = [Z(1,1,c), Z(1,:,c), Z(1,end,c);
    Z(:,1,c), Z(:,:,c),      Z(:,end,c);
    Z(end,1,c), Z(end,:,c), Z(end,end,c)];
end
Zp = cat(3, Zp{:});
