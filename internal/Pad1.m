% Copyright 2017 Google Inc.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%      http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function Zp = Pad1(Z)
% Repeated padding (1px wide) to handle boundary condition

Zp = {};
for c = 1:size(Z,3)
  Zp{c} = [Z(1,1,c), Z(1,:,c), Z(1,end,c);
    Z(:,1,c), Z(:,:,c),      Z(:,end,c);
    Z(end,1,c), Z(end,:,c), Z(end,end,c)];
end
Zp = cat(3, Zp{:});
