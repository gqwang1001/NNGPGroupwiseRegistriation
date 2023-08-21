function warns = checkLogMat(A)

% if ~isfloat(A) || ~ismatrix(A)
%     error(message('MATLAB:logm:inputType'));
% end
% 
% if size(A,1) ~= size(A,2)
%      error(message('MATLAB:logm:inputMustBeSquare'));
% end

% maxroots = 100;
exitflag = 0;

% if ~matlab.internal.math.allfinite(A)
%     L = NaN(size(A),class(A));
%     return
% end
warns = false;
% Check for triangularity.
% schurInput = matlab.internal.math.isschur(A);
% if schurInput
%     T = A;
% else
    % Assume A has finite elements.
%     [Q, T] = matlab.internal.math.nofinitecheck.schur(A);
    [Q, T] = eig(A);

% end
% stayReal = isreal(A);

% Compute the logarithm.
if isdiag(T)      % Check if T is diagonal.
    d = diag(T);
    if any(real(d) <= 0 & imag(d) == 0)
        warns = true;
%         warning(message('MATLAB:logm:nonPosRealEig'));
    end
%     if schurInput
%         L = diag(log(d));
%     else
%         logd = log(d);
%         L = (Q.*logd.')*Q';
%         if isreal(logd)
%             L = (L+L')/2;
%         end
%     end
else
%     n = size(T,1);
    % Check for negative real eigenvalues.
    ei = ordeig(T);
    warns = any(ei == 0);
    if any(real(ei) < 0 & imag(ei) == 0 )
        warns = true;
%         if stayReal
%             if schurInput
%                 % Output will be complex - change to complex Schur form.
%                 Q = eye(n, class(T));
%                 schurInput = false; % Need to undo rsf2csf at end.
%             end
%             [Q, T] = rsf2csf(Q, T);
%         end
    end
%     if warns
%         warning(message('MATLAB:logm:nonPosRealEig'));
%     end
%     % Get block structure of Schur factor.
%     blockformat = qtri_struct(T);
%     % Get parameters.
%     [s, m, Troot, exitflag] = logm_params(T, maxroots);
%     % Compute Troot - I = T(1/2^s) - I more accurately.
%     Troot = recompute_diag_blocks_sqrt(Troot, T, blockformat, s);
%     % Compute Pade approximant.
%     L = pade_approx(Troot, m);   
%     % Scale back up.
%     L = 2^s * L;
%     % Recompute diagonal blocks.
%     L = recompute_diag_blocks_log(L, T, blockformat);
%     % Combine if needed
%     if ~schurInput
%         L = Q*L*Q';
%     end
end
end