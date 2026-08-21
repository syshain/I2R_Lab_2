function [q_solution, success, error_norm] = ik_solve(T_target, q0, joint_limits)
    % Solve inverse kinematics numerically
    % Input: T_target = 4x4 target transformation matrix
    %        q0 = initial guess (6-element vector in radians)
    %        joint_limits = 6x2 matrix [min_rad, max_rad] for each joint
    % Output: q_solution = joint angles in radians
    %         success = true if solution found
    %         error_norm = final error norm
    
    % Set up optimization options
    options = optimoptions('fsolve', ...
        'Algorithm', 'levenberg-marquardt', ...
        'Display', 'iter-detailed', ...
        'MaxFunctionEvaluations', 5000, ...
        'MaxIterations', 500, ...
        'FunctionTolerance', 1e-10, ...
        'StepTolerance', 1e-12, ...
        'OptimalityTolerance', 1e-6);
    
    % If joint limits are provided, use lsqnonlin instead of fsolve
    if nargin == 3 && ~isempty(joint_limits)
        % Use lsqnonlin to enforce joint limits
        lb = joint_limits(:,1);
        ub = joint_limits(:,2);
        
        % Objective function for lsqnonlin (returns the same error vector)
        obj_func = @(q) ik_objective(q, T_target);
        
        [q_solution, resnorm, residual, exitflag] = lsqnonlin(obj_func, q0, lb, ub, options);
        
        % Check if solution was found
        if exitflag > 0
            success = true;
        else
            success = false;
            warning('IK solver did not converge. Exit flag: %d', exitflag);
        end
        
        error_norm = norm(residual);
    else
        % Use fsolve without joint limits
        obj_func = @(q) ik_objective(q, T_target);
        
        [q_solution, fval, exitflag] = fsolve(obj_func, q0, options);
        
        if exitflag > 0
            success = true;
        else
            success = false;
            warning('IK solver did not converge. Exit flag: %d', exitflag);
        end
        
        error_norm = norm(fval);
    end
end
