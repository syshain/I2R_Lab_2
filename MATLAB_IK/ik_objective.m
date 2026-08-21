function err = ik_objective(q, T_target)
    % Compute error between current forward kinematics and target pose
    % Input: q = [q1, q2, q3, q4, q5, q6] in radians
    %        T_target = 4x4 target transformation matrix
    % Output: err = 6-element error vector [position_error; orientation_error]
    
    % Compute current pose
    T_current = fk_lite6(q);
    
    % Position error (meters)
    err_pos = T_current(1:3,4) - T_target(1:3,4);
    
    % Orientation error (from rotation matrix difference)
    R_current = T_current(1:3,1:3);
    R_target = T_target(1:3,1:3);
    
    % Compute rotation error using axis-angle representation
    R_error = R_current * R_target';
    
    % Extract rotation angle from the error matrix
    % Using: angle = arccos((trace(R) - 1)/2)
    % The axis is then derived from the matrix elements
    trace_R = trace(R_error);
    angle_rad = acos(max(-1, min(1, (trace_R - 1) / 2))); % Clamp to [-1, 1]
    
    if abs(angle_rad) < 1e-8
        err_rot = [0; 0; 0];
    else
        % Axis-angle representation
        err_rot = (angle_rad / (2 * sin(angle_rad))) * ...
                  [R_error(3,2) - R_error(2,3);
                   R_error(1,3) - R_error(3,1);
                   R_error(2,1) - R_error(1,2)];
    end
    
    % Combined error
    err = [err_pos; err_rot];
end
