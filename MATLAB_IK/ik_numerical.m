%% UFACTORY Lite 6: Numerical Inverse Kinematics
% Uses the Levenberg-Marquardt algorithm (fsolve) to find joint angles
% that achieve a target end-effector pose.
% All functions are defined at the bottom of this script.

clear; clc; close all;

%% ============================================================
% PART 1: MAIN DEMONSTRATION
% ============================================================

% Define joint limits for UFACTORY Lite 6 (from your lab worksheet)
% Format: [min_rad, max_rad]
joint_limits_deg = [
    -360, 360;      % J1
    -150, 150;      % J2
    -3.5, 300;      % J3 (note: min is -3.5 degrees)
    -360, 360;      % J4
    -124, 124;      % J5
    -360, 360       % J6
];
joint_limits_rad = deg2rad(joint_limits_deg);


% --- Test Configuration 1: Home Position ---
disp('--- Test 1: Home Position ---');
q_home = [-180, 0, 4, 0, -45, 0]; % Home joint angles
q_home = deg2rad(q_home);
T_home = fk_lite6(q_home);

% Try to solve with a perturbed initial guess
q0 = [0.1, -1.2, 0.1, 0.1, 0.1, 0.1]; % Perturbed guess
[q_sol, success, error_norm] = ik_solve(T_home, q0, joint_limits_rad);

if success
    disp('Solution found!');
    disp(['Joint angles (deg): ', num2str(rad2deg(q_sol))]);
    disp(['Final error norm: ', num2str(error_norm, '%.2e')]);
    
    % Verify by computing forward kinematics
    T_verify = fk_lite6(q_sol);
    pos_error = norm(T_verify(1:3,4) - T_home(1:3,4)) * 1000; % in mm
    orientation_error = T_verify(1:3,1:3)*T_home(1:3,1:3)';
    angle_rad = acos(max(-1, min(1, (trace(orientation_error) - 1) / 2))); % Clamp to [-1, 1]

    disp(['Position error: ', num2str(pos_error, '%.3f'), ' mm']);
    disp(['Angle error: ', num2str(angle_rad, '%.3f'), ' rad']);
else
    disp('IK solver failed to find a solution.');
end

disp(' ');

% --- Test Configuration 2: Elbow Down ---
disp('--- Test 2: Elbow Down ---');
q_elbow = [-180, -18.8, 4, 0, -45, 0];
q_elbow = deg2rad(q_elbow);
T_elbow = fk_lite6(q_elbow);

% Different initial guess to find elbow down solution
q0_elbow = [0.1, -0.8, 0.3, 0.1, 0.1, 0.1];
[q_sol, success, error_norm] = ik_solve(T_elbow, q0_elbow, joint_limits_rad);

if success
    disp('Solution found!');
    disp(['Joint angles (deg): ', num2str(rad2deg(q_sol))]);
    disp(['Final error norm: ', num2str(error_norm, '%.2e')]);
    
    T_verify = fk_lite6(q_sol);
    pos_error = norm(T_verify(1:3,4) - T_elbow(1:3,4)) * 1000; % in mm
    orientation_error = T_verify(1:3,1:3)*T_elbow(1:3,1:3)';
    angle_rad = acos(max(-1, min(1, (trace(orientation_error) - 1) / 2))); % Clamp to [-1, 1]

    disp(['Position error: ', num2str(pos_error, '%.3f'), ' mm']);
    disp(['Angle error: ', num2str(angle_rad, '%.3f'), ' rad']);
else
    disp('IK solver failed to find a solution.');
end

disp(' ');

% --- Test Configuration 3: Complex Pose ---
disp('--- Test 3: Complex Pose ---');
q_complex = [-302.2 -19.9 37.2 -207.4 -108.7 -39.9];
q_complex = deg2rad(q_complex);
T_complex = fk_lite6(q_complex);

q0_complex = [0.3, -0.8, 0.5, 0.2, 0.3, 0.1];
[q_sol, success, error_norm] = ik_solve(T_complex, q0_complex, joint_limits_rad);

if success
    disp('Solution found!');
    disp(['Joint angles (deg): ', num2str(rad2deg(q_sol))]);
    disp(['Final error norm: ', num2str(error_norm, '%.2e')]);
    
    T_verify = fk_lite6(q_sol);
    pos_error = norm(T_verify(1:3,4) - T_complex(1:3,4)) * 1000;
    orientation_error = T_verify(1:3,1:3)*T_complex(1:3,1:3)';
    angle_rad = acos(max(-1, min(1, (trace(orientation_error) - 1) / 2))); % Clamp to [-1, 1]

    disp(['Position error: ', num2str(pos_error, '%.3f'), ' mm']);
    disp(['Angle error: ', num2str(angle_rad, '%.3f'), ' rad']);
else
    disp('IK solver failed to find a solution.');
end
%%
pose=[0 0 0 0 0 0]; %x y z yaw pitch roll
pos_xyz = pose(1:3)/1000;
eul_xyz = deg2rad([pose(4) pose(5) pose(6)]); % Check this
target_T = pose_to_matrix(pos_xyz, eul_xyz);
% Different initial guesses
q_guesses = [[0.1, -1.2, 0.1, 0.1, 0.1, 0.1];   % Slightly above home
[0.1, -0.8, -0.1, 0.1, 0.1, 0.1];  % Elbow slightly up
[0.1, -1.5, 0.3, 0.1, 0.1, 0.1];   % Different arm angle
[0.1, -1.0, -0.3, 0.1, 0.1, 0.1];   % Another variation
deg2rad([30 45 120 90 45 200])
];

% q_guess0 = mean(joint_limits_rad');
disp('Different solutions for the same target pose (home position):');

% [q_sol, success, error_norm] = ik_solve(target_T, q_guess0, joint_limits_rad);
% if success && error_norm < 1e-4
%     q_deg = rad2deg(q_sol);
%     fprintf('Solution %d: [%6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f] deg\n', i, q_deg(1), q_deg(2), q_deg(3), q_deg(4), q_deg(5), q_deg(6));
% else
%     fprintf("Solulu delulu\n");
% end
% 
for i = 1:size(q_guesses, 1)
    [q_sol, success, error_norm] = ik_solve(target_T, q_guesses(i,:), joint_limits_rad);
    if success && error_norm < 1e-4
        q_deg = rad2deg(q_sol);
        fprintf('Solution %d: [%6.1f, %6.1f, %6.1f, %6.1f, %6.1f, %6.1f] deg\n', i, q_deg(1), q_deg(2), q_deg(3), q_deg(4), q_deg(5), q_deg(6));
    end
end