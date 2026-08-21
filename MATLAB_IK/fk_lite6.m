function T = fk_lite6(q)
    % Forward Kinematics for UFACTORY Lite 6
    % Input: q = [q1, q2, q3, q4, q5, q6] in radians
    % Output: T = 4x4 homogeneous transformation matrix (base to end-effector)
    
    % DH Parameters (from your lab worksheet)
    % [theta_offset_deg, d_mm, alpha_deg, a_mm]
    dh = [
        0,     243.3,  -90,    0;
        -90,   0,      180,    200;
        -90,   0,      90,     87;
        0,     227.6,  90,     0;
        0,     0,      -90,    0;
        0,     61.5,   0,      0
    ];

    % Convert to meters and radians
    d = dh(:,2) / 1000;
    a = dh(:,4) / 1000;
    alpha = deg2rad(dh(:,3));
    theta_offset = deg2rad(dh(:,1));

    dh = [
        -0.0026, 242.604, -1.5766, -1.18;
        -1.5706, 1.362, 3.1404, 198.971;
        -1.5686, -1.362, 1.5693, 88.725;
        0.0097, 226.737, 1.5747, -1.375;
        -0.0025,0.478, -1.5716, -0.234;
        -0.0056, 62.276, 0.0037, -0.456
        ];

    d = dh(:,2) / 1000;
    a = dh(:,4) / 1000;
    alpha = dh(:,3);
    theta_offset = dh(:,1);
    
    % Compute transformation matrices
    T = eye(4);
    for i = 1:6
        theta = q(i) + theta_offset(i);
        
        % DH transformation matrix
        Ti = [cos(theta), -sin(theta)*cos(alpha(i)),  sin(theta)*sin(alpha(i)), a(i)*cos(theta);
              sin(theta),  cos(theta)*cos(alpha(i)), -cos(theta)*sin(alpha(i)), a(i)*sin(theta);
              0,           sin(alpha(i)),             cos(alpha(i)),             d(i);
              0,           0,                        0,                         1];
        T = T * Ti;
    end
end
