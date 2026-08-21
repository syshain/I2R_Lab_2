function T = pose_to_matrix(pos_xyz, euler_xyz)
    % Convert pose to homogeneous matrix
    % pos_xyz = [x, y, z] in meters
    % euler_xyz = [roll, pitch, yaw] in radians (XYZ fixed angles)
    R = eul2rotm(euler_xyz([3,2,1]), 'zyx');
    T = [R, pos_xyz'; 0 0 0 1];
end
