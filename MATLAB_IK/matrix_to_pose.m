function [pos_xyz, euler_xyz] = matrix_to_pose(T)
    % Extract pose from homogeneous matrix
    pos_xyz = T(1:3,4)';
    R = T(1:3,1:3);
    euler_xyz = rotm2eul(R,'zyx');
    euler_xyz = euler_xyz([3,2,1]);
end
