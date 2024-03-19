import numpy as np
import scipy.linalg


class KalmanFilter(object):
    """
    A simplified Kalman filter for tracking objects in world coordinates (x, y).
    
    The 4-dimensional state space,
        x, y, vx, vy
    contains the object position (x, y) and velocity (vx, vy).
    
    Object motion follows a constant velocity model. The object location
    (x, y) is taken as direct observation of the state space (linear observation model).
    """

    def __init__(self):
        ndim, dt = 2, 1.  # Now working with a 2-dimensional state (x, y)

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.
        
        Parameters:
            measurement : ndarray
                Detected object coordinates (x, y) with center position.
        
        Returns:
            (ndarray, ndarray)
                Returns the mean vector (4 dimensional) and covariance matrix (4x4
                dimensional) of the new track. Unobserved velocities are initialized
                to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[1],  # std_x
            2 * self._std_weight_position * measurement[1],  # std_y
            10 * self._std_weight_velocity * measurement[1],  # std_vx
            10 * self._std_weight_velocity * measurement[1]]  # std_vy
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.
        
        Parameters:
            mean : ndarray
                The 4 dimensional mean vector of the object state at the previous
                time step.
            covariance : ndarray
                The 4x4 dimensional covariance matrix of the object state at the
                previous time step.
        
        Returns:
            (ndarray, ndarray)
                Returns the mean vector and covariance matrix of the predicted
                state.
        """
        std_pos = [
            self._std_weight_position * mean[3],  # std_x
            self._std_weight_position * mean[3]]  # std_y
        std_vel = [
            self._std_weight_velocity * mean[3],  # std_vx
            self._std_weight_velocity * mean[3]]  # std_vy
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.
        
        Parameters:
            mean : ndarray
                The predicted state's mean vector (4 dimensional).
            covariance : ndarray
                The state's covariance matrix (4x4 dimensional).
            measurement : ndarray
                The 2-dimensional measurement vector (x, y), where (x, y)
                is the center position of the object.
        
        Returns:
            (ndarray, ndarray)
                Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.
        
        Parameters:
            mean : ndarray
                The state's mean vector (4 dimensional array).
            covariance : ndarray
                The state's covariance matrix (4x4 dimensional).
        
        Returns:
            (ndarray, ndarray)
                Returns the projected mean and covariance matrix of the given state
                estimate.
        """
        std = [
            self._std_weight_position * mean[3],  # std_x
            self._std_weight_position * mean[3]]  # std_y
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov


if __name__ == '__main__':
    # Simulate object movement and noisy measurements
    np.random.seed(42)  # For reproducible results

    # True initial position and velocity
    true_position = np.array([0, 0])
    true_velocity = np.array([1, 0.5])  # Moves right and slightly down

    # Generate noisy measurements
    num_measurements = 20
    measurements = [true_position + true_velocity * i + np.random.normal(0, 0.5, 2) for i in range(num_measurements)]

    # Initialize the Kalman Filter
    kf = KalmanFilter()
    mean, covariance = kf.initiate(measurements[0])

    # Lists to store filter predictions and updates
    predictions = [mean[:2]]
    updates = []

    # Process measurements and update Kalman Filter
    for measurement in measurements[1:]:
        # Prediction step
        mean, covariance = kf.predict(mean, covariance)
        predictions.append(mean[:2])  # Store the prediction

        # Update step
        mean, covariance = kf.update(mean, covariance, measurement)
        updates.append(mean[:2])  # Store the updated estimate

    # Visualization
    import matplotlib.pyplot as plt

    # Extract x and y coordinates for plotting
    true_positions = np.array([true_position + true_velocity * i for i in range(num_measurements)])
    measured_positions = np.array(measurements)
    predicted_positions = np.array(predictions)
    updated_positions = np.array(updates)

    plt.figure(figsize=(10, 8))
    plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', label='True Position')
    plt.plot(measured_positions[:, 0], measured_positions[:, 1], 'rx', label='Measured Position')
    plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], 'b--', label='Predicted Position')
    plt.plot(updated_positions[:, 0], updated_positions[:, 1], 'ko-', label='Updated Position')
    plt.legend()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Kalman Filter Tracking')
    plt.show()
