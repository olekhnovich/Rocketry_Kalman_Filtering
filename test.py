import numpy as np

import math
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from pyproj import Proj, transform
from scipy.spatial.transform import Rotation
from scipy.stats import chi2
from scipy.interpolate import interp1d

#initialization
gps_position = [[0,0,0],
                [0,0,0],
                [0,0,0]] #meters
gps_velocity = [[0,0,0],
                [0,0,0],
                [0,0,0]] #meters/second
init_position = np.zeros(3)  
init_velocity = np.zeros(3)
init_attitude = np.zeros(3)
init_angular_rate = np.zeros(3)
gps_time=0

class State:
    def __init__(self):
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0]) 
        self.accel_bias = np.array([0.0, 0.0, 0.0])
        self.gyro_bias = np.array([0.0, 0.0, 0.0])

class GPSData:
    def __init__(self, position, velocity,gnss_t):
        self.position = position
        self.velocity = velocity
        self.num_satellites = 4 # Placeholder value. Fix this later
        self.gps_time = gnss_t if np.isscalar(gnss_t) else gnss_t.item()

def quat_to_dcm(q):
    # q = [w, x, y, z]
    w,x, y, z, = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w),   1-2*(x*x+y*y)]
    ])
    return R

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    q3 = np.array([w,x,y,z])
    return q3

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def gps_check(num_satellites):
    if num_satellites < 4:
        return False
    return True

#constants
deg_to_rad = 0.01745329252
rad_to_deg = 1/deg_to_rad
micro_g_to_meters_per_second_squared = 9.80665*.000010

#convert gps from lat, lon, alt to suitable coordinates. Have mulitple written since different simulation softwares/sensors output different format.
#As of now, ENU is used
def lla_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref):
    # WGS84 ellipsoid
    proj_wgs84 = Proj(proj='latlong', datum='WGS84')
    proj_enu = Proj(proj='aeqd', lat_0=lat_ref, lon_0=lon_ref, datum='WGS84')
    x, y = transform(proj_wgs84, proj_enu, lon, lat)
    z = alt - alt_ref
    return np.array([x, y, z])
def lla_to_ned(lat, lon, alt, lat_ref, lon_ref, alt_ref):
    # WGS84 ellipsoid
    proj_wgs84 = Proj(proj='latlong', datum='WGS84')
    proj_ned = Proj(proj='aeqd', lat_0=lat_ref, lon_0=lon_ref, datum='WGS84')
    x, y = transform(proj_wgs84, proj_ned, lon, lat)
    return np.array([x, y, (alt - alt_ref)]) 

# Standard gravity constant (m/s^2)
GRAVITY = 9.80665
# Navigation gravity vector for ENU frame (x-east, y-north, z-up): gravity points down
GRAVITY_NAV_ENU = np.array([0.0, 0.0, -GRAVITY])

def compensate_lever_arm(accel_nosecone, gyro, lever_arm_body):
    """
    Remove rotational acceleration from nosecone accelerometer
    
    Args:
        accel_nosecone: [ax, ay, az] measured at nosecone in body frame
        gyro: [wx, wy, wz] angular rates in body frame (rad/s)
        lever_arm_body: [lx, ly, lz] vector from CG to nosecone in body frame
        
    Returns:
        accel_cg: corrected acceleration at center of mass
    """
    omega = np.array(gyro)
    r = np.array(lever_arm_body)
    
    # Centripetal: ω × (ω × r)
    centripetal = np.cross(omega, np.cross(omega, r))
    
    # Subtract to get CG acceleration
    # (we ignore angular acceleration α × r for simplicity unless you have high-rate gyro data)
    accel_cg = accel_nosecone - centripetal
    
    return accel_cg
#TODO implement actual data retrieval


#- Initialize Kalman filter covariance matrices


#state vector: [position(3), velocity(3), quaternion(4), accel_bias(3), gyro_bias(3)]
# inputs: state, imu_measurement(n by 3), dt(currently hard coded to 0.01s)
# output: updated imu state
def imu_propagation(state, imu_measurement):
    dt = 0.01
    position = state.position
    velocity = state.velocity
    quaternion = state.quaternion
    ba = state.accel_bias
    bg = state.gyro_bias

    body_frame_accel = np.array([imu_measurement[0], imu_measurement[1], imu_measurement[2]], dtype=float) - ba
    body_frame_gyro = np.array([imu_measurement[3], imu_measurement[4], imu_measurement[5]], dtype=float) - bg

    # attitude update
    theta = body_frame_gyro * dt
    angle = norm(theta)
    if angle > 1e-8:
        axis = theta / angle
        # Create quaternion in [w,x, y, z] format
        dq = np.array([
            math.cos(angle/2.0), # w
            axis[0] * math.sin(angle/2.0), # x
            axis[1] * math.sin(angle/2.0), # y
            axis[2] * math.sin(angle/2.0) # z
])
    else:
        # small-angle: dq ≈ [θ/2, 1] in [x,y,z,w] format
        dq = np.array([1,0.5 * theta[0], 0.5 * theta[1], 0.5 * theta[2] ])
    
    dq = dq / norm(dq)  # Normalize
    quaternion = quat_mul(quaternion, dq)
    quaternion = quaternion / norm(quaternion)

    # Rotate acceleration to nav frame
    C_b_n = quat_to_dcm(quaternion)
    accel_nav = C_b_n @ body_frame_accel#+ GRAVITY_NAV_ENU

    # Integrate
    vel_new = velocity + accel_nav * dt
    pos_new = position + velocity * dt + 0.5 * accel_nav * dt**2

    state.position = pos_new
    state.velocity = vel_new
    state.quaternion = quaternion
    return state

#P = error covariance matrix
#F = state transition matrix
#Q = process noise covariance matrix
def covariance_propagation(P, state, dt,imu_measurement):


    F = create_F(state,dt,imu_measurement)
    Q=create_Q(state,dt)
    P_next = F @ P @ F.T + Q
    
    return P_next

def create_F(state,dt, imu_measurement):
    F=np.eye(15)
    C_b_n = quat_to_dcm(state.quaternion)
    F[0:3,3:6] = np.eye(3)*dt
    ba = state.accel_bias
    body_frame_accel = [imu_measurement[0] - ba[0],
                        imu_measurement[1] - ba[1],
                        imu_measurement[2] - ba[2]] #meters/second^2
    # @TODO accel_body = state.last_accel  // Store this from IMU measurement
    F[3:6,6:9] = -C_b_n @ skew(body_frame_accel)*dt
    F[3:6,9:12] = -C_b_n*dt #accel bias
    F[6:9,12:15] = -np.eye(3)*dt #gyro bias
    return F

def create_Q(state,dt):
    #from rocketPy
    gyro_noise_density = np.array([0.01, 0.01, 0.05]) 
    gyro_rw_density    = np.array([0.01, 0.02, 0.05]) 
    accel_noise_density = 0.05       
    accel_rw_density = 0.03             
    accel_const_bias = 1.0   

    #Spectral Densities
    S_a = accel_noise_density**2          
    S_ba = accel_rw_density**2            
    S_g = (gyro_noise_density**2)         
    S_bg = (gyro_rw_density**2)  
    Q = np.zeros((15,15))
    Q[3:6,3:6] = np.eye(3) *  S_a * dt   
    Q[6:9,6:9] = np.diag(S_g * dt)       
    Q[9:12,9:12] = np.eye(3)*S_ba * dt               
    Q[12:15,12:15] = np.diag(S_bg * dt) 

    #an,gn = 0.05**2 *dt, 0.05**2*dt # m/s^2 / sqrt(Hz), rad/s / sqrt(Hz)
    #a_walk,g_walk = 0.02**2*dt, 0.01**2*dt # m/s^2 / sqrt(s), rad/s / sqrt(s)
    #Q=np.zeros((15,15))
    #Q[3:6,3:6] = np.eye(3)*(an**2)*dt #accel noise affects velocity
    #Q[6:9,6:9] = np.eye(3)*(gn**2)*dt #gyro noise affects attitude
    #Q[9:12,9:12] = np.eye(3)*(a_walk**2)*dt #accel bias random walk
    #Q[12:15,12:15] = np.eye(3)*(g_walk**2)*dt #gyro bias random walk
    return Q

def create_measurement_covariance(gps_data):

    R = np.zeros((3,3))

    gnss_sr = 1.0
    pos_acc = 12.0      
    alt_acc = 20.00      
    vel_sigma_guess = 0.25  #@TODO REPLACE PLACEHOLDER
    #R[0:2,0:2] = np.eye(2) * pos_acc**2   # East, North
    #R[2,2] = alt_acc**2                   # Up (or Down depending on frame) 
    #R[3:6,3:6] = np.eye(3) * (vel_sigma_guess**2)
    R[0,0] = pos_acc**2
    R[1,1] = pos_acc**2
    R[2,2] = alt_acc**2
    #@TODO replace placeholders
    #position_variance = 1**2
    #velocity_variance = 1**2  
    #R[0:3, 0:3] = np.eye(3) * position_variance
    #R[3:6, 3:6] = np.eye(3) * velocity_variance
    return R

#Update step with GPS measurement
def gps_update(state,P,gps_data):

    # @TODO check if data is bad, then skip update
    pos_pred = state.position
    vel_pred = state.velocity

    #residual = np.concatenate([gps_data.position - pos_pred,
     #                         gps_data.velocity - vel_pred])
    
    residual = gps_data.position - pos_pred
    # H is used to convert between state and measurement space
    H = np.zeros((3, 15))
    H[0:3, 0:3] = np.eye(3)
    #H[3:6, 3:6] = np.eye(3)

    # Measurement noise covariance (in measurement space)
    R = create_measurement_covariance(gps_data) 
    S= H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    error = K @ residual
    #print("GPS update @ t =", gps_data.gps_time)
    I_KH = np.eye(15) - K @ H
    P_new = I_KH @ P @ I_KH.T + K @ R @ K.T
    return error, P_new
    

def state_correct(state, error, P):
    state.position += error[0:3]
    state.velocity += error[3:6]
    orientation_error = error[6:9]
    orientation_error_norm = norm(orientation_error)
    
    if orientation_error_norm > 1e-8:
        axis = orientation_error / orientation_error_norm
        dq = np.array([math.cos(orientation_error_norm/2),
            axis[0] * math.sin(orientation_error_norm/2),
            axis[1] * math.sin(orientation_error_norm/2),
            axis[2] * math.sin(orientation_error_norm/2)
            
        ])  # [x, y, z, w] format
    else:
        dq = np.array([1,
            orientation_error[0]/2,
            orientation_error[1]/2,
            orientation_error[2]/2
           
        ])
    
    state.quaternion = quat_mul(dq,state.quaternion)
    state.quaternion = state.quaternion / norm(state.quaternion)
    state.accel_bias += error[9:12]
    state.gyro_bias += error[12:15]
    return state

def imu_data_available():
    # @TODO implement actual check
    return True
def gps_data_available():
    # @TODO implement actual check
    return True

def initialize_covariance():
    P = np.eye(15)
    P[0:3,0:3] *= 12**2  # position 
    P[3:6,3:6] *= 0.75**2     # velocity 
    P[6:9,6:9] *= (np.pi)**2  # attitude unknown
    P[9:12,9:12] *= 0.000001**2   # accel bias
    P[12:15,12:15] *= 0.000001**2 # gyro bias
    return P


#data = pandas.read_csv(r'C:\Users\labra\OneDrive\Documents\School\WISP\simulationTrajectory.csv')
# Rename columns to match expected format for sensor and state data.
# The CSV should have columns: t_imu, t_gps, x, y, z, x_gps, y_gps, z_gps, vx, vy, vz, ax, ay, az, w, gx, gy, gz
#data.columns = ["t_imu", "x", "y", "z","x_gps","y_gps","z_gps","vx","vy","vz","ax","ay","az","w","gx","gy","gz"]
#t = data['t_imu'].values
#t_gps = data['t_gps'].values
#pos_x = data['x'].values
#pos_x_gps = data['x_gps'].values
#pos_y_gps = data['y_gps'].values
#pos_z_gps = data['z_gps'].values
#pos_y = data['y'].values
#pos_z = data['z'].values
#vel_x_imu = data['vx'].values
#vel_y_imu = data['vy'].values
#vel_z_imu = data['vz'].values
# Compute GPS velocities by numerical differentiation of positions
#vel_x_gps = np.gradient(pos_x, t)
#vel_y_gps = np.gradient(pos_y, t)
#vel_z_gps = np.gradient(pos_z, t)

#imu_csv = pandas.read_csv('imu_data.csv')
#gps_csv = pandas.read_csv('gps_data.csv')
#data = pandas.read_csv('simulationTrajectory.csv')

#gps_data = [GPSData(pos, vel) for pos, vel in zip(data[['x_gps','y_gps','z_gps']].to_numpy(), data[['vx','vy','vz']].to_numpy())]
#imu_data = data[['ax','ay','az','w','gx','gy','gz']].to_numpy()
#print(gps_data[0])

#gps_pos = data[['x_gps','y_gps','z_gps']].to_numpy()


#import and setup data
# Load raw data
accel_csv = pd.read_csv('Defiance_Simulation_accel_noisy_nosecone.csv')
gyro_csv = pd.read_csv('Defiance_Simulation_gyro_noisy.csv')
gnss_csv = pd.read_csv('Defiance_Simulation_gnss_clean.csv')
gt_csv = pd.read_csv('Defiance_Simulation_gnss_ground_truth.csv')

accel_csv.columns = ["t", "ax", "ay", "az"]
at = accel_csv["t"].values
ax = accel_csv["ax"].values
ay = accel_csv["ay"].values
az = accel_csv["az"].values

gyro_csv.columns = ["t", "wx", "wy", "wz"]
gt = gyro_csv["t"].values
wx_deg = gyro_csv["wx"].values  # Keep in degrees for now
wy_deg = gyro_csv["wy"].values
wz_deg = gyro_csv["wz"].values

# Convert to radians ONCE
wx_rad = np.radians(wx_deg)
wy_rad = np.radians(wy_deg)
wz_rad = np.radians(wz_deg)

print(f"\n=== Initial Sensor Check ===")
print(f"First 5 accel (nosecone): ax={ax[0:5]}")
print(f"First 5 gyro (deg/s): wx={wx_deg[0:5]}")
print(f"First 5 gyro (rad/s): wx={wx_rad[0:5]}")

# Lever arm compensation
LEVER_ARM_BODY = np.array([0.0, 0.0, 0.563])  # meters

ax_corrected = []
ay_corrected = []
az_corrected = []

for i in range(len(ax)):
    accel_nosecone = np.array([ax[i], ay[i], az[i]])
    gyro = np.array([wx_rad[i], wy_rad[i], wz_rad[i]])  # Use radians
    accel_cg = compensate_lever_arm(accel_nosecone, gyro, LEVER_ARM_BODY)
    ax_corrected.append(accel_cg[0])
    ay_corrected.append(accel_cg[1])
    az_corrected.append(accel_cg[2])

ax = np.array(ax_corrected)
ay = np.array(ay_corrected)
az = np.array(az_corrected)
# Load GPS data
gnss_csv.columns = ["t","latitude","longitude","altitude"]
gnss_t = gnss_csv["t"].values
lat = gnss_csv["latitude"].values
lon = gnss_csv["longitude"].values
alt = gnss_csv["altitude"].values

gt_csv.columns = ["t","latitude","longitude","altitude"]
gt_t = gt_csv["t"].values
gt_lat = gt_csv["latitude"].values
gt_lon = gt_csv["longitude"].values
gt_alt = gt_csv["altitude"].values

# Convert GPS coordinates
lat_ref = lat[0]
lon_ref = lon[0]
alt_ref = alt[0]

gps_pos = lla_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref)
true_pos = lla_to_enu(gt_lat, gt_lon, gt_alt, lat_ref, lon_ref, alt_ref)

pos_x_gps, pos_y_gps, pos_z_gps = gps_pos[0], gps_pos[1], gps_pos[2]
pos_x, pos_y, pos_z = true_pos[0], true_pos[1], true_pos[2]

vel_x_gps = np.gradient(pos_x_gps, gnss_t)
vel_y_gps = np.gradient(pos_y_gps, gnss_t)
vel_z_gps = np.gradient(pos_z_gps, gnss_t)

gps_data = [GPSData(np.array([pos_x_gps[i], pos_y_gps[i], pos_z_gps[i]]),
                    np.array([vel_x_gps[i], vel_y_gps[i], vel_z_gps[i]]), 
                    gnss_t[i])  # Not np.array() - keep scalar
            for i in range(len(pos_x_gps))]

#print(f"Total GPS measurements: {len(gps_data)}")
#print(f"GPS time range: {gps_data[0].gps_time:.2f} to {gps_data[-1].gps_time:.2f}")
imu_data = np.column_stack([ax, ay, az, wx_rad,  wy_rad,  wz_rad,at])


pos_log = []
vel_log = []
quat_log = []
t_log = []

static_accel_data = imu_data[0:50, 0:3] 
avg_accel = np.mean(static_accel_data, axis=0)
ax_avg, ay_avg, az_avg = avg_accel
roll_rad = math.atan2(-ay_avg, az_avg) 
pitch_rad = math.atan2(ax_avg, math.sqrt(ay_avg**2 + az_avg**2))
yaw_rad = 0.0
initial_quat_wxyz = Rotation.from_euler('xyz', [roll_rad, pitch_rad, yaw_rad]).as_quat() # [x, y, z, w]
initial_quat_wlast = [initial_quat_wxyz[3], initial_quat_wxyz[0], initial_quat_wxyz[1], initial_quat_wxyz[2]] # Convert to [w, x, y, z]

def loop():

    state = State()
    
    state.position = gps_data[0].position.copy()
    state.velocity = gps_data[0].velocity.copy()
    state.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # A guess. We likely won't know this. Account by setting initial uncertainty to be really high
    
    P = initialize_covariance()
    
    imu_index = 0
    gps_index = 0
    gps_update_count = 0
    
    
    while imu_index < len(imu_data):
        imu_measurement = imu_data[imu_index]
        current_time = imu_measurement[6]
        
        # Propagate with IMU
        state = imu_propagation(state, imu_measurement)
        P = covariance_propagation(P, state, 0.01, imu_measurement)
        
        # Check if we should do GPS update
        # GPS update happens when IMU time catches up to next GPS measurement
        if gps_index < len(gps_data):
            next_gps_time = gps_data[gps_index].gps_time
            time_diff = current_time - next_gps_time
            
            # If IMU time is past GPS time (within tolerance), do update
            if abs(time_diff) < 0.015:  # Within 15ms
                pos_residual = gps_data[gps_index].position - state.position
                vel_residual = gps_data[gps_index].velocity - state.velocity
                error, P = gps_update(state, P, gps_data[gps_index])
                state = state_correct(state, error, P)
                
                gps_index += 1
                gps_update_count += 1
        
        # Log state
        pos_log.append(state.position.copy())
        vel_log.append(state.velocity.copy())
        quat_log.append(state.quaternion.copy())
        t_log.append(current_time)
        
        imu_index += 1
    
    print(f"\nFilter complete. Applied {gps_update_count} GPS updates out of {len(gps_data)} available.")
    return pos_log, vel_log, quat_log, t_log

print(f"\n=== Initial Sensor Check ===")
print(f"First 5 accel (nosecone): ax={ax[0:5]}")
print(f"First 5 gyro (deg/s): wx={wx_deg[0:5]}")
print(f"First 5 gyro (rad/s): wx={wx_rad[0:5]}")
print(f"\n=== After Lever Arm Compensation ===")
print(f"First 5 CG accel: ax={ax[0:5]}")
print(f"Expected at rest: ~[0, 0, -9.8] m/s^2")
print(f"Actual first value: [{ax[0]:.2f}, {ay[0]:.2f}, {az[0]:.2f}]")
print(f"\n=== GPS Data Check ===")
print(f"First GPS position (ENU): [{pos_x_gps[0]:.2f}, {pos_y_gps[0]:.2f}, {pos_z_gps[0]:.2f}]")
print(f"First GPS velocity: [{vel_x_gps[0]:.2f}, {vel_y_gps[0]:.2f}, {vel_z_gps[0]:.2f}]")
print(f"First ground truth: [{pos_x[0]:.2f}, {pos_y[0]:.2f}, {pos_z[0]:.2f}]")


pos_log, vel_log, quat_log, t_log = loop()
pos_log = np.array(pos_log)
vel_log = np.array(vel_log) 
quat_log = np.array(quat_log)
t_log = np.array(t_log)


figure = plt.figure(1)
ax=figure.add_subplot(111,projection='3d')
ax.plot(pos_log[:,0], pos_log[:,1], pos_log[:,2], label='Estimated Position')
ax.plot(pos_x, pos_y,pos_z, label='Ground Truth', linestyle='dashed')
ax.plot(pos_x_gps, pos_y_gps, pos_z_gps, label='GPS Position')
ax.legend()
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
plt.tight_layout()

#downsample ground truth and plot the errors
interp_gt_x = interp1d(gt_t, pos_x, fill_value="extrapolate")
pos_x_interp = interp_gt_x(t_log)
interp_gt_y = interp1d(gt_t, pos_y, fill_value="extrapolate")
pos_y_interp = interp_gt_y(t_log)
interp_gt_z = interp1d(gt_t, pos_z, fill_value="extrapolate")
pos_z_interp = interp_gt_z(t_log)

interp_gps_x = interp1d(gnss_t, pos_x_gps, fill_value="extrapolate")
pos_x_gps_interp = interp_gps_x(t_log)
interp_gps_y = interp1d(gnss_t, pos_y_gps, fill_value="extrapolate")
pos_y_gps_interp = interp_gps_y(t_log)
interp_gps_z = interp1d(gnss_t, pos_z_gps, fill_value="extrapolate")
pos_z_gps_interp = interp_gps_z(t_log)

fig2 = plt.figure(2)
ax1 = fig2.add_subplot(111)
#ax1.plot(pos_x-pos_log[:,0], label='Position X Error')
ax1.plot(t_log,pos_y_interp-pos_log[:,1], label='Position Y Error')
ax1.plot(t_log,pos_y_interp-pos_y_gps_interp, label='GPS Y Error')
ax1.plot(t_log,pos_y_interp-pos_y_interp, label =  'Ground Truth')
ax1.set_ylabel('Position Error (m)')
ax1.legend()
plt.tight_layout()


fig3 = plt.figure(3)
ax2 = fig3.add_subplot(111)
ax2.plot(t_log,pos_z_interp-pos_log[:,2], label='Position Z Error')
ax2.plot(t_log,pos_z_interp-pos_z_gps_interp, label='GPS Z Error')
ax2.plot(t_log,pos_z_interp-pos_z_interp, label = 'Ground Truth')
ax2.set_ylabel('Position Error (m)')
ax2.legend()
plt.title('Position Z Error')
plt.tight_layout()

fig4 = plt.figure(4)
ax3 = fig4.add_subplot(111)
ax3.plot(t_log,pos_x_interp-pos_log[:,0], label='Position X Error')
ax3.plot(t_log,pos_x_interp-pos_x_gps_interp, label='GPS X Error')
ax3.plot(t_log,pos_x_interp-pos_x_interp, label =  'Ground Truth')
ax3.set_ylabel('Position Error (m)')
ax3.legend()
plt.tight_layout()
plt.show()


#ax1 = figure.add_subplot(311)
#ax1.plot(t_log, pos_log[:,0], label='Estimated Position X')
#ax1.plot(data['t_gps'], gps_pos[:,0], label='GPS Position X', linestyle='dashed')
#ax1.set_ylabel('X (m)')
#ax1.legend()
#plt.tight_layout()
#plt.show()

#dx=np.mean((pos_log[:,0]-pos_x)**2)
#dy=np.mean((pos_log[:,1]-pos_y)**2)
#dx_gps =np.mean((pos_x_gps-pos_x)**2)
#dy_gps =np.mean((pos_y_gps-pos_y)**2)







