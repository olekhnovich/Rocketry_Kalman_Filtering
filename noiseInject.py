import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dt = 0.01  


accel_acc = 0.05 
gyro_acc = 0.1 
gps_pos_acc = 12.0 

# --- LOAD DATA ---
df = pd.read_csv('openWispRocketSimulationOne.csv', comment='#')
df.columns = ["Time (s)", "Altitude (m)", "Altitude above sea level (m)", "Vertical velocity (m/s)",
              "Total velocity (m/s)", "Vertical acceleration (m/sÂ²)", "Total acceleration (m/sÂ²)",
              "Position East of launch (m)", "Position North of launch (m)","Lateral distance (m)",
              "Lateral direction (Â°)","Lateral velocity (m/s)","Lateral acceleration (m/sÂ²)","Latitude (Â° N)",
              "Longitude (Â° E)","Angle of attack (Â°)","Roll rate (r/s)","Pitch rate (r/s)","Yaw rate (r/s)",
              "Vertical orientation (zenith) (Â°)","Lateral orientation (azimuth) (Â°)","Mass (g)",
              "Motor mass (g)","Longitudinal moment of inertia (kgÂ·mÂ²)","Rotational moment of inertia (kgÂ·mÂ²)",
              "Gravitational acceleration (m/sÂ²)","CP location (cm)","CG location (cm)","Stability margin calibers (â€‹)",
              "Thrust (N)","Thrust-to-weight ratio (â€‹)","Drag force (N)","Drag coefficient (â€‹)",
              "Friction drag coefficient (â€‹)","Pressure drag coefficient (â€‹)","Base drag coefficient (â€‹)",
              "Axial drag coefficient (â€‹)","Normal force coefficient (â€‹)","Pitch moment coefficient (â€‹)",
              "Yaw moment coefficient (â€‹)","Side force coefficient (â€‹)","Roll moment coefficient (â€‹)",
              "Roll forcing coefficient (â€‹)","Roll damping coefficient (â€‹)","Pitch damping coefficient (â€‹)",
              "Wind velocity (m/s)","Wind direction (Â°)","Air temperature (Â°C)","Air pressure (mbar)",
              "Air density (g/cmÂ³)","Speed of sound (m/s)","Mach number (â€‹)","Reynolds number (â€‹)","Reference length (cm)",
              "Reference area (cmÂ²)","Simulation time step (s)","Computation time (s)","Coriolis acceleration (m/sÂ²)"]

t = df.filter(like='Time (s)').iloc[:, 0].values
lat = df.filter(like='Latitude (Â° N)').iloc[:, 0].values
lon = df.filter(like='Longitude (Â° E)').iloc[:, 0].values
alt = df.filter(like='Altitude (m)').iloc[:, 0].values

#store ground truth
#OpenRocket outputs only lateral acceleration. We assume the XZ plane has been rotate s.t. there is no motion along Y.
accel_z_truth = df.filter(like='Vertical acceleration (m/sÂ²)').iloc[:, 0].values + 9.81
accel_x_truth = df.filter(like='Lateral acceleration (m/sÂ²)').iloc[:, 0].values 
accel_y_truth = np.zeros_like(accel_x_truth) 

gyro_x_truth = np.radians(df.filter(like='Pitch rate (r/s)').iloc[:, 0].values)
gyro_y_truth = np.radians(df.filter(like='Yaw rate (r/s)').iloc[:, 0].values)
gyro_z_truth = np.radians(df.filter(like='Roll rate (r/s)').iloc[:, 0].values)

#np.random.seed(42)

ax_noisy = accel_x_truth + np.random.normal(0, accel_acc, len(t))
ay_noisy = accel_y_truth + np.random.normal(0, accel_acc, len(t))
az_noisy = accel_z_truth + np.random.normal(0, accel_acc, len(t))


wx_noisy = gyro_x_truth + np.random.normal(0, gyro_acc, len(t))
wy_noisy = gyro_y_truth + np.random.normal(0, gyro_acc, len(t))
wz_noisy = gyro_z_truth + np.random.normal(0, gyro_acc, len(t))


lat_noisy = lat + (np.random.normal(0, gps_pos_acc, len(t)) / 111139) 
lon_noisy = lon + (np.random.normal(0, gps_pos_acc, len(t)) / 111139)
alt_noisy = alt + np.random.normal(0, gps_pos_acc, len(t)) 

df_accel = pd.DataFrame({'t': t, 'ax': ax_noisy, 'ay': ay_noisy, 'az': az_noisy})
df_accel.to_csv('WISP_Simulation_accel_noisy_nosecone.csv', index=False)

df_gyro = pd.DataFrame({'t': t, 'wx': np.degrees(wx_noisy), 'wy': np.degrees(wy_noisy), 'wz': np.degrees(wz_noisy)})
df_gyro.to_csv('WISP_Simulation_gyro_noisy.csv', index=False)

df_gnss = pd.DataFrame({'t': t, 'latitude': lat_noisy, 'longitude': lon_noisy, 'altitude': alt_noisy})
df_gnss.to_csv('WISP_Simulation_gnss_clean.csv', index=False)

df_gt = pd.DataFrame({'t': t, 'latitude': lat, 'longitude': lon, 'altitude': alt})
df_gt.to_csv('WISP_Simulation_gnss_ground_truth.csv', index=False)

print("Success! Generated 4 sensor files for your EKF.")