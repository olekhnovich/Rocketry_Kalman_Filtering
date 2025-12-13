import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from rocketpy import Environment, Flight, Function, Rocket
from rocketpy.motors import CylindricalTank, Fluid, HybridMotor
from rocketpy.motors.tank import MassFlowRateBasedTank
from rocketpy import Accelerometer, Barometer, GnssReceiver, Gyroscope
import pandas as pd

flight_date = datetime.date(2024, 8, 24)
env = Environment(latitude=47.966527, longitude=-81.87413, elevation=1383.4)

env.set_date((flight_date.year, flight_date.month, flight_date.day, 0))
env.set_atmospheric_model(type="custom_atmosphere", wind_v=1.0, wind_u=-2.9)

#env.info()

oxidizer_liq = Fluid(name="N2O_l", density=960)
oxidizer_gas = Fluid(name="N2O_g", density=1.9277)

tank_shape = CylindricalTank(0.0665, 1.79)

oxidizer_tank = MassFlowRateBasedTank(
    name="oxidizer_tank",
    geometry=tank_shape,
    flux_time=(6.5),
    liquid=oxidizer_liq,
    gas=oxidizer_gas,
    initial_liquid_mass=20,
    initial_gas_mass=0,
    liquid_mass_flow_rate_in=0,
    liquid_mass_flow_rate_out=20 / 6.5,
    gas_mass_flow_rate_in=0,
    gas_mass_flow_rate_out=0,
)

def thrust_fuction(t):
    return 5750 * 2 ** (-t / 200)

hybrid_motor = HybridMotor(
    
    thrust_source=str(Path("C:/Users/labra/OneDrive/Documents/School/WISP/Thrust_curve.csv")),
    dry_mass=13.832,
    dry_inertia=(1.801, 1.801, 0.0305),
    center_of_dry_mass_position=780 / 1000,
    reshape_thrust_curve=False,
    grain_number=1,
    grain_separation=0,
    grain_outer_radius=0.0665,
    grain_initial_inner_radius=0.061,
    grain_initial_height=1.25,
    grain_density=920,
    nozzle_radius=0.0447,
    throat_radius=0.0234,
    interpolation_method="linear",
    grains_center_of_mass_position=0.377,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)
hybrid_motor.add_tank(tank=oxidizer_tank, position=2.2)

defiance = Rocket(
    radius=0.07,
    mass=37.211,
    # inertia = (180.142, 180.142, 0.262),
    inertia=(94.14, 94.14, 0.09),
    center_of_mass_without_motor=3.29,
    power_off_drag = str(Path("C:/Users/labra/OneDrive/Documents/School/WISP/DragCurve.csv")),
    power_on_drag=str(Path("C:/Users/labra/OneDrive/Documents/School/WISP/DragCurve.csv")),
    coordinate_system_orientation="tail_to_nose",
)

defiance.add_motor(hybrid_motor, position=0.2)
defiance.add_nose(length=0.563, kind="vonKarman", position=4.947)
defiance.add_trapezoidal_fins(
    n=3, span=0.115, root_chord=0.4, tip_chord=0.2, position=0.175
)
defiance.add_tail(top_radius=0.07, bottom_radius=0.064, length=0.0597, position=0.1)
defiance.add_parachute(name="main", cd_s=2.2, trigger=305, sampling_rate=100, lag=0)
defiance.add_parachute(
    name="drogue", cd_s=1.55, trigger="apogee", sampling_rate=100, lag=0
)

accel_noisy_nosecone = Accelerometer(
    sampling_rate=100,
    consider_gravity=True,
    #orientation=(0, 0, 0),
    #measurement_range=70,
    #resolution=0.4882,
    noise_density=0.05,
    random_walk_density=0.02,
    constant_bias=1,
    #operating_temperature=25,
    #temperature_bias=0.02,
    #temperature_scale_factor=0.02,
    #cross_axis_sensitivity=0.02,
    name="Accelerometer in Nosecone",
)
accel_clean_cdm = Accelerometer(
    sampling_rate=100,
    consider_gravity=True,
    orientation=[
        [0.25, -0.0581, 0.9665],
        [0.433, 0.8995, -0.0581],
        [-0.8661, 0.433, 0.25],
    ],
    name="Accelerometer in CDM",
)
defiance.add_sensor(accel_noisy_nosecone, 1.278)
defiance.add_sensor(accel_clean_cdm, -0.10482544178314143)  # , 127/2000)
gyro_clean = Gyroscope(sampling_rate=100)
gyro_noisy = Gyroscope(
    sampling_rate=100,
    #resolution=0.001064225153655079,
    orientation=(0, 0, 0),
    noise_density=[0, 0.03, 0.05],
    noise_variance=1,

    random_walk_density=[0, 0.01, 0.02],
    random_walk_variance=[1, 1, 1.05],
    constant_bias=[0, 0.3, 0.5],
    #operating_temperature=25,
    #temperature_bias=[0, 0.01, 0.02],
    #temperature_scale_factor=[0, 0.01, 0.02],
    #cross_axis_sensitivity=0.5,
    #acceleration_sensitivity=[0, 0.0008, 0.0017],
    name="Gyroscope",
)


gyro_ground_truth = Gyroscope(
    sampling_rate=1000,
    name="Gyro Ground Truth",
    orientation=(0, 0, 0),
    noise_density=[0, 0.0, 0.0],
    random_walk_density=[0, 0.0, 0.002],
    random_walk_variance=[0, 0, 0.0],
    constant_bias=[0, 0.0, 0.0],
)
defiance.add_sensor(gyro_ground_truth, -0.1048 + 0.5)  # , +127/2000)
defiance.add_sensor(gyro_clean, -0.10482544178314143)  # +0.5, 127/2000)
defiance.add_sensor(gyro_noisy, (1.278 - 0.4, 127 / 2000 - 127 / 4000, 0))

gnss_ground_truth = GnssReceiver(
    sampling_rate=500,              # high fs
    position_accuracy=0,            # perfect
    altitude_accuracy=0,
    name="GNSS Ground Truth",
)
defiance.add_sensor(gnss_ground_truth, (-0.1048 + 0.5, +127 / 2000, 0))

gnss_clean = GnssReceiver(
    sampling_rate=1,
    position_accuracy=12,
    altitude_accuracy=20,
)
defiance.add_sensor(gnss_clean, (-0.10482544178314143 + 0.5, +127 / 2000, 0))

test_flight = Flight(
    rocket=defiance, environment=env, inclination=85, heading=90, rail_length=10
)

test_flight.prints.apogee_conditions()
test_flight.plots.trajectory_3d()
#print(test_flight)
test_flight.prints.apogee_conditions()


#TODO Save sensor data individually. Test to see the order in which the sensor data is saved in the flight data object.
# Then combine the sensors into a single csv and transform it to a format mathcing the KF code's expectations.

accel_noisy_nosecone.export_measured_data("Defiance_Simulation_accel_noisy_nosecone.csv")
gyro_noisy.export_measured_data("Defiance_Simulation_gyro_noisy.csv")
gnss_clean.export_measured_data("Defiance_Simulation_gnss_clean.csv")
gnss_ground_truth.export_measured_data("Defiance_Simulation_gnss_ground_truth.csv")