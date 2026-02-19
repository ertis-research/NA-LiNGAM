adabyron_renames = {
    'uma_adabyron_smartmeter_kwh_consumed': 'Consumption_Wh',
    'uma_adabyron_toshiba_AC_moda-north_activeEnergy': 'AC_A_North_Energy_Wh',
    'uma_adabyron_toshiba_AC_modb-north_activeEnergy': 'AC_B_North_Energy_Wh',
    'uma_adabyron_toshiba_AC_moda-south_activeEnergy': 'AC_A_South_Energy_Wh',
    'uma_adabyron_toshiba_AC_modb-south_activeEnergy': 'AC_B_South_Energy_Wh',
    'uma_adabyron_AirTreatmentUnit_climate_a_activeEnergy': 'Air_Treatment_Climate_A_Energy_Wh',
    'uma_adabyron_AirTreatmentUnit_climate_b_activeEnergy': 'Air_Treatment_Climate_B_Energy_Wh',
    'uma_adabyron_AirTreatmentUnit_heat_pump_activeEnergy': 'Air_Treatment_Heat_Pump_Energy_Wh',
    'uma_adabyron_weatherStation_dewPoint': 'Weather_Dew_Point',
    'uma_adabyron_weatherStation_relativeHumidity': 'Weather_Humidity',
    'uma_adabyron_weatherStation_solarRadiation': 'Weather_Solar_Radiation',
    'uma_adabyron_weatherStation_temperature': 'Weather_Temperature'
}

adabyron_simplified_vars = """
    Consumption_Wh

    AC_A_Energy_Wh
    AC_B_Energy_Wh

    Air_Treatment_Climate_A_Energy_Wh
    Air_Treatment_Climate_B_Energy_Wh
    Air_Treatment_Heat_Pump_Energy_Wh

    Weather_Dew_Point
    Weather_Humidity
    Weather_Solar_Radiation
    Weather_Temperature
""".split()

adabyron_posible_edges = [
    # --- Climate into building ---
    ("Weather_Dew_Point", "Air_Treatment_Heat_Pump_Energy_Wh"),
    ("Weather_Humidity", "Air_Treatment_Heat_Pump_Energy_Wh"),
    ("Weather_Humidity", "AC_A_Energy_Wh"),
    ("Weather_Humidity", "AC_B_Energy_Wh"),
    ("Weather_Solar_Radiation", "AC_A_Energy_Wh"),
    ("Weather_Solar_Radiation", "AC_B_Energy_Wh"),
    ("Weather_Temperature", "AC_A_Energy_Wh"),
    ("Weather_Temperature", "AC_B_Energy_Wh"),

    # --- Building to consumption ---
    ("Air_Treatment_Climate_A_Energy_Wh", "AC_A_Energy_Wh"),
    ("Air_Treatment_Climate_A_Energy_Wh", "Air_Treatment_Heat_Pump_Energy_Wh"),
    ("Air_Treatment_Climate_B_Energy_Wh", "AC_B_Energy_Wh"),
    ("Air_Treatment_Climate_B_Energy_Wh", "Air_Treatment_Heat_Pump_Energy_Wh"),
    ("AC_A_Energy_Wh", "Consumption_Wh"),
    ("AC_B_Energy_Wh", "Consumption_Wh"),
    ("Air_Treatment_Heat_Pump_Energy_Wh", "Consumption_Wh")
]