import numpy as np
from scipy import stats

ReadingTypeIdDict = {
    2001: 'KWH',
    2002: 'KW',
    2004: 'I',
    2019: 'IR',
    2020: 'IY',
    2021: 'IB',
    2047: 'ITHD',
    2026: 'ITHDR',
    2027: 'ITHDY',
    2028: 'ITHDB',
    2008: 'KVA',
    2022: 'KVAH',
    2051: 'INeutral',
    2012: 'VLL',
    2013: 'VLL_R',
    2014: 'VLL_Y',
    2015: 'VLL_B',
    2003: 'VLN',
    2016: 'VLN_R',
    2017: 'VLN_Y',
    2018: 'VLN_B',
    2006: 'PF',
    2046: 'VTHD',
    2023: 'VTHD_R',
    2024: 'VTHD_Y',
    2025: 'VTHD_B',
    2009: 'kVA_R',
    2010: 'kVA_Y',
    2011: 'kVA_B',
    2032: 'kW_R',
    2033: 'kW_Y',
    2041: 'kVAh (DELIVERED)',
    2038: 'POWER FACTOR (DELIVERED)',
    2005: 'FREQUENCY',
    2007: 'kVAR',
    2029: 'kVARh (INDUCTIVE RECEIVED)',
    2030: 'kVARh (CAPACITIVE RECEIVED)',
    2031: 'MAXIMUM DEMAND',
    2034: 'kW_B',
    2035: 'kVAR_R',
    2036: 'kVAR_Y',
    2037: 'kVAR_B',
    2039: 'CURRENT(DELIVERED)',
    2040: 'kWH (DELIVERED)',
    2044: 'POWER FACTOR (RECEIVED)',
    2048: 'PF_R',
    2049: 'PF_Y',
    2050: 'PF_B',
    2042: 'kVARh (INDUCTIVE DELIVERED)',
    2043: 'kVARh (CAPACITIVE DELIVERED)',
    2045: 'CURRENT (RECEIVED)',
    2052: 'RISING DEMAND',
    2053: 'FORECAST DEMAND',
    2596: 'Max_Apparent_Power',
    2692: 'Raw_PF',
    2693: 'Raw_PF_R',
    2694: 'Raw_PF_Y',
    2695: 'Raw_PF_B',


    # New Parameters for Motor analytics starting from 5000

    5000: 'Vp',    
    5001: 'Vn',
    5002: 'Vz',
    5003: 'Ip',
    5004: 'In',
    5005: 'Iz',
    5006: 'Zp',
    5007: 'Zn',
    5008: 'Zz',
    5009: 'PF_Mod_R',
    5010: 'PF_Mod_B',
    5011: 'PF_Mod_Y',
    5012: 'Error_R',
    5013: 'Error_B',
    5014: 'Error_Y',
    5015: 'KW_Exp',
    5035: 'I_Exp',
    5016: 'V_Imb_IEEE',
    5017: 'I_Imb_IEEE',
    5018: 'IUF',
    5019: 'VUF',
    5020: 'VLL_R_Min',
    5021: 'VLL_R_Max',
    5022: 'VLL_B_Min',
    5023: 'VLL_B_Max',
    5024: 'VLL_Y_Min',
    5025: 'VLL_Y_Max',
    5026: 'HealthIndex',
    5027: 'LoadingIndex',
    5028: 'Error_R_NN',
    5029: 'Error_B_NN',
    5030: 'Error_Y_NN',
    5031: 'OffTime',
    5032: 'OverLoadedTime',
    5033: 'UnderLoadedTime',
    5034: 'NormalLoadedTime',
    5039: 'V_Imb_NEEMA',
    5040: 'I_Imb_NEEMA',
    5041: 'Imb_Most_Deviant_Phase_Voltage',
    5042: 'Imb_Most_Deviant_Phase_Current',
    5043: 'THD_Most_Deviant_Phase_Voltage',
    5044: 'THD_Most_Deviant_Phase_Current',
    5045: 'V_Ratio',
    5046: 'I_Ratio',
    2110: 'Motor_RPM',
    2133: 'Set_Frequency',
    2864: 'Motor_Rotation_Count',
    2865: 'Heat_Sink_Temperature',
    2866: 'Torque_current_detection',
    2867: 'Excitation_current_detection',
    2868: 'U_ph_output_current',
    2869: 'V_ph_output_current',
    2870: 'W_ph_output_current',
    2871: 'Internal_DC_Link_voltage', 
    2863: 'Motor_speed_in_min-1' 
}


AGG_DICT = {
    'KWH': lambda x: np.max(x) - np.min(x),
    'KW': np.average,
    'I': np.average,
    'IR': np.average,
    'IY': np.average,
    'IB': np.average,
    'ITHD': np.average,
    'ITHDR': np.average,
    'ITHDY': np.average,
    'ITHDB': np.average,
    'KVA': np.average,
    'KVAH': lambda x: np.max(x) - np.min(x),
    'INeutral': np.average,
    'VLL': np.average,
    'VLL_R': np.average,
    'VLL_Y': np.average,
    'VLL_B': np.average,
    'VLN': np.average,
    'VLN_R': np.average,
    'VLN_Y': np.average,
    'VLN_B': np.average,
    'PF': np.average,
    'VTHD': np.average,
    'VTHD_R': np.average,
    'VTHD_Y': np.average,
    'VTHD_B': np.average,
    'kVA_R': np.average,
    'kVA_Y': np.average,
    'kVA_B': np.average,
    'kW_R': np.average,
    'kW_Y': np.average,
    'kVAh (DELIVERED)': lambda x: np.max(x) - np.min(x),
    'POWER FACTOR (DELIVERED)': np.average,
    'FREQUENCY': np.average,
    'kVAR': np.average,
    'kVARh (INDUCTIVE RECEIVED)': lambda x: np.max(x) - np.min(x),
    'kVARh (CAPACITIVE RECEIVED)': lambda x: np.max(x) - np.min(x),
    'MAXIMUM DEMAND': np.average,
    'kW_B': np.average,
    'kVAR_R': np.average,
    'kVAR_Y': np.average,
    'kVAR_B': np.average,
    'CURRENT(DELIVERED)': np.average,
    'kWH (DELIVERED)': lambda x: np.max(x) - np.min(x),
    'POWER FACTOR (RECEIVED)': np.average,
    'PF_R': np.average,
    'PF_Y': np.average,
    'PF_B': np.average,
    'kVARh (INDUCTIVE DELIVERED)': lambda x: np.max(x) - np.min(x),
    'kVARh (CAPACITIVE DELIVERED)': lambda x: np.max(x) - np.min(x),
    'CURRENT (RECEIVED)': np.average,
    'RISING DEMAND': np.average,
    'FORECAST DEMAND': np.average,

    'Vp': np.average,    
    'Vn': np.average,
    'Vz': np.average,
    'Ip': np.average,
    'In': np.average,
    'Iz': np.average,
    'Zp': np.average,
    'Zn': np.average,
    'Zz': np.average,
    'PF_Mod_R': np.average,
    'PF_Mod_B': np.average,
    'PF_Mod_Y': np.average,
    'Error_R': np.average,
    'Error_B': np.average,
    'Error_Y': np.average,
    'KW_Exp': np.average,
    'I_Exp': np.average,
    'V_Imb_IEEE': np.average,
    'I_Imb_IEEE': np.average,
    'IUF': np.average,
    'VUF': np.average,
    'VLL_R_Min': np.min, 
    'VLL_R_Max': np.max,
    'VLL_B_Min': np.min,
    'VLL_B_Max': np.max,
    'VLL_Y_Min': np.min,
    'VLL_Y_Max': np.max,
    'V_Imb_NEEMA': np.average,
    'I_Imb_NEEMA': np.average,
    'Imb_Most_Deviant_Phase_Voltage': stats.mode,
    'Imb_Most_Deviant_Phase_Current': stats.mode,
    'THD_Most_Deviant_Phase_Voltage': stats.mode,
    'THD_Most_Deviant_Phase_Current': stats.mode,

    '2001': lambda x: np.max(x) - np.min(x),
    '2002': np.average,
    '2004': np.average,
    '2019': np.average,
    '2020': np.average,
    '2021': np.average,
    '2047': np.average,
    '2026': np.average,
    '2027': np.average,
    '2028': np.average,
    '2008': np.average,
    '2022': lambda x: np.max(x) - np.min(x),
    '2051': np.average,
    '2012': np.average,
    '2013': np.average,
    '2014': np.average,
    '2015': np.average,
    '2003': np.average,
    '2016': np.average,
    '2017': np.average,
    '2018': np.average,
    '2006': np.average,
    '2046': np.average,
    '2023': np.average,
    '2024': np.average,
    '2025': np.average,
    '2009': np.average,
    '2010': np.average,
    '2011': np.average,
    '2032': np.average,
    '2033': np.average,
    '2041': lambda x: np.max(x) - np.min(x),
    '2038': np.average,
    '2005': np.average,
    '2007': np.average,
    '2029': lambda x: np.max(x) - np.min(x),
    '2030': lambda x: np.max(x) - np.min(x),
    '2031': np.average,
    '2034': np.average,
    '2035': np.average,
    '2036': np.average,
    '2037': np.average,
    '2039': np.average,
    '2040': lambda x: np.max(x) - np.min(x),
    '2044': np.average,
    '2048': np.average,
    '2049': np.average,
    '2050': np.average,
    '2042': lambda x: np.max(x) - np.min(x),
    '2043': lambda x: np.max(x) - np.min(x),
    '2045': np.average,
    '2052': np.average,
    '2053': np.average,

    # New Parameters for Motor analytics starting from 5000

    '5000': np.average,
    '5001': np.average,
    '5002': np.average,
    '5003': np.average,
    '5004': np.average,
    '5005': np.average,
    '5006': np.average,
    '5007': np.average,
    '5008': np.average,
    '5009': np.average,
    '5010': np.average,
    '5011': np.average,
    '5012': np.average,
    '5013': np.average,
    '5014': np.average,
    '5015': np.average,
    '5016': np.average,
    '5017': np.average,
    '5018': np.average,
    '5019': np.average,

    '2013': np.min,
    '2013': np.max,
    '2015': np.min,
    '2015': np.max,
    '2014': np.min,
    '2014': np.max


}

ReadingTypeIdDictInv = {
    'KWH': 2001,
    'KW': 2002,
    'I': 2004,
    'IR': 2019,
    'IY': 2020,
    'IB': 2021,
    'ITHD': 2047,
    'ITHDR': 2026,
    'ITHDY': 2027,
    'ITHDB': 2028,
    'KVA': 2008,
    'KVAH': 2022,
    'INeutral': 2051,
    'VLL': 2012,
    'VLL_R': 2013,
    'VLL_Y': 2014,
    'VLL_B': 2015,
    'VLN': 2003,
    'VLN_R': 2016,
    'VLN_Y': 2017,
    'VLN_B': 2018,
    'PF': 2006,
    'VTHD': 2046,
    'VTHD_R': 2023,
    'VTHD_Y': 2024,
    'VTHD_B': 2025,
    'kVA_R': 2009,
    'kVA_Y': 2010,
    'kVA_B': 2011,
    'kW_R': 2032,
    'kW_Y': 2033,
    'kVAh (DELIVERED)': 2041,
    'POWER FACTOR (DELIVERED)': 2038,
    'FREQUENCY': 2005,
    'kVAR': 2007,
    'kVARh (INDUCTIVE RECEIVED)': 2029,
    'kVARh (CAPACITIVE RECEIVED)': 2030,
    'MAXIMUM DEMAND': 2031,
    'kW_B': 2034,
    'kVAR_R': 2035,
    'kVAR_Y': 2036,
    'kVAR_B': 2037,
    'CURRENT(DELIVERED)': 2039,
    'kWH (DELIVERED)': 2040,
    'POWER FACTOR (RECEIVED)': 2044,
    'PF_R': 2048,
    'PF_Y': 2049,
    'PF_B': 2050,
    'kVARh (INDUCTIVE DELIVERED)': 2042,
    'kVARh (CAPACITIVE DELIVERED)': 2043,
    'CURRENT (RECEIVED)': 2045,
    'RISING DEMAND': 2052,
    'FORECAST DEMAND': 2053,

    'Vp': 5000,
    'Vn': 5001,
    'Vz': 5002,
    'Ip': 5003,
    'In': 5004,
    'Iz': 5005,
    'Zp': 5006,
    'Zn': 5007,
    'Zz': 5008,
    'PF_Mod_R': 5009,
    'PF_Mod_B': 5010,
    'PF_Mod_Y': 5011,
    'Error_R': 5012,
    'Error_B': 5013,
    'Error_Y': 5014,
    'KW_Exp': 5015,
    'V_Imb': 5016,
    'I_Imb': 5017,
    'IUF': 5018,
    'VUF': 5019,
    'I_Exp': 5035,

    'HealthIndex': 5026,
    'LoadingIndex': 5027,

    'VLL_R_Min': 5020,
    'VLL_R_Max': 5021,
    'VLL_B_Min': 5022,
    'VLL_B_Max': 5023,
    'VLL_Y_Min': 5024,
    'VLL_Y_Max': 5025,

    'Error_R_NN': 5028, 
    'Error_B_NN': 5029, 
    'Error_Y_NN': 5030,

    'OffTime': 5031,
    'OverLoadedTime': 5032,
    'UnderLoadedTime': 5033,
    'NormalLoadedTime': 5034
}

StatIdDict = {

    5000: 5,
    5001: 5,
    5002: 5,
    5003: 5,
    5004: 5,
    5005: 5,
    5006: 5,
    5007: 5,
    5008: 5,
    5009: 5,
    5010: 5,
    5011: 5,
    5012: 5,
    5013: 5,
    5014: 5,
    5015: 5,
    5016: 5,
    5017: 5,
    5018: 5,
    5019: 5,
    5020: 7,
    5021: 6,
    5022: 7,
    5023: 6,
    5024: 7,
    5025: 6,
    5026: 5,
    5027: 5,
    5028: 5,
    5029: 5,
    5030: 5,
    5031: 5,
    5032: 5,
    5033: 5,
    5034: 5,
}
