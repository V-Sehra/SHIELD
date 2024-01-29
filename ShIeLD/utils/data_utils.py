

tissue_type_name = ['normalLiver', 'core', 'rim']
tissue_dict = {'normalLiver': 0,
               'core': 1,
               'rim': 2}

def get_tissue_type_name(tissue_type_id):
    return tissue_type_name[tissue_type_id]

def get_tissue_type_id(tissue_type_name):
    return tissue_dict[tissue_type_name]

def turn_pixel_to_meter(pixel_radius):
    pixel_to_miliMeter_factor = 2649.291339
    mycro_meter_radius = pixel_radius * (10**3/pixel_to_miliMeter_factor)
    return(round(mycro_meter_radius))