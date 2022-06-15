from configparser import ConfigParser, NoSectionError

def parse_parameters(config_file):
    config = ConfigParser()
    config.read(config_file)
    config_result = {}
    config_result["date"] = config["date"]["date"]
    config_result["boxsize"] = config['snap']['boxsize']
    config_result["ytfilename"] = config['snap']['ytfilename']
    config_result["caesarfilename"] = config['snap']['caesarfilename']
    config_result["mode"] = config['sample']['mode']
    if config_result["mode"] == "randomized":
        config_result["n_galaxies_sample"] = config['sample']['n_galaxies_sample']
        config_result["min_mass"] = config['sample']['min_mass']
        config_result["max_mass"] = config['sample']['max_mass']
    try:
        config_result["sbatch"] = {k: v for k, v in config.items("sbatch")}
    except NoSectionError:
        config_result["sbatch"] = {}
    return config_result