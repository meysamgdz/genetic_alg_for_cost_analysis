import numpy as np

def calculate_noise(band_width: float = 40e6) -> float:
    """
    Calculate thermal noise power in dBm.

    Args:
        band_width (float): Bandwidth in Hz (default: 40 MHz)

    Returns:
        float: Noise power in dBm
    """
    temperature = 300  # 275 + 25°C
    boltzmanns_constant = 1.38e-23  # m^2.kg.s^(-2).K^(-1)
    noise_figure = 8  # dB
    noise_figure_linear = 10 ** (noise_figure / 10)
    thermal_noise = band_width * temperature * boltzmanns_constant * noise_figure_linear  # Watt
    return 10 * np.log10(thermal_noise * 1e3)


def calculate_threshold(tx_power: float, noise: float, localization_acc: float = 0.5) -> float:
    """
    Calculate the path loss threshold for required localization accuracy.

    Args:
        tx_power (float): Transmit power in dBm
        noise (float): Noise power in dBm
        localization_acc (float): Required localization accuracy in meters

    Returns:
        float: Path loss threshold in dB
    """
    req_snr = 62 * np.exp(-localization_acc)
    pl_threshold = tx_power - noise - req_snr
    return pl_threshold