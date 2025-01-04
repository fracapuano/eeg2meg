import numpy as np
import warnings

class SensorProcessing:
    """
    Class for processing multi-channel real-world signals via:
      1) Robust scaling to [-1, +1],
      2) Mu-law companding,
      3) Quantization to mu-dependant levels.
    """

    def __init__(self, mu: float = 255.0):
        """
        Initialize the SensorProcessing class.

        :param mu: The mu parameter for mu-law companding (commonly 255).
        """
        self.mu = mu

    def robust_scale(
        self,
        input_data: np.ndarray,
        bottom_p: float = 1.0,
        top_p: float = 99.0
    ) -> np.ndarray:
        """
        Clips the input_data to the [bottom_p, top_p] percentile range,
        then scales to the range [-1, +1].

        :param input_data: (H, W) array of raw sensor data.
        :param bottom_p: Lower percentile for clipping (default 1st).
        :param top_p: Upper percentile for clipping (default 99th).
        :return: (H, W) array scaled to [-1, +1].
        """
        # Flatten to compute percentiles globally
        flat_data = input_data.flatten()
        low_val = np.percentile(flat_data, bottom_p)
        high_val = np.percentile(flat_data, top_p)

        # Clip to [low_val, high_val]
        clipped = np.clip(input_data, low_val, high_val)

        # Avoid division by zero if high_val == low_val
        if np.isclose(high_val, low_val, atol=1e-12):
            msg = (
                "WARNING: ranges too close, all zeros returned"
                f"low value: {low_val} / high value: {high_val}"
                f"absolute difference: {abs(low_val - high_val)}"
            )
            warnings.warn(
                msg,
                UserWarning,
                )

            # If the data has effectively no variation, just return zeros in [-1, +1]
            return np.zeros_like(input_data, dtype=np.float32)

        # Scale to [0, 1]
        scaled_0_1 = (clipped - low_val) / (high_val - low_val)

        # Map [0, 1] -> [-1, +1]
        scaled_minus1_1 = 2.0 * scaled_0_1 - 1.0

        return scaled_minus1_1.astype(np.float32)

    def mulaw(self, input_data: np.ndarray) -> np.ndarray:
        """
        Apply mu-law companding to an array in the range [-1, +1].

        :param input_data: (H, W) array of floats in [-1, +1].
        :return: (H, W) array after mu-law companding, also in [-1, +1].
        """
        # mu-law formula: f(x) = sign(x) * ln(1 + mu * |x|) / ln(1 + mu)
        # Ensure numerical stability by clipping to [-1, +1]
        input_data = np.clip(input_data, -1.0, 1.0)

        # Sign
        sign = np.sign(input_data)

        # Magnitude
        magnitude = np.abs(input_data)

        # Apply mu-law
        numerator = np.log1p(self.mu * magnitude)   # log(1 + mu*|x|)
        denominator = np.log1p(self.mu)             # log(1 + mu)
        companded = sign * (numerator / denominator)

        return companded.astype(np.float32)

    def quantize(self, input_data: np.ndarray) -> np.ndarray:
        """
        Quantize mu-law output from [-1, +1] into 8-bit discrete levels [0..255].

        :param input_data: (H, W) array in [-1, +1].
        :return: (H, W) array of uint8 discrete values in [0..255].
        """
        # 1) Map [-1, +1] -> [0, 1]
        mapped_0_1 = (input_data + 1.0) / 2.0

        # 2) Scale to [0, 255]
        scaled_0_255 = mapped_0_1 * 255.0

        # 3) Round and clip to ensure in [0..255], then convert to uint8
        discrete = np.round(scaled_0_255).clip(0, 255).astype(np.uint8)

        return discrete

    def to_discrete(
        self,
        input_data: np.ndarray,
        bottom_p: float = 1.0,
        top_p: float = 99.0
    ) -> np.ndarray:
        """
        Main processing method that:
          1. Robust-scales the input data to [-1, +1],
          2. Applies mu-law transform,
          3. Quantizes to 8-bit values.

        :param input_data: (H, W) raw sensor data.
        :param bottom_p: percentile for lower clipping in robust_scale.
        :param top_p: percentile for upper clipping in robust_scale.
        :return: (H, W) array of discrete values in [0..255].
        """
        # Step 1: Robust scale to [-1, +1]
        scaled = self.robust_scale(input_data, bottom_p, top_p)

        # Step 2: Mu-law
        companded = self.mulaw(scaled)

        # Step 3: Quantize to 8-bit [0, 255]
        discrete = self.quantize(companded)

        return discrete

