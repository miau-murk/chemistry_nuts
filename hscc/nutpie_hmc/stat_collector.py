import arviz as az

class Statistics:
    def __init__(self, inferencedata=None):
        """
        Initialize Statistics with optional arviz.InferenceData object.
        """
        if inferencedata is not None:
            self.data = inferencedata
        else:
            self.data = None

    def from_trace(self, trace):
        """
        Collect statistics from nutpie/PyMC trace and convert to arviz.InferenceData.
        Automatically includes all sample_stats available.
        """
        self.data = az.from_nutpie(trace)
        # Optionally: Check/convert mass_matrix or other custom fields

    def save_to_nc(self, filename):
        """
        Save all statistics to NetCDF file (.nc).
        """
        if self.data is None:
            raise ValueError("No inference data available to save.")
        self.data.to_netcdf(filename)

    def load_from_nc(self, filename):
        """
        Load statistics from NetCDF file (.nc).
        """
        self.data = az.InferenceData.from_netcdf(filename)

    def get_stat(self, stat_name):
        """
        Return given sample statistic (e.g. step_size, mass_matrix, gradient, divergence).
        """
        if self.data is None:
            raise ValueError("No inference data loaded.")
        if hasattr(self.data, "sample_stats") and hasattr(self.data.sample_stats, stat_name):
            return getattr(self.data.sample_stats, stat_name)
        else:
            raise AttributeError(f"Statistic {stat_name} not found.")

    def get_all_stats(self):
        """
        Return dictionary with all sample_stats available in trace.
        """
        if self.data is None:
            raise ValueError("No inference data loaded.")
        # Get all stats as dict of xarray objects
        if hasattr(self.data, "sample_stats"):
            return {k: getattr(self.data.sample_stats, k) for k in self.data.sample_stats.data_vars}
        else:
            return {}

